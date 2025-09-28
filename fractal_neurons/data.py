from __future__ import annotations

import fnmatch
import glob
import os
from dataclasses import dataclass
import random
from typing import Iterable, List, Tuple, Iterator

import torch
from torch.utils.data import Dataset
from .progress import Progress, fmt_duration


DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    "**/.cache/**",
    "**/.venv/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/*.bin",
    "**/*.so",
    "**/*.dylib",
    "**/*.o",
    # Skip volatile/special FS by default
    "/proc/**",
    "/sys/**",
    "/dev/**",
    "/run/**",
    "/tmp/**",
]


def _match_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _list_files(include_globs: List[str], exclude_globs: List[str]) -> List[str]:
    files: List[str] = []
    for pat in include_globs:
        files.extend(glob.glob(os.path.expanduser(pat), recursive=True))
    # Keep only files
    files = [f for f in files if os.path.isfile(f)]
    # Exclude
    ex = [os.path.expanduser(p) for p in (exclude_globs or [])]
    out = []
    for f in files:
        if _match_any(f, ex):
            continue
        out.append(f)
    return sorted(set(out))


def _walk_all_files(root: str, exclude_globs: List[str], max_size_bytes: int, verbose: bool = False) -> List[str]:
    root = os.path.expanduser(root or "/")
    ex = [os.path.expanduser(p) for p in (exclude_globs or [])]
    out: List[str] = []
    scanned = 0
    t0 = os.path.getmtime if False else None  # placeholder to avoid linter; not used
    import time as _time
    t_last = _time.time()
    t_start = t_last
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded directories in-place for efficiency
        dirnames[:] = [d for d in dirnames if not _match_any(os.path.join(dirpath, d), ex)]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if _match_any(fp, ex):
                continue
            try:
                if not os.path.isfile(fp):
                    continue
                size = os.path.getsize(fp)
                if size < 0 or size > max_size_bytes:
                    continue
                out.append(fp)
                scanned += 1
                if verbose:
                    now = _time.time()
                    if now - t_last >= 2.0:
                        elapsed = now - t_start
                        rate = scanned / max(elapsed, 1e-6)
                        print(f"[scan] files scanned={scanned} matched={len(out)} elapsed={fmt_duration(elapsed)} rate={rate:.1f}/s")
                        t_last = now
            except OSError:
                continue
    return sorted(set(out))


@dataclass
class SystemDataConfig:
    include_globs: List[str]
    exclude_globs: List[str]
    seq_len: int = 512
    stride: int = 512
    allow_all: bool = False
    root: str = "/"         # used when allow_all is true
    max_file_mb: int = 16    # skip files larger than this
    verbose: bool = False
    max_index_items: int = 500_000  # cap to avoid huge indices on full system scans


class SystemByteDataset(Dataset):
    """
    Byte-level dataset over whitelisted local files. Builds an index of (file_path, offset) windows.
    """

    def __init__(self, cfg: SystemDataConfig):
        exclude = (cfg.exclude_globs or []) + DEFAULT_EXCLUDES
        max_bytes = int(cfg.max_file_mb) * 1024 * 1024
        if cfg.allow_all:
            root = os.path.expanduser(cfg.root or "/")
            if not os.path.isdir(root):
                raise RuntimeError(f"Root directory not found or not a directory: {root}")
            self.files = _walk_all_files(root, exclude, max_bytes, verbose=cfg.verbose)
        else:
            if not cfg.include_globs:
                raise RuntimeError("include_globs must be provided when allow_all is false")
            self.files = _list_files(cfg.include_globs, exclude)
        if not self.files:
            mode = "root scan" if cfg.allow_all else "include_globs"
            raise RuntimeError(f"No files found for {mode} after excludes or size limits. Check paths/excludes/limits.")
        if cfg.verbose:
            print(f"[dataset] system files: {len(self.files)} matched; seq_len={cfg.seq_len} stride={cfg.stride}")
        self.seq_len = int(cfg.seq_len)
        self.stride = int(cfg.stride)
        self.index: List[Tuple[str, int]] = []
        max_items = int(getattr(cfg, "max_index_items", 500_000))
        seen = 0
        # Pre-compute total windows for progress (when verbose)
        total_windows = 0
        if cfg.verbose:
            for f in self.files:
                try:
                    size = os.path.getsize(f)
                except OSError:
                    continue
                if size < 0:
                    continue
                total_windows += max(1, (size + self.stride - 1) // self.stride)
            prog = Progress(total=total_windows or 1, name="index")
        else:
            prog = None  # type: ignore
        for f in self.files:
            try:
                size = os.path.getsize(f)
            except OSError:
                continue
            if size <= 0:
                continue
            # Windows with stride
            for off in range(0, max(1, size), self.stride):
                if len(self.index) < max_items:
                    self.index.append((f, off))
                else:
                    # reservoir sampling to keep a bounded, roughly uniform sample
                    j = random.randint(0, seen)
                    if j < max_items:
                        self.index[j] = (f, off)
                seen += 1
                if prog is not None:
                    prog.update(seen, info=f"kept={len(self.index)} cap={max_items}")
        if prog is not None:
            prog.finalize()
        if cfg.verbose:
            cap_info = " (capped)" if len(self.index) >= max_items else ""
            print(f"[dataset] system windows indexed: {len(self.index)}{cap_info}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        path, off = self.index[i]
        with open(path, "rb") as fh:
            fh.seek(off)
            data = fh.read(self.seq_len)
        b = torch.zeros(self.seq_len, dtype=torch.long)
        if data:
            n = min(len(data), self.seq_len)
            b[:n] = torch.tensor(list(data[:n]), dtype=torch.long)
        return b  # [T] values in 0..255


class MaskedByteCollator:
    def __init__(self, mask_rate: float = 0.15, vocab_size: int = 257, mask_id: int = 256):
        self.mask_rate = mask_rate
        self.vocab_size = vocab_size
        self.mask_id = mask_id

    def __call__(self, batch: List[torch.Tensor]):
        # batch: list of [T] tensors
        x = torch.stack(batch, dim=0)  # [B, T]
        B, T = x.shape
        # Random mask
        mask = torch.rand(B, T) < self.mask_rate
        targets = x.clone()
        inputs = x.clone()
        inputs[mask] = self.mask_id
        return {
            "tokens": inputs,  # [B, T]
            "targets": targets,  # [B, T]
            "loss_mask": mask,   # [B, T] bool
        }


class MaskedTokenCollator:
    def __init__(self, tokenizer, seq_len: int, mask_rate: float = 0.15):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.mask_rate = mask_rate

    def __call__(self, batch: List[object]):
        # batch of strings
        if isinstance(batch[0], torch.Tensor):
            # bytes already tokenized elsewhere
            x = torch.stack(batch, dim=0)
        else:
            ids_list = [self.tok.encode(str(text), self.seq_len) for text in batch]
            x = torch.tensor(ids_list, dtype=torch.long)
        B, T = x.shape
        mask = torch.rand(B, T) < self.mask_rate
        targets = x.clone()
        inputs = x.clone()
        if getattr(self.tok, "mask_id", None) is not None:
            inputs[mask] = int(self.tok.mask_id)
        else:
            # Replace masked positions with random tokens when no mask token exists
            rand = torch.randint(low=0, high=int(self.tok.vocab_size), size=(B, T))
            inputs[mask] = rand[mask]
        return {"tokens": inputs, "targets": targets, "loss_mask": mask}
