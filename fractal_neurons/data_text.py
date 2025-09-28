from __future__ import annotations

import glob
import os
import random
import time
from typing import Iterator, List, Optional
import fnmatch
from torch.utils.data import get_worker_info
from multiprocessing import Value, Lock
from hashlib import sha1


def _hash_key(root: str, includes: List[str], excludes: List[str], recursive: bool) -> str:
    h = sha1()
    h.update(os.path.abspath(os.path.expanduser(root)).encode())
    for p in sorted(includes or []):
        h.update(b"|i|")
        h.update(p.encode())
    for p in sorted(excludes or []):
        h.update(b"|e|")
        h.update(p.encode())
    h.update(b"|r|")
    h.update(str(bool(recursive)).encode())
    return h.hexdigest()[:16]

from torch.utils.data import IterableDataset
from .progress import Progress
from .tokenizer import TokenizerConfig, build_tokenizer
import torch


class TextDirStream(IterableDataset):
    """
    Stream text lines from a directory tree, matching globs.
    Suitable for use with MaskedTokenCollator and an HF tokenizer.
    """

    def __init__(self, root: str, include_globs: Optional[List[str]] = None, exclude_globs: Optional[List[str]] = None,
                 encoding: str = "utf-8", recursive: bool = True, verbose: bool = False,
                 use_cache: bool = True, cache_dir: Optional[str] = None, refresh_cache: bool = False,
                 shuffle_buffer: int = 0, reshuffle_each_epoch: bool = True, shuffle_seed: Optional[int] = None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.include_globs = include_globs or [
            "**/*.txt", "**/*.md", "**/*.jsonl", "**/*.json", "**/*.log", "**/*.py"
        ]
        self.exclude_globs = exclude_globs or ["**/.git/**", "**/.cache/**", "**/node_modules/**", "**/__pycache__/**"]
        self.encoding = encoding
        self.recursive = recursive
        self.verbose = verbose
        self.use_cache = bool(use_cache)
        self.refresh_cache = bool(refresh_cache)
        self.cache_dir = os.path.join(self.root, ".fractal_cache") if cache_dir is None else os.path.expanduser(cache_dir)
        self.shuffle_buffer = max(0, int(shuffle_buffer or 0))
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self.shuffle_seed = shuffle_seed if shuffle_seed is None else int(shuffle_seed)
        # Build the file list once in the main process; workers will inherit
        self.files: List[str] = self._load_or_scan()
        self._total = len(self.files)
        # Shared progress counter across workers for coherent ETA
        try:
            self._counter = Value('i', 0)
            self._lock = Lock()
        except Exception:
            self._counter = None
            self._lock = None

    def _list_files(self) -> List[str]:
        """
        Fast, single-walk file listing with include/exclude globs.
        Avoids multiple glob() passes and prunes directories early.
        """
        root = self.root
        includes = [os.path.join(root, p) for p in (self.include_globs or ["**/*"]) ]
        excludes = [os.path.join(root, p) for p in (self.exclude_globs or [])]
        out: List[str] = []
        # Walk once, prune excluded dirs in-place
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune directories aggressively using excludes
            dir_full = [os.path.join(dirpath, d) for d in dirnames]
            keep_dirs = []
            for d, full in zip(dirnames, dir_full):
                if any(fnmatch.fnmatch(full, pat) for pat in excludes):
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs

            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if not os.path.isfile(fp):
                    continue
                if excludes and any(fnmatch.fnmatch(fp, pat) for pat in excludes):
                    continue
                if includes and not any(fnmatch.fnmatch(fp, pat) for pat in includes):
                    continue
                out.append(fp)
        return sorted(set(out))

    def _cache_path(self) -> Optional[str]:
        if not self.use_cache:
            return None
        key = _hash_key(self.root, self.include_globs, self.exclude_globs, self.recursive)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(self.cache_dir, f"textdir_files_{key}.txt")

    def _load_or_scan(self) -> List[str]:
        cp = self._cache_path()
        files: List[str] = []
        if cp and (not self.refresh_cache) and os.path.isfile(cp):
            try:
                with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            files.append(line)
                if files:
                    if self.verbose:
                        print(f"[textdir] loaded cached file list: {len(files)} from {cp}")
                    return files
                if self.verbose:
                    print(f"[textdir] cache empty at {cp}, re-scanning {self.root}")
            except Exception:
                files = []
        # Build fresh and write cache (only from main or worker 0)
        wi = get_worker_info()
        files = self._list_files()
        if not files:
            includes = ", ".join(self.include_globs or []) or "<none>"
            excludes = ", ".join(self.exclude_globs or []) or "<none>"
            raise RuntimeError(
                "textdir: no files matched. Set data.text_root to a directory with text files "
                f"and adjust data.text_globs if needed. root={self.root} includes={includes} excludes={excludes}"
            )
        if cp and (wi is None or wi.id == 0):
            try:
                tmp = cp + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(f"# root: {self.root}\n")
                    f.write(f"# recursive: {self.recursive}\n")
                    f.write(f"# include_globs: {self.include_globs}\n")
                    f.write(f"# exclude_globs: {self.exclude_globs}\n")
                    for p in files:
                        f.write(p + "\n")
                os.replace(tmp, cp)
                if self.verbose:
                    print(f"[textdir] cached file list: {len(files)} to {cp}")
            except Exception:
                pass
        return files

    def __iter__(self) -> Iterator[str]:
        files = self.files
        # Shard across PyTorch DataLoader workers if present
        wi = get_worker_info()
        master_log = False
        if wi is not None and wi.num_workers > 1:
            files = files[wi.id :: wi.num_workers]
            master_log = (wi.id == 0)
        else:
            master_log = True

        if master_log and self.verbose:
            print(f"[textdir] files: {self._total} under {self.root}")
        prog = Progress(total=self._total or 1, name="textdir read") if (master_log and self.verbose) else None
        buf_size = self.shuffle_buffer
        buffer: List[str] = []
        rng = None
        if buf_size > 0:
            base_seed = self.shuffle_seed if self.shuffle_seed is not None else int(time.time() * 1000)
            if self.reshuffle_each_epoch:
                base_seed += int(time.time() * 1000) & 0xFFFF
            if wi is not None:
                base_seed += wi.id
            rng = random.Random(base_seed)

        def _emit(line: str):
            nonlocal buffer
            if buf_size > 0:
                buffer.append(line)
                if len(buffer) >= buf_size:
                    if rng is not None:
                        rng.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
            else:
                yield line
        processed_local = 0
        MAX_WHOLE_READ = 32 * 1024 * 1024  # 32MB threshold for whole-file reads
        for path in files:
            try:
                # Handle jsonl vs plain text
                if path.endswith(".jsonl"):
                    try:
                        if os.path.getsize(path) <= MAX_WHOLE_READ:
                            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                                data = f.read()
                            lines_iter = data.splitlines()
                        else:
                            lines_iter = None
                    except Exception:
                        lines_iter = None
                    if lines_iter is None:
                        f = open(path, "r", encoding=self.encoding, errors="replace")
                        lines_iter = f
                    for raw in lines_iter:
                        line = raw.strip()
                        if not line:
                            continue
                        # naive extraction of a 'text' field if present
                        if line.startswith("{") and '"text"' in line:
                            try:
                                import json
                                obj = json.loads(line)
                                txt = obj.get("text")
                                if isinstance(txt, str) and txt.strip():
                                    yield from _emit(txt)
                                    continue
                            except Exception:
                                pass
                        yield from _emit(line)
                elif path.endswith(".json"):
                    try:
                        import json
                        with open(path, "r", encoding=self.encoding, errors="replace") as f:
                            obj = json.load(f)
                            if isinstance(obj, dict):
                                txt = obj.get("text") or obj.get("content") or obj.get("body")
                                if isinstance(txt, str) and txt.strip():
                                    yield from _emit(txt)
                            elif isinstance(obj, list):
                                for item in obj:
                                    if isinstance(item, dict):
                                        txt = item.get("text") or item.get("content") or item.get("body")
                                        if isinstance(txt, str) and txt.strip():
                                            yield from _emit(txt)
                    except Exception:
                        # Fallback to raw text read
                        try:
                            if os.path.getsize(path) <= MAX_WHOLE_READ:
                                with open(path, "r", encoding=self.encoding, errors="replace") as f:
                                    data = f.read()
                                for line in data.splitlines():
                                    line = line.strip()
                                    if line:
                                        yield from _emit(line)
                            else:
                                with open(path, "r", encoding=self.encoding, errors="replace") as f:
                                    for line in f:
                                        s = line.strip()
                                        if s:
                                            yield from _emit(s)
                        except Exception:
                            pass
                else:
                    # Fast path: read whole file then splitlines, with size guard
                    try:
                        if os.path.getsize(path) <= MAX_WHOLE_READ:
                            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                                data = f.read()
                            for line in data.splitlines():
                                line = line.strip()
                                if line:
                                    yield from _emit(line)
                        else:
                            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                                for line in f:
                                    s = line.strip()
                                    if s:
                                        yield from _emit(s)
                    except Exception:
                        pass
            except OSError:
                continue
            finally:
                processed_local += 1
                # Update shared counter
                if self._counter is not None and self._lock is not None:
                    try:
                        with self._lock:
                            self._counter.value += 1
                            n_all = int(self._counter.value)
                    except Exception:
                        n_all = processed_local
                else:
                    n_all = processed_local
                if prog is not None:
                    prog.update(n_all)
        if buffer:
            if rng is not None:
                rng.shuffle(buffer)
            for item in buffer:
                yield item
            buffer = []
        if prog is not None:
            prog.finalize()

    # Indicate to the training loop that we implement worker sharding
    supports_workers: bool = True


class TextDirTokenizedStream(IterableDataset):
    """
    Like TextDirStream but performs HF tokenization inside DataLoader workers.
    Yields torch.LongTensor token ids of shape [seq_len].
    """
    def __init__(self, root: str, seq_len: int, tokenizer_name_or_path: str, truncation_side: str = "right",
                 include_globs: Optional[List[str]] = None, exclude_globs: Optional[List[str]] = None,
                 encoding: str = "utf-8", recursive: bool = True, verbose: bool = False,
                 use_cache: bool = True, cache_dir: Optional[str] = None, refresh_cache: bool = False,
                 shuffle_buffer: int = 0, reshuffle_each_epoch: bool = True, shuffle_seed: Optional[int] = None):
        super().__init__()
        # Reuse TextDirStream for file discovery and caching
        self.reader = TextDirStream(
            root=root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            encoding=encoding,
            recursive=recursive,
            verbose=verbose,
            use_cache=use_cache,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            shuffle_buffer=shuffle_buffer,
            reshuffle_each_epoch=reshuffle_each_epoch,
            shuffle_seed=shuffle_seed,
        )
        self.seq_len = int(seq_len)
        self.tok_cfg = TokenizerConfig(type="hf", name_or_path=tokenizer_name_or_path, truncation_side=truncation_side)
        self._tok = None
        self.verbose = verbose

    def _tok_or_init(self):
        if self._tok is None:
            self._tok = build_tokenizer(self.tok_cfg)
        return self._tok

    def __iter__(self) -> Iterator[torch.Tensor]:
        tok = self._tok_or_init()
        for text in self.reader.__iter__():
            # text may be a string; convert and encode
            try:
                ids = tok.encode(str(text), self.seq_len)
                yield torch.tensor(ids, dtype=torch.long)
            except Exception:
                # Skip problematic lines
                continue

    supports_workers: bool = True
