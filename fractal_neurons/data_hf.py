from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Iterator, Dict, Any

import torch
from torch.utils.data import IterableDataset, Dataset
from .progress import Progress


@dataclass
class HFDatasetConfig:
    path: str           # e.g., "openwebtext", "wikipedia", "c4"
    name: Optional[str] = None  # e.g., "20220301.en" for wikipedia
    split: str = "train"
    streaming: bool = True
    text_field: Optional[str] = None  # if None, tries common defaults
    seq_len: int = 512
    return_text: bool = False  # when True, yield raw strings for external tokenization
    verbose: bool = False


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "The 'datasets' library is required for HF datasets.\n"
            "Install with: pip install datasets\n"
            "Optionally set HF_DATASETS_CACHE to control cache location."
        ) from e


def _load_hf_iter(cfg: HFDatasetConfig):
    from datasets import load_dataset

    ds = load_dataset(cfg.path, cfg.name, split=cfg.split, streaming=cfg.streaming)
    return ds


def _detect_text(obj: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in obj:
        return obj[preferred]
    # common field names
    for k in ("text", "content", "article", "paragraph", "document"):
        if k in obj and isinstance(obj[k], str):
            return obj[k]
    # concatenate all string fields as fallback
    parts = [str(v) for v in obj.values() if isinstance(v, str)]
    return "\n".join(parts) if parts else None


class HFDatasetStream(IterableDataset):
    def __init__(self, cfg: HFDatasetConfig):
        super().__init__()
        _require_datasets()
        self.cfg = cfg

    def __iter__(self) -> Iterator[torch.Tensor]:
        ds = _load_hf_iter(self.cfg)
        if self.cfg.verbose:
            print(f"[hf] streaming={self.cfg.streaming} path={self.cfg.path} name={self.cfg.name} split={self.cfg.split} return_text={self.cfg.return_text}")
        if self.cfg.return_text:
            for ex in ds:
                text = _detect_text(ex, self.cfg.text_field)
                if not text:
                    continue
                yield text
        else:
            buf = bytearray()
            seq_len = self.cfg.seq_len
            for ex in ds:
                text = _detect_text(ex, self.cfg.text_field)
                if not text:
                    continue
                buf.extend(text.encode("utf-8", errors="replace"))
                # emit fixed windows
                while len(buf) >= seq_len:
                    chunk = bytes(buf[:seq_len])
                    del buf[:seq_len]
                    yield torch.tensor(list(chunk), dtype=torch.long)


class HFDatasetMap(Dataset):
    def __init__(self, cfg: HFDatasetConfig):
        super().__init__()
        _require_datasets()
        from datasets import load_dataset
        self.cfg = cfg
        self.ds = load_dataset(cfg.path, cfg.name, split=cfg.split, streaming=False)
        # Precompute windows by indices (approximate by sample)
        # Here we build a list of (sample_index, offset) pairs by slicing each sample text
        self.index = []
        total = len(self.ds)
        prog = Progress(total=total or 1, name="hf index")
        for i in range(total):
            rec = self.ds[i]
            text = _detect_text(rec, cfg.text_field)
            if not text:
                prog.update(i + 1)
                continue
            b = text.encode("utf-8", errors="replace")
            for off in range(0, max(1, len(b)), cfg.seq_len):
                self.index.append((i, off))
            prog.update(i + 1, info=f"windows={len(self.index)}")
        prog.finalize()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        i, off = self.index[idx]
        rec = self.ds[i]
        text = _detect_text(rec, self.cfg.text_field) or ""
        b = text.encode("utf-8", errors="replace")
        chunk = b[off:off + self.cfg.seq_len]
        t = torch.zeros(self.cfg.seq_len, dtype=torch.long)
        if chunk:
            n = min(len(chunk), self.cfg.seq_len)
            t[:n] = torch.tensor(list(chunk[:n]), dtype=torch.long)
        return t
