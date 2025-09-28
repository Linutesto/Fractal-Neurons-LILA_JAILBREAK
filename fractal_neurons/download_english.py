# -*- coding: utf-8 -*-
"""
English text downloader with safe auto-subset handling and trust_remote_code.
Usage (examples):
  python -m fractal_neurons.download_english \
    --dataset c4 --subset en --split train --out data/english_corpus

  python -m fractal_neurons.download_english \
    --dataset bookcorpusopen --split train --out data/english_corpus
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Iterable, Dict, Any

from datasets import (
    load_dataset,
    get_dataset_config_names,
    Dataset,
    DatasetDict,
    IterableDataset,
)


# ------------------------------- helpers ------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _auto_text_key(example: Dict[str, Any]) -> Optional[str]:
    """Pick the first likely text field from an example."""
    if not example:
        return None
    # common priorities
    for k in ("text", "content", "article", "body"):
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            return k
    # fallback: first non-empty string
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return k
    return None


def _iter_examples_streaming(
    ds: IterableDataset,
    text_key: Optional[str],
) -> Iterable[str]:
    text_key_local = text_key
    for ex in ds:
        if text_key_local is None:
            text_key_local = _auto_text_key(ex)
            if text_key_local is None:
                continue
        v = ex.get(text_key_local, None)
        if isinstance(v, str) and v.strip():
            yield v


def _iter_examples_map(
    dset: Dataset,
    text_key: Optional[str],
) -> Iterable[str]:
    n = len(dset)
    tkey = text_key
    # try determine on first row if needed
    if n > 0 and tkey is None:
        tkey = _auto_text_key(dset[0])

    for i in range(n):
        ex = dset[i]
        key = tkey or _auto_text_key(ex)
        if key is None:
            continue
        v = ex.get(key, None)
        if isinstance(v, str) and v.strip():
            yield v


def _resolve_subset(
    dataset_name: str,
    subset: Optional[str],
    trust_remote_code: bool,
) -> Optional[str]:
    """
    Returns a valid subset string or None. If a subset is provided but not found,
    warn and return None.
    """
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=trust_remote_code)
    except Exception:
        # Some datasets don't expose configs or require auth/etc; fall back silently.
        configs = None

    if not subset:
        return None

    if configs is None:
        print(f"[warn] Could not list configs for '{dataset_name}'. Ignoring subset '{subset}'.")
        return None

    if subset in configs:
        return subset

    print(
        f"[warn] BuilderConfig '{subset}' not found for '{dataset_name}'. "
        f"Available: {configs[:10]}{'...' if len(configs) > 10 else ''}. Using no subset."
    )
    return None


# ------------------------------- main logic ----------------------------------


def download(
    dataset: str,
    split: str,
    out_dir: str,
    subset: Optional[str] = None,
    text_key: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = False,
) -> str:
    """
    Download/stream an English dataset and write a JSONL with {"text": ...} lines.
    Returns the path to the created file.
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "english.jsonl")

    subset = _resolve_subset(dataset, subset, trust_remote_code=trust_remote_code)

    load_kwargs_common = dict(
        split=split,
        trust_remote_code=trust_remote_code,
    )

    print(
        f"[info] Loading dataset='{dataset}' subset='{subset or ''}' "
        f"split='{split}' streaming={streaming} trust_remote_code={trust_remote_code}"
    )

    if streaming:
        # Streaming
        ds = load_dataset(dataset, subset, streaming=True, **load_kwargs_common) if subset \
            else load_dataset(dataset, streaming=True, **load_kwargs_common)

        # If DatasetDict, pick split again (some streamers return dicts)
        if isinstance(ds, DatasetDict):
            ds = ds[split]  # type: ignore[index]

        assert isinstance(ds, IterableDataset), "Expected an IterableDataset in streaming mode."
        it = _iter_examples_streaming(ds, text_key=text_key)
    else:
        # Normal map-style loading
        ds = load_dataset(dataset, subset, **load_kwargs_common) if subset \
            else load_dataset(dataset, **load_kwargs_common)

        # Materialize the desired split
        if isinstance(ds, DatasetDict):
            ds = ds[split]  # type: ignore[index]
        assert isinstance(ds, Dataset), "Expected a map-style Dataset in non-streaming mode."
        it = _iter_examples_map(ds, text_key=text_key)

    # Write JSONL
    appended = 0
    file_exists = os.path.exists(out_path)
    if file_exists and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as existing:
            existing.seek(-1, os.SEEK_END)
            last = existing.read(1)
        if last not in {b"\n", b"\r"}:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n")
    if file_exists:
        print(f"[info] Appending to existing {out_path}")
    else:
        print(f"[info] Creating {out_path}")

    with open(out_path, "a", encoding="utf-8") as f:
        for text in it:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            appended += 1

    if appended == 0:
        print(f"[done] No new documents added (existing file retained) → {out_path}")
    else:
        print(f"[done] Appended {appended} documents → {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Download English corpus to JSONL.")
    p.add_argument("--dataset", required=True, help="HF dataset path (e.g., c4, wikimedia/wikipedia)")
    p.add_argument("--subset", default=None, help="Optional BuilderConfig (e.g., 'en' for c4).")
    p.add_argument("--split", default="train", help="Split name (default: train).")
    p.add_argument("--out", required=True, help="Output directory (will be created).")
    p.add_argument("--text-key", default=None, help="Preferred text field (auto if blank).")
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming load (good for huge datasets).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset.",
    )

    args = p.parse_args()

    download(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        out_dir=args.out,
        text_key=args.text_key if args.text_key else None,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
