#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def reservoir_sample(stream_path: Path, sample_size: int) -> list[str]:
    sample: list[str] = []
    with stream_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sample[j] = line
    return sample


def extract_text(record: dict) -> str:
    for key in ("text", "content", "article", "paragraph", "document"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix conversational and English corpora into a single JSONL")
    parser.add_argument("--conversations", default="data/conversational_corpus/conversations.jsonl")
    parser.add_argument("--english", default="data/english_corpus/english.jsonl")
    parser.add_argument("--output", default="data/mixed_corpus/mixed_data.jsonl")
    parser.add_argument("--english-sample", type=int, default=500)
    args = parser.parse_args()

    conv_path = Path(args.conversations).expanduser()
    english_path = Path(args.english).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conversational_data = []
    if conv_path.exists():
        with conv_path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                conversational_data.append({
                    "text": f"{payload.get('prompt', '').strip()} {payload.get('response', '').strip()}".strip()
                })
    print(f"Loaded {len(conversational_data)} conversational examples from {conv_path}")

    english_records = []
    if english_path.exists() and args.english_sample > 0:
        sample_lines = reservoir_sample(english_path, args.english_sample)
        for line in sample_lines:
            try:
                data = json.loads(line)
                text = extract_text(data)
            except json.JSONDecodeError:
                text = line.strip()
            if text:
                english_records.append({"text": text})
    print(f"Loaded {len(english_records)} English examples from {english_path}")

    combined = conversational_data + english_records
    random.shuffle(combined)
    print(f"Combined dataset has {len(combined)} examples → {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
