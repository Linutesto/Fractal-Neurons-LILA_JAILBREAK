#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert prompt/response JSONL into chat-format SFT records")
    parser.add_argument("--input", default="data/conversational_corpus/conversations.jsonl")
    parser.add_argument("--output", default="data/chat_sft.jsonl")
    parser.add_argument("--system", default="You are a helpful assistant.")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            record = json.loads(line)
            system = record.get("system", args.system).strip()
            user = record.get("prompt", "").strip()
            assistant = record.get("response", "").strip()
            if not user or not assistant:
                continue
            text = (
                f"<|system|> {system}<|end|>\n"
                f"<|user|> {user}<|end|>\n"
                f"<|assistant|> {assistant}<|end|>\n"
            )
            dst.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote chat SFT file → {out_path}")


if __name__ == "__main__":
    main()
