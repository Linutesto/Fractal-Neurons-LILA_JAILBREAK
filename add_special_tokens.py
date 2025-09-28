from tokenizers import Tokenizer
from pathlib import Path
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True)
    args = parser.parse_args()

    tok_dir = Path(args.tokenizer_dir)
    tok_json = tok_dir / "tokenizer.json"

    if not tok_json.exists():
        print(f"tokenizer.json not found in {tok_dir}")
        return

    tok = Tokenizer.from_file(str(tok_json))

    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    tok.add_special_tokens(special_tokens)

    tok.save(str(tok_json))

    # also ensure special_tokens_map.json exists & matches IDs
    stm_path = tok_dir / "special_tokens_map.json"
    if stm_path.exists():
        with open(stm_path, "r") as f:
            stm = json.load(f)
    else:
        stm = {}

    stm.update({
        "system_token": "<|system|>",
        "user_token": "<|user|>",
        "assistant_token": "<|assistant|>",
        "end_token": "<|end|>",
    })

    with open(stm_path, "w") as f:
        json.dump(stm, f, indent=2)

    print(f"Added special tokens to {tok_dir}")

if __name__ == "__main__":
    main()
