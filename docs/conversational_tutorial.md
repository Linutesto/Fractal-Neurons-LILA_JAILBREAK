# Conversational Training Tutorial

This guide walks through the end-to-end workflow for building a conversational Fractal Neurons model. It covers dataset preparation, command-line training, and the interactive menu experience so you can reproduce the new conversational pipelines with minimal guesswork.

---

## 1. Prerequisites

- Python 3.9+
- An existing Fractal Neurons checkout (`git clone ...`)
- Virtual environment activated and dependencies installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece tokenizers datasets pyyaml
```

Set the tokenizer path once (the menu does this for you automatically):

```bash
export FRACTAL_TOKENIZER_PATH="runs/fsi_en_v1/tokenizer"
export USE_EXISTING_TOKENIZER=1
```

If you have an older tokenizer that outputs byte-level artefacts (`Ġ`, `Ċ`, …), run:

```bash
python fix_tokenizer_decoder.py --tokenizer-dir "$FRACTAL_TOKENIZER_PATH"
python add_special_tokens.py --tokenizer-dir "$FRACTAL_TOKENIZER_PATH"
```

---

## 2. Prepare Conversational Data

### 2.1 Synthetic Conversations

Generate 500 prompt/response pairs:

```bash
python generate_conversations.py
```
Outputs: `data/conversational_corpus/conversations.jsonl`

### 2.2 Convert to Chat SFT Format

Wrap each prompt/response in system/user/assistant tags and create an SFT dataset:

```bash
python tools/make_chat_sft.py < data/conversational_corpus/conversations.jsonl \
  > data/chat_sft.jsonl
```

### 2.3 Mix with English Corpus

Blend the conversational set with a 500-line reservoir sample from the English corpus:

```bash
python mix_datasets.py
```
Outputs: `data/mixed_corpus/mixed_data.jsonl`

---

## 3. Command-Line Training Recipes

### 3.1 Conversational Model (Synthetic Only)

```bash
python -m fractal_neurons.train --config configs/conversational_v1.yaml
```

### 3.2 Mixed Conversational + English Corpus

```bash
python -m fractal_neurons.train --config configs/mixed_corpus_v1.yaml
```

### 3.3 Chat Supervised Finetune from an Existing Checkpoint

```bash
python -m fractal_neurons.train \
  --config configs/finetune_chat.yaml \
  --init_ckpt runs/fsi_en_v1/ema_final_20000.pt \
  --device cuda --resume
```

> **Tip:** `--init_ckpt` can be any compatible checkpoint. The config accepts `init.load_optimizer`, `init.load_step`, and `init.load_ema` if you need finer control.

### 3.4 Generation & Chat

```bash
python -m fractal_neurons.generate \
  --ckpt runs/conversational_v1/latest.pt \
  --chat \
  --seq_len 768 \
  --max_new_tokens 200 \
  --temperature 0.85 \
  --top_k 40 --top_p 0.9 --repetition_penalty 1.05
```

---

## 4. Evolution & Distillation Notes

Self-distillation has an `--accept-all` flag so early conversational models (which may produce short or noisy text) still generate training data:

```bash
python -m fractal_neurons.distill \
  --ckpt runs/conversational_v1/latest.pt \
  --prompt-file prompts/seed_prompts.txt \
  --output data/self_distilled_v1/distill.jsonl \
  --limit 2000 --seq-len 512 --max-new-tokens 192 --steps 8 \
  --temperature 0.9 --top_k 50 --accept-all
```

The automation scripts (`run_fsi_pipeline.sh`, `run_fsi_loop.sh`) already pass `--accept-all` and reuse the tokenizer path environment variables.

---

## 5. Menu Walkthrough

Launch the menu:

```bash
python menu.py
```

Key conversational options:

1. **Train Conversational Model** – Launches `configs/conversational_v1.yaml` (choose data root if you relocated the JSONL).
2. **Train on Mixed Corpus** – Runs the mixed dataset pipeline (calls `mix_datasets.py` if needed, then `configs/mixed_corpus_v1.yaml`).
3. **Finetune for Chat** – Wraps the chat SFT procedure, prompting for a base checkpoint and launching `configs/finetune_chat.yaml`.
4. **Fix Tokenizer Decoder** – Calls `fix_tokenizer_decoder.py` for the directory you specify.
5. **Download/Finetune English Corpus** – Same helpers as before; useful if you need additional base data.

Every menu action logs to `logs/menu_history.*`, so you can review the exact commands the wizard executed.

Generation defaults inside the menu now use `temperature=0.8`, `top_k=50`, and you can toggle `top_p`/`repetition_penalty` interactively.

---

## 6. Troubleshooting Checklist

- **Tokenizer artefacts:** run the decoder fixer and ensure `FRACTAL_TOKENIZER_PATH` points to your trained tokenizer.
- **Empty distillation output:** re-run with `--accept-all` or provide custom prompts.
- **CUDA not available:** use `--device cpu --num_workers 0` for quick sanity checks, but full training is designed for CUDA.
- **Menu crashes on missing data:** confirm `data.conversational_corpus/conversations.jsonl` and `data/chat_sft.jsonl` exist, or regenerate them with the scripts above.

---

Happy chatting!
