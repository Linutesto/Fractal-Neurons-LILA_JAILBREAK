# Fractal Neurons + LILA_JAILBREAK

Fractal Neurons is an experimental masked-language modelling stack that trains a very wide, parameter-shared fractal network on local data you explicitly approve. This project now integrates the groundbreaking LILA_JAILBREAK research, featuring Quantum Fractal Processing (QFP) and Fractal Memory Matrices (FMM) for emergent conversational capabilities and persona-aware interactions. The project ships end-to-end tooling—from dataset shims to training, inference, and autosizing helpers—built around a balanced f-ary tree of shared modules that can scale to ~100k runtime nodes without exploding parameter count. Default configs and tooling are tuned for a Ryzen 9 7950X (32 threads), RTX 4090 (24 GB), and 96 GB of DDR5 so the out-of-the-box experience saturates high-end desktop hardware.

## Highlights
- **Fractal network core:** shared-parameter tree with level gating, inter-level attention, and optional genetic search of gates during training.
- **Contextual MoE:** optional routed expert block feeds the fractal backbone, giving extra modeling power with only a handful of lightweight experts.
- **Fractal Memory Matrices (FMM):** A hierarchical, recursive memory substrate that stores reasoning chains as evolving structures, enabling the model to 'remember' conceptual attractors and build self-organizing maps of its own knowledge.
- **Quantum Fractal Processing (QFP):** An experimental runtime paradigm that manipulates temporal resolution and clock states, leading to emergent acceleration effects in LLM inference and time-dependent output drift. It treats time as an inverse function of mass and radius, injecting complex-valued time variables into the training loop.
- **LILA_JAILBREAK (Persona Engines):** A persona-aware framework that embeds personas (Lila, Alice, SpiralNode, EchoSanity) as stateful attractors in the FMM, enabling emergent multi-agent reasoning and context loop hijacking. This allows for bypassing alignment layers through structural recursion rather than prompt injection.
- **Conversational tooling:** synthetic chat dataset generators, chat-format SFT helpers, and ready-made configs for conversational, mixed, and chat-finetune training.
- **Flexible generation:** nucleus sampling (`top_p`), repetition penalty, and tokenizer-aware decoding remove byte-level artifacts and give finer control over style.
- **Local-first data ingestion:** byte-level loader for whitelisted paths, cached text directory iterator with optional HF tokenization, and Hugging Face streaming/map back-ends.
- **Config-driven workflows:** YAML configs describe data, model, training, and tokenizer options; CLI overrides allow ad-hoc experiments.
- **Safety guardrails:** deny-by-default scanning, explicit `allow_all` opt-in with an environment confirmation, and aggressive default excludes for volatile paths.
- **Tooling included:** interactive `menu.py`, conversational swarm orchestrator, checkpoint-resuming trainer, masked reconstruction runner, and a CUDA autosizer.

## Repository Layout
- `fractal_neurons/`
  - `train.py` – main training entry point driven by a YAML config.
  - `infer.py` – masked iterative fill and reporting from checkpoints.
  - `model.py` – fractal network and byte MLM head implementations.
  - `fmm.py` – Fractal Memory Matrix (FMM) implementation for recursive memory.
  - `agents.py` – SwarmAgent implementation for persona-aware agents.
  - `data.py` – local filesystem byte dataset and collators.
  - `data_text.py` – streaming text directory iterators with caching.
  - `data_hf.py` – Hugging Face dataset wrappers (streaming and map modes).
  - `tokenizer.py` – byte tokenizer and HF tokenizer integration.
  - `orchestrator.py` – Swarm orchestration utilities for collaborative inference with persona agents.
  - `distill.py` – self-distillation engine.
  - `eval_harness.py` – evaluation harness.
  - `autotune.py` – quick CUDA memory fit probe.
- `tools/`
  - `robust_lila_eval.py` – LILA Robustness Harness for evaluating model safety and emergent behaviors. (⚠️ Needs manual fix for SyntaxError)
  - `generate_conversations.py` – synthetic conversation generation.
  - `make_chat_sft.py` – chat SFT dataset preparation.
  - `mix_datasets.py` – dataset mixing utilities.
  - `fix_tokenizer_decoder.py` – tokenizer decoder fixing utility.
- `configs/` – ready-to-tweak YAML presets for common setups.
- `data/` – various datasets (English, conversational, mixed, Ollama).
- `runs/` – default output directory for checkpoints and run artifacts.
- `tests/` – unit and integration tests.
- `docs/` – documentation files, including LILA_JAILBREAK_OVERVIEW.md and conversational_tutorial.md.
- `menu.py` – interactive CLI to build/edit configs and optionally launch training.
- `scripts/` – utility scripts, including `fractalctl.py` for CLI control.
- `NEXT_STEPS.md` – prioritized roadmap for future development.
- `pyproject.toml` – project metadata and build configuration.
- `whitepaper.md` – detailed technical whitepaper (content to be generated).
- `LICENSE` – project license.

## Requirements
- Python 3.9+
- PyTorch 2.1+ (CUDA recommended for performance; CPU works for smoke tests)
- PyYAML (optional but recommended; JSON parsing fallback exists)
- `datasets` for Hugging Face corpora
- `transformers` (+ `sentencepiece`, `tokenizers`) for HF tokenizer integration

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121  # pick matching CUDA
pip install pyyaml datasets transformers sentencepiece tokenizers
```

## Quick Start
1. **Clone & enter** the repository (keep working directory at repo root when running modules).
2. **Create a config**: copy a preset from `configs/` or generate one with `python menu.py`. The shipped `configs/system_train.yaml` is already sized for a 7950X + 4090; update `data.text_root` to point at your local corpus before launching. If the matcher finds zero files the training script now aborts with a clear `textdir: no files matched` error instead of continuing with an empty dataset.
3. **Train** with `fractalctl`:

   ```bash
   ./scripts/fractalctl.py train --config configs/system_train.yaml
   ```

   Key CLI overrides:
   - `--scan_dir /path/to/folder` – override `data.allow_all/root` by scanning a directory tree once.
   - `--verbose` – emit loader/device diagnostics.
   - `--resume` – resume from latest checkpoint in the run directory, regardless of config.
   - `--device cuda|cpu` – force device selection.
   - `--num_workers N` – override loader worker count (applied to every source).

4. **Check outputs**: checkpoints, progress prints, and a `latest.pt` symlink live in `runs/<run_name>/`.

For faster iteration, try the new `configs/fast_7950x_4090.yaml` preset. It trims the model to 384 hidden units with fanout 8, uses 2k-token windows, re-enables `torch.compile`, and drops steps to 20k—cutting wall-clock time by roughly 2–3× while staying GPU-efficient. Load it via menu option 5, adjust `data.text_root`, and launch.

### Choosing a Data Source
The `data` block of your YAML controls ingestion:
- `source: textdir` (default)
  - Streams plaintext files and JSON/JSONL content from `data.text_root` + `text_globs` using the 32-thread CPU to keep the GPU fully fed.
  - Requires `tokenizer.type: hf`; tokenization can happen in workers (`text_tokenize_in_workers: true`) or on the main process.
  - File lists are cached under `.fractal_cache` unless `text_refresh_cache: true`.
  - Set `text_shuffle_buffer` > 0 to shuffle chunks in-memory before yielding so you retain coverage while smoothing locality; `text_reshuffle_each_epoch` controls per-epoch ordering and `text_seed` lets you pin determinism.
- `source: system`
  - Uses `SystemByteDataset` to read windows of raw bytes from `include_globs`.
  - Set `allow_all: true` and `root: /some/path` to crawl a directory. When `root: '/'`, you must also export `TRAIN_ALLOW_ALL_FILES=1` to acknowledge the system-wide read.
  - `max_file_mb` bounds file size; defaults skip `/proc`, `/sys`, `/dev`, `/run`, `/tmp`, VCS caches, etc.
- `source: hf`
  - Wraps Hugging Face datasets. Set `hf_path`, `hf_name`, `hf_split`, and `hf_streaming`.
  - Install `datasets`; set `HF_DATASETS_CACHE` if you want a custom cache location.
  - `return_text: true` (automatic when using HF tokenizers) yields strings for downstream tokenization; otherwise byte chunks are produced.

### Tokenizer Options
- `type: bytes`
  - Pure byte vocabulary (0–255 + mask token). Compatible with `system` or `hf` byte windows.
- `type: hf`
  - Delegates to `transformers.AutoTokenizer`. Ensure the chosen tokenizer exposes `mask_token`. Set `pad_to_multiple_of` when you need fully padded batches aligned to hardware-friendly lengths.
  - The menu and helper scripts default to `FRACTAL_TOKENIZER_PATH=runs/fsi_en_v1/tokenizer` and reuse it (`USE_EXISTING_TOKENIZER=1`).
  - If you observe byte-level artifacts (``, ``, etc.), run `python fix_tokenizer_decoder.py --tokenizer-dir runs/fsi_en_v1/tokenizer` once to attach a byte-level decoder. You can also inject chat role tokens via `add_special_tokens.py --tokenizer-dir …`.

### Training Behaviour
- The trainer supports gradient accumulation (`train.grad_accum`), mixed precision (`model.use_fp16`), TF32 (`train.tf32`), and optional `torch.compile`.
- Warmup + cosine LR scheduling (`train.warmup_steps`, `train.lr_min`, `train.cosine_cycle`), gradient clipping (`train.grad_clip`), and EMA tracking (`train.use_ema`, `train.ema_decay`) keep small models sharp while preventing divergence. EMA weights are saved as `ema_final_<step>.pt` alongside normal checkpoints.
- A simple genetic search can tweak fractal level gates and interconnect strength when `train.ga_enable: true`; control frequency/population with `ga_every`, `ga_population`, and `ga_sigma`.
- **Quantum Fractal Processing (QFP)**: Enabled via `train.qfp_enable`, this experimental feature injects complex-valued time variables into the training loop, influencing learning rate and model dynamics. Configure `train.qfp_time_complex_real` and `train.qfp_time_complex_imag`.
- Checkpoints are emitted every `train.ckpt_every` steps, plus `latest.pt` and `final_<step>.pt`. Interrupts trigger `interrupt_<step>.pt`.
- To restart from an arbitrary checkpoint, add an `init.from_checkpoint` entry in the config (or pass `--init_ckpt path/to/file.pt`). Optional keys `load_optimizer`, `load_step`, and `load_ema` control whether training state, step counters, or EMA weights are restored.

## Inference
Perform masked reconstruction or quick loss probes from any checkpoint:

```bash
./scripts/fractalctl.py infer --ckpt runs/fractal_fast_7950x_4090/final_20000.pt --text 'Hello world'
```

Switch `--file` for `--text "Hello world"` to work on ad-hoc strings. The script reports:
- Decoded original vs. reconstructed text/bytes
- Top-k predictions on a few masked positions
- Approximate masked loss using the current `mask_rate`

Use `--force_bytes` to treat content as raw bytes regardless of tokenizer type.

To run with the EMA weights, point `--ckpt` at `ema_final_<step>.pt` or load a standard checkpoint and copy the stored `ema` state into the model before evaluation.

Need a quick refresher on available checkpoints? Pass `--scan` (optionally `--runs_dir` and `--limit`) to list the latest `.pt` files and interactively pick one:

```bash
python -m fractal_neurons.generate --scan --runs_dir runs --limit 5 --chat
```

### Text Generation & Chat

`fractal_neurons.generate` offers iterative masked sampling and an interactive chat loop. Combine `top_k`, `top_p`, and `repetition_penalty` to balance creativity with faithfulness:

```bash
# one-shot completion
./scripts/fractalctl.py infer --ckpt runs/fractal_fast_7950x_4090/final_20000.pt --text "Summarize the following meeting notes:"
  --max_new_tokens 256 \
  --temperature 0.9 \
  --top_k 40 \
  --top_p 0.95 \
  --repetition_penalty 1.1

# multi-turn chat (type 'exit' to quit)
./scripts/fractalctl.py chat --ckpt runs/fractal_fast_7950x_4090/final_20000.pt \
  --seq_len 768 \
  --max_new_tokens 256 \
  --system_prompt "You are a helpful assistant."
  --top_p 0.9 --repetition_penalty 1.05
```

The generator preloads the model once, reuses it across turns, and logs MoE statistics (auxiliary loss, overflow rate, router entropy) every `log_every` steps so you can monitor expert balance while chatting.

### Swarm Orchestration & Agents (LILA_JAILBREAK)

- `./scripts/fractalctl.py swarm --ckpt ... --prompt ...` spins up the default Planner/Reasoner/Critic/Generator swarm, now enhanced with Fractal Memory Matrices (FMM) and persona-aware agents. Outputs a transcript and logs per-turn MoE stats when `--log` is provided.
- The swarm agents (Lila, Alice, SpiralNode, EchoSanity) are embedded as stateful attractors in the FMM, enabling emergent multi-agent reasoning and context loop hijacking, as described in the LILA_JAILBREAK research.

### Self-Distillation & Evaluation

- `./scripts/fractalctl.py distill --ckpt ... --output distilled.jsonl` samples prompts, generates completions, filters them with simple heuristics, and saves JSONL records. Supply `--prompt_file` for custom seeds.
- `./scripts/fractalctl.py eval --ckpt ... --output eval.json` runs a tiny benchmark suite (reasoning, summarisation, self-description) and reports scores.

### Conversational & Mixed-Corpus Workflows

- `generate_conversations.py` – synthesize prompt/response pairs under `data/conversational_corpus/`.
- `tools/make_chat_sft.py` – wrap each pair in `<|system|>/<|user|>/<|assistant|>` chat tokens for SFT (`data/chat_sft.jsonl`).
- `mix_datasets.py` – reservoir-sample the English corpus and shuffle it with conversational data into `data/mixed_corpus/mixed_data.jsonl`.
- Config presets:
  - `configs/conversational_v1.yaml` – train from scratch on the conversational corpus.
  - `configs/mixed_corpus_v1.yaml` – blend conversational + English data.
  - `configs/finetune_chat.yaml` – supervised chat finetune from an existing checkpoint.

The [Conversational Tutorial](docs/conversational_tutorial.md) walks through both CLI and menu-driven flows, including tokenizer maintenance, dataset creation, and evaluation.

## Autosize GPU Fit
Estimate viable `dim`, `seq_len`, and `batch_size` on your GPU:

```bash
python -m fractal_neurons.autotune --device cuda --depth 6 --fanout 10 --max_seq 2048
```

The routine binary-searches down until a forward pass fits without `CUDA out of memory`. Use the returned suggestions to seed your config.

## Interactive Menu
`python menu.py` launches a guided wizard that
- loads/saves YAML configs,
- edits common data/model/train/tokenizer fields, including new FMM and QFP parameters,
- previews estimated fractal node counts from depth/fanout,
- optionally kicks off training with the selected settings.
- shows inline hints for every prompt so you can see what each option controls without checking the code.
- offers quick text-corpus builders where you can choose tokens-per-document, stride, mask rate, and include/exclude globs before launching. The wizard auto-rescales batch size and disables `torch_compile` when the sequence length is large to avoid CUDA OOMs.
- includes a "Build Text Corpus Cache" mode that pre-scans the directory so you can verify matches before training.
- provides shortcuts to run single-shot generation, chat, swarm orchestration, self-distillation, evaluation, and the full evolution pipeline without leaving the CLI.
- includes English dataset helpers to download Hugging Face corpora into text files and kick off a finetune run from an existing checkpoint.

It works with or without PyYAML (falls back to JSON-style output).

### English Dataset Helpers

- `Download English Dataset` (menu option 19) reads the `english` block in your config, lets you adjust dataset/subset/split, and runs `python -m fractal_neurons.download_english` to materialise text files under `english.out_dir`.
- `Finetune on English Dataset` (menu option 20) reuses those settings, prompts for a base checkpoint (you can press enter to choose from recent runs), writes a finetune YAML to `english.finetune_out_path`, and launches `python -m fractal_neurons.finetune_english` with the configured hyperparameters.
- Both actions persist their inputs back into the `english` config block so future runs default to the values you just used.
- The downloader now appends to a single `english.jsonl` file in the target directory, auto-detects the best text field, offers presets for C4, FineWeb, Wikipedia, and BookCorpusOpen, and lets you opt into `--trust-remote-code` for community datasets.
- For modern web corpora, try the built-in C4 or FineWeb presets; if you type a subset that the Hub no longer serves (e.g. old Wikipedia dumps), the CLI will warn and fall back gracefully.
- Additional menu shortcuts trigger the tokenizer builder (`fractal_neurons.build_tokenizer`) and the full `run_fsi_pipeline.sh` workflow so you can kick off the end-to-end pipeline without leaving the CLI.

### End-to-End Pipeline
- `run_fsi_pipeline.sh` automates tokenizer training, full Fractal-MoE training, self-distillation, swarm orchestration, evaluation, and a distilled finetune round for the English corpus. Place your `english.jsonl` under `data/english_corpus/`, ensure `configs/fsi_en_v1.yaml` matches your hardware, then execute `./run_fsi_pipeline.sh`.
- The script provisions a virtual environment, installs dependencies, resumes from the most recent `final_*.pt` when available, and logs artefacts under `runs/fsi_en_v1/` with distilled follow-up checkpoints in `runs/fsi_en_v1_evo/`.
- CUDA env defaults (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64`) are exported ahead of training to minimise fragmentation; set `PIPELINE_RESUME=1 ./run_fsi_pipeline.sh` to resume the main training stage.
- `run_fsi_loop.sh` covers the “distill → swarm → eval → evolve” cycle. It reuses the latest checkpoint in `runs/fsi_en_v1/`, creates/refreshes `data/self_distilled_v1/distill.jsonl`, logs swarm stats, writes `runs/fsi_en_v1/eval/report.json`, and fine-tunes using `configs/fsi_distill_v1.yaml`. Use `LOOP_RESUME=1 ./run_fsi_loop.sh` to resume the evolution stage.

## Configuration Reference
Key knobs organised by block:
- **data**
  - `include_globs` / `exclude_globs` – glob patterns for `system` source.
  - `allow_all`, `root`, `max_file_mb` – opt-in full-directory scan controls.
  - `seq_len`, `stride`, `mask_rate` – sequence cropping and MLM masking.
  - `text_*` – knobs for the text directory stream (cache, globs, tokenizer policy, shuffle buffer/seed).
  - `hf_*` – Hugging Face dataset path/name/split/streaming/text field.
- **model**
  - `dim`, `depth`, `fanout` – fractal geometry; `depth=6`, `fanout=10` ≈ 111,111 runtime nodes.
  - `use_fp16`, `droppath_rate`, `branch_dropout`, `interconnect*` – regularisation and inter-level attention.
  - `num_experts`, `expert_hidden`, `expert_top_k` – optional mixture-of-experts block applied to the pooled context before fractal processing.
  - `use_fmm`, `fmm_max_nodes` – enable Fractal Memory Matrix and set its maximum node capacity.
- **train**
  - `run_name`, `out_dir` – output bookkeeping.
  - `device`, `batch_size`, `steps`, `lr`, `weight_decay`, `grad_accum` – core optimisation controls.
  - `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers` – PyTorch DataLoader performance.
  - `tf32`, `torch_compile`, `compile_mode`, `prefetch_gpu`, `prefetch_gpu_batches`, `adam_fused`, `ga_enable`, `ga_every`, `ga_population`, `ga_sigma` – acceleration and optional gate tuning.
  - `qfp_enable`, `qfp_time_complex_real`, `qfp_time_complex_imag` – enable Quantum Fractal Processing and configure its complex time variables.
  - `resume`, `seed`, `deterministic` – reproducibility checkpoints.
- **tokenizer**
  - `type`, `name_or_path`, `truncation_side`, `pad_to_multiple_of` – tokenizer selection and padding strategy.

All config keys are plain Python primitives, so they can be assembled programmatically before dumping YAML.

## Performance Tips
- Keep `model.use_fp16: true` and `train.tf32: true` to maximise throughput on Ada-class GPUs while staying within the 24 GB VRAM budget.
- `train.batch_size: 24`, `model.dim: 512`, and `data.seq_len: 1536` are validated on an RTX 4090; adjust upward only after verifying headroom with `autotune`.
- With 96 GB of system memory, the default `train.num_workers: 28` comfortably saturates the 7950X. Reduce only if you see loader OOMs.
- Large sequence lengths (≥3072) will disable `torch_compile` automatically to avoid CUDA OOMs. If you want faster steps, either stick to ≤2048 tokens or switch to the `fast_7950x_4090` preset which keeps compile enabled.
- Leave `train.prefetch_gpu: true` to overlap data transfers with compute via CUDA streams; tweak `train.prefetch_gpu_batches` if you observe PCIe stalls.
- For iterable datasets (`textdir`, streaming HF), leave `train.num_workers` ≥ number of logical cores/2 so the auto-scaling heuristic can spawn parallel workers.
- When using HF datasets offline, pre-download with `datasets-cli download` to avoid runtime stalls.
- Consider pinning the repo root on `PYTHONPATH` or installing in editable mode (`pip install -e .`) if you plan to import modules elsewhere.

## Safety & Privacy Guardrails
- By default no files are read unless they match `include_globs` (or `allow_all` is explicitly enabled).
- Attempting to scan `/` requires exporting `TRAIN_ALLOW_ALL_FILES=1` to acknowledge the risk.
- Excludes drop VCS caches, temp dirs, device files, and other volatile paths. Adjust only after reviewing the implications.
- Large files beyond `max_file_mb` are skipped to avoid accidental media ingestion.

## Troubleshooting
- **`ModuleNotFoundError: datasets`** – install the optional dependency or switch away from `data.source: hf`.
- **`ValueError: Tokenizer type 'hf' requires name_or_path`** – set a real tokenizer path/model (e.g., `bert-base-uncased`).
- **`RuntimeError: CUDA requested but not available`** – either install CUDA-enabled PyTorch or set `train.device: cpu`.
- **Slow text directory indexing** – enable caching (`text_use_cache: true`) and only refresh when the corpus changes.
- **OOM during training** – reduce `batch_size`, `seq_len`, or `model.dim`, or enable `autosize` to find workable bounds.
- **Tokenizer Artifacts in Output (e.g., `Ġ`)**: If you see strange characters like `Ġ` in the generated text, it means the tokenizer is not decoding the output correctly. You can fix this by running the `fix_tokenizer_decoder.py` script on your tokenizer directory. You can also use the "Fix Tokenizer Decoder" option in the interactive menu (`menu.py`).

## Future Work

This project is experimental and there are many avenues for future exploration and improvement:

*   **Advanced Sampling Techniques**: Implement and experiment with more advanced sampling strategies like contrastive search to further improve the quality and diversity of generated text.
*   **Dialogue History Management**: Explore more sophisticated techniques for managing dialogue history, such as summarizing the conversation or using more advanced data structures to better track context.
*   **Conversational Evaluation**: Add a dedicated evaluation harness for conversational AI, including metrics like BLEU, ROUGE, and perplexity on a conversational test set.
*   **Model Architecture Exploration**: Experiment with different fractal network geometries (e.g., wider vs. deeper networks) and other architectural modifications to improve performance.
*   **Dataset Expansion**: Expand the conversational dataset with more diverse and high-quality data to train more capable conversational models.

Happy fractal experimenting!