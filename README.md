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
- `whitepaper.md` – detailed technical whitepaper.
- `DATA_GENERATOR.md` – documentation for autonomous conversational data generation.
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
  - If you observe byte-level artifacts (`Ġ`, `Ċ`, etc.), run `python fix_tokenizer_decoder.py --tokenizer-dir runs/fsi_en_v1/tokenizer` once to attach a byte-level decoder. You can also inject chat role tokens via `add_special_tokens.py --tokenizer-dir …`.

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

Switch `--file` for `--text 