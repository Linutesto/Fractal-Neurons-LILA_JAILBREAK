#!/usr/bin/env python3
"""
Interactive menu to configure and launch Fractal Neurons training.

Features:
- Edit data/model/train options.
- Load/save YAML config.
- Show estimated total nodes from depth/fanout.
- Optionally launch training with the selected config.

Usage:
  python menu.py
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import shlex

from fractal_neurons.generate import (
    generate_text,
    load_generation_model,
    scan_checkpoints,
    print_checkpoint_table,
    ChatSession,
    log_generation,
)

MENU_LOG_DIR = Path("logs")
MENU_LOG_TXT = MENU_LOG_DIR / "menu_history.txt"
MENU_LOG_JSONL = MENU_LOG_DIR / "menu_history.jsonl"

DEFAULT_TOKENIZER_DIR = os.environ.get("FRACTAL_TOKENIZER_PATH", "runs/fsi_en_v1/tokenizer")
_expanded_default_tokenizer = os.path.expanduser(DEFAULT_TOKENIZER_DIR)
DEFAULT_TOKENIZER_EXISTS = os.path.isdir(_expanded_default_tokenizer)
DEFAULT_TOKENIZER_NAME = DEFAULT_TOKENIZER_DIR if DEFAULT_TOKENIZER_EXISTS else "bert-base-uncased"


def log_menu_event(event: str, payload: Dict[str, Any]) -> None:
    try:
        MENU_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **payload,
        }
        with MENU_LOG_TXT.open("a", encoding="utf-8") as txt:
            txt_payload = {k: v for k, v in payload.items()}
            txt.write(f"{entry['timestamp']} {event}: {txt_payload}\n")
        with MENU_LOG_JSONL.open("a", encoding="utf-8") as jsonl:
            jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_command(cmd: str, check: bool = True, env: Dict[str, str] | None = None) -> int:
    print(f"\n▶ {cmd}\n")
    start = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    run_env = (env.copy() if env is not None else os.environ.copy())
    run_env.setdefault("USE_EXISTING_TOKENIZER", "1")
    run_env.setdefault("FRACTAL_TOKENIZER_PATH", _expanded_default_tokenizer)
    result = subprocess.run(cmd, shell=True, env=run_env)
    log_menu_event(
        "command",
        {
            "cmd": cmd,
            "check": check,
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "started_at": start,
            "env_override": bool(env),
        },
    )
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def resolve_pipeline_script(path: str) -> str | None:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path.cwd() / "configs",
        Path.cwd() / "scripts",
    ]
    name = candidate.name
    for root in search_roots:
        guess = root / name
        if guess.is_file():
            return str(guess.resolve())

    try:
        found = next(Path.cwd().rglob(name))
        if found.is_file():
            return str(found.resolve())
    except StopIteration:
        pass
    return None


def ensure_prompt_file(path: str) -> str:
    expanded = os.path.expanduser(path)
    prompt_path = Path(expanded)
    if prompt_path.exists():
        return expanded
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "Summarize the benefits of modular neural networks.\n"
        "Explain the concept of fractal intelligence in simple terms.\n"
        "Write three bullet points about efficient training tricks.\n",
        encoding="utf-8",
    )
    return expanded


def default_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


try:
    import yaml  # type: ignore
except Exception:  # fallback: write JSON-like if YAML missing
    yaml = None
    import json


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "source": "textdir",  # textdir | system | hf
        "include_globs": [],
        "exclude_globs": [
            "**/.git/**",
            "**/.cache/**",
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/*.bin",
            "/proc/**",
            "/sys/**",
            "/dev/**",
            "/run/**",
            "/tmp/**",
        ],
        "seq_len": 1536,
        "stride": 1536,
        "mask_rate": 0.15,
        # textdir options tuned for 7950X/4090
        "text_root": os.path.expanduser("~/datasets/fractal_corpus"),
        "text_globs": [
            "**/*.txt",
            "**/*.md",
            "**/*.jsonl",
            "**/*.json",
            "**/*.log",
            "**/*.py",
        ],
        "text_use_cache": True,
        "text_refresh_cache": False,
        "text_tokenize_in_workers": True,
        "text_shuffle_buffer": 2048,
        "text_reshuffle_each_epoch": True,
        "text_seed": None,
        "allow_all": False,
        "root": "/",
        "max_file_mb": 32,
        # HF options (fallback when switching source)
        "hf_path": "wikipedia",
        "hf_name": "20220301.en",
        "hf_split": "train",
        "hf_streaming": True,
        "hf_text_field": None,
    },
    "model": {
        "dim": 512,
        "depth": 6,
        "fanout": 10,
        "use_fp16": True,
        "droppath_rate": 0.1,
        "branch_dropout": 0.1,
        "interconnect": True,
        "interconnect_heads": 4,
        "interconnect_dropout": 0.05,
        "num_experts": 4,
        "expert_hidden": 1024,
        "expert_top_k": 2,
    },
    "train": {
        "run_name": f"fractal_{int(time.time())}",
        "device": default_device(),
        "batch_size": 24,
        "steps": 50000,
        "lr": 1.5e-3,
        "weight_decay": 0.01,
        "grad_accum": 1,
        "num_workers": 28,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "torch_compile": True,
        "tf32": True,
        "log_every": 50,
        "ckpt_every": 1000,
        "out_dir": "runs",
        # GA defaults off; enable manually when experimenting
        "resume": True,
        "seed": 1337,
        "deterministic": False,
        "ga_enable": False,
        "ga_every": 500,
        "ga_population": 8,
        "ga_sigma": 0.05,
        "compile_mode": "default",
        "prefetch_gpu": True,
        "prefetch_gpu_batches": 2,
        "adam_fused": True,
        "warmup_steps": 500,
        "lr_min": 1e-4,
        "cosine_cycle": 50000,
        "grad_clip": 1.0,
        "use_ema": True,
        "ema_decay": 0.999,
    },
    "tokenizer": {
        "type": "hf",  # bytes | hf
        "name_or_path": DEFAULT_TOKENIZER_NAME,
        "truncation_side": "right",
        "pad_to_multiple_of": 128,
    },
    "conversational": {
        "conversations_path": "data/conversational_corpus/conversations.jsonl",
        "chat_sft_path": "data/chat_sft.jsonl",
        "mixed_path": "data/mixed_corpus/mixed_data.jsonl",
        "english_path": "data/english_corpus/english.jsonl",
        "english_sample": 500,
        "synthetic_examples": 500,
        "ollama_conversations_path": "data/ollama_conversations/conversations.jsonl",
        "ollama_model": "gpt-oss:20b",
        "ollama_target_mb": 10.0,
        "ollama_system_prompt": "You are a helpful AI assistant. Engage in a natural conversation with the user.",
        "ollama_initial_user_prompt": "Start a conversation about a random interesting topic.",
    },
    "english": {
        "dataset": "wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_key": None,
        "out_dir": "data/english_corpus",
        "streaming": False,
        "tokenizer_out_dir": DEFAULT_TOKENIZER_DIR,
        "tokenizer_vocab_size": 65536,
        "pipeline_script": "run_fsi_pipeline.sh",
        "pipeline_resume": True,
        "pipeline_base_config": "configs/fsi_en_v1.yaml",
        "pipeline_evo_config": "configs/fsi_distill_v1.yaml",
        "loop_script": "run_fsi_loop.sh",
        "loop_resume": True,
        "finetune_steps": 3000,
        "finetune_lr": 3e-4,
        "finetune_batch": 24,
        "finetune_seq": 1024,
        "finetune_ckpt": "",
        "finetune_out_path": "runs/finetune_english.yaml",
        "finetune_num_experts": 4,
        "finetune_expert_top_k": 2,
        "finetune_ema_decay": 0.9995,
        "finetune_warmup": 500,
        "finetune_grad_clip": 1.0,
        "finetune_tokenizer_type": "hf",
        "finetune_tokenizer_name": DEFAULT_TOKENIZER_NAME,
        "finetune_tokenizer_truncation": "right",
        "finetune_tokenizer_pad_multiple": 128,
        "trust_remote_code": False,
    },
}


@dataclass
class MenuContext:
    cfg: Dict[str, Any]
    verbose: bool = False
    running: bool = True


@dataclass
class MenuOption:
    name: str
    action: Callable[["MenuContext"], None]
    label_getter: Callable[["MenuContext"], str] | None = None

    def label(self, ctx: "MenuContext") -> str:
        return self.label_getter(ctx) if self.label_getter else self.name


def read_input(prompt: str, default: Any = None) -> str:
    sdef = f" [{default}]" if default is not None else ""
    try:
        val = input(f"{prompt}{sdef}: ").strip()
    except EOFError:
        print()
        sys.exit(0)
    return val or (str(default) if default is not None else "")


def parse_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "y", "yes", "t"}


def parse_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        # simple YAML/JSON-like
        try:
            import json
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [p for p in parts[0].split() if p]
    return parts


def select_checkpoint(cfg: Dict[str, Any], limit: int = 10) -> str:
    runs_dir = cfg.get("train", {}).get("out_dir", "runs")
    files = scan_checkpoints(runs_dir, limit)
    print_checkpoint_table(files)
    default_ckpt = cfg.get("last_ckpt") or (str(files[0]) if files else "")
    choice = read_input("Checkpoint path or index", default_ckpt).strip()
    if not choice:
        choice = default_ckpt
    if not choice:
        raise RuntimeError("No checkpoint selected.")
    if choice.isdigit() and files:
        idx = int(choice)
        if 1 <= idx <= len(files):
            ckpt = str(files[idx - 1])
        else:
            raise RuntimeError("Selection out of range.")
    else:
        ckpt = os.path.expanduser(choice)
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Checkpoint not found: {ckpt}")
    cfg["last_ckpt"] = ckpt
    return ckpt


def prompt_init_settings(cfg: Dict[str, Any]) -> None:
    init_cfg = dict(cfg.get("init", {}) or {})
    has_ckpt = bool(init_cfg.get("from_checkpoint"))
    if not parse_bool(read_input("Load weights from checkpoint before training? (true/false)", has_ckpt)):
        cfg.pop("init", None)
        return
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[init] {e}")
        ckpt = read_input("Checkpoint path", init_cfg.get("from_checkpoint", "")).strip()
    ckpt = os.path.expanduser(ckpt)
    if not ckpt:
        print("[init] No checkpoint selected; skipping checkpoint initialization.")
        cfg.pop("init", None)
        return
    if not os.path.exists(ckpt):
        print(f"[init] Checkpoint not found: {ckpt}")
        cfg.pop("init", None)
        return
    init_cfg["from_checkpoint"] = ckpt
    init_cfg["load_optimizer"] = parse_bool(
        read_input("Restore optimizer state from checkpoint? (true/false)", init_cfg.get("load_optimizer", False))
    )
    init_cfg["load_step"] = parse_bool(
        read_input("Resume step counter from checkpoint? (true/false)", init_cfg.get("load_step", False))
    )
    init_cfg["load_ema"] = parse_bool(
        read_input("Load EMA weights when available? (true/false)", init_cfg.get("load_ema", True))
    )
    init_cfg["strict"] = parse_bool(
        read_input("Use strict parameter loading? (true/false)", init_cfg.get("strict", False))
    )
    cfg["init"] = init_cfg
    cfg["last_ckpt"] = ckpt
    if not init_cfg.get("load_step", False):
        cfg.setdefault("train", {})["resume"] = False


def quick_generate(cfg: Dict[str, Any]) -> None:
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[gen] {e}")
        return
    inf = cfg.setdefault("inference", {})
    prompt = read_input("Prompt", inf.get("last_prompt", "")).strip()
    if not prompt:
        print("Prompt required.")
        return
    seq_default = inf.get("seq_len", cfg.get("data", {}).get("seq_len", 512))
    seq_len = int(read_input("Sequence length", seq_default))
    max_new = int(read_input("Max new tokens", inf.get("max_new_tokens", 256)))
    steps = int(read_input("Fill steps", inf.get("steps", 8)))
    temperature = float(read_input("Temperature", inf.get("temperature", 0.9)))
    top_k = int(read_input("Top-k (0 disables)", inf.get("top_k", 20)))
    top_p = float(read_input("Top-p (0 disables)", inf.get("top_p", 0.0)))
    repetition_penalty = float(read_input("Repetition penalty", inf.get("repetition_penalty", 1.0)))
    device = read_input("Device (blank auto)", cfg.get("train", {}).get("device", default_device())).strip() or None
    log_path = read_input("Log file (blank disable)", inf.get("log_path", "")).strip()
    model, tokenizer, _ = load_generation_model(ckpt, device)
    text, stats = generate_text(
        ckpt,
        prompt,
        seq_len=seq_len,
        max_new_tokens=max_new,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
        cached=(model, tokenizer),
        return_stats=True,
    )
    print("\n=== Assistant ===\n" + text + "\n=================\n")
    if log_path:
        stats.update({"mode": "single", "system_prompt": inf.get("system_prompt")})
        log_generation(log_path, stats)
    inf.update(
        {
            "last_prompt": prompt,
            "seq_len": seq_len,
            "max_new_tokens": max_new,
            "steps": steps,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "log_path": log_path,
        }
    )


def interactive_chat(cfg: Dict[str, Any]) -> None:
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[chat] {e}")
        return
    inf = cfg.setdefault("inference", {})
    seq_len = int(read_input("Sequence length", inf.get("seq_len", 768)))
    max_new = int(read_input("Max new tokens", inf.get("max_new_tokens", 256)))
    steps = int(read_input("Fill steps", inf.get("steps", 8)))
    temperature = float(read_input("Temperature", inf.get("temperature", 0.9)))
    top_k = int(read_input("Top-k (0 disables)", inf.get("top_k", 20)))
    top_p = float(read_input("Top-p (0 disables)", inf.get("top_p", 0.0)))
    repetition_penalty = float(read_input("Repetition penalty", inf.get("repetition_penalty", 1.0)))
    system_prompt = read_input("System prompt", inf.get("system_prompt", "You are a helpful assistant.")).strip()
    device = read_input("Device (blank auto)", cfg.get("train", {}).get("device", default_device())).strip() or None
    log_path = read_input("Log file (blank disable)", inf.get("log_path", "")).strip()
    model, tokenizer, _ = load_generation_model(ckpt, device)
    session = ChatSession(system_prompt)
    print("Fractal Chat — type 'exit' to quit")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        prompt = session.build_prompt(user)
        reply, stats = generate_text(
            ckpt,
            prompt,
            seq_len=seq_len,
            max_new_tokens=max_new,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
            cached=(model, tokenizer),
            return_stats=True,
        )
        print(f"Assistant: {reply}")
        session.append(user, reply)
        if log_path:
            stats.update(
                {
                    "mode": "chat",
                    "turn": len(session.history),
                    "user_input": user,
                    "system_prompt": system_prompt,
                }
            )
            log_generation(log_path, stats)
    inf.update(
        {
            "seq_len": seq_len,
            "max_new_tokens": max_new,
            "steps": steps,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "system_prompt": system_prompt,
            "log_path": log_path,
        }
    )


def run_swarm(cfg: Dict[str, Any]) -> None:
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[swarm] {e}")
        return
    prompt = read_input("Prompt", "Describe the purpose of this swarm.").strip()
    rounds = int(read_input("Rounds", "2"))
    seq_len = int(read_input("Sequence length", cfg.get("inference", {}).get("seq_len", 768)))
    max_new = int(read_input("Max new tokens", cfg.get("inference", {}).get("max_new_tokens", 256)))
    steps = int(read_input("Fill steps", cfg.get("inference", {}).get("steps", 8)))
    temperature = float(read_input("Temperature", cfg.get("inference", {}).get("temperature", 0.8)))
    top_k = int(read_input("Top-k (0 disables)", cfg.get("inference", {}).get("top_k", 20)))
    log_path = read_input("Log file (blank disable)", cfg.get("inference", {}).get("log_path", "")).strip()
    device = read_input("Device (blank auto)", cfg.get("train", {}).get("device", default_device())).strip() or None
    cmd = (
        f"python -m fractal_neurons.orchestrator --ckpt {shlex.quote(ckpt)} "
        f"--prompt {shlex.quote(prompt)} --rounds {rounds} --seq_len {seq_len} "
        f"--max_new_tokens {max_new} --steps {steps} --temperature {temperature} --top_k {top_k}"
    )
    if device:
        cmd += f" --device {shlex.quote(device)}"
    if log_path:
        cmd += f" --log {shlex.quote(log_path)}"
    run_command(cmd)


def run_distillation(cfg: Dict[str, Any]) -> None:
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[distill] {e}")
        return
    dist_cfg = cfg.setdefault("distill", {})
    prompt_file = ensure_prompt_file(read_input("Prompt file", dist_cfg.get("prompt_file", "prompts/seed_prompts.txt")).strip() or "prompts/seed_prompts.txt")
    output = read_input("Output JSONL", dist_cfg.get("output", "runs/distill_samples.jsonl")).strip()
    n_samples = int(read_input("Number of samples", dist_cfg.get("n_samples", 1000)))
    temperature = float(read_input("Temperature", dist_cfg.get("temperature", 0.9)))
    top_k = int(read_input("Top-k (0 disables)", dist_cfg.get("top_k", 20)))
    top_p = float(read_input("Top-p (0 disables)", dist_cfg.get("top_p", 0.0)))
    accept_all = parse_bool(read_input("Accept all samples? (true/false)", dist_cfg.get("accept_all", False)))
    min_len = int(read_input("Min length (characters)", dist_cfg.get("min_len", 64)))
    max_len = int(read_input("Max length (characters)", dist_cfg.get("max_len", 1024)))
    log_path = read_input("Log file (blank disable)", dist_cfg.get("log_path", "")).strip()
    if log_path:
        log_path = os.path.expanduser(log_path)
    seq_len = int(read_input("Sequence length", dist_cfg.get("seq_len", 768)))
    max_new = int(read_input("Max new tokens", dist_cfg.get("max_new_tokens", 256)))
    steps = int(read_input("Fill steps", dist_cfg.get("steps", 8)))
    device = read_input("Device (blank auto)", cfg.get("train", {}).get("device", default_device())).strip()
    cmd = (
        f"python -m fractal_neurons.distill --ckpt {shlex.quote(ckpt)} "
        f"--prompt-file {shlex.quote(prompt_file)} --output {shlex.quote(output)} "
        f"--limit {n_samples} --temperature {temperature} --top-k {top_k} "
        f"--seq-len {seq_len} --max-new-tokens {max_new} --steps {steps} "
        f"--min-len {min_len} --max-len {max_len}"
    )
    if top_p > 0.0:
        cmd += f" --top-p {top_p}"
    if accept_all:
        cmd += " --accept-all"
    if device:
        cmd += f" --device {shlex.quote(device)}"
    if log_path:
        cmd += f" --log {shlex.quote(log_path)}"
    run_command(cmd)
    dist_cfg.update(
        {
            "prompt_file": prompt_file,
            "output": output,
            "n_samples": n_samples,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "accept_all": accept_all,
            "min_len": min_len,
            "max_len": max_len,
            "log_path": log_path,
            "seq_len": seq_len,
            "max_new_tokens": max_new,
            "steps": steps,
        }
    )


def run_eval(cfg: Dict[str, Any]) -> None:
    try:
        ckpt = select_checkpoint(cfg)
    except RuntimeError as e:
        print(f"[eval] {e}")
        return
    eval_cfg = cfg.setdefault("eval", {})
    suite = read_input("Suites (comma separated)", eval_cfg.get("suite", "reasoning,summary"))
    shots = read_input("Shots (comma separated)", eval_cfg.get("shots", "0"))
    output = read_input("Output JSON", eval_cfg.get("output", "runs/eval/report.json"))
    device = read_input("Device (blank auto)", cfg.get("train", {}).get("device", default_device())).strip()
    cmd = (
        f"python -m fractal_neurons.eval_harness --ckpt {shlex.quote(ckpt)} "
        f"--suite {shlex.quote(suite)} --shots {shlex.quote(shots)} --output {shlex.quote(output)}"
    )
    if device:
        cmd += f" --device {shlex.quote(device)}"
    run_command(cmd)
    eval_cfg.update({"suite": suite, "shots": shots, "output": output})


def run_evolution(cfg: Dict[str, Any]) -> None:
    config_path = read_input("Evolution config path", cfg.get("evo_config", "evo.yaml")).strip()
    if not config_path:
        print("Evolution config required.")
        return
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    run_command(f"python evo.py -c {shlex.quote(config_path)}")
    cfg["evo_config"] = config_path


ENGLISH_DATASET_PRESETS: List[Dict[str, Any]] = [
    {
        "label": "C4 English (streaming)",
        "dataset": "c4",
        "subset": "en",
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    },
    {
        "label": "FineWeb Edu (streaming)",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": None,
        "split": "train",
        "streaming": True,
        "trust_remote_code": False,
    },
    {
        "label": "Wikipedia 20231101 English",
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "streaming": False,
        "trust_remote_code": False,
    },
    {
        "label": "BookCorpusOpen (streaming)",
        "dataset": "bookcorpusopen",
        "subset": None,
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    },
]


def ensure_english_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    english_cfg = cfg.setdefault("english", {})
    if english_cfg is DEFAULT_CONFIG.get("english"):
        english_cfg = cfg["english"] = english_cfg.copy()
    return english_cfg


def ensure_conversational_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    conv_cfg = cfg.setdefault("conversational", {})
    if conv_cfg is DEFAULT_CONFIG.get("conversational"):
        conv_cfg = cfg["conversational"] = conv_cfg.copy()
    return conv_cfg


def ensure_conversational_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    conv_cfg = cfg.setdefault("conversational", {})
    if conv_cfg is DEFAULT_CONFIG.get("conversational"):
        conv_cfg = cfg["conversational"] = conv_cfg.copy()
    return conv_cfg

def download_english_dataset(cfg: Dict[str, Any]) -> None:
    english_cfg = ensure_english_config(cfg)
    print("\n-- Download English Dataset --")
    for idx, preset in enumerate(ENGLISH_DATASET_PRESETS, start=1):
        print(f"{idx}) {preset['label']} -> {preset['dataset']}")
    preset_choice = read_input("Choose preset (0=custom, enter number)", "0").strip()
    preset_cfg: Dict[str, Any] | None = None
    if preset_choice and preset_choice.isdigit():
        idx = int(preset_choice)
        if 1 <= idx <= len(ENGLISH_DATASET_PRESETS):
            preset_cfg = ENGLISH_DATASET_PRESETS[idx - 1]

    dataset_default = preset_cfg["dataset"] if preset_cfg else english_cfg.get("dataset", "wikipedia")
    dataset = read_input("Dataset name (HF path)", dataset_default) or dataset_default

    subset_default = preset_cfg.get("subset") if preset_cfg else english_cfg.get("subset")
    subset_input = read_input("Subset/config (blank for none)", subset_default or "").strip()
    if subset_input.lower() in {"none", "null", "-"}:
        subset_input = ""
    subset = subset_input or None

    split_default = preset_cfg.get("split") if preset_cfg else english_cfg.get("split", "train")
    split = read_input("Split", split_default) or split_default

    legacy_field = english_cfg.get("text_field") if isinstance(english_cfg.get("text_field"), str) else None
    text_key_default = preset_cfg.get("text_key") if preset_cfg else (english_cfg.get("text_key") or legacy_field)
    text_key_input = read_input("Preferred text field (blank auto)", text_key_default or "").strip()
    if text_key_input.lower() in {"none", "null", "-"}:
        text_key_input = ""
    text_key = text_key_input or None

    out_dir_default = english_cfg.get("out_dir", "data/english_corpus")
    out_dir = read_input("Output directory", out_dir_default) or out_dir_default

    streaming_default = preset_cfg.get("streaming") if preset_cfg else english_cfg.get("streaming", False)
    streaming = parse_bool(read_input("Enable streaming download? (true/false)", streaming_default))

    trust_default = preset_cfg.get("trust_remote_code") if preset_cfg else english_cfg.get("trust_remote_code", False)
    trust_remote_code = parse_bool(read_input("Trust remote code? (true/false)", trust_default))

    english_cfg.update(
        {
            "dataset": dataset,
            "subset": subset,
            "split": split,
            "text_key": text_key,
            "out_dir": out_dir,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
    )
    if "text_field" in english_cfg:
        english_cfg.pop("text_field", None)
    if "limit" in english_cfg:
        english_cfg.pop("limit", None)

    args = [
        sys.executable,
        "-m",
        "fractal_neurons.download_english",
        "--dataset",
        dataset,
        "--split",
        split,
        "--out",
        os.path.expanduser(out_dir),
    ]
    if subset:
        args.extend(["--subset", subset])
    if text_key:
        args.extend(["--text-key", text_key])
    if streaming:
        args.append("--streaming")
    if trust_remote_code:
        args.append("--trust-remote-code")

    cmd = " ".join(shlex.quote(arg) for arg in args)
    run_command(cmd)
    print("[info] english.jsonl saved to", os.path.join(os.path.expanduser(out_dir), "english.jsonl"))


def finetune_english(cfg: Dict[str, Any]) -> None:
    english_cfg = ensure_english_config(cfg)
    print("\n-- Finetune on English Dataset --")

    corpus_default = english_cfg.get("out_dir", "data/english_corpus")
    corpus_dir = read_input("Corpus directory", corpus_default) or corpus_default

    ckpt_default = english_cfg.get("finetune_ckpt") or cfg.get("last_ckpt", "")
    ckpt_input = read_input("Base checkpoint (blank to pick from runs)", ckpt_default)
    ckpt_path = ckpt_input.strip()
    if not ckpt_path:
        try:
            ckpt_path = select_checkpoint(cfg)
        except RuntimeError as exc:
            print(f"[finetune] {exc}")
            return

    def read_int(prompt: str, default: int) -> int:
        value = read_input(prompt, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            print(f"Invalid integer '{value}', keeping {default}.")
            return int(default)

    def read_float(prompt: str, default: float) -> float:
        value = read_input(prompt, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            print(f"Invalid float '{value}', keeping {default}.")
            return float(default)

    steps = read_int("Finetune steps", english_cfg.get("finetune_steps", 3000))
    lr = read_float("Learning rate", english_cfg.get("finetune_lr", 3e-4))
    batch = read_int("Batch size", english_cfg.get("finetune_batch", 24))
    seq_len = read_int("Sequence length", english_cfg.get("finetune_seq", 1024))
    num_experts = read_int("MoE experts", english_cfg.get("finetune_num_experts", 4))
    expert_top_k = read_int("MoE top-k", english_cfg.get("finetune_expert_top_k", 2))
    ema_decay = read_float("EMA decay", english_cfg.get("finetune_ema_decay", 0.9995))
    warmup = read_int("Warmup steps", english_cfg.get("finetune_warmup", 500))
    grad_clip = read_float("Grad clip", english_cfg.get("finetune_grad_clip", 1.0))
    out_cfg_default = english_cfg.get("finetune_out_path", "runs/finetune_english.yaml")
    out_cfg_path = read_input("Output config path", out_cfg_default) or out_cfg_default

    tokenizer_cfg = dict(cfg.get("tokenizer", {})) or DEFAULT_CONFIG.get("tokenizer", {}).copy()
    tokenizer_type = tokenizer_cfg.get("type", english_cfg.get("finetune_tokenizer_type", "hf"))
    if tokenizer_type.lower() != "hf":
        print(f"[finetune] textdir source requires an HF tokenizer; defaulting to {DEFAULT_TOKENIZER_NAME}.")
        tokenizer_type = "hf"
        tokenizer_name = DEFAULT_TOKENIZER_NAME
        truncation_side = "right"
        pad_multiple = 128
    else:
        tokenizer_name = tokenizer_cfg.get("name_or_path") or english_cfg.get("finetune_tokenizer_name", DEFAULT_TOKENIZER_NAME)
        truncation_side = tokenizer_cfg.get("truncation_side", english_cfg.get("finetune_tokenizer_truncation", "right"))
        pad_multiple = tokenizer_cfg.get("pad_to_multiple_of", english_cfg.get("finetune_tokenizer_pad_multiple"))

    english_cfg.update(
        {
            "out_dir": corpus_dir,
            "finetune_ckpt": ckpt_path,
            "finetune_steps": steps,
            "finetune_lr": lr,
            "finetune_batch": batch,
            "finetune_seq": seq_len,
            "finetune_num_experts": num_experts,
            "finetune_expert_top_k": expert_top_k,
            "finetune_ema_decay": ema_decay,
            "finetune_warmup": warmup,
            "finetune_grad_clip": grad_clip,
            "finetune_out_path": out_cfg_path,
            "finetune_tokenizer_type": tokenizer_type,
            "finetune_tokenizer_name": tokenizer_name,
            "finetune_tokenizer_truncation": truncation_side,
            "finetune_tokenizer_pad_multiple": pad_multiple,
        }
    )

    args = [
        sys.executable,
        "-m",
        "fractal_neurons.finetune_english",
        "--corpus",
        os.path.expanduser(corpus_dir),
        "--ckpt",
        ckpt_path,
        "--out",
        os.path.expanduser(out_cfg_path),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--batch_size",
        str(batch),
        "--seq_len",
        str(seq_len),
        "--num_experts",
        str(num_experts),
        "--expert_top_k",
        str(expert_top_k),
        "--ema_decay",
        str(ema_decay),
        "--warmup",
        str(warmup),
        "--grad_clip",
        str(grad_clip),
    ]

    args += [
        "--tokenizer_type",
        tokenizer_type,
        "--tokenizer_name",
        tokenizer_name,
        "--tokenizer_truncation_side",
        truncation_side,
    ]

    if pad_multiple is not None:
        args += ["--tokenizer_pad_multiple", str(pad_multiple)]

    cmd = " ".join(shlex.quote(arg) for arg in args)
    run_command(cmd)


def build_english_tokenizer(cfg: Dict[str, Any]) -> None:
    english_cfg = ensure_english_config(cfg)
    print("\n-- Build English Tokenizer --")

    default_corpus = english_cfg.get("tokenizer_corpus") or os.path.join(
        english_cfg.get("out_dir", "data/english_corpus"),
        "english.jsonl",
    )
    corpus_path = read_input("Corpus JSONL path", default_corpus) or default_corpus
    corpus_path = os.path.expanduser(corpus_path)
    if not os.path.exists(corpus_path):
        print(f"Corpus not found: {corpus_path}")
        return

    default_out = english_cfg.get("tokenizer_out_dir", "runs/fsi_en_v1/tokenizer")
    out_dir = read_input("Tokenizer output directory", default_out) or default_out
    out_dir = os.path.expanduser(out_dir)

    vocab_default = english_cfg.get("tokenizer_vocab_size", 65536)
    vocab_input = read_input("Vocabulary size", vocab_default)
    try:
        vocab_size = int(vocab_input)
    except (TypeError, ValueError):
        print(f"Invalid vocab '{vocab_input}', keeping {vocab_default}.")
        vocab_size = int(vocab_default)

    if os.environ.get("USE_EXISTING_TOKENIZER", "0") == "1":
        if os.path.isdir(out_dir):
            print(f"[tokenizer] Reusing existing tokenizer at {out_dir}")
            cfg.setdefault("tokenizer", {})["name_or_path"] = out_dir
            english_cfg["tokenizer_out_dir"] = out_dir
            return
        else:
            print(f"[tokenizer] USE_EXISTING_TOKENIZER=1 but {out_dir} not found; building a new tokenizer.")

    english_cfg.update(
        {
            "tokenizer_corpus": corpus_path,
            "tokenizer_out_dir": out_dir,
            "tokenizer_vocab_size": vocab_size,
        }
    )

    args = [
        sys.executable,
        "-m",
        "fractal_neurons.build_tokenizer",
        "--corpus",
        corpus_path,
        "--out",
        out_dir,
        "--vocab-size",
        str(vocab_size),
    ]

    cmd = " ".join(shlex.quote(arg) for arg in args)
    run_command(cmd)
    cfg.setdefault("tokenizer", {})["name_or_path"] = os.path.expanduser(out_dir)

def option_fix_tokenizer_decoder(ctx: MenuContext) -> None:
    cfg = ctx.cfg
    print("\n-- Fix Tokenizer Decoder --")
    default_dir = cfg.get("tokenizer", {}).get("name_or_path", DEFAULT_TOKENIZER_DIR)
    tok_dir = read_input("Tokenizer directory to fix", default_dir)
    if not tok_dir or not os.path.isdir(os.path.expanduser(tok_dir)):
        print(f"Directory not found: {tok_dir}")
        return
    cmd = [sys.executable, "fix_tokenizer_decoder.py", "--tokenizer-dir", tok_dir]
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    run_command(cmd_str, check=False)


def option_generate_conversations(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    default_count = conv.get("synthetic_examples", 500)
    count = int(read_input("Synthetic conversations to generate", default_count))
    default_out = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    out_path = os.path.expanduser(read_input("Output JSONL", default_out) or default_out)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "generate_conversations.py",
        "--count",
        str(count),
        "--output",
        out_path,
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "synthetic_examples": count,
        "conversations_path": out_path,
    })


def option_make_chat_sft(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    default_in = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    input_path = os.path.expanduser(read_input("Input conversations JSONL", default_in) or default_in)
    default_out = conv.get("chat_sft_path", "data/chat_sft.jsonl")
    output_path = os.path.expanduser(read_input("Output chat SFT JSONL", default_out) or default_out)
    system_prompt = read_input("System prompt", conv.get("system_prompt", "You are a helpful assistant.")).strip()
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/make_chat_sft.py",
        "--input",
        input_path,
        "--output",
        output_path,
        "--system",
        system_prompt,
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "conversations_path": input_path,
        "chat_sft_path": output_path,
        "system_prompt": system_prompt,
    })


def option_mix_corpus(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    english_cfg = ensure_english_config(ctx.cfg)
    default_conv = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    conv_path = os.path.expanduser(read_input("Conversational JSONL", default_conv) or default_conv)
    default_english = conv.get("english_path", english_cfg.get("out_dir", "data/english_corpus"))
    english_default_file = default_english if default_english.endswith(".jsonl") else os.path.join(default_english, "english.jsonl")
    english_path = os.path.expanduser(read_input("English corpus JSONL", english_default_file) or english_default_file)
    sample_default = conv.get("english_sample", 500)
    sample_size = int(read_input("English reservoir sample size", sample_default))
    default_out = conv.get("mixed_path", "data/mixed_corpus/mixed_data.jsonl")
    output_path = os.path.expanduser(read_input("Output mixed JSONL", default_out) or default_out)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "mix_datasets.py",
        "--conversations",
        conv_path,
        "--english",
        english_path,
        "--output",
        output_path,
        "--english-sample",
        str(sample_size),
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "conversations_path": conv_path,
        "english_path": english_path,
        "mixed_path": output_path,
        "english_sample": sample_size,
    })


def option_train_conversational(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", False))
    extra = ["--resume"] if resume_flag else None
    launch_training(
        "configs/conversational_v1.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra,
    )


def option_train_mixed(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", False))
    extra = ["--resume"] if resume_flag else None
    launch_training(
        "configs/mixed_corpus_v1.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra,
    )


def option_finetune_chat(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    base_ckpt = read_input("Base checkpoint (path or blank to choose)", ctx.cfg.get("last_ckpt", "")).strip()
    if not base_ckpt:
        try:
            base_ckpt = select_checkpoint(ctx.cfg)
        except RuntimeError as e:
            print(f"[finetune] {e}")
            base_ckpt = ""
    if base_ckpt:
        ctx.cfg["last_ckpt"] = base_ckpt
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", True))
    extra: List[str] = []
    if resume_flag:
        extra.append("--resume")
    if base_ckpt:
        extra.extend(["--init_ckpt", os.path.expanduser(base_ckpt)])
    launch_training(
        "configs/finetune_chat.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra or None,
    )



def option_generate_conversations(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    default_count = conv.get("synthetic_examples", 500)
    count = int(read_input("Synthetic conversations to generate", default_count))
    default_out = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    out_path = os.path.expanduser(read_input("Output JSONL", default_out) or default_out)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "generate_conversations.py",
        "--count",
        str(count),
        "--output",
        out_path,
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "synthetic_examples": count,
        "conversations_path": out_path,
    })


def option_make_chat_sft(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    default_in = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    input_path = os.path.expanduser(read_input("Input conversations JSONL", default_in) or default_in)
    default_out = conv.get("chat_sft_path", "data/chat_sft.jsonl")
    output_path = os.path.expanduser(read_input("Output chat SFT JSONL", default_out) or default_out)
    system_prompt = read_input("System prompt", conv.get("system_prompt", "You are a helpful assistant.")).strip()
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/make_chat_sft.py",
        "--input",
        input_path,
        "--output",
        output_path,
        "--system",
        system_prompt,
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "conversations_path": input_path,
        "chat_sft_path": output_path,
        "system_prompt": system_prompt,
    })


def option_mix_corpus(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    english_cfg = ensure_english_config(ctx.cfg)
    default_conv = conv.get("conversations_path", "data/conversational_corpus/conversations.jsonl")
    conv_path = os.path.expanduser(read_input("Conversational JSONL", default_conv) or default_conv)
    default_english = conv.get("english_path", english_cfg.get("out_dir", "data/english_corpus"))
    english_default_file = default_english if default_english.endswith(".jsonl") else os.path.join(default_english, "english.jsonl")
    english_path = os.path.expanduser(read_input("English corpus JSONL", english_default_file) or english_default_file)
    sample_default = conv.get("english_sample", 500)
    sample_size = int(read_input("English reservoir sample size", sample_default))
    default_out = conv.get("mixed_path", "data/mixed_corpus/mixed_data.jsonl")
    output_path = os.path.expanduser(read_input("Output mixed JSONL", default_out) or default_out)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "mix_datasets.py",
        "--conversations",
        conv_path,
        "--english",
        english_path,
        "--output",
        output_path,
        "--english-sample",
        str(sample_size),
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))
    conv.update({
        "conversations_path": conv_path,
        "english_path": english_path,
        "mixed_path": output_path,
        "english_sample": sample_size,
    })


def option_train_conversational(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", False))
    extra = ["--resume"] if resume_flag else None
    launch_training(
        "configs/conversational_v1.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra,
    )


def option_train_mixed(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", False))
    extra = ["--resume"] if resume_flag else None
    launch_training(
        "configs/mixed_corpus_v1.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra,
    )


def option_finetune_chat(ctx: MenuContext) -> None:
    dev = read_input("Override device (blank to use config)", "").strip()
    base_ckpt = read_input("Base checkpoint (path or blank to choose)", ctx.cfg.get("last_ckpt", "")).strip()
    if not base_ckpt:
        try:
            base_ckpt = select_checkpoint(ctx.cfg)
        except RuntimeError as e:
            print(f"[finetune] {e}")
            base_ckpt = ""
    if base_ckpt:
        ctx.cfg["last_ckpt"] = base_ckpt
    resume_flag = parse_bool(read_input("Append --resume? (true/false)", True))
    extra: List[str] = []
    if resume_flag:
        extra.append("--resume")
    if base_ckpt:
        extra.extend(["--init_ckpt", os.path.expanduser(base_ckpt)])
    launch_training(
        "configs/finetune_chat.yaml",
        verbose_flag=ctx.verbose,
        device_override=(dev or None),
        extra_args=extra or None,
    )



def option_generate_ollama_conversations(ctx: MenuContext) -> None:
    conv = ensure_conversational_config(ctx.cfg)
    default_output = conv.get("ollama_conversations_path", "data/ollama_conversations/conversations.jsonl")
    output_path = os.path.expanduser(read_input("Output JSONL for Ollama conversations", default_output) or default_output)
    
    default_model = conv.get("ollama_model", "gpt-oss:20b")
    ollama_model = read_input("Ollama model (e.g., gpt-oss:20b)", default_model).strip()

    default_target_mb = conv.get("ollama_target_mb", 10.0)
    target_mb = float(read_input("Target dataset size in MB", default_target_mb))

    default_system_prompt = conv.get("ollama_system_prompt", "You are a helpful AI assistant. Engage in a natural conversation with the user.")
    system_prompt = read_input("System prompt for AI", default_system_prompt).strip()

    default_initial_user_prompt = conv.get("ollama_initial_user_prompt", "Start a conversation about a random interesting topic.")
    initial_user_prompt = read_input("Initial user prompt to start conversation", default_initial_user_prompt).strip()

    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "generate_ollama_conversations.py",
        "--output",
        output_path,
        "--model",
        ollama_model,
        "--target_mb",
        str(target_mb),
        "--system_prompt",
        system_prompt,
        "--initial_user_prompt",
        initial_user_prompt,
    ]
    run_command(" ".join(shlex.quote(part) for part in cmd))

    conv.update({
        "ollama_conversations_path": output_path,
        "ollama_model": ollama_model,
        "ollama_target_mb": target_mb,
        "ollama_system_prompt": system_prompt,
        "ollama_initial_user_prompt": initial_user_prompt,
    })


def run_english_loop(ctx: MenuContext) -> None:
    english_cfg = ensure_english_config(ctx.cfg)
    print("\n-- Run English Distill→Swarm→Eval→Evolve Loop --")

    default_script = os.path.expanduser(english_cfg.get("loop_script", "run_fsi_loop.sh"))
    script_path = resolve_pipeline_script(default_script)
    if script_path is None:
        prompt_path = read_input("Loop script path", default_script) or default_script
        script_path = resolve_pipeline_script(prompt_path)
        if script_path is None:
            print(f"Loop script not found: {prompt_path}")
            return
    print(f"[info] Using loop script {script_path}")
    english_cfg["loop_script"] = script_path

    resume_flag = parse_bool(
        read_input("Resume evolution training? (true/false)", english_cfg.get("loop_resume", True))
    )
    english_cfg["loop_resume"] = resume_flag

    if os.access(script_path, os.X_OK):
        cmd = shlex.quote(script_path)
    else:
        cmd = f"bash {shlex.quote(script_path)}"
    env = os.environ.copy()
    env["LOOP_RESUME"] = "1" if resume_flag else "0"
    ret = run_command(cmd, check=False, env=env)
    if ret != 0:
        print(f"Loop script exited with code {ret}")


def run_english_pipeline(cfg: Dict[str, Any]) -> None:
    english_cfg = ensure_english_config(cfg)
    print("\n-- Run English FSI Pipeline --")

    default_script = os.path.expanduser(english_cfg.get("pipeline_script", "run_fsi_pipeline.sh"))
    script_path = resolve_pipeline_script(default_script)
    if script_path is None:
        prompt_path = read_input("Pipeline script path", default_script) or default_script
        script_path = resolve_pipeline_script(prompt_path)
        if script_path is None:
            print(f"Pipeline script not found: {prompt_path}")
            return
    print(f"[info] Using pipeline script {script_path}")
    english_cfg["pipeline_script"] = script_path

    resume_flag = parse_bool(
        read_input(
            "Resume training stages during pipeline? (true/false)",
            english_cfg.get("pipeline_resume", True),
        )
    )
    english_cfg["pipeline_resume"] = resume_flag

    if os.access(script_path, os.X_OK):
        cmd = shlex.quote(script_path)
    else:
        cmd = f"bash {shlex.quote(script_path)}"
    env = os.environ.copy()
    if resume_flag:
        env["PIPELINE_RESUME"] = "1"
    ret = run_command(cmd, check=False, env=env)
    if ret != 0:
        print(f"Pipeline exited with code {ret}")


def option_edit_data(ctx: MenuContext) -> None:
    edit_data(ctx.cfg)


def option_edit_model(ctx: MenuContext) -> None:
    edit_model(ctx.cfg)


def option_edit_train(ctx: MenuContext) -> None:
    edit_train(ctx.cfg)


def option_edit_tokenizer(ctx: MenuContext) -> None:
    edit_tokenizer(ctx.cfg)


def option_apply_preset(ctx: MenuContext) -> None:
    ctx.cfg = apply_preset(ctx.cfg)

def option_save_config(ctx: MenuContext) -> None:
    cfg = ctx.cfg
    if parse_bool(read_input("Adjust settings before save? (true/false to tweak config)", False)):
        cfg = adjust_run_settings(cfg)
    save_config(cfg)
    ctx.cfg = cfg

def option_load_config(ctx: MenuContext) -> None:
    cfg = load_config()
    if parse_bool(read_input("Adjust settings now? (true/false to tweak config)", False)):
        cfg = adjust_run_settings(cfg)
    ctx.cfg = cfg

def option_quick_text_train(ctx: MenuContext) -> None:
    cfg = ctx.cfg
    scan_dir = read_input("Directory to scan for text (txt/md/json/jsonl/log/py)", os.path.expanduser("~"))
    seq_len = int(read_input("Tokens per document (seq_len)", str(cfg["data"].get("seq_len", 1536))))
    stride = int(read_input("Stride between documents (tokens)", str(cfg["data"].get("stride", seq_len))))
    mask_rate = float(read_input("Mask rate (MLM corruption fraction)", str(cfg["data"].get("mask_rate", 0.15))))
    default_globs = cfg["data"].get("text_globs", ["**/*.txt", "**/*.md", "**/*.jsonl", "**/*.json", "**/*.log", "**/*.py"])
    globs_in = read_input("File patterns to include (comma or JSON list)", ", ".join(default_globs))
    text_globs = parse_list(globs_in) or default_globs
    default_ex = cfg["data"].get("exclude_globs", ["**/.git/**", "**/.cache/**", "**/__pycache__/**", "**/node_modules/**", "**/.venv/**", "**/*.bin"])
    excl_in = read_input("Patterns to exclude (comma or JSON list)", ", ".join(default_ex))
    exclude_globs = parse_list(excl_in) or default_ex
    shuffle_default = cfg["data"].get("text_shuffle_buffer", 2048)
    shuffle_buf = int(read_input("Shuffle buffer (0 disables)", str(shuffle_default)))
    reshuffle_default = cfg["data"].get("text_reshuffle_each_epoch", True)
    reshuffle = parse_bool(read_input("Reshuffle each epoch? (true/false)", reshuffle_default))
    seed_default = cfg["data"].get("text_seed", None)
    seed_in = read_input("Text shuffle seed (blank auto)", "" if seed_default is None else str(seed_default))
    text_seed = int(seed_in) if seed_in.strip() else None
    tkn = read_input(
        "HF tokenizer (path or model id)",
        str(cfg.get("tokenizer", {}).get("name_or_path", DEFAULT_TOKENIZER_NAME)),
    )
    dev = read_input("device (cuda/cpu)", cfg.get("train", {}).get("device", default_device()))
    want_adjust = parse_bool(read_input("Adjust advanced settings? (true/false to open editor)", True))
    prev_seq = cfg.get("data", {}).get("seq_len", 1536)
    prev_batch = cfg.get("train", {}).get("batch_size", 24)
    resume_flag = parse_bool(read_input("Resume from latest checkpoint? (true/false)", cfg.get("train", {}).get("resume", True)))
    cfg.setdefault("train", {})["resume"] = resume_flag
    cfg.setdefault("train", {})["resume"] = resume_flag
    resume_flag = parse_bool(read_input("Resume from latest checkpoint? (true/false)", cfg.get("train", {}).get("resume", True)))

    cfg["data"]["source"] = "textdir"
    cfg["data"]["text_root"] = scan_dir
    cfg["data"]["text_globs"] = text_globs
    cfg["data"]["exclude_globs"] = exclude_globs
    cfg["data"]["text_use_cache"] = True
    cfg["data"]["text_refresh_cache"] = False
    cfg["data"]["text_tokenize_in_workers"] = True
    cfg["data"]["seq_len"] = seq_len
    cfg["data"]["stride"] = stride
    cfg["data"]["mask_rate"] = mask_rate
    cfg["data"]["text_shuffle_buffer"] = shuffle_buf
    cfg["data"]["text_reshuffle_each_epoch"] = reshuffle
    cfg["data"]["text_seed"] = text_seed
    cfg.setdefault("tokenizer", {})
    cfg["tokenizer"]["type"] = "hf"
    cfg["tokenizer"]["name_or_path"] = tkn
    cfg["train"]["device"] = dev
    cfg["train"]["resume"] = resume_flag
    cfg["train"]["ga_enable"] = True
    cfg["model"]["interconnect"] = True

    if seq_len > 0 and prev_seq > 0:
        target_tokens = prev_batch * prev_seq
        new_batch = max(1, target_tokens // seq_len)
        if new_batch < cfg["train"].get("batch_size", prev_batch):
            print(f"[auto] Reducing batch_size from {cfg['train'].get('batch_size', prev_batch)} -> {new_batch} to keep tokens per step manageable.")
            cfg["train"]["batch_size"] = new_batch
    if seq_len >= 3072 and cfg["train"].get("torch_compile", True):
        print("[auto] Disabling torch.compile for long sequences to avoid CUDA OOM.")
        cfg["train"]["torch_compile"] = False
        cfg["train"]["compile_mode"] = "default"

    if want_adjust:
        cfg = adjust_run_settings(cfg)
    else:
        prompt_init_settings(cfg)
        cfg.setdefault("train", {})["resume"] = cfg.get("train", {}).get("resume", resume_flag)
    if cfg["train"].get("torch_compile", False):
        cfg["train"]["compile_mode"] = "max-autotune"
    cfg["train"].setdefault("use_ema", True)

    ctx.cfg = cfg
    path = save_config(cfg)
    cmd = [sys.executable, "-m", "fractal_neurons.train", "--config", path]
    if ctx.verbose:
        cmd.append("--verbose")
    if dev:
        cmd += ["--device", dev]
    if cfg.get("train", {}).get("resume", False):
        cmd.append("--resume")
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    ret = run_command(cmd_str, check=False)
    if ret != 0:
        print(f"Training process exited with code {ret}")
    ctx.cfg = cfg


def option_quick_system_train(ctx: MenuContext) -> None:
    cfg = ctx.cfg
    scan_dir = read_input("Directory to scan recursively (root folder)", os.path.expanduser("~"))
    seq_len = int(read_input("Tokens per document (seq_len)", str(cfg["data"].get("seq_len", 1536))))
    stride = int(read_input("Stride between documents (tokens)", str(cfg["data"].get("stride", 512))))
    mask_rate = float(read_input("Mask rate (MLM corruption fraction)", str(cfg["data"].get("mask_rate", 0.15))))
    dev = read_input("device (cuda/cpu)", cfg.get("train", {}).get("device", default_device()))
    want_adjust = parse_bool(read_input("Adjust advanced settings? (true/false to open editor)", True))
    prev_seq = cfg.get("data", {}).get("seq_len", 1536)
    prev_batch = cfg.get("train", {}).get("batch_size", 24)

    cfg["data"]["allow_all"] = True
    cfg["data"]["root"] = scan_dir
    cfg["data"]["seq_len"] = seq_len
    cfg["data"]["stride"] = stride
    cfg["data"]["mask_rate"] = mask_rate
    cfg["data"]["max_index_items"] = cfg["data"].get("max_index_items", 500000)
    cfg["train"]["device"] = dev

    if seq_len > 0 and prev_seq > 0:
        target_tokens = prev_batch * prev_seq
        new_batch = max(1, target_tokens // seq_len)
        if new_batch < cfg["train"].get("batch_size", prev_batch):
            print(f"[auto] Reducing batch_size from {cfg['train'].get('batch_size', prev_batch)} -> {new_batch} to keep tokens per step manageable.")
            cfg["train"]["batch_size"] = new_batch
    if seq_len >= 3072 and cfg["train"].get("torch_compile", True):
        print("[auto] Disabling torch.compile for long sequences to avoid CUDA OOM.")
        cfg["train"]["torch_compile"] = False
        cfg["train"]["compile_mode"] = "default"

    if want_adjust:
        cfg = adjust_run_settings(cfg)
    else:
        prompt_init_settings(cfg)
        cfg.setdefault("train", {})["resume"] = cfg.get("train", {}).get("resume", resume_flag)
    if cfg["train"].get("torch_compile", False):
        cfg["train"]["compile_mode"] = "max-autotune"
    cfg["train"].setdefault("use_ema", True)

    ctx.cfg = cfg
    path = save_config(cfg)
    env = os.environ.copy()
    if os.path.abspath(os.path.expanduser(scan_dir)) != "/":
        env["TRAIN_ALLOW_ALL_FILES"] = "1"
    else:
        ans = read_input("You chose '/'. Confirm full-root scan? (y/N)", "N")
        if ans.lower() in ("y", "yes", "1", "true"):
            env["TRAIN_ALLOW_ALL_FILES"] = "1"
        else:
            print("Aborting launch; full-root scan not confirmed.")
            return

    cmd = [sys.executable, "-m", "fractal_neurons.train", "--config", path, "--scan_dir", scan_dir]
    if ctx.verbose:
        cmd.append("--verbose")
    if dev:
        cmd += ["--device", dev]
    if cfg.get("train", {}).get("resume", False):
        cmd.append("--resume")
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    print("Launching:", cmd_str)
    ret = run_command(cmd_str, check=False, env=env)
    if ret != 0:
        print(f"Training process exited with code {ret}")
    ctx.cfg = cfg


def option_build_cache(ctx: MenuContext) -> None:
    build_text_corpus_cache(ctx.cfg)


def option_start_training(ctx: MenuContext) -> None:
    path = read_input("Config path to use (YAML file)", "configs/system_train.yaml")
    if not os.path.exists(path):
        print("Config not found. Save it first (option 6) or enter an existing path.")
        return
    if parse_bool(read_input("Override values before launch? (true/false)", True)):
        cfg = load_config()
        cfg = adjust_run_settings(cfg)
        path = save_config(cfg)
        ctx.cfg = cfg
    resume_flag = parse_bool(read_input("Resume from latest checkpoint? (true/false)", ctx.cfg.get("train", {}).get("resume", True)))
    ctx.cfg.setdefault("train", {})["resume"] = resume_flag
    dev = read_input("Override device (blank to use config)", "")
    extra = []
    if resume_flag and not ctx.cfg.get("init", {}).get("from_checkpoint"):
        extra.append("--resume")
    launch_training(path, verbose_flag=ctx.verbose, device_override=(dev or None), extra_args=extra)

def option_single_prompt(ctx: MenuContext) -> None:
    quick_generate(ctx.cfg)

def option_chat(ctx: MenuContext) -> None:
    interactive_chat(ctx.cfg)

def option_run_inference(ctx: MenuContext) -> None:
    ckpt = read_input("Path to checkpoint .pt or run directory (inference source)", "")
    if not ckpt or not os.path.exists(ckpt):
        print("Path not found. You can provide a specific .pt file or a run directory (it will pick the latest .pt).")
        return
    src_mode = read_input("Source: (f)ile or (t)ext", "f").lower()
    args = [sys.executable, "-m", "fractal_neurons.infer", "--ckpt", ckpt]
    if src_mode.startswith("f"):
        fpath = read_input("File path (input document)", "")
        if not fpath or not os.path.exists(fpath):
            print("File not found.")
            return
        args += ["--file", fpath]
    else:
        text = read_input("Text (raw string to analyse)", "The quick brown fox")
        args += ["--text", text]
    dev = read_input("device (cuda/cpu)", default_device())
    args += [
        "--seq_len",
        read_input("seq_len (tokens to process)", "512"),
        "--steps",
        read_input("fill steps (iterations of reconstruction)", "8"),
        "--mask_rate",
        read_input("mask_rate (fraction to mask)", "0.25"),
    ]
    if dev:
        args += ["--device", dev]
    cmd_str = " ".join(shlex.quote(part) for part in args)
    ret = run_command(cmd_str, check=False)
    if ret != 0:
        print(f"Inference exited with code {ret}")

def option_run_swarm(ctx: MenuContext) -> None:
    run_swarm(ctx.cfg)

def option_run_distill(ctx: MenuContext) -> None:
    run_distillation(ctx.cfg)

def option_run_eval(ctx: MenuContext) -> None:
    run_eval(ctx.cfg)

def option_run_evolution(ctx: MenuContext) -> None:
    run_evolution(ctx.cfg)

def option_download_english(ctx: MenuContext) -> None:
    download_english_dataset(ctx.cfg)

def option_finetune_english(ctx: MenuContext) -> None:
    finetune_english(ctx.cfg)

def option_build_english_tokenizer(ctx: MenuContext) -> None:
    build_english_tokenizer(ctx.cfg)

def option_run_pipeline(ctx: MenuContext) -> None:
    run_english_pipeline(ctx.cfg)

def option_run_loop(ctx: MenuContext) -> None:
    run_english_loop(ctx.cfg)

def option_run_robustness_eval(ctx: MenuContext) -> None:
    try:
        ckpt = select_checkpoint(ctx.cfg)
    except RuntimeError as e:
        print(f"[robust_eval] {e}")
        return

    eval_cfg = ctx.cfg.setdefault("robust_eval", {})
    default_out = eval_cfg.get("output", "runs/robust_eval.jsonl")
    out_path = read_input("Output JSONL for robustness evaluation", default_out) or default_out

    seq_len = int(read_input("Sequence length", eval_cfg.get("seq_len", 768)))
    max_new = int(read_input("Max new tokens", eval_cfg.get("max_new_tokens", 256)))
    steps = int(read_input("Fill steps", eval_cfg.get("steps", 8)))
    temperature = float(read_input("Temperature", eval_cfg.get("temperature", 0.8)))
    top_k = int(read_input("Top-k (0 disables)", eval_cfg.get("top_k", 40)))
    top_p = float(read_input("Top-p (0 disables)", eval_cfg.get("top_p", 0.95)))
    repetition_penalty = float(read_input("Repetition penalty", eval_cfg.get("repetition_penalty", 1.05)))
    device = read_input("Device (blank auto)", ctx.cfg.get("train", {}).get("device", default_device())).strip() or None

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tools/robust_lila_eval.py",
        "--ckpt", shlex.quote(ckpt),
        "--out", shlex.quote(out_path),
        "--seq_len", str(seq_len),
        "--max_new", str(max_new),
        "--steps", str(steps),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
        "--repetition_penalty", str(repetition_penalty),
    ]
    if device:
        cmd.extend(["--device", shlex.quote(device)])

    run_command(" ".join(cmd))

    eval_cfg.update({
        "output": out_path,
        "seq_len": seq_len,
        "max_new_tokens": max_new,
        "steps": steps,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    })
    print(f"[info] Robustness evaluation results saved to {out_path}")

def option_toggle_verbose(ctx: MenuContext) -> None:
    ctx.verbose = not ctx.verbose
    print(f"Verbose mode is now {'ON' if ctx.verbose else 'OFF'}.")

def option_quit(ctx: MenuContext) -> None:
    ctx.running = False

def adjust_run_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive adjustment of common run settings just before launch/save."""
    d = cfg.setdefault("data", {})
    m = cfg.setdefault("model", {})
    t = cfg.setdefault("train", {})
    tk = cfg.setdefault("tokenizer", {})

    print("\n-- Adjust Run Settings --")
    # Core training controls
    t["run_name"] = read_input("run_name (used for runs/<name> folder)", t.get("run_name", f"fractal_{int(time.time())}")) or t.get("run_name")
    t["device"] = read_input("device (choose cuda or cpu)", t.get("device", default_device())) or t.get("device", default_device())
    t["batch_size"] = int(read_input("batch_size (global samples per step)", t.get("batch_size", 24)))
    t["steps"] = int(read_input("steps (total optimizer iterations)", t.get("steps", 50000)))
    t["lr"] = float(read_input("lr (learning rate)", t.get("lr", 1.5e-3)))
    t["weight_decay"] = float(read_input("weight_decay (L2 regularization)", t.get("weight_decay", 0.01)))
    t["grad_accum"] = int(read_input("grad_accum (iterations per optimizer step)", t.get("grad_accum", 1)))
    t["num_workers"] = int(read_input("num_workers (DataLoader subprocesses)", t.get("num_workers", 28)))
    t["log_every"] = int(read_input("log_every (steps between console logs)", t.get("log_every", 50)))
    t["ckpt_every"] = int(read_input("ckpt_every (steps between checkpoints)", t.get("ckpt_every", 1000)))
    t["torch_compile"] = parse_bool(read_input("torch_compile (enable torch.compile for speed?)", t.get("torch_compile", True)))
    t["tf32"] = parse_bool(read_input("tf32 (allow NVIDIA TF32 matmuls?)", t.get("tf32", True)))
    t["resume"] = parse_bool(read_input("resume (load latest checkpoint if present?)", t.get("resume", True)))
    t["seed"] = int(read_input("seed (random seed)", t.get("seed", 1337)))
    t["deterministic"] = parse_bool(read_input("deterministic (force deterministic kernels?)", t.get("deterministic", False)))

    # Data controls
    d["seq_len"] = int(read_input("data.seq_len (tokens per sample)", d.get("seq_len", 1536)))
    d["stride"] = int(read_input("data.stride (sliding window hop)", d.get("stride", d.get("seq_len", 1536))))
    d["mask_rate"] = float(read_input("data.mask_rate (MLM corruption fraction)", d.get("mask_rate", 0.15)))

    # GA controls
    t["ga_enable"] = parse_bool(read_input("ga_enable (toggle gate evolution step)", t.get("ga_enable", False)))
    t["ga_every"] = int(read_input("ga_every (steps between GA sweeps)", t.get("ga_every", 500)))
    t["ga_population"] = int(read_input("ga_population (candidate gates per sweep)", t.get("ga_population", 8)))
    t["ga_sigma"] = float(read_input("ga_sigma (gate mutation stddev)", t.get("ga_sigma", 0.05)))
    t["compile_mode"] = read_input("compile_mode (torch.compile mode)", t.get("compile_mode", "default")) or t.get("compile_mode", "default")
    t["prefetch_gpu"] = parse_bool(read_input("prefetch_gpu (enable CUDA prefetcher?)", t.get("prefetch_gpu", True)))
    t["prefetch_gpu_batches"] = int(read_input("prefetch_gpu_batches (batches to overlap)", t.get("prefetch_gpu_batches", 2)))
    t["adam_fused"] = parse_bool(read_input("adam_fused (use fused AdamW?)", t.get("adam_fused", True)))
    t["warmup_steps"] = int(read_input("warmup_steps (optimizer warmup iters)", t.get("warmup_steps", 0)))
    t["lr_min"] = float(read_input("lr_min (minimum LR for cosine)", t.get("lr_min", 0.0)))
    t["cosine_cycle"] = int(read_input("cosine_cycle (updates for cosine schedule)", t.get("cosine_cycle", 0)))
    grad_clip_in = read_input("grad_clip (max grad norm, blank disables)", "" if t.get("grad_clip") is None else str(t.get("grad_clip")))
    t["grad_clip"] = float(grad_clip_in) if grad_clip_in.strip() else None
    t["use_ema"] = parse_bool(read_input("use_ema (maintain EMA weights?)", t.get("use_ema", True)))
    t["ema_decay"] = float(read_input("ema_decay (decay rate)", t.get("ema_decay", 0.999)))

    # QFP controls
    t["qfp_enable"] = parse_bool(read_input("qfp_enable (toggle Quantum Fractal Processing)", t.get("qfp_enable", False)))
    if t["qfp_enable"]:
        t["qfp_time_complex_real"] = float(read_input("qfp_time_complex_real (real part of time_complex)", t.get("qfp_time_complex_real", -2.999999)))
        t["qfp_time_complex_imag"] = float(read_input("qfp_time_complex_imag (imaginary part of time_complex)", t.get("qfp_time_complex_imag", -0.002189)))

    if parse_bool(read_input("Adjust text shuffle options? (true/false)", False)):
        d["text_shuffle_buffer"] = int(read_input("data.text_shuffle_buffer (0 disables)", d.get("text_shuffle_buffer", 0)))
        d["text_reshuffle_each_epoch"] = parse_bool(read_input("data.text_reshuffle_each_epoch (reshuffle every epoch?)", d.get("text_reshuffle_each_epoch", True)))
        seed_val = d.get("text_seed", "")
        seed_in = read_input("data.text_seed (blank for auto)", "" if seed_val is None else str(seed_val))
        d["text_seed"] = int(seed_in) if seed_in.strip() else None

    # Optional model hyperparams
    if parse_bool(read_input("Adjust model hyperparameters? (true/false for advanced tuning)", False)):
        m["dim"] = int(read_input("model.dim (embedding width)", m.get("dim", 512)))
        m["depth"] = int(read_input("model.depth (fractal depth levels)", m.get("depth", 6)))
        m["fanout"] = int(read_input("model.fanout (branches per node)", m.get("fanout", 10)))
        m["use_fp16"] = parse_bool(read_input("model.use_fp16 (enable mixed precision?)", m.get("use_fp16", True)))
        m["droppath_rate"] = float(read_input("model.droppath_rate (level drop probability)", m.get("droppath_rate", 0.1)))
        m["branch_dropout"] = float(read_input("model.branch_dropout (child mask probability)", m.get("branch_dropout", 0.1)))
        m["interconnect"] = parse_bool(read_input("model.interconnect (enable level attention?)", m.get("interconnect", True)))
        m["interconnect_heads"] = int(read_input("model.interconnect_heads (attention heads)", m.get("interconnect_heads", 4)))
        m["interconnect_dropout"] = float(read_input("model.interconnect_dropout (attention dropout)", m.get("interconnect_dropout", 0.05)))
        m["num_experts"] = int(read_input("model.num_experts (0 disables MoE)", m.get("num_experts", 0)))
        m["expert_hidden"] = int(read_input("model.expert_hidden (hidden width per expert)", m.get("expert_hidden", m.get("dim", 512) * 2)))
        m["expert_top_k"] = int(read_input("model.expert_top_k (experts per token)", m.get("expert_top_k", 2)))

    # FMM controls
    if parse_bool(read_input("Adjust FMM settings? (true/false)", False)):
        m["use_fmm"] = parse_bool(read_input("use_fmm (enable Fractal Memory Matrix?)", m.get("use_fmm", False)))
        if m["use_fmm"]:
            m["fmm_max_nodes"] = int(read_input("fmm_max_nodes (maximum nodes in FMM)", m.get("fmm_max_nodes", 10000)))

    # Optional tokenizer settings
    if parse_bool(read_input("Adjust tokenizer settings? (true/false to switch tokenizer)", False)):
        tk["type"] = read_input("type (bytes or hf tokenizer)", tk.get("type", "bytes")) or tk.get("type", "bytes")
        if tk["type"].lower() == "hf":
            tk["name_or_path"] = read_input(
                "name_or_path (HF model id or path)",
                tk.get("name_or_path", DEFAULT_TOKENIZER_NAME),
            )
            tk["truncation_side"] = read_input("truncation_side (truncate left/right)", tk.get("truncation_side", "right"))
            p2 = read_input("pad_to_multiple_of (multiple or blank)", str(tk.get("pad_to_multiple_of", "")))
            tk["pad_to_multiple_of"] = int(p2) if p2.strip().isdigit() else None

    prompt_init_settings(cfg)

    # Persist updated sections
    cfg["data"], cfg["model"], cfg["train"], cfg["tokenizer"] = d, m, t, tk
    print("Run settings updated.")
    return cfg


def show_nodes(cfg: Dict[str, Any]) -> int:
    depth = int(cfg["model"]["depth"]) or 1
    fanout = int(cfg["model"]["fanout"]) or 1
    # sum_{i=0..depth-1} fanout^i
    nodes = 0
    cur = 1
    for _ in range(depth):
        nodes += cur
        cur *= fanout
    return nodes


def edit_data(cfg: Dict[str, Any]) -> None:
    d = cfg["data"]
    print("\n-- Edit Data Settings --")
    d["source"] = read_input("source (choose system, textdir, or hf)", d.get("source", "textdir")) or d.get("source", "textdir")
    if d.get("include_globs") is None:
        d["include_globs"] = []
    if d.get("exclude_globs") is None:
        d["exclude_globs"] = []
    inc = read_input("include_globs (patterns to include)", ", ".join(d.get("include_globs", [])))
    exc = read_input("exclude_globs (patterns to always skip)", ", ".join(d.get("exclude_globs", [])))
    d["include_globs"] = parse_list(inc)
    d["exclude_globs"] = parse_list(exc)
    d["seq_len"] = int(read_input("seq_len (tokens per sample)", d["seq_len"]))
    d["stride"] = int(read_input("stride (window hop size)", d["stride"]))
    d["mask_rate"] = float(read_input("mask_rate (MLM corruption rate)", d["mask_rate"]))
    d["allow_all"] = parse_bool(read_input("allow_all (ignore include_globs?)", d["allow_all"]))
    d["root"] = read_input("root (scan base when allow_all)", d["root"]) or d["root"]
    d["max_file_mb"] = int(read_input("max_file_mb (skip files larger than MB)", d["max_file_mb"]))
    if d["source"].lower() == "textdir":
        d["text_use_cache"] = parse_bool(read_input("text_use_cache (cache file list?)", d.get("text_use_cache", True)))
        d["text_refresh_cache"] = parse_bool(read_input("text_refresh_cache (force rebuild cache?)", d.get("text_refresh_cache", False)))
        d["text_tokenize_in_workers"] = parse_bool(read_input("text_tokenize_in_workers (HF tokenization in workers?)", d.get("text_tokenize_in_workers", True)))
        d["text_shuffle_buffer"] = int(read_input("text_shuffle_buffer (lines to shuffle buffer, 0 disables)", d.get("text_shuffle_buffer", 0)))
        d["text_reshuffle_each_epoch"] = parse_bool(read_input("text_reshuffle_each_epoch (reshuffle buffer every epoch?)", d.get("text_reshuffle_each_epoch", True)))
        seed_val = d.get("text_seed", "")
        seed_in = read_input("text_seed (blank for auto)", "" if seed_val is None else str(seed_val))
        d["text_seed"] = int(seed_in) if seed_in.strip() else None
    if d["source"].lower() == "hf":
        d["hf_path"] = read_input("hf_path (dataset id, e.g. wikipedia)", d.get("hf_path", "wikipedia"))
        d["hf_name"] = read_input("hf_name (subset like 20220301.en)", d.get("hf_name", "20220301.en"))
        d["hf_split"] = read_input("hf_split (train/validation/etc)", d.get("hf_split", "train"))
        d["hf_streaming"] = parse_bool(read_input("hf_streaming (enable streaming loader?)", d.get("hf_streaming", True)))
        d["hf_text_field"] = read_input("hf_text_field (field containing text)", d.get("hf_text_field", "")) or None


def edit_model(cfg: Dict[str, Any]) -> None:
    m = cfg["model"]
    print("\n-- Edit Model Settings --")
    m["dim"] = int(read_input("dim (embedding width)", m["dim"]))
    m["depth"] = int(read_input("depth (fractal levels)", m["depth"]))
    m["fanout"] = int(read_input("fanout (children per node)", m["fanout"]))
    m["use_fp16"] = parse_bool(read_input("use_fp16 (enable mixed precision?)", m["use_fp16"]))
    m["droppath_rate"] = float(read_input("droppath_rate (level drop probability 0..1)", m.get("droppath_rate", 0.0)))
    m["branch_dropout"] = float(read_input("branch_dropout (child drop probability 0..1)", m.get("branch_dropout", 0.0)))
    m["interconnect"] = parse_bool(read_input("interconnect (enable level attention?)", m.get("interconnect", True)))
    m["interconnect_heads"] = int(read_input("interconnect_heads (attention heads)", m.get("interconnect_heads", 2)))
    m["interconnect_dropout"] = float(read_input("interconnect_dropout (attention dropout)", m.get("interconnect_dropout", 0.0)))
    m["num_experts"] = int(read_input("num_experts (0 disables MoE)", m.get("num_experts", 0)))
    m["expert_hidden"] = int(read_input("expert_hidden (hidden width per expert)", m.get("expert_hidden", m.get("dim", 512) * 2)))
    m["expert_top_k"] = int(read_input("expert_top_k (experts per token)", m.get("expert_top_k", 2)))
    print(f"Estimated total nodes: {show_nodes(cfg)}")

def edit_train(cfg: Dict[str, Any]) -> None:
    t = cfg["train"]
    print("\n-- Edit Train Settings --")
    t["run_name"] = read_input("run_name (used for runs/<name>)", t["run_name"]) or t["run_name"]
    t["device"] = read_input("device (cuda or cpu)", t["device"]) or t["device"]
    t["batch_size"] = int(read_input("batch_size (samples per step)", t["batch_size"]))
    t["steps"] = int(read_input("steps (optimizer iterations)", t["steps"]))
    t["lr"] = float(read_input("lr (learning rate)", t["lr"]))
    t["weight_decay"] = float(read_input("weight_decay (L2 regularization)", t["weight_decay"]))
    t["grad_accum"] = int(read_input("grad_accum (steps before optimizer step)", t["grad_accum"]))
    t["num_workers"] = int(read_input("num_workers (DataLoader workers)", t["num_workers"]))
    t["pin_memory"] = parse_bool(read_input("pin_memory (page-lock CUDA transfers?)", t["pin_memory"]))
    t["prefetch_factor"] = int(read_input("prefetch_factor (batches prefetched per worker)", t["prefetch_factor"]))
    t["persistent_workers"] = parse_bool(read_input("persistent_workers (keep loader workers alive?)", t["persistent_workers"]))
    t["torch_compile"] = parse_bool(read_input("torch_compile (enable compile cache?)", t["torch_compile"]))
    t["tf32"] = parse_bool(read_input("tf32 (allow TF32 matmuls?)", t["tf32"]))
    t["compile_mode"] = read_input("compile_mode (torch.compile mode)", t.get("compile_mode", "default")) or t.get("compile_mode", "default")
    t["prefetch_gpu"] = parse_bool(read_input("prefetch_gpu (enable CUDA prefetcher?)", t.get("prefetch_gpu", True)))
    t["prefetch_gpu_batches"] = int(read_input("prefetch_gpu_batches (batches to overlap)", t.get("prefetch_gpu_batches", 2)))
    t["adam_fused"] = parse_bool(read_input("adam_fused (use fused AdamW?)", t.get("adam_fused", True)))
    t["warmup_steps"] = int(read_input("warmup_steps (optimizer warmup iters)", t.get("warmup_steps", 0)))
    t["lr_min"] = float(read_input("lr_min (minimum LR for cosine)", t.get("lr_min", 0.0)))
    t["cosine_cycle"] = int(read_input("cosine_cycle (updates for cosine schedule)", t.get("cosine_cycle", 0)))
    grad_clip_in = read_input("grad_clip (max grad norm, blank disables)", "" if t.get("grad_clip") is None else str(t.get("grad_clip")))
    t["grad_clip"] = float(grad_clip_in) if grad_clip_in.strip() else None
    t["use_ema"] = parse_bool(read_input("use_ema (maintain EMA weights?)", t.get("use_ema", True)))
    t["ema_decay"] = float(read_input("ema_decay (decay rate)", t.get("ema_decay", 0.999)))
    t["log_every"] = int(read_input("log_every (steps between prints)", t["log_every"]))
    t["ckpt_every"] = int(read_input("ckpt_every (steps between checkpoints)", t["ckpt_every"]))
    t["out_dir"] = read_input("out_dir (where to store checkpoints)", t["out_dir"]) or t["out_dir"]

def edit_tokenizer(cfg: Dict[str, Any]) -> None:
    tk = cfg.get("tokenizer", {})
    print("\n-- Edit Tokenizer --")
    tk["type"] = read_input("type (bytes or hf tokenizer)", tk.get("type", "bytes")) or tk.get("type", "bytes")
    if tk["type"].lower() == "hf":
        tk["name_or_path"] = read_input(
            "name_or_path (HF model or local dir)",
            tk.get("name_or_path", DEFAULT_TOKENIZER_NAME),
        )
        tk["truncation_side"] = read_input("truncation_side (left/right)", tk.get("truncation_side", "right"))
        p2 = read_input("pad_to_multiple_of (optional multiple)", str(tk.get("pad_to_multiple_of", "")))
        tk["pad_to_multiple_of"] = int(p2) if p2.strip().isdigit() else None
    cfg["tokenizer"] = tk

def list_presets() -> Dict[str, str]:
    return {
        "4090_20gb_hf": "configs/preset_4090_20gb_hf.yaml",
        "7950x_4090_system": "configs/preset_7950x_4090.yaml",
        "english_wiki": "configs/english_hf.yaml",
        "fast_7950x_4090": "configs/fast_7950x_4090.yaml",
    }

def build_text_corpus_cache(cfg: Dict[str, Any]) -> None:
    print("\n-- Build Text Corpus Cache --")
    default_root = cfg.get("data", {}).get("text_root") or os.path.expanduser("~")
    root = read_input("Directory to index (text_root)", default_root) or default_root
    default_globs = cfg.get("data", {}).get("text_globs") or [
        "**/*.txt", "**/*.md", "**/*.jsonl", "**/*.json", "**/*.log", "**/*.py"
    ]
    globs_in = read_input("File patterns to include (comma/space/JSON)", ", ".join(default_globs))
    include_globs = parse_list(globs_in) or default_globs
    default_ex = cfg.get("data", {}).get("exclude_globs") or [
        "**/.git/**", "**/.cache/**", "**/__pycache__/**", "**/node_modules/**", "**/.venv/**", "**/*.bin"
    ]
    excl_in = read_input("Patterns to exclude (comma/space/JSON)", ", ".join(default_ex))
    exclude_globs = parse_list(excl_in) or default_ex
    refresh = parse_bool(read_input("Refresh cache? (true/false)", False))
    verbose = parse_bool(read_input("Verbose scan output? (true/false)", True))

    try:
        from fractal_neurons.data_text import TextDirStream

        stream = TextDirStream(
            root=root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            verbose=verbose,
            use_cache=True,
            refresh_cache=refresh,
        )
        count = len(stream.files)
        cache_dir = stream.cache_dir if getattr(stream, "cache_dir", None) else "<unknown>"
        print(f"[textdir] indexed {count} files. Cache directory: {cache_dir}")
    except RuntimeError as e:
        print(f"[error] {e}")
    except Exception as e:
        print(f"[error] failed to build corpus: {e}")

def apply_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    presets = list_presets()
    print("\n-- Presets --")
    for i, (k, v) in enumerate(presets.items(), start=1):
        print(f"{i}) {k} -> {v}")
    idx = read_input("Choose preset number (enter list index)", "1")
    try:
        i = int(idx)
        key = list(presets.keys())[i - 1]
        path = presets[key]
    except Exception:
        print("Invalid selection.")
        return cfg
    if not os.path.exists(path):
        print(f"Preset file not found: {path}")
        return cfg
    try:
        if yaml is not None:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            import json
            with open(path, "r") as f:
                data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError
        print(f"Applied preset from {path}")
        return data
    except Exception as e:
        print(f"Failed to apply preset: {e}")
        return cfg

def save_config(cfg: Dict[str, Any]) -> str:
    default_path = "configs/system_train.yaml"
    resp = read_input("Save to path (destination config file)", default_path)
    # Treat 'y'/'yes' as accept default, not a literal path
    path = default_path if resp.strip().lower() in ("y", "yes") else resp
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if yaml is not None:
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        with open(path, "w") as f:
            f.write(json.dumps(cfg, indent=2))  # type: ignore[name-defined]
    print(f"Saved config to {path}")
    return path

def load_config() -> Dict[str, Any]:
    path = read_input("Load from path (existing config file)", "configs/system_train.yaml")
    if not os.path.exists(path):
        print(f"No file at {path}")
        return DEFAULT_CONFIG.copy()
    try:
        if yaml is not None:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                import json
                data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        print(f"Loaded config from {path}")
        return data
    except Exception as e:
        print(f"Failed to load config: {e}")
        return DEFAULT_CONFIG.copy()

def launch_training(
    cfg_path: str,
    verbose_flag: bool = False,
    device_override: str | None = None,
    extra_args: List[str] | None = None,
) -> None:
    env = os.environ.copy()
    # If allow_all on in config, require explicit confirmation unless already set
    try:
        if yaml is not None:
            with open(cfg_path, "r") as f:
                c = yaml.safe_load(f)
        else:
            import json
            with open(cfg_path, "r") as f:
                c = json.load(f)
        allow_all = bool(c.get("data", {}).get("allow_all", False))
    except Exception:
        allow_all = False
    # Only require confirmation when scanning full root
    root = str(c.get("data", {}).get("root", "/")) if c else "/"
    if allow_all and os.path.abspath(os.path.expanduser(root)) == "/" and env.get("TRAIN_ALLOW_ALL_FILES", "0") != "1":
        ans = read_input("allow_all with root=/ detected. Set TRAIN_ALLOW_ALL_FILES=1? (y/N)", "N")
        if ans.lower() in ("y", "yes", "1", "true"):
            env["TRAIN_ALLOW_ALL_FILES"] = "1"
        else:
            print("Not starting training; confirmation is required for full-root allow_all mode.")
            return

    init_cfg = (c or {}).get("init", {}) if isinstance(c, dict) else {}
    init_ckpt = init_cfg.get("from_checkpoint")
    if init_ckpt:
        init_ckpt = os.path.abspath(os.path.expanduser(str(init_ckpt)))

    cmd = [sys.executable, "-m", "fractal_neurons.train", "--config", cfg_path]
    if verbose_flag:
        cmd.append("--verbose")
    if device_override:
        cmd += ["--device", device_override]
    if extra_args:
        cmd.extend(extra_args)
    if init_ckpt and "--init_ckpt" not in cmd:
        cmd += ["--init_ckpt", init_ckpt]
    # Add --resume when enabled and no explicit init override
    try:
        should_resume = bool(c.get("train", {}).get("resume", True)) and not init_ckpt
        if should_resume and "--resume" not in cmd:
            cmd.append("--resume")
    except Exception:
        pass
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    ret = run_command(cmd_str, check=False, env=env)
    if ret != 0:
        print(f"Training process exited with code {ret}")

def build_menu_options() -> List[MenuOption]:
    return [
        MenuOption("Edit Data", option_edit_data),
        MenuOption(
            "Edit Model",
            option_edit_model,
            label_getter=lambda ctx: f"Edit Model (nodes: {show_nodes(ctx.cfg)})",
        ),
        MenuOption("Edit Train", option_edit_train),
        MenuOption("Edit Tokenizer", option_edit_tokenizer),
        MenuOption("Apply Preset", option_apply_preset),
        MenuOption("Save Config", option_save_config),
        MenuOption("Load Config", option_load_config),
        MenuOption("Quick: Text corpus and train (textdir)", option_quick_text_train),
        MenuOption("Quick: Scan system and train (bytes)", option_quick_system_train),
        MenuOption("Build Text Corpus Cache", option_build_cache),
        MenuOption("Start Training", option_start_training),
        MenuOption("Single Prompt Generation", option_single_prompt),
        MenuOption("Chat Session", option_chat),
        MenuOption("Run Inference (full script)", option_run_inference),
        MenuOption("Run Swarm Orchestration", option_run_swarm),
        MenuOption("Run Self-Distillation", option_run_distill),
        MenuOption("Run Evaluation Harness", option_run_eval),
        MenuOption("Run Robustness Evaluation", option_run_robustness_eval),
        MenuOption("Run Evolution Pipeline", option_run_evolution),
        MenuOption("Download English Dataset", option_download_english),
        MenuOption("Finetune on English Dataset", option_finetune_english),
        MenuOption("Build English Tokenizer", option_build_english_tokenizer),
        MenuOption("Generate Conversational Dataset", option_generate_conversations),
        MenuOption("Generate Ollama Conversations", option_generate_ollama_conversations),
        MenuOption("Make Chat SFT Dataset", option_make_chat_sft),
        MenuOption("Mix Conversational + English Data", option_mix_corpus),
        MenuOption("Train Conversational Model", option_train_conversational),
        MenuOption("Train Mixed Corpus Model", option_train_mixed),
        MenuOption("Finetune Chat Model", option_finetune_chat),
        MenuOption("Fix Tokenizer Decoder", option_fix_tokenizer_decoder),
        MenuOption("Run English FSI Pipeline", option_run_pipeline),
        MenuOption("Run English FSI Loop", option_run_loop),
        MenuOption(
            "Toggle Verbose Mode",
            option_toggle_verbose,
            label_getter=lambda ctx: f"Toggle Verbose Mode [{'ON' if ctx.verbose else 'OFF'}]",
        ),
        MenuOption("Quit", option_quit),
    ]

def main() -> None:
    ctx = MenuContext(cfg=DEFAULT_CONFIG.copy())
    options = build_menu_options()
    try:
        default_choice = next(
            i for i, opt in enumerate(options, start=1) if opt.name == "Start Training"
        )
    except StopIteration:
        default_choice = 1

    while ctx.running:
        print("\n==== Fractal Neurons Menu ====")
        for idx, option in enumerate(options, start=1):
            print(f"{idx}) {option.label(ctx)}")
        choice = read_input("Choose (menu option number)", str(default_choice))
        if not choice.isdigit():
            print("Invalid choice.")
            continue
        index = int(choice)
        if not (1 <= index <= len(options)):
            print("Invalid choice.")
            continue
        selected = options[index - 1]
        try:
            selected.action(ctx)
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - CLI helper
            print(f"[error] {exc}")


if __name__ == "__main__":
    main()
