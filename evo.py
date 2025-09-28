#!/usr/bin/env python3
"""Evolution orchestrator for Fractal Neurons."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parent


def run(cmd: str, check: bool = True) -> None:
    print(f"\n▶ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def avg_score(report: Dict) -> float:
    tasks = report.get("tasks", {})
    scores = []
    for value in tasks.values():
        if isinstance(value, dict) and "score" in value:
            scores.append(float(value["score"]))
        elif isinstance(value, (int, float)):
            scores.append(float(value))
    return sum(scores) / max(1, len(scores))


def compare_reports(parent_report: Dict, child_report: Dict, rules: Dict) -> Tuple[bool, Dict[str, float]]:
    parent_avg = avg_score(parent_report)
    child_avg = avg_score(child_report)
    rel_gain = 100.0 * (child_avg - parent_avg) / max(1e-6, parent_avg)
    overflow = child_report.get("moe", {}).get("overflow_pct_avg", 0.0)
    ok = (
        child_avg >= rules.get("min_avg", 0.0)
        and rel_gain >= rules.get("rel_gain_pct", 0.0)
        and overflow <= rules.get("max_overflow_pct", 100.0)
    )
    return ok, {
        "parent_avg": parent_avg,
        "child_avg": child_avg,
        "rel_gain_pct": rel_gain,
        "overflow_pct": overflow,
    }


def ensure_prompts(prompt_file: Path) -> Path:
    if prompt_file.exists():
        return prompt_file
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        "Summarize the latest research on mixture-of-experts models.\n"\
        "List three creative applications for fractal neural networks.\n"\
        "Explain why layer gating can help stabilize deep architectures.\n"
    )
    return prompt_file


def run_distillation(cfg: Dict, parent_ckpt: str, distill_dir: Path) -> Path:
    dist_cfg = cfg["distill"]
    prompt_file = ensure_prompts(Path(dist_cfg.get("prompt_file", "prompts/seed_prompts.txt")).expanduser())
    out_path = distill_dir / dist_cfg.get("output", "distill_samples.jsonl")
    log_path = distill_dir / "distill.log"
    cmd = (
        f"python -m fractal_neurons.distill --ckpt {parent_ckpt} "
        f"--prompt-file {prompt_file} --output {out_path} "
        f"--limit {dist_cfg.get('n_samples', 1000)} --temperature {dist_cfg.get('temperature', 0.9)} "
        f"--top-k {dist_cfg.get('top_k', 20)} --seq-len {dist_cfg.get('seq_len', 768)} "
        f"--max-new-tokens {dist_cfg.get('max_new_tokens', 256)} --steps {dist_cfg.get('steps', 8)} "
        f"--min-len {dist_cfg.get('min_len', 64)} --max-len {dist_cfg.get('max_len', 1024)} "
        f"--log {log_path}"
    )
    run(cmd)
    return out_path


def filter_dataset(cfg: Dict, distilled_path: Path, filtered_path: Path) -> Path:
    filt_cfg = cfg.get("filter", {})
    min_chars = int(filt_cfg.get("min_chars", 0))
    seen: set[str] = set()
    kept = 0
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with distilled_path.open("r", encoding="utf-8") as inp, filtered_path.open("w", encoding="utf-8") as out:
        for line in inp:
            obj = json.loads(line)
            text = obj.get("completion", "")
            if len(text) < min_chars:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Filtered dataset: {kept} examples -> {filtered_path}")
    return filtered_path


def materialize_corpus(filtered_jsonl: Path, corpus_dir: Path) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with filtered_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            data = json.loads(line)
            text = data.get("completion", "")
            (corpus_dir / f"sample_{idx}.txt").write_text(text, encoding="utf-8")
    return corpus_dir


def write_finetune_config(cfg: Dict, parent_ckpt: str, corpus_dir: Path, out_path: Path) -> Path:
    ft = cfg["finetune"]
    config = {
        "data": {
            "source": "textdir",
            "text_root": str(corpus_dir),
            "seq_len": ft.get("seq_len", 1024),
            "text_tokenize_in_workers": True,
            "text_shuffle_buffer": 2048,
        },
        "model": {
            "num_experts": ft.get("num_experts", 4),
            "expert_top_k": ft.get("expert_top_k", 2),
        },
        "train": {
            "steps": ft.get("num_steps", 2000),
            "lr": ft.get("lr", 3e-4),
            "warmup": ft.get("warmup", 500),
            "ema_decay": ft.get("ema_decay", 0.9995),
            "grad_clip": ft.get("grad_clip", 1.0),
            "seq_len": ft.get("seq_len", 1024),
            "batch_size": ft.get("batch_size", 24),
            "resume": False,
            "device": "cuda",
        },
        "init": {
            "from_checkpoint": parent_ckpt,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out_path


def latest_ema(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("ema_final_*.pt"))
    if not candidates:
        candidates = list(Path("runs").glob("**/ema_final_*.pt"))
    if not candidates:
        raise RuntimeError("No EMA checkpoints found")
    return max(candidates, key=os.path.getmtime)


def run_evolution(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    work_dir = Path(cfg["work_dir"]).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    exp_root = work_dir / f"{cfg['exp_name']}_{timestamp()}"
    exp_root.mkdir(parents=True, exist_ok=True)

    parent_ckpt = cfg["parent_ckpt"]

    # Distillation
    distill_dir = exp_root / "distill"
    distill_dir.mkdir()
    distilled_jsonl = run_distillation(cfg, parent_ckpt, distill_dir)

    # Filtering
    filtered_jsonl = filter_dataset(cfg, distilled_jsonl, distill_dir / cfg.get("filter", {}).get("out_jsonl", "distill_filtered.jsonl"))

    # Finetune
    child_dir = exp_root / "child"
    child_dir.mkdir()
    corpus_dir = materialize_corpus(filtered_jsonl, child_dir / "corpus")
    finetune_cfg = write_finetune_config(cfg, parent_ckpt, corpus_dir, child_dir / "finetune.yaml")
    run(f"python -m fractal_neurons.train --config {finetune_cfg}")
    child_ckpt = latest_ema(Path("runs"))
    (child_dir / "ckpt.txt").write_text(str(child_ckpt))

    # Evaluation
    eval_dir = exp_root / "eval"
    eval_dir.mkdir()
    suite = cfg.get("eval", {}).get("suite", ["reasoning", "summary"])
    shots = cfg.get("eval", {}).get("shots", [0])
    suite_str = ",".join(suite)
    shots_str = ",".join(map(str, shots))
    parent_eval_path = eval_dir / "parent.json"
    child_eval_path = eval_dir / "child.json"
    run(
        f"python -m fractal_neurons.eval_harness --ckpt {parent_ckpt} --suite {suite_str} --shots {shots_str} --output {parent_eval_path}"
    )
    run(
        f"python -m fractal_neurons.eval_harness --ckpt {child_ckpt} --suite {suite_str} --shots {shots_str} --output {child_eval_path}"
    )

    parent_report = load_json(parent_eval_path)
    child_report = load_json(child_eval_path)
    promote, stats = compare_reports(parent_report, child_report, cfg["eval"]["promote_if"])
    (eval_dir / "decision.json").write_text(json.dumps({"promote": promote, **stats}, indent=2))

    promoted_symlink = work_dir / f"{cfg['exp_name']}_PROMOTED.pt"
    if promote:
        if promoted_symlink.exists() or promoted_symlink.is_symlink():
            promoted_symlink.unlink()
        promoted_symlink.symlink_to(Path(child_ckpt).resolve())
        print(f"\n✅ Promoted child model -> {promoted_symlink}")
    else:
        print("\nChild model did not meet promotion criteria. Parent retained.")

    # Housekeeping
    keep_last = cfg.get("keep_last", 3)
    experiments = sorted(work_dir.glob(f"{cfg['exp_name']}_*"))
    for old in experiments[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated evolution runner")
    parser.add_argument("--config", "-c", default="evo.yaml")
    args = parser.parse_args()
    run_evolution(Path(args.config).expanduser())


if __name__ == "__main__":
    main()
