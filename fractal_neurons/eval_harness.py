"""Lightweight evaluation harness for Fractal Neurons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .generate import load_generation_model, generate_text

SUITES: Dict[str, Dict[str, Tuple[str, object]]] = {
    "reasoning": {
        "math_easy": ("What is 12 + 15?", "27"),
        "logic": ("If all fractals are recursive and recursive things are interesting, are fractals interesting?", "yes"),
        "self": ("In one sentence, describe what you are.", "I am"),
    },
    "summary": {
        "summary": (
            "Summarize in one sentence: Fractal models reuse parameters across hierarchical structures to stay compact.",
            ["reuse", "parameters", "hierarchical"],
        ),
    },
}


def score_reasoning(output: str, answer: str) -> float:
    return 1.0 if answer.lower() in output.lower() else 0.0


def score_keywords(output: str, keywords: List[str]) -> float:
    hits = sum(1 for kw in keywords if kw.lower() in output.lower())
    return hits / len(keywords)


def evaluate(ckpt: str, device: str | None, suites: List[str], out_path: str) -> Dict[str, object]:
    model, tokenizer, _ = load_generation_model(ckpt, device)
    tasks_report: Dict[str, Dict[str, object]] = {}
    moe_overflows: List[float] = []
    for suite in suites:
        entries = SUITES.get(suite.strip().lower())
        if not entries:
            continue
        for name, (prompt, target) in entries.items():
            text, stats = generate_text(
                ckpt,
                prompt,
                seq_len=512,
                max_new_tokens=128,
                steps=6,
                temperature=0.7,
                top_k=20,
                device=device,
                cached=(model, tokenizer),
                return_stats=True,
            )
            if isinstance(target, list):
                score = score_keywords(text, target)
            else:
                score = score_reasoning(text, str(target))
            stats.update({"prompt": prompt, "output": text, "score": score})
            tasks_report[f"{suite}/{name}"] = stats
            if "moe_overflow_mean" in stats and stats["moe_overflow_mean"] is not None:
                moe_overflows.append(float(stats["moe_overflow_mean"]))
    result: Dict[str, object] = {
        "tasks": {name: {"score": info["score"]} for name, info in tasks_report.items()},
        "details": tasks_report,
    }
    if moe_overflows:
        result["moe"] = {"overflow_pct_avg": 100.0 * sum(moe_overflows) / len(moe_overflows)}
    if out_path:
        path = Path(out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="Run lightweight evaluation suite")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", "--out", default="eval_results.json")
    ap.add_argument("--suite", default="reasoning,summary")
    ap.add_argument("--shots", default="0")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    result = evaluate(args.ckpt, args.device, suites, args.output)
    for name, info in result.get("tasks", {}).items():
        print(f"{name}: {info['score']:.2f}")


if __name__ == "__main__":
    main()
