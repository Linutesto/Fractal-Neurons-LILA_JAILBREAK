"""Command-line agent shell that runs the fractal swarm orchestrator."""
from __future__ import annotations

import argparse

from .orchestrator import SwarmOrchestrator, default_roles, build_default_generator


def main():
    ap = argparse.ArgumentParser(description="Fractal agent shell")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    generator = build_default_generator(
        args.ckpt,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    orchestrator = SwarmOrchestrator(default_roles(), generator, rounds=args.rounds, log_path=args.log)
    summary = orchestrator.run(args.prompt)
    print("=== Swarm Transcript ===")
    for role, content in summary["history"]:
        print(f"[{role}] {content}\n")


if __name__ == "__main__":
    main()
