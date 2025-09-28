#!/usr/bin/env python3
import argparse, subprocess

parser = argparse.ArgumentParser(description="Fractal Neurons Control CLI")
parser.add_argument("action", choices=["train", "infer", "chat", "distill", "evolve", "menu"])
args = parser.parse_args()

cmds = {
    "train": "python -m fractal_neurons.train --config configs/system_train.yaml",
    "infer": "python -m fractal_neurons.infer --ckpt runs/fractal_fast_7950x_4090/final_20000.pt --text 'Hello world'",
    "chat": "python -m fractal_neurons.generate --ckpt runs/fractal_fast_7950x_4090/final_20000.pt --chat",
    "distill": "python -m fractal_neurons.distill --ckpt runs/fractal_fast_7950x_4090/final_20000.pt --output distilled.jsonl",
    "evolve": "python -m fractal_neurons.evo --config configs/fsi_distill_v1.yaml",
    "menu": "python menu.py"
}

subprocess.run(cmds[args.action], shell=True)