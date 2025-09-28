import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # fallback minimal parser if PyYAML not present
    yaml = None

from .model import FractalModel, FractalModelConfig
from .data import SystemByteDataset, SystemDataConfig, MaskedByteCollator, MaskedTokenCollator
from .data_hf import HFDatasetConfig, HFDatasetStream, HFDatasetMap
from .data_text import TextDirStream, TextDirTokenizedStream
from .tokenizer import TokenizerConfig, build_tokenizer


class _CUDABatchPrefetcher:
    def __init__(self, loader, device: torch.device, pin_memory: bool = True, prefetch_batches: int = 2):
        if device.type != "cuda":
            raise RuntimeError("_CUDABatchPrefetcher requires a CUDA device")
        self.loader = iter(loader)
        self.device = device
        self.pin_memory = pin_memory
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.stream = torch.cuda.Stream(device=device)
        self.queue: deque[Dict[str, torch.Tensor]] = deque()
        for _ in range(self.prefetch_batches):
            self._enqueue()

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=self.pin_memory) for k, v in batch.items()}

    def _enqueue(self) -> None:
        try:
            batch = next(self.loader)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            self.queue.append(self._to_device(batch))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if not self.queue:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.queue.popleft()
        self._enqueue()
        return batch


class EMAManager:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        self.shadow: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            shadow = self.shadow[name]
            if shadow.device != param.device or shadow.dtype != param.dtype:
                shadow = shadow.to(param.device, dtype=param.dtype)
                self.shadow[name] = shadow
            shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state.items()}

@dataclass
class TrainConfig:
    run_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_accum: int = 1
    seed: int = 1337
    deterministic: bool = False
    resume: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    torch_compile: bool = False
    tf32: bool = True
    log_every: int = 50
    ckpt_every: int = 200
    out_dir: str = "runs"
    verbose: bool = False
    compile_mode: str = "default"
    prefetch_gpu: bool = True
    prefetch_gpu_batches: int = 2
    adam_fused: bool = True
    warmup_steps: int = 0
    lr_min: float = 0.0
    cosine_cycle: int = 0
    grad_clip: Optional[float] = None
    use_ema: bool = False
    ema_decay: float = 0.999
    # Simple genetic step on fractal gates
    ga_enable: bool = False
    ga_every: int = 500
    ga_population: int = 5
    ga_sigma: float = 0.1
    # QFP parameters
    qfp_enable: bool = False
    qfp_time_complex_real: float = -2.999999
    qfp_time_complex_imag: float = -0.002189


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        # very small hand-rolled subset: assume it's JSON-like YAML
        import json
        with open(path, "r") as f:
            txt = f.read()
        return json.loads(txt)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scan_dir", default=None, help="Scan all files under this directory (overrides data.allow_all/root)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the run directory")
    parser.add_argument(
        "--init_ckpt",
        default=None,
        help="Initialize weights from a specific checkpoint without relying on the run directory",
    )
    parser.add_argument("--device", default=None, help="Force device override (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count for all sources")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Data config
    data_cfg = cfg.get("data", {})
    # Override: --scan_dir implies allow_all on that directory
    override_allow_all = False
    override_root = None
    if args.scan_dir:
        override_allow_all = True
        override_root = os.path.abspath(os.path.expanduser(args.scan_dir))

    root_cfg = str(data_cfg.get("root", "/"))
    allow_all_cfg = bool(data_cfg.get("allow_all", False))
    # Gate only when scanning full filesystem root
    allow_all_effective = override_allow_all or allow_all_cfg
    root_effective = override_root or root_cfg
    if allow_all_effective and os.path.abspath(os.path.expanduser(root_effective)) == "/":
        allow_all_effective = allow_all_effective and (os.environ.get("TRAIN_ALLOW_ALL_FILES", "0") == "1")

    sdc = SystemDataConfig(
        include_globs=data_cfg.get("include_globs") or [],
        exclude_globs=data_cfg.get("exclude_globs") or [],
        seq_len=int(data_cfg.get("seq_len", 512)),
        stride=int(data_cfg.get("stride", 512)),
        allow_all=allow_all_effective,
        root=root_effective,
        max_file_mb=int(data_cfg.get("max_file_mb", 16)),
        max_index_items=int(data_cfg.get("max_index_items", 500_000)),
        verbose=bool(data_cfg.get("verbose", False)),
    )

    # Tokenizer
    tok_cfg = cfg.get("tokenizer", {})
    tcfg = TokenizerConfig(
        type=str(tok_cfg.get("type", "bytes")),
        name_or_path=tok_cfg.get("name_or_path"),
        truncation_side=str(tok_cfg.get("truncation_side", "right")),
        pad_to_multiple_of=tok_cfg.get("pad_to_multiple_of"),
    )
    tokenizer = build_tokenizer(tcfg)

    # Model config (vocab from tokenizer)
    model_cfg = cfg.setdefault("model", {})
    fmc = FractalModelConfig(
        vocab_size=int(getattr(tokenizer, "vocab_size", model_cfg.get("vocab_size", 257))),
        dim=int(model_cfg.get("dim", 128)),
        depth=int(model_cfg.get("depth", 6)),
        fanout=int(model_cfg.get("fanout", 10)),
        use_fp16=bool(model_cfg.get("use_fp16", False)),
        droppath_rate=float(model_cfg.get("droppath_rate", 0.0)),
        branch_dropout=float(model_cfg.get("branch_dropout", 0.0)),
        interconnect=bool(model_cfg.get("interconnect", True)),
        interconnect_heads=int(model_cfg.get("interconnect_heads", 2)),
        interconnect_dropout=float(model_cfg.get("interconnect_dropout", 0.0)),
        num_experts=int(model_cfg.get("num_experts", 0)),
        expert_hidden=int(model_cfg.get("expert_hidden", 0)),
        expert_top_k=int(model_cfg.get("expert_top_k", 2)),
        router_temperature=float(model_cfg.get("router_temperature", 1.0)),
        moe_aux_lambda=float(model_cfg.get("moe_aux_lambda", 0.0)),
        capacity_factor=float(model_cfg.get("capacity_factor", 1.1)),
    )
    model_cfg["vocab_size"] = fmc.vocab_size
    model_cfg.setdefault("router_temperature", fmc.router_temperature)
    model_cfg.setdefault("moe_aux_lambda", fmc.moe_aux_lambda)
    model_cfg.setdefault("capacity_factor", fmc.capacity_factor)

    # Train config
    train_cfg = cfg.get("train", {})
    tc = TrainConfig(
        run_name=str(train_cfg.get("run_name", f"fractal_{int(time.time())}")),
        device=str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        batch_size=int(train_cfg.get("batch_size", 8)),
        steps=int(train_cfg.get("steps", 1000)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        grad_accum=int(train_cfg.get("grad_accum", 1)),
        seed=int(train_cfg.get("seed", 1337)),
        deterministic=bool(train_cfg.get("deterministic", False)),
        resume=bool(train_cfg.get("resume", False)),
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(train_cfg.get("persistent_workers", True)),
        torch_compile=bool(train_cfg.get("torch_compile", False)),
        tf32=bool(train_cfg.get("tf32", True)),
        log_every=int(train_cfg.get("log_every", 50)),
        ckpt_every=int(train_cfg.get("ckpt_every", 200)),
        out_dir=str(train_cfg.get("out_dir", "runs")),
        verbose=bool(train_cfg.get("verbose", False)),
        ga_enable=bool(train_cfg.get("ga_enable", False)),
        ga_every=int(train_cfg.get("ga_every", 500)),
        ga_population=int(train_cfg.get("ga_population", 5)),
        ga_sigma=float(train_cfg.get("ga_sigma", 0.1)),
        compile_mode=str(train_cfg.get("compile_mode", "default")),
        prefetch_gpu=bool(train_cfg.get("prefetch_gpu", True)),
        prefetch_gpu_batches=int(train_cfg.get("prefetch_gpu_batches", 2)),
        adam_fused=bool(train_cfg.get("adam_fused", True)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        lr_min=float(train_cfg.get("lr_min", 0.0)),
        cosine_cycle=int(train_cfg.get("cosine_cycle", 0)),
        grad_clip=(float(train_cfg.get("grad_clip")) if train_cfg.get("grad_clip") is not None else None),
        use_ema=bool(train_cfg.get("use_ema", False)),
        ema_decay=float(train_cfg.get("ema_decay", 0.999)),
        qfp_enable=bool(train_cfg.get("qfp_enable", False)),
        qfp_time_complex_real=float(train_cfg.get("qfp_time_complex_real", -2.999999)),
        qfp_time_complex_imag=float(train_cfg.get("qfp_time_complex_imag", -0.002189)),
    )
    if args.verbose:
        tc.verbose = True
        sdc.verbose = True
    if args.resume:
        tc.resume = True
    if args.device:
        tc.device = args.device
    if args.num_workers is not None:
        tc.num_workers = int(args.num_workers)

    init_cfg = dict(cfg.get("init", {}) or {})
    if args.init_ckpt:
        init_cfg["from_checkpoint"] = args.init_ckpt
    init_ckpt_path = init_cfg.get("from_checkpoint")
    if init_ckpt_path:
        init_ckpt_path = os.path.abspath(os.path.expanduser(str(init_ckpt_path)))
    init_load_optimizer = bool(init_cfg.get("load_optimizer", False))
    init_load_step = bool(init_cfg.get("load_step", False))
    init_load_ema = init_cfg.get("load_ema", True)
    init_strict = bool(init_cfg.get("strict", False))
    if init_ckpt_path:
        cfg.setdefault("init", {})
        cfg["init"]["from_checkpoint"] = init_ckpt_path
        cfg["init"]["load_optimizer"] = init_load_optimizer
        cfg["init"]["load_step"] = init_load_step
        cfg["init"]["load_ema"] = init_load_ema
        cfg["init"]["strict"] = init_strict

    # QFP Initialization
    time_complex = None
    temporal_factor = 1.0
    if tc.qfp_enable:
        time_complex = complex(tc.qfp_time_complex_real, tc.qfp_time_complex_imag)
        temporal_factor = torch.exp(torch.tensor(time_complex.real)).item()
        print(f"[QFP] Enabled with time_complex={time_complex}, temporal_factor={temporal_factor:.4f}")

    # Reproducibility controls
    try:
        torch.manual_seed(tc.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tc.seed)
        if tc.deterministic:
            try:
                import numpy as _np  # type: ignore
                _np.random.seed(tc.seed)
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                import torch.backends.cudnn as _cudnn
                _cudnn.deterministic = True
                _cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

    # Dataset and loader
    source = str(data_cfg.get("source", "system")).lower()
    if source == "hf":
        hf_cfg = HFDatasetConfig(
            path=str(data_cfg.get("hf_path")),
            name=data_cfg.get("hf_name"),
            split=str(data_cfg.get("hf_split", "train")),
            streaming=bool(data_cfg.get("hf_streaming", True)),
            text_field=data_cfg.get("hf_text_field"),
            seq_len=int(data_cfg.get("seq_len", 512)),
            return_text=(tcfg.type != "bytes"),
            verbose=tc.verbose,
        )
        if hf_cfg.streaming:
            ds = HFDatasetStream(hf_cfg)
        else:
            ds = HFDatasetMap(hf_cfg)
    elif source == "textdir":
        if tcfg.type == "bytes":
            raise RuntimeError("textdir source requires a text tokenizer (tokenizer.type: hf)")
        text_root = str(data_cfg.get("text_root", os.path.expanduser("~")))
        text_globs = data_cfg.get(
            "text_globs",
            ["**/*.txt", "**/*.md", "**/*.jsonl", "**/*.json", "**/*.log", "**/*.py"],
        ) or []
        exclude_globs = data_cfg.get("exclude_globs", []) or []
        text_use_cache = bool(data_cfg.get("text_use_cache", True))
        text_cache_dir = data_cfg.get("text_cache_dir")
        text_refresh_cache = bool(data_cfg.get("text_refresh_cache", False))
        tokenize_in_workers = bool(data_cfg.get("text_tokenize_in_workers", True))
        text_shuffle_buffer = int(data_cfg.get("text_shuffle_buffer", 0))
        text_seed_raw = data_cfg.get("text_seed")
        text_seed = (
            int(text_seed_raw)
            if isinstance(text_seed_raw, (int, float)) or (isinstance(text_seed_raw, str) and text_seed_raw.strip())
            else None
        )
        text_reshuffle = bool(data_cfg.get("text_reshuffle_each_epoch", True))
        if tokenize_in_workers:
            if tc.verbose:
                print("[textdir] tokenize_in_workers=True (parallel HF tokenization)")
            ds = TextDirTokenizedStream(
                root=text_root,
                seq_len=int(data_cfg.get("seq_len", 512)),
                tokenizer_name_or_path=str(getattr(tcfg, "name_or_path", "")),
                truncation_side=str(getattr(tcfg, "truncation_side", "right")),
                include_globs=text_globs,
                exclude_globs=exclude_globs,
                verbose=tc.verbose,
                use_cache=text_use_cache,
                cache_dir=text_cache_dir,
                refresh_cache=text_refresh_cache,
                shuffle_buffer=text_shuffle_buffer,
                reshuffle_each_epoch=text_reshuffle,
                shuffle_seed=text_seed,
            )
        else:
            ds = TextDirStream(
                root=text_root,
                include_globs=text_globs,
                exclude_globs=exclude_globs,
                verbose=tc.verbose,
                use_cache=text_use_cache,
                cache_dir=text_cache_dir,
                refresh_cache=text_refresh_cache,
                shuffle_buffer=text_shuffle_buffer,
                reshuffle_each_epoch=text_reshuffle,
                shuffle_seed=text_seed,
            )
    else:
        if tcfg.type != "bytes":
            raise RuntimeError("System file dataset supports only 'bytes' tokenizer. Use HF source for text tokenizers.")
        ds = SystemByteDataset(sdc)
    if tcfg.type != "bytes" and source in ("hf", "textdir"):
        collate = MaskedTokenCollator(
            tokenizer,
            seq_len=int(data_cfg.get("seq_len", 512)),
            mask_rate=float(data_cfg.get("mask_rate", 0.15)),
        )
    else:
        mask_id = int(getattr(tokenizer, "mask_id", fmc.vocab_size - 1) or (fmc.vocab_size - 1))
        collate = MaskedByteCollator(
            mask_rate=float(data_cfg.get("mask_rate", 0.15)),
            vocab_size=fmc.vocab_size,
            mask_id=mask_id,
        )
    # IterableDataset cannot be shuffled, and prefetch/persistent options differ
    is_iterable = isinstance(ds, torch.utils.data.IterableDataset)
    iterable_allows_workers = is_iterable and bool(getattr(ds, "supports_workers", False))
    # Auto-scale workers for textdir-like iterable datasets if user left a low default
    if iterable_allows_workers:
        try:
            import os as _os
            cpu = int(_os.cpu_count() or 4)
        except Exception:
            cpu = 4
        target = min(cpu, 32)
        if tc.num_workers < max(4, cpu // 2):
            if tc.verbose:
                print(f"[loader] bumping num_workers from {tc.num_workers} -> {target} (auto) for iterable source")
            tc.num_workers = target
            # modest prefetch for many workers
            tc.prefetch_factor = max(tc.prefetch_factor, 2)
            tc.persistent_workers = True
    dl = DataLoader(
        ds,
        batch_size=tc.batch_size,
        shuffle=(not is_iterable),
        num_workers=(tc.num_workers if iterable_allows_workers else (0 if is_iterable else tc.num_workers)),
        collate_fn=collate,
        pin_memory=(tc.pin_memory and (not is_iterable or iterable_allows_workers)),
        prefetch_factor=(None if ((is_iterable and not iterable_allows_workers) or tc.num_workers <= 0) else tc.prefetch_factor),
        persistent_workers=(False if ((is_iterable and not iterable_allows_workers) or tc.num_workers <= 0) else tc.persistent_workers),
    )
    if tc.verbose:
        workers_eff = (tc.num_workers if iterable_allows_workers else (0 if is_iterable else tc.num_workers))
        pin_eff = (tc.pin_memory and (not is_iterable or iterable_allows_workers))
        prefetch_eff = (None if ((is_iterable and not iterable_allows_workers) or tc.num_workers <= 0) else tc.prefetch_factor)
        persistent_eff = (False if ((is_iterable and not iterable_allows_workers) or tc.num_workers <= 0) else tc.persistent_workers)
        print(f"[loader] iterable={is_iterable} workers={workers_eff} pin={pin_eff} prefetch={prefetch_eff} persistent={persistent_eff}")

    # Model
    model = FractalModel(fmc)
    total_nodes = model.total_nodes()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] nodes={total_nodes} params={n_params:,} dim={fmc.dim} depth={fmc.depth} fanout={fmc.fanout} vocab={fmc.vocab_size}")
    if total_nodes < 100000:
        print("[warn] total nodes < 100k. Increase depth or fanout in config.")

    # Device selection and validation
    if tc.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set train.device=cpu or install CUDA.")
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    if tc.verbose:
        if device.type == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                print(f"[device] cuda: {name} sm{cap[0]}{cap[1]}")
            except Exception:
                print("[device] cuda: available")
        else:
            if torch.cuda.is_available() and tc.device != "cuda":
                print("[device] running on CPU while CUDA is available (set --device cuda)")
    if tc.tf32 and device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            print("TF32 enabled; cuDNN benchmark on.")
        except Exception:
            pass
    if tc.torch_compile:
        try:
            model = torch.compile(model, mode=tc.compile_mode or "default")
            print(f"torch.compile enabled (mode={tc.compile_mode}).")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")
    model = model.to(device)
    ema = EMAManager(model, tc.ema_decay) if tc.use_ema else None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=fmc.use_fp16 and device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=fmc.use_fp16 and device.type == "cuda")

    fused_ok = tc.adam_fused and device.type == "cuda"
    opt_kwargs = dict(lr=tc.lr, weight_decay=tc.weight_decay)
    if fused_ok:
        try:
            opt = optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
            if tc.verbose:
                print("[optim] Using fused AdamW")
        except TypeError:
            if tc.verbose:
                print("[optim] fused AdamW unavailable; falling back to standard implementation")
            opt = optim.AdamW(model.parameters(), **opt_kwargs)
    else:
        opt = optim.AdamW(model.parameters(), **opt_kwargs)

    run_dir = os.path.join(tc.out_dir, tc.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Optional resume
    start_step = 0
    if tc.resume:
        try:
            import glob as _glob
            ckpts = sorted(_glob.glob(os.path.join(run_dir, "*.pt")))
            if ckpts:
                latest = ckpts[-1]
                state = torch.load(latest, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
                    if "opt" in state:
                        opt.load_state_dict(state["opt"])
                    if ema is not None and "ema" in state:
                        ema.load_state_dict(state["ema"])
                    start_step = int(state.get("step", 0))
                    print(f"[resume] loaded {latest} at step {start_step}")
        except Exception as e:
            print(f"[resume] failed: {e}")

    # Optional init checkpoint load after resume handling so explicit init overrides resume state
    if init_ckpt_path:
        try:
            state = torch.load(init_ckpt_path, map_location="cpu")
            state_model = state.get("model", state)
            if init_strict:
                model.load_state_dict(state_model, strict=True)
                missing = unexpected = []
            else:
                missing, unexpected = model.load_state_dict(state_model, strict=False)
                if missing:
                    print(f"[init] missing parameters: {missing}")
                if unexpected:
                    print(f"[init] unexpected parameters: {unexpected}")
            if init_load_optimizer and "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                except Exception as e:
                    print(f"[init] optimizer load failed: {e}")
            if ema is not None and init_load_ema and "ema" in state:
                try:
                    ema.load_state_dict(state["ema"])
                except Exception as e:
                    print(f"[init] ema load failed: {e}")
            if init_load_step and "step" in state:
                start_step = int(state.get("step", start_step))
            else:
                start_step = 0
            print(f"[init] loaded checkpoint {init_ckpt_path}")
        except Exception as e:
            print(f"[init] failed to load {init_ckpt_path}: {e}")

    step = start_step
    running = 0.0
    model.train()
    import time as _time
    t0 = _time.time()
    t_start = t0
    seq_len = int(data_cfg.get("seq_len", 512))
    effective_updates = max(1, math.ceil(tc.steps / max(1, tc.grad_accum)))

    def _lr_for_update(update_idx: int) -> float:
        base_lr = tc.lr
        if tc.qfp_enable:
            base_lr *= temporal_factor # Apply QFP temporal factor to base LR

        if tc.warmup_steps > 0 and update_idx < tc.warmup_steps:
            warm = base_lr * float(update_idx + 1) / float(tc.warmup_steps)
            return warm
        cycle = tc.cosine_cycle if tc.cosine_cycle > 0 else effective_updates
        cycle = max(cycle, tc.warmup_steps + 1)
        if cycle <= tc.warmup_steps:
            return max(tc.lr_min, base_lr)
        progress = (update_idx - tc.warmup_steps) / float(max(1, cycle - tc.warmup_steps))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = tc.lr_min + (base_lr - tc.lr_min) * cosine
        return max(lr, tc.lr_min)

    def _set_lr(lr_value: float) -> None:
        for pg in opt.param_groups:
            pg["lr"] = lr_value

    def _batch_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=tc.pin_memory)
            else:
                out[k] = v
        return out

    try:
        current_lr = opt.param_groups[0]["lr"] if opt.param_groups else tc.lr
        moe_aux_running = 0.0
        moe_overflow_running = 0.0
        moe_entropy_running = 0.0
        moe_steps = 0
        while step < tc.steps:
            if device.type == "cuda" and tc.prefetch_gpu:
                data_iter: Iterator[Dict[str, torch.Tensor]] = _CUDABatchPrefetcher(
                    dl,
                    device,
                    pin_memory=tc.pin_memory,
                    prefetch_batches=tc.prefetch_gpu_batches,
                )
            else:
                data_iter = (_batch_to_device(batch) for batch in dl)

            saw_batch = False
            for batch in data_iter:
                saw_batch = True
                update_idx = step // max(1, tc.grad_accum)
                current_lr = _lr_for_update(update_idx)
                _set_lr(current_lr)
                tokens = batch["tokens"]
                targets = batch["targets"]
                loss_mask = batch["loss_mask"]

                with torch.amp.autocast(device_type="cuda", enabled=fmc.use_fp16 and device.type == "cuda"):
                    # Pass t_context to the model's forward method
                    _, loss = model(tokens, loss_mask=loss_mask, targets=targets, t_context=t_context)
                moe_info = getattr(model, "last_moe_info", None)
                aux_loss = getattr(model, "last_aux_loss", None)
                if aux_loss is not None and model.moe_aux_lambda > 0:
                    loss = loss + model.moe_aux_lambda * aux_loss
                if moe_info is not None:
                    moe_steps += 1
                    if "aux_loss_scalar" in moe_info:
                        moe_aux_running += float(moe_info["aux_loss_scalar"])
                    if "overflow_rate" in moe_info:
                        moe_overflow_running += float(moe_info["overflow_rate"])
                    if "router_entropy_scalar" in moe_info:
                        moe_entropy_running += float(moe_info["router_entropy_scalar"])

                loss = loss / tc.grad_accum
                scaler.scale(loss).backward()

                if (step + 1) % tc.grad_accum == 0:
                    scaler.unscale_(opt)
                    if tc.grad_clip is not None:
                        clip_grad_norm_(model.parameters(), tc.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

                running += loss.item() * tc.grad_accum
                step += 1
                if step % tc.log_every == 0:
                    dt = _time.time() - t0
                    toks = tc.batch_size * seq_len * tc.log_every
                    tps = toks / max(dt, 1e-6)
                    # percentage and ETA
                    elapsed = _time.time() - t_start
                    steps_done = step
                    pct = 100.0 * (steps_done / max(1, tc.steps))
                    steps_per_sec = tc.log_every / max(dt, 1e-6)
                    remaining_steps = max(0, tc.steps - steps_done)
                    eta_sec = remaining_steps / max(steps_per_sec, 1e-6)

                    def _fmt(s):
                        s = int(s)
                        h = s // 3600
                        m = (s % 3600) // 60
                        sec = s % 60
                        return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

                    avg_aux = moe_aux_running / max(1, moe_steps)
                    avg_overflow = moe_overflow_running / max(1, moe_steps)
                    avg_entropy = moe_entropy_running / max(1, moe_steps)
                    print(
                        f"[{pct:5.1f}%] step {step}/{tc.steps} "
                        f"loss={running/tc.log_every:.4f} throughput={tps:.0f} tok/s "
                        f"lr={current_lr:.6f} aux={avg_aux:.4f} ovf={avg_overflow:.3f} ent={avg_entropy:.3f} "
                        f"elapsed={_fmt(elapsed)} eta={_fmt(eta_sec)}"
                    )
                    running = 0.0
                    t0 = _time.time()
                    moe_aux_running = moe_overflow_running = moe_entropy_running = 0.0
                    moe_steps = 0
                # Optional simple GA on fractal connectivity gates
                if tc.ga_enable and (step % tc.ga_every == 0):
                    try:
                        with torch.no_grad():
                            fractal = getattr(model, "fractal", None)
                            if fractal is None:
                                raise RuntimeError("model.fractal not available for GA")
                            targets = []
                            bases = []
                            attr_names = []
                            for attr in ("level_gate", "ic_gate"):
                                tensor = getattr(fractal, attr, None)
                                if isinstance(tensor, torch.nn.Parameter):
                                    attr_names.append(attr)
                                    bases.append(tensor.data.clone())
                                    targets.append(tensor)
                            if not targets:
                                raise RuntimeError("no fractal gates found for GA mutation")

                            # evaluate current parameters as baseline
                            with torch.amp.autocast(device_type="cuda", enabled=fmc.use_fp16 and device.type == "cuda"):
                                # Pass t_context to the model's forward method during GA evaluation
                                _, base_loss = model(tokens, loss_mask=loss_mask, targets=targets, t_context=t_context)
                            best_loss = float(base_loss.detach().item()) if base_loss is not None else 1e9
                            best_states = [base.clone() for base in bases]

                            for _ in range(max(1, tc.ga_population)):
                                candidates = []
                                for base in bases:
                                    cand = base + tc.ga_sigma * torch.randn_like(base)
                                    candidates.append(cand)
                                for param, cand in zip(targets, candidates):
                                    param.data.copy_(cand)
                                with torch.amp.autocast(device_type="cuda", enabled=fmc.use_fp16 and device.type == "cuda"):
                                    # Pass t_context to the model's forward method during GA evaluation
                                    _, cand_loss = model(tokens, loss_mask=loss_mask, targets=targets, t_context=t_context)
                                val = float(cand_loss.detach().item()) if cand_loss is not None else 1e9
                                if val < best_loss:
                                    best_loss = val
                                    best_states = [cand.clone() for cand in candidates]
                            # restore best parameters
                            for param, best_state in zip(targets, best_states):
                                param.data.copy_(best_state)
                        if tc.verbose:
                            joined = ", ".join(attr_names)
                            print(f"[ga] updated fractal gates ({joined}) best_loss={best_loss:.4f}")
                    except Exception as _e:
                        for param, base in zip(targets if 'targets' in locals() else [], bases if 'bases' in locals() else []):
                            param.data.copy_(base)
                        if tc.verbose:
                            print(f"[ga] skipped due to error: {_e}")

                if step % tc.ckpt_every == 0:
                    ckpt_obj = {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": cfg,
                        "step": step,
                    }
                    if ema is not None:
                        ckpt_obj["ema"] = ema.state_dict()
                    ckpt_path = os.path.join(run_dir, f"ckpt_{step}.pt")
                    torch.save(ckpt_obj, ckpt_path)
                    # also maintain a moving latest pointer
                    latest_path = os.path.join(run_dir, "latest.pt")
                    try:
                        torch.save(ckpt_obj, latest_path)
                    except Exception:
                        pass
                    # checkpoint progress context
                    elapsed = _time.time() - t_start
                    steps_done = step
                    pct = 100.0 * (steps_done / max(1, tc.steps))
                    steps_per_sec = max(1e-6, steps_done / max(elapsed, 1e-6))
                    remaining_steps = max(0, tc.steps - steps_done)
                    eta_sec = remaining_steps / steps_per_sec

                    def _fmt(s):
                        s = int(s)
                        h = s // 3600
                        m = (s % 3600) // 60
                        sec = s % 60
                        return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

                    print(f"[{pct:5.1f}%] saved checkpoint: {ckpt_path} eta={_fmt(eta_sec)}")
                if step >= tc.steps:
                    break
            if not saw_batch:
                src = str(data_cfg.get("source", "textdir"))
                if src == "textdir":
                    root_info = data_cfg.get("text_root") or data_cfg.get("root") or "<unset>"
                    detail = f"text_root={root_info}"
                elif src == "system":
                    detail = f"root={data_cfg.get('root', '<unset>')}"
                else:
                    detail = src
                raise RuntimeError(
                    "Data loader produced zero batches. Ensure your data source has files to read "
                    f"(source={src} {detail})."
                )
            if step >= tc.steps:
                break
    except KeyboardInterrupt:
        # Save interrupt checkpoint
        ckpt_path = os.path.join(run_dir, f"interrupt_{step}.pt")
        try:
            ckpt_obj = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg,
                "step": step,
            }
            if ema is not None:
                ckpt_obj["ema"] = ema.state_dict()
            torch.save(ckpt_obj, ckpt_path)
            print(f"interrupt: saved checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"interrupt: failed to save checkpoint: {e}")

    # Final checkpoint
    ckpt_path = os.path.join(run_dir, f"final_{step}.pt")
    final_ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "cfg": cfg,
        "step": step,
    }
    if ema is not None:
        final_ckpt["ema"] = ema.state_dict()
    torch.save(final_ckpt, ckpt_path)
    if ema is not None:
        ema_path = os.path.join(run_dir, f"ema_final_{step}.pt")
        torch.save({
            "model": ema.state_dict(),
            "cfg": cfg,
            "step": step,
        }, ema_path)
        print(f"saved EMA weights: {ema_path}")
    total_time = _time.time() - t_start
    def _fmt_total(s):
        s = int(s)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"
    print(f"done. saved final checkpoint: {ckpt_path} | total { _fmt_total(total_time) }")


if __name__ == "__main__":
    main()
