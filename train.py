# ===== FILE: train.py =====
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange

# Project modules (v2.0 전용 구성)
from data import CosmosVideoDataset
from genie.config import GenieConfig  # 만약 패키지 구조라면: from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT  # 패키지 구조면: from genie.st_mask_git import STMaskGIT


logger = get_logger(__name__)


# -------------------------------
# Utilities
# -------------------------------
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    vv = v.lower()
    if vv in ("true", "1", "yes", "y", "on"):
        return True
    if vv in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def save_checkpoint(
    accelerator: Accelerator,
    model: STMaskGIT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config: GenieConfig,
    output_dir: Path,
    step: int,
    tag: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint_step{step}" if tag is None else f"checkpoint_{tag}"
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), ckpt_dir / "pytorch_model.bin")
    config.save_pretrained(str(ckpt_dir / "config.json"))
    torch.save({"optimizer": optimizer.state_dict()}, ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save({"scheduler": scheduler.state_dict()}, ckpt_dir / "scheduler.pt")

    if accelerator.is_main_process:
        logger.info(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint_if_any(
    accelerator: Accelerator,
    model: STMaskGIT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    resume_from: Optional[str],
):
    if not resume_from:
        return 0
    ckpt_dir = Path(resume_from)
    model_path = ckpt_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"No model weights at {model_path}")

    state = torch.load(model_path, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(state, strict=True)

    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu")["optimizer"])
    sch_path = ckpt_dir / "scheduler.pt"
    if scheduler is not None and sch_path.exists():
        scheduler.load_state_dict(torch.load(sch_path, map_location="cpu")["scheduler"])

    # global step heuristic
    try:
        step_str = ckpt_dir.name.split("step")[-1]
        return int(step_str)
    except Exception:
        return 0


def build_scheduler(optimizer, warmup_steps, max_steps, name: str = "cosine"):
    try:
        from transformers import get_scheduler
        return get_scheduler(name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    except Exception:
        # Fallback: simple cosine
        class _Cosine:
            def __init__(self, opt, warmup, total):
                self.opt = opt
                self.warmup = warmup
                self.total = total
                self.step_idx = 0
                self.base_lrs = [g["lr"] for g in opt.param_groups]
            def step(self):
                self.step_idx += 1
                t = min(self.step_idx / max(1, self.total), 1.0)
                if self.step_idx <= self.warmup:
                    scale = self.step_idx / max(1, self.warmup)
                else:
                    # cosine
                    tt = (self.step_idx - self.warmup) / max(1, self.total - self.warmup)
                    scale = 0.5 * (1.0 + math.cos(math.pi * tt))
                for i, g in enumerate(self.opt.param_groups):
                    g["lr"] = self.base_lrs[i] * scale
            def state_dict(self): return {"step_idx": self.step_idx, "base_lrs": self.base_lrs}
            def load_state_dict(self, s): self.step_idx = s.get("step_idx", 0); self.base_lrs = s.get("base_lrs", self.base_lrs)
            def get_last_lr(self): return [g["lr"] for g in self.opt.param_groups]
        return _Cosine(optimizer, warmup_steps, max_steps)


# -------------------------------
# Argparse
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ST-MaskGIT (v2.0 Cosmos 8×8×8)")

    # Data
    p.add_argument("--train_data_dir", type=str, required=True, help="Path to shard root containing tokens_{rank}.bin, states_{rank}.bin")
    p.add_argument("--val_data_dir", type=str, default=None, help="Optional path for validation shard")
    p.add_argument("--rank", type=int, default=0, help="Shard rank id (filename suffix)")

    # Model
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)

    # Optimization
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Batch / Loader
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=str2bool, default=True)

    # Eval / Save
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from", type=str, default=None)

    # Logging
    p.add_argument("--output_dir", type=str, default="outputs/cosmos_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_with", type=str, default=None, choices=[None, "wandb", "tensorboard", "comet"], help="Experiment tracker via accelerate")
    p.add_argument("--project_name", type=str, default="stmaskgit-cosmos-v2")
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    accelerator = Accelerator(log_with=args.log_with)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if args.log_with is not None:
        accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={args.log_with: {"name": args.run_name}} if args.run_name else None)

    # ---- Dataset (v2.0 전용) ----
    train_set = CosmosVideoDataset(args.train_data_dir, rank=args.rank)
    val_set = CosmosVideoDataset(args.val_data_dir, rank=args.rank) if args.val_data_dir else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=args.pin_memory,
            drop_last=False,
        )

    # ---- Config & Model (v2.0 기본값 내장) ----
    config = GenieConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        # v2.0 공통값은 config.py의 기본으로 잡힘 (T=6, S=1024, vocab 40×3, prefix on)
    )
    model = STMaskGIT(config)

    # ---- Optim / Scheduler ----
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=tuple(args.betas), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_train_steps, name="cosine")

    # ---- Prepare with accelerate ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    # ---- Optionally resume ----
    global_step = load_checkpoint_if_any(accelerator, model, optimizer, scheduler, args.resume_from)

    # ---- Train loop ----
    accelerator.print(f"Num train batches: {len(train_loader)}")
    if val_loader is not None:
        accelerator.print(f"Num val batches: {len(val_loader)}")

    model.train()
    best_eval = float("inf")
    running_loss = 0.0
    running_acc = 0.0
    log_every = max(10, args.eval_steps // 5)

    for step, batch in enumerate(train_loader, start=global_step + 1):
        # batch: {"input_ids":[B,6*32*32], "labels":[B,6*32*32], "states_future":[B,17,25]}
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                states_future=batch["states_future"],
            )
            loss = outputs.loss
            accelerator.backward(loss)

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach().float().item()
        if outputs.acc is not None:
            running_acc += outputs.acc.detach().float().item()

        # logging
        if step % log_every == 0 or step == 1:
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            logs = {
                "train/loss": running_loss / log_every,
                "train/acc": running_acc / max(1, (log_every if outputs.acc is not None else 0) or 1),
                "lr": lr,
                "step": step,
            }
            accelerator.log(logs, step=step)
            if accelerator.is_main_process:
                logger.info(json.dumps({k: float(v) if isinstance(v, torch.Tensor) else v for k, v in logs.items()}))
            running_loss, running_acc = 0.0, 0.0

        # eval
        if args.eval_steps and step % args.eval_steps == 0 and val_loader is not None:
            model.eval()
            eval_loss, eval_acc, n_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vout = model(
                        input_ids=vbatch["input_ids"],
                        labels=vbatch["labels"],
                        states_future=vbatch["states_future"],
                    )
                    eval_loss += vout.loss.detach().float().item()
                    if vout.acc is not None:
                        eval_acc += vout.acc.detach().float().item()
                    n_batches += 1
            eval_loss /= max(1, n_batches)
            eval_acc /= max(1, n_batches)
            accelerator.log({"eval/loss": eval_loss, "eval/acc": eval_acc, "step": step}, step=step)
            if accelerator.is_main_process:
                logger.info(f"[eval] step={step} loss={eval_loss:.4f} acc={eval_acc:.4f}")

            # best checkpoint (by eval loss)
            if eval_loss < best_eval and accelerator.is_main_process:
                best_eval = eval_loss
                save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, step, tag="best")
            model.train()

        # save
        if args.save_steps and step % args.save_steps == 0 and accelerator.is_main_process:
            save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, step)

        if step >= args.max_train_steps:
            break

    # final save
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, step, tag="final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
