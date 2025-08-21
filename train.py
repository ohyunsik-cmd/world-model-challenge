# ===== FILE: train.py =====
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from einops import rearrange

# ---- Project modules (v2.0 전용 구성) ----
from data import CosmosVideoDataset, list_available_shards, build_concat_cosmos
from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT

logger = get_logger(__name__)


# -------------------------------
# Arg helpers
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


# -------------------------------
# Checkpoint I/O
# -------------------------------
def save_checkpoint(
    accelerator: Accelerator,
    model: STMaskGIT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config: GenieConfig,
    output_dir: Path,
    step: int,
    tag: Optional[str] = None,
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
        try:
            torch.save({"scheduler": scheduler.state_dict()}, ckpt_dir / "scheduler.pt")
        except Exception:
            pass

    if accelerator.is_main_process:
        logger.info(f"Saved checkpoint to {ckpt_dir}")


def _infer_step_from_dir(ckpt_dir: Path) -> int:
    # checkpoint_step{N} or checkpoint_best/final
    name = ckpt_dir.name
    if name.startswith("checkpoint_step"):
        try:
            return int(name.split("checkpoint_step")[-1])
        except Exception:
            return 0
    return 0


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    cands = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_step")]
    if not cands:
        return None
    cands.sort(key=_infer_step_from_dir)
    return cands[-1]


def load_checkpoint_if_any(
    accelerator: Accelerator,
    model: STMaskGIT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    resume_from: Optional[str],
) -> int:
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
        try:
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu")["scheduler"])
        except Exception:
            pass

    # 추정 global step
    step = _infer_step_from_dir(ckpt_dir)
    if accelerator.is_main_process:
        logger.info(f"Resumed from {ckpt_dir} (step≈{step})")
    return step


# -------------------------------
# Scheduler
# -------------------------------
def build_scheduler(optimizer, warmup_steps, max_steps, name: str = "cosine"):
    try:
        from transformers import get_scheduler
        return get_scheduler(name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    except Exception:
        class _Cosine:
            def __init__(self, opt, warmup, total):
                self.opt = opt
                self.warmup = int(warmup)
                self.total = int(max(1, total))
                self.step_idx = 0
                self.base_lrs = [g["lr"] for g in opt.param_groups]
            def step(self):
                self.step_idx += 1
                # linear warmup
                if self.step_idx <= self.warmup:
                    scale = self.step_idx / max(1, self.warmup)
                else:
                    t = (self.step_idx - self.warmup) / max(1, self.total - self.warmup)
                    scale = 0.5 * (1.0 + math.cos(math.pi * t))
                for i, g in enumerate(self.opt.param_groups):
                    g["lr"] = self.base_lrs[i] * scale
            def state_dict(self):
                return {"step_idx": self.step_idx, "base_lrs": self.base_lrs}
            def load_state_dict(self, s):
                self.step_idx = s.get("step_idx", 0)
                self.base_lrs = s.get("base_lrs", self.base_lrs)
            def get_last_lr(self):
                return [g["lr"] for g in self.opt.param_groups]
        return _Cosine(optimizer, warmup_steps, max_steps)


# -------------------------------
# Argparse
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ST-MaskGIT on 1xGPT v2.0 (Cosmos 8×8×8)")

    # Data
    p.add_argument("--train_data_dir", type=str, required=True,
                   help="Shard root (contains video_{rank}.bin, states_{rank}.bin, metadata.json, data/metadata/metadata_{rank}.json)")
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--rank", type=int, default=0, help="Shard rank id suffix")

    # Model
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)

    # Optim
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_train_steps", type=int, default=30000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loader
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=str2bool, default=True)

    # Eval/Save
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--auto_resume", type=str2bool, default=True, help="resume from latest checkpoint in output_dir if present")

    # Logging
    p.add_argument("--output_dir", type=str, default="outputs/cosmos_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_with", type=str, default="wandb", choices=[None, "wandb", "tensorboard", "comet"])
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
        accelerator.init_trackers(
            args.project_name,
            config=vars(args),
            init_kwargs={args.log_with: {"name": args.run_name}} if args.run_name else None,
        )

    # ---- Dataset (v2.0 Cosmos) ----

    if args.rank == -1:
        # 모든 shard를 묶어서 하나의 큰 데이터셋으로
        train_ranks = list_available_shards(args.train_data_dir)
        print(f"DEBUG: Found {len(train_ranks)} shards: {train_ranks}")  # 강제로 출력
        train_set = build_concat_cosmos(args.train_data_dir, train_ranks)

        if args.val_data_dir:
            val_ranks = list_available_shards(args.val_data_dir)
            print(f"DEBUG: Found {len(val_ranks)} val shards: {val_ranks}")
            val_set = build_concat_cosmos(args.val_data_dir, val_ranks)
        else:
            val_set = None

        if accelerator.is_main_process:
            logger.info(f"[concat] train shards: {len(train_ranks)} -> {train_ranks[:8]}{'...' if len(train_ranks)>8 else ''}")
            if val_set is not None:
                logger.info(f"[concat] val   shards: {len(val_ranks)} -> {val_ranks[:8]}{'...' if len(val_ranks)>8 else ''}")
    else:
        # 단일 샤드 모드(기존 동작)
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

    # ---- Config & Model (v2.0 기본값: T=6, S=1024, vocab 40×3 등은 config 내부/모델에서 사용) ----
    config = GenieConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        # 나머지 v2.0 값은 프로젝트의 config.py 기본/모델 내부 상수 사용
    )
    model = STMaskGIT(config)

    # ---- Optim / Scheduler ----
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=tuple(args.betas), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_train_steps, name="cosine")

    # ---- Prepare with accelerate ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ---- (Auto) Resume ----
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume:
        cand = latest_checkpoint(output_dir)
        if cand is not None:
            resume_path = str(cand)

    completed_steps = 0
    if resume_path:
        completed_steps = load_checkpoint_if_any(accelerator, model, optimizer, scheduler, resume_path)

    # ---- SIGTERM/SIGINT 핸들러: 중도 종료 시 최신 체크포인트 저장 ----
    def _graceful_save(*_):
        if accelerator.is_main_process:
            logger.warning("Signal received: saving a last-minute checkpoint...")
        save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, max(completed_steps, 0), tag="interrupt")
        raise SystemExit(0)

    if accelerator.is_main_process:
        signal.signal(signal.SIGTERM, _graceful_save)
        signal.signal(signal.SIGINT, _graceful_save)

    # ---- Training loop (에폭 반복하며 max_train_steps까지) ----
    accelerator.print(f"Num train batches per epoch: {len(train_loader)}")
    if val_loader is not None:
        accelerator.print(f"Num val batches: {len(val_loader)}")

    model.train()
    best_eval = float("inf")
    running_loss = 0.0
    running_acc = 0.0
    step_time_ma = None
    log_every = max(10, args.eval_steps // 5 if args.eval_steps else 50)
    start_wall = time.time()

    # 작은 유틸
    def _now_mem_gb():
        if not torch.cuda.is_available():
            return 0.0, 0.0
        dev = torch.cuda.current_device()
        return (
            torch.cuda.memory_allocated(dev) / (1024 ** 3),
            torch.cuda.max_memory_allocated(dev) / (1024 ** 3),
        )

    epoch = 0
    while completed_steps < args.max_train_steps:
        epoch += 1
        for batch in train_loader:
            if completed_steps >= args.max_train_steps:
                break

            t0 = time.time()
            try:
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        states_future=batch["states_future"],
                    )
                    loss = outputs.loss

                    accelerator.backward(loss)

                    if args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError as e:
                # OOM 안전장치: 스텝 스킵 + 그래드 초기화
                if "CUDA out of memory" in str(e):
                    if accelerator.is_main_process:
                        logger.error("CUDA OOM: skipping step, clearing cache.")
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise

            completed_steps += 1

            # 통계
            running_loss += float(loss.detach())
            if getattr(outputs, "acc", None) is not None:
                running_acc += float(outputs.acc.detach())

            # ETA/진행률
            dt = time.time() - t0
            step_time_ma = dt if step_time_ma is None else (0.9 * step_time_ma + 0.1 * dt)
            done = completed_steps
            total = args.max_train_steps
            pct = min(100.0, 100.0 * done / max(1, total))
            remain = max(0, total - done)
            eta_s = remain * step_time_ma
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            mem_gb, max_mem_gb = _now_mem_gb()

            # 주기 로그 (스텝 기준)
            if (completed_steps % log_every == 0) or (completed_steps == 1):
                denom = max(1, (log_every if getattr(outputs, "acc", None) is not None else 1))
                logs = {
                    "train/loss": running_loss / log_every,
                    "train/acc": running_acc / denom,
                    "lr": lr,
                    "step": completed_steps,
                    "epoch": epoch,
                    "progress/percent": pct,
                    "speed/step_time_s_ma": step_time_ma,
                    "speed/sps_tokens": None,  # 필요시 채워넣기
                    "gpu/mem_gb": mem_gb,
                    "gpu/max_mem_gb": max_mem_gb,
                    "time/elapsed_min": (time.time() - start_wall) / 60.0,
                    "time/eta_min": eta_s / 60.0,
                }
                accelerator.log(logs, step=completed_steps)
                if accelerator.is_main_process:
                    logger.info(json.dumps({k: (float(v) if isinstance(v, (float, int)) else v) for k, v in logs.items()}))
                running_loss = 0.0
                running_acc = 0.0

            # 평가 (스텝 기준)
            if args.eval_steps and (completed_steps % args.eval_steps == 0) and (val_loader is not None):
                model.eval()
                eval_loss, eval_acc, n_batches = 0.0, 0.0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vout = model(
                            input_ids=vbatch["input_ids"],
                            labels=vbatch["labels"],
                            states_future=vbatch["states_future"],
                        )
                        eval_loss += float(vout.loss.detach())
                        if getattr(vout, "acc", None) is not None:
                            eval_acc += float(vout.acc.detach())
                        n_batches += 1
                eval_loss /= max(1, n_batches)
                eval_acc /= max(1, n_batches)
                accelerator.log({"eval/loss": eval_loss, "eval/acc": eval_acc, "step": completed_steps}, step=completed_steps)
                if accelerator.is_main_process:
                    logger.info(f"[eval] step={completed_steps} loss={eval_loss:.4f} acc={eval_acc:.4f}")

                # best (eval loss)
                if accelerator.is_main_process and eval_loss < best_eval:
                    best_eval = eval_loss
                    save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, completed_steps, tag="best")
                model.train()

            # 주기 저장 (스텝 기준)
            if args.save_steps and (completed_steps % args.save_steps == 0) and accelerator.is_main_process:
                save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, completed_steps)

    # 최종 저장
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, scheduler, config, output_dir, completed_steps, tag="final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
