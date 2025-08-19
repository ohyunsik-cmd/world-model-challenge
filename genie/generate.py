#!/usr/bin/env python3
"""
Generates token sequences from a trained ST-MaskGIT model.

Supports:
- v2.0 Cosmos dataset (`--dataset cosmos`): uses [past3 | MASK*future3] with optional state-prefix conditioning.
- v1.1 raw token dataset (`--dataset v11`) for backward compatibility.

The script saves tokens to:
  <output_dir>/video.bin
and metadata to:
  <output_dir>/metadata.json
so downstream visualize/comic tools can consume them.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

# ---- Dataset imports (v2.0 prefers Cosmos) ----
# Expect you added data_cosmos.py per the new spec.
try:
    from data_cosmos import CosmosVideoDataset  # new dataset (v2.0)
except Exception:
    CosmosVideoDataset = None

# v1.1 fallback (kept for convenience)
from data import RawTokenDataset

# Model
from st_mask_git import STMaskGIT  # assumes your patched version that accepts states_future in compute/generate
# If your STMaskGIT class lives under a package, update the import accordingly.


def parse_args():
    p = argparse.ArgumentParser(description="Generate future frames (tokens) from a trained model.")
    # Common
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Path or HF repo to a HuggingFace-style checkpoint (config + weights).")
    p.add_argument("--output_dir", type=str, default="data/gen_generated",
                   help="Directory to save generated outputs (video.bin, metadata.json).")
    p.add_argument("--maskgit_steps", type=int, default=2,
                   help="Number of MaskGIT sampling steps per frame.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="If <=1e-8, greedy sampling (argmax); else categorical sampling.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)

    # Which dataset
    p.add_argument("--dataset", type=str, default="cosmos", choices=["cosmos", "v11"],
                   help="cosmos (v2.0) or v11 (legacy)")

    # ------ v2.0 (Cosmos) ------
    p.add_argument("--root", type=str, default="val_v2.0",
                   help="Root dir for Cosmos shard (contains *_{rank}.bin).")
    p.add_argument("--rank", type=int, default=0, help="Shard rank to read.")
    p.add_argument("--example_ind", type=int, default=0,
                   help="Clip index to run generation on (0 <= i < num_clips-1).")
    p.add_argument("--window_size", type=int, default=6,
                   help="Temporal window (past3 + future3 => 6).")
    p.add_argument("--num_prompt_frames", type=int, default=3,
                   help="How many context frames we condition on (3 for Cosmos past).")

    # ------ v1.1 (RawTokenDataset) ------
    p.add_argument("--val_data_dir", type=str, default="data/val_v1.1",
                   help="[v11] directory with metadata.json, video.bin")
    p.add_argument("--stride", type=int, default=15, help="[v11] stride between frames")
    p.add_argument("--v11_window_size", type=int, default=16,
                   help="[v11] temporal window size (T)")

    # Optional: disable teacher-forcing along time
    p.add_argument("--teacher_force_time", action="store_true",
                   help="Teacher-force across time (useful for debugging).")

    return p.parse_args()


@torch.no_grad()
def run_cosmos(args):
    assert CosmosVideoDataset is not None, "data_cosmos.py (CosmosVideoDataset) not found."

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Load dataset (tokens_{rank}.bin -> [num_clips,3,32,32], etc.)
    ds = CosmosVideoDataset(root=args.root, rank=args.rank,
                            mask_token_id=None,  # dataset builds input/label tensors itself
                            ignore_index=-100)

    # Pick one example
    ex = ds[args.example_ind]
    # input_ids_THW: [6, 32, 32] = [past3, MASK*future3]
    # labels_THW:    [6, 32, 32] = [IGNORE*3, future3]
    input_ids_THW = ex["input_ids_THW"].unsqueeze(0).to(device)   # (1, 6, 32, 32)
    labels_THW = ex["labels_THW"].unsqueeze(0).to(device)         # (1, 6, 32, 32)
    states_future = ex.get("states_future")                        # [17, 25]
    if states_future is not None:
        states_future = states_future.unsqueeze(0).to(device)      # (1, 17, 25)

    T, H, W = input_ids_THW.shape[1:]
    assert T == args.window_size, f"Cosmos window_size mismatch: got {T}, expected {args.window_size}"
    assert args.num_prompt_frames < T, "num_prompt_frames must be < window_size."

    # Load model
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    # Mask future (already masked by dataset, but re-assert)
    prompt_THW = input_ids_THW.clone()
    if hasattr(model, "mask_token_id"):
        prompt_THW[:, args.num_prompt_frames:] = model.mask_token_id

    # Generate for timesteps [num_prompt_frames .. T-1]
    samples = []
    for t in range(args.num_prompt_frames, T):
        # teacher-forced time (optional)
        if args.teacher_force_time:
            prompt_THW = labels_THW.clone()
            if hasattr(model, "mask_token_id"):
                prompt_THW[:, t:] = model.mask_token_id

        # MaskGIT step with (optional) state prefix conditioning
        # NOTE: This assumes your patched STMaskGIT.maskgit_generate accepts `states_future`.
        sample_HW, _ = model.maskgit_generate(
            prompt_THW, out_t=t,
            maskgit_steps=args.maskgit_steps,
            temperature=args.temperature,
            # new arg in your patched version (ignored if model not in prefix mode)
            states_future=states_future
        )

        samples.append(sample_HW)
        if not args.teacher_force_time:
            # autoregressive
            prompt_THW[:, t] = sample_HW

    # Compose outputs for saving:
    # [<prompt frames><predicted future frames><ground truth future frames>]
    pred_future_THW = torch.stack(samples, dim=1)                             # (1, T - num_prompt, H, W)
    outputs_THW = torch.cat([
        labels_THW[:, :args.num_prompt_frames],                               # context (GT)
        pred_future_THW,                                                      # predictions
        labels_THW[:, args.num_prompt_frames:],                               # GT future (for comic)
    ], dim=1)  # (1, num_prompt + (T-num_prompt) + (T-num_prompt), H, W)

    # Save tokens + metadata
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use dataset dtype to store
    token_dtype = np.dtype("int32")
    np.asarray(outputs_THW.squeeze(0).cpu().numpy(), dtype=token_dtype).tofile(out_dir / "video.bin")

    # Metadata compatible with visualize/comic pipeline (captioning)
    meta = {
        "dataset": "cosmos_v2.0",
        "num_images": int(outputs_THW.size(1)),
        "s": H,                 # latent side = 32
        "h": H, "w": W,         # kept for compatibility with some readers
        "t": args.window_size,  # original temporal window (6)
        "window_size": args.window_size,
        "num_prompt_frames": args.num_prompt_frames,
        "token_dtype": "int32",
        # extra notes
        "comment": "Outputs = [prompt | predictions | gtruth_future] for comic visualization",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[cosmos] Saved tokens to: {out_dir / 'video.bin'}")
    print(f"[cosmos] Saved metadata to: {out_dir / 'metadata.json'}")


@torch.no_grad()
def run_v11(args):
    """Legacy path to match older RawTokenDataset generate.py behavior."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ds = RawTokenDataset(args.val_data_dir, window_size=args.v11_window_size, stride=args.stride)
    s = ds.metadata["s"]
    example_THW = ds[0]["input_ids"].reshape(1, args.v11_window_size, s, s).to(device)

    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    prompt_THW = example_THW.clone()
    if hasattr(model, "mask_token_id"):
        prompt_THW[:, args.num_prompt_frames:] = model.mask_token_id

    samples = []
    for t in range(args.num_prompt_frames, args.v11_window_size):
        if args.teacher_force_time:
            prompt_THW = example_THW.clone()
            if hasattr(model, "mask_token_id"):
                prompt_THW[:, t:] = model.mask_token_id

        sample_HW, _ = model.maskgit_generate(
            prompt_THW, out_t=t, maskgit_steps=args.maskgit_steps, temperature=args.temperature
        )
        samples.append(sample_HW)
        if not args.teacher_force_time:
            prompt_THW[:, t] = sample_HW

    outputs = torch.stack(samples, dim=1)
    outputs = torch.cat([example_THW[:, :args.num_prompt_frames], outputs], dim=1)
    outputs = torch.cat([outputs, example_THW[:, args.num_prompt_frames:]], dim=1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_dtype = np.dtype(ds.metadata.get("token_dtype", "uint32"))
    outputs.cpu().numpy().astype(token_dtype).tofile(out_dir / "video.bin")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "dataset": "v1.1_raw",
            "num_images": int(outputs.shape[1]),
            "s": s,
            "h": s, "w": s,
            "t": args.v11_window_size,
            "window_size": args.v11_window_size,
            "num_prompt_frames": args.num_prompt_frames,
            "token_dtype": str(token_dtype),
        }, f, indent=2)

    print(f"[v11] Saved tokens to: {out_dir / 'video.bin'}")
    print(f"[v11] Saved metadata to: {out_dir / 'metadata.json'}")


def main():
    args = parse_args()
    if args.dataset == "cosmos":
        run_cosmos(args)
    else:
        run_v11(args)


if __name__ == "__main__":
    main()
