# ===== FILE: data.py =====
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import glob, json
from pathlib import Path
from torch.utils.data import ConcatDataset


# ------------------------------
# v1.1 레거시 (그대로 유지 가능)
# ------------------------------
class RawTokenDataset(TorchDataset):
    """Simple memmap-backed dataset for legacy v1.1 (per-frame MAGVIT tokens)."""
    def __init__(
        self,
        data_dir: str | Path,
        window_size: int,
        stride: int = 1,
        filter_interrupts: bool = True,
        filter_overlaps: bool = False,
    ):
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
        video_tokens_path = data_dir / "video.bin"
        segment_ids_path = data_dir / "segment_ids.bin"

        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(video_tokens_path, dtype=token_dtype, mode="r", shape=shape)

        if segment_ids_path.exists():
            self.segment_ids = np.memmap(segment_ids_path, dtype=np.int32, mode="r", shape=(self.metadata["num_images"],))
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        self.window_size, self.stride = int(window_size), int(stride)
        self.video_len = (self.window_size - 1) * self.stride

        # valid start indices
        self.valid_start_inds: list[int] = []
        for start_ind in range(len(self.data) - self.video_len):
            if not (filter_interrupts and self.segment_ids is not None
                    and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

        if filter_overlaps:
            filtered: list[int] = []
            for start_ind in self.valid_start_inds:
                overlapping = {start_ind - i * self.stride for i in range(1, self.window_size)}
                for prev in filtered[-self.window_size * self.stride:]:
                    if prev in overlapping:
                        break
                else:
                    filtered.append(start_ind)
            self.valid_start_inds = filtered

    def __len__(self) -> int:
        return len(self.valid_start_inds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy(self.data[start_ind : start_ind + self.video_len + 1 : self.stride].astype(np.int64))
        x = x.flatten()
        attn = torch.ones_like(x)
        return {"input_ids": x, "labels": x, "attention_mask": attn}


# ------------------------------
# v2.0 Cosmos (DV 8×8×8)
# ------------------------------

FRAMES_PER_CLIP = 17      # raw 17 frames per clip
LATENT_T = 3              # 17 → 3 latent timesteps
H_LATENT = 32
W_LATENT = 32
T_PAST = 3
T_FUTURE = 3
T_TOTAL = T_PAST + T_FUTURE  # 6

IMAGE_VOCAB_SIZE = 64000
DEFAULT_MASK_TOKEN_ID = IMAGE_VOCAB_SIZE
DEFAULT_IGNORE_INDEX = -100

def list_available_shards(root: str) -> list[int]:
    """data/<split>/metadata/metadata_*.json 검사해서 사용 가능한 shard index들을 반환."""
    meta_dir = Path(root) / "metadata"
    if meta_dir.exists():
        files = sorted(glob.glob(str(meta_dir / "metadata_*.json")))
        ranks = []
        for f in files:
            # .../metadata_12.json -> 12
            stem = Path(f).stem
            ind = int(stem.split("_")[-1])
            ranks.append(ind)
        return ranks
    # fallback: metadata.json에 num_shards 있으면 0..num_shards-1 반환
    mj = json.load(open(Path(root) / "metadata.json"))
    num = int(mj.get("num_shards", 1))
    return list(range(num))

def build_concat_cosmos(root: str, ranks: list[int]) -> ConcatDataset:
    """여러 shards를 하나로 합친 ConcatDataset."""
    parts = [CosmosVideoDataset(root, rank=r) for r in ranks]
    return ConcatDataset(parts)

def _first_existing(root: Path, candidates: List[str]) -> Optional[Path]:
    """Return the first existing path under root for the given relative paths."""
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def _edge_repeat_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    Take arr[start : start+length] with edge-repeat padding if it spills over.
    arr: [F, ...]  →  returns [length, ...]
    """
    F = arr.shape[0]
    end = start + length
    if start >= F:
        return np.repeat(arr[F - 1:F], length, axis=0)
    if end <= F:
        return arr[start:end]
    valid = arr[start:F]
    pad = np.repeat(arr[F - 1:F], end - F, axis=0)
    return np.concatenate([valid, pad], axis=0)


class CosmosVideoDataset(TorchDataset):
    """
    Cosmos v2.0 dataset loader for folder-split layout.

    Expected directory layout (example):
      <split>/
        metadata.json
        metadata/            # shard metas
          metadata_0.json
          metadata_1.json
          ...
        videos/
          video_0.bin        # [num_clips, 3, 32, 32] (int32 preferred; uint16 also supported)
          video_1.bin
        robot_states/
          states_0.bin       # [F, 25] float32
          states_1.bin
        segment_indices/     # optional
          segment_idx_0.bin  # [F] int32
          segment_idx_1.bin

    We also support a few fallbacks for legacy names:
      - tokens_{rank}.bin, video_{rank}.bin at split root
      - states_{rank}.bin at split root
      - segment_idx_{rank}.bin or segment_ids_{rank}.bin at split root
    """

    def __init__(
        self,
        root: str | Path,
        rank: int = 0,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ):
        super().__init__()
        self.root = Path(root)
        self.rank = int(rank)
        self.mask_token_id = int(mask_token_id)
        self.ignore_index = int(ignore_index)

        # ---------- global metadata ----------
        global_meta_path = self.root / "metadata.json"
        if not global_meta_path.exists():
            raise FileNotFoundError(f"Missing global metadata at {global_meta_path}")
        self.meta_all: Dict[str, Any] = json.load(open(global_meta_path, "r"))

        # ---------- shard metadata ----------
        # primary: <split>/metadata/metadata_{rank}.json
        # fallbacks: <split>/metadata_{rank}.json, <split>/../metadata/metadata_{rank}.json
        shard_meta_path = _first_existing(self.root, [
            f"metadata/metadata_{self.rank}.json",
            f"metadata_{self.rank}.json",
            f"../metadata/metadata_{self.rank}.json",
        ])
        if shard_meta_path is None:
            raise FileNotFoundError(
                f"Missing shard metadata for rank={self.rank}. "
                f"Tried: metadata/metadata_{self.rank}.json, metadata_{self.rank}.json, ../metadata/metadata_{self.rank}.json under {self.root}"
            )
        self.meta_shard: Dict[str, Any] = json.load(open(shard_meta_path, "r"))
        if "shard_num_frames" not in self.meta_shard:
            raise KeyError(f"'shard_num_frames' missing in {shard_meta_path}")
        self.total_frames: int = int(self.meta_shard["shard_num_frames"])

        # ---------- locate data files ----------
        # videos
        self.tokens_path = _first_existing(self.root, [
            f"videos/video_{self.rank}.bin",    # preferred
            f"video_{self.rank}.bin",           # fallback at split root
            f"tokens_{self.rank}.bin",          # legacy
        ])
        if self.tokens_path is None:
            raise FileNotFoundError(
                f"Missing video tokens for rank={self.rank}. Looked for "
                f"'videos/video_{self.rank}.bin', 'video_{self.rank}.bin', 'tokens_{self.rank}.bin' under {self.root}"
            )

        # robot states
        self.states_path = _first_existing(self.root, [
            f"robot_states/states_{self.rank}.bin",  # preferred
            f"states_{self.rank}.bin",               # fallback at split root
        ])
        if self.states_path is None:
            raise FileNotFoundError(
                f"Missing robot states for rank={self.rank}. Looked for "
                f"'robot_states/states_{self.rank}.bin', 'states_{self.rank}.bin' under {self.root}"
            )

        # segment indices (optional)
        self.segidx_path = _first_existing(self.root, [
            f"segment_indices/segment_idx_{self.rank}.bin",
            f"segment_idx_{self.rank}.bin",
            f"segment_ids_{self.rank}.bin",
        ])

        # ---------- shape constants ----------
        self.latent_h = H_LATENT
        self.latent_w = W_LATENT
        self.t_past = T_PAST
        self.t_future = T_FUTURE
        self.t_total = T_TOTAL
        self.frames_per_clip = FRAMES_PER_CLIP

        # ---------- memmap ----------
        # states: [F, 25]
        self.states_mm = np.memmap(self.states_path, dtype=np.float32, mode="r", shape=(self.total_frames, 25))

        # num_clips from F
        self.num_clips = int(math.ceil(self.total_frames / self.frames_per_clip))

        # tokens: [num_clips, 3, 32, 32]  (int32 → fallback uint16)
        def _try_tokens(dtype: np.dtype):
            try:
                return np.memmap(self.tokens_path, dtype=dtype, mode="r",
                                 shape=(self.num_clips, self.t_past, self.latent_h, self.latent_w))
            except Exception:
                return None

        self.tokens_mm = _try_tokens(np.int32)
        if self.tokens_mm is None:
            self.tokens_mm = _try_tokens(np.uint16)

        if self.tokens_mm is None:
            raise FileNotFoundError(f"Could not load tokens file in either int32 or uint16 format at {data_dir}")

        # optional segment idx
        if self.segidx_path is not None:
            try:
                self.segment_idx_mm = np.memmap(self.segidx_path, dtype=np.int32, mode="r", shape=(self.total_frames,))
            except Exception:
                self.segment_idx_mm = None
        else:
            self.segment_idx_mm = None

        # info (reference)
        self.metadata: Dict[str, Any] = {
            "dataset": "cosmos_v2.0",
            "hz": int(self.meta_all.get("hz", 30)),
            "rank": self.rank,
            "shard_num_frames": self.total_frames,
            "num_clips": self.num_clips,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "t_total": self.t_total,
            "mask_token_id": self.mask_token_id,
            "ignore_index": self.ignore_index,
            "paths": {
                "global_meta": str(global_meta_path),
                "shard_meta": str(shard_meta_path),
                "tokens": str(self.tokens_path),
                "states": str(self.states_path),
                "segment_idx": str(self.segidx_path) if self.segidx_path is not None else None,
            },
        }

    def __len__(self) -> int:
        # pair (i -> past, i+1 -> future); last clip has no future partner
        return max(0, self.num_clips - 1)

    def _clip_states(self, clip_idx: int) -> np.ndarray:
        """Return [17,25] states for given clip_idx with edge-repeat padding."""
        start = clip_idx * self.frames_per_clip
        return _edge_repeat_slice(self.states_mm, start, self.frames_per_clip)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i < 0 or i >= self.__len__():
            raise IndexError(f"Index {i} out of range for num_samples={self.__len__()}")

        # past/future latent tokens
        past3 = self.tokens_mm[i]           # [3,32,32]
        fut3  = self.tokens_mm[i + 1]       # [3,32,32]

        # inputs/labels (THW)
        masked_future3 = np.full_like(fut3, fill_value=self.mask_token_id, dtype=np.int32)
        ignore_past3   = np.full_like(past3, fill_value=self.ignore_index, dtype=np.int32)

        input_ids_THW = np.concatenate([past3, masked_future3], axis=0)   # [6,32,32]
        labels_THW    = np.concatenate([ignore_past3, fut3], axis=0)      # [6,32,32]

        # states (edge-repeat)
        states_future = self._clip_states(i + 1)                           # [17,25]

        # torch tensors (flatten to [T*H*W])
        input_ids = torch.from_numpy(input_ids_THW.astype(np.int64)).view(self.t_total, -1).flatten()
        labels    = torch.from_numpy(labels_THW.astype(np.int64)).view(self.t_total, -1).flatten()

        return {
            "input_ids": input_ids,                 # [6*1024]
            "labels": labels,                       # [6*1024] (past 3 are ignore_index)
            "states_future": torch.from_numpy(states_future.copy()).float(),  # [17,25]
            # 디버깅/시각화를 원하면 아래 2개도 유용
            "input_ids_THW": torch.from_numpy(input_ids_THW.astype(np.int64)),  # [6,32,32]
            "labels_THW": torch.from_numpy(labels_THW.astype(np.int64)),        # [6,32,32]
        }
