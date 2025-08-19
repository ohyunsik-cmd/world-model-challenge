# ===== FILE: evaluate.py =====
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

# 로컬 모듈
sys.path.append(os.getcwd())
from eval_utils import (
    compute_loss_future_only_v2,
    compute_acc_future_only_v2,
)
from genie.st_mask_git import STMaskGIT


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GENIE (v2.0, future-only CE/Acc).")

    # 데이터 경로 / 로더
    p.add_argument("--val_data_dir", type=str, required=True,
                   help="샤드/메타가 들어있는 v2.0 검증 폴더 (예: data/val_v2.0)")
    p.add_argument("--rank", type=int, default=0, help="읽을 샤드 rank")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")

    # 모델 & 평가
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="HuggingFace-style 체크포인트 경로")
    p.add_argument("--max_batches", type=int, default=None,
                   help="디버그용: 지정 시 해당 배치 수까지만 평가")

    # v2.0 고정 하이퍼파라미터(필요시 덮어쓸 수 있게 인자화)
    p.add_argument("--T_total", type=int, default=6)          # past3 + future3
    p.add_argument("--future_start", type=int, default=3)     # 미래 시작 인덱스
    p.add_argument("--factored_vocab_size", type=int, default=40)  # V
    p.add_argument("--num_factored_vocabs", type=int, default=3)    # F
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--ignore_index", type=int, default=-100)

    return p.parse_args()


def _load_dataset(val_dir: str, rank: int):
    """
    CosmosVideoDataset가 있으면 사용하고, 없으면 RawTokenDataset로 폴백.
    CosmosVideoDataset:
      __getitem__가 dict 반환:
        - input_ids: [6*H*W], labels: [6*H*W], states_future: [17,25] (옵션)
    RawTokenDataset (v1 포맷)일 경우에는 window_size/stride 등을 강제 지정해야 하는데,
    v2.0 용 평가에서는 Cosmos 사용을 권장.
    """
    # 1) Cosmos 우선
    try:
        from data import CosmosVideoDataset
        ds = CosmosVideoDataset(root=val_dir, rank=rank)
        use_cosmos = True
        return ds, use_cosmos
    except Exception:
        # 2) 폴백: RawTokenDataset (가능하면 window_size=6로)
        from data import RawTokenDataset
        # RawTokenDataset은 window_size/stride 가 필요. v2.0에 맞추어 window_size=6.
        ds = RawTokenDataset(val_dir, window_size=6, stride=1, filter_overlaps=True)
        use_cosmos = False
        return ds, use_cosmos


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터셋
    val_dataset, use_cosmos = _load_dataset(args.val_data_dir, args.rank)

    def _collate(batch):
        # Cosmos/RawTokenDataset 둘 다 dict를 반환한다고 가정하고, 기본 스택
        out = {}
        keys = batch[0].keys()
        for k in keys:
            if batch[0][k] is None:
                out[k] = None
            else:
                out[k] = torch.stack([torch.as_tensor(ex[k]) for ex in batch])
        return out

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=_collate,
    )

    # 모델 로드
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    T = args.T_total
    H = args.H
    W = args.W
    V = args.factored_vocab_size
    F = args.num_factored_vocabs
    Tf = T - args.future_start  # 3

    # 집계 메트릭
    total_loss = 0.0
    total_acc = 0.0
    total_frames = 0  # 배치 샘플 수 기준이 아니라, 배치 내 유효 프레임 수(미래3*B)로 스케일링해도 되고,
                      # 여기선 단순히 샘플 수(=배치 수) 평균으로 처리.

    # 평가 루프
    for bi, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        input_ids = batch["input_ids"].to(device)      # [B, T*H*W]
        labels = batch["labels"].to(device)            # [B, T*H*W] (과거는 IGNORE일 수 있음)
        states_future = batch.get("states_future", None)
        if states_future is not None:
            states_future = states_future.to(device)   # [B,17,25] 예상

        B = input_ids.size(0)

        # [B, T, H, W]로 변환
        x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # 모델 로짓 계산: [B, C=V*F, T, H, W]
        logits_CTHW = model.compute_logits(x_THW)

        # 미래 3프레임 로짓만 선택 후, [B, V, F, Tf, H, W]로 재배열
        logits_future = logits_CTHW[:, :, args.future_start:]  # [B, C, Tf, H, W]
        factored_logits = rearrange(
            logits_future,
            "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
            vocab_size=V,
            num_vocabs=F,
        )  # [B, V, F, Tf, H, W] (Tf==3)

        # 손실 (미래 프레임만 CE)
        loss = compute_loss_future_only_v2(
            labels=labels,
            factored_logits=factored_logits,
            T_total=T,
            future_start=args.future_start,
            num_factored_vocabs=F,
            factored_vocab_size=V,
            H=H,
            W=W,
            ignore_index=args.ignore_index,
            reduction="mean",
        )

        # 정확도 (미래 프레임만)
        #   per-factor argmax -> [B, F, Tf, H, W] -> (F 축을 팩토리 합성해) [B, Tf, H, W]
        preds_factor = factored_logits.argmax(dim=1)  # [B, F, Tf, H, W]
        preds_THWF = rearrange(preds_factor, "b f t h w -> b t h w f")
        # unfactorize
        from genie.factorization_utils import unfactorize_token_ids
        preds_future_THW = unfactorize_token_ids(
            preds_THWF, num_factored_vocabs=F, factored_vocab_size=V
        )  # [B, Tf, H, W]

        # 라벨/샘플 전체 T축 텐서로 맞추기 위해 과거 3프레임은 dummy로 채워 concat
        pad = torch.full((B, args.future_start, H, W), fill_value=args.ignore_index, device=preds_future_THW.device)
        preds_THW = torch.cat([pad, preds_future_THW], dim=1)  # [B, T, H, W]
        acc = compute_acc_future_only_v2(
            samples=preds_THW,
            labels=rearrange(labels, "b (t h w) -> b t h w", t=T, h=H, w=W),
            T_total=T,
            future_start=args.future_start,
            H=H,
            W=W,
            ignore_index=args.ignore_index,
        )

        total_loss += float(loss)
        total_acc += float(acc)
        total_frames += 1

        if args.max_batches is not None and (bi + 1) >= args.max_batches:
            break

    mean_loss = total_loss / max(total_frames, 1)
    mean_acc = total_acc / max(total_frames, 1)

    print(f"[Eval v2.0] loss_future3_ce: {mean_loss:.6f}  acc_future3: {mean_acc:.4f}")
    # 필요시 WandB로도 기록하려면 여기서 wandb.init / wandb.log 호출 추가하면 됨.


if __name__ == "__main__":
    main()
