# ===== FILE: eval_utils.py =====
from __future__ import annotations

from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F
from einops import rearrange

# factorized vocab 라벨 변환 유틸 (프로젝트 내 모듈)
from factorization_utils import factorize_labels, unfactorize_token_ids


"""
v2.0 평가 유틸 요약
- T_total = 6 (past3 + future3), future_start = 3
- vocab factorization: num_factored_vocabs = 3, factored_vocab_size = 40  => 총 채널 120
- 모델 출력 로짓: [B, V=40, F=3, T_future=3, H, W]
- 라벨: [B, T_total*H*W] 또는 [B, T_total, H, W]
- Loss/Accuracy: '미래 3프레임'만 집계 (과거는 무시/IGNORE)
"""


# ---------------------------
# 형태 표준화 헬퍼
# ---------------------------
def _ensure_labels_THW(labels: torch.LongTensor, *, T_total: int, H: int, W: int) -> torch.LongTensor:
    """
    labels: [B, T_total*H*W] or [B, T_total, H, W]
    return: [B, T_total, H, W]
    """
    if labels.dim() == 2:
        B = labels.size(0)
        return labels.view(B, T_total, H, W)
    if labels.dim() == 4:
        assert labels.size(1) == T_total and labels.size(2) == H and labels.size(3) == W, \
            f"labels shape mismatch: got {tuple(labels.shape)}, expect (B,{T_total},{H},{W})"
        return labels
    raise ValueError(f"Unsupported labels shape: {tuple(labels.shape)}")


def _ensure_samples_THW(samples: torch.LongTensor, *, T_total: int, H: int, W: int) -> torch.LongTensor:
    """
    samples: [B, T_total*H*W] or [B, T_total, H, W]
    return: [B, T_total, H, W]
    """
    if samples.dim() == 2:
        B = samples.size(0)
        return samples.view(B, T_total, H, W)
    if samples.dim() == 4:
        assert samples.size(1) == T_total and samples.size(2) == H and samples.size(3) == W, \
            f"samples shape mismatch: got {tuple(samples.shape)}, expect (B,{T_total},{H},{W})"
        return samples
    raise ValueError(f"Unsupported samples shape: {tuple(samples.shape)}")


# ---------------------------
# Loss / Accuracy (미래 3프레임만)
# ---------------------------
@torch.no_grad()
def compute_acc_future_only_v2(
    samples: torch.LongTensor,
    labels: torch.LongTensor,
    *,
    T_total: int = 6,
    future_start: int = 3,
    H: int = 32,
    W: int = 32,
    ignore_index: int = -100,
) -> float:
    """
    v2.0용 토큰 정확도: 미래 구간만 평균 (argmax된 샘플과 GT 일치율)

    Args:
        samples: [B, T_total*H*W] or [B, T_total, H, W]
        labels : [B, T_total*H*W] or [B, T_total, H, W] (과거 3프레임은 IGNORE가 들어있을 수 있음)
    """
    samples_THW = _ensure_samples_THW(samples, T_total=T_total, H=H, W=W)
    labels_THW = _ensure_labels_THW(labels,  T_total=T_total, H=H, W=W)

    preds = samples_THW[:, future_start:]  # [B, 3, H, W]
    gts   = labels_THW[:, future_start:]   # [B, 3, H, W]

    valid = (gts != ignore_index).float()
    correct = ((preds == gts).float() * valid).sum()
    total = valid.sum().clamp_min(1.0)
    return (correct / total).item()


def compute_loss_future_only_v2(
    labels: torch.LongTensor,                 # [B, T_total*H*W] or [B, T_total, H, W]
    factored_logits: torch.FloatTensor,       # [B, V, F, T_future, H, W]
    *,
    T_total: int = 6,
    future_start: int = 3,
    num_factored_vocabs: int = 3,             # F
    factored_vocab_size: int = 40,            # V
    H: Optional[int] = None,
    W: Optional[int] = None,
    ignore_index: int = -100,
    reduction: str = "mean",                  # "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    v2.0(Cosmos 8×8×8)용 CE:
    - 미래 구간(t >= future_start)만 집계
    - factored_logits: [B, V=40, F=3, T_future=3, H, W]
    """
    assert factored_logits.dim() == 6, f"factored_logits must be [B,V,F,T,H,W], got {tuple(factored_logits.shape)}"
    B, Vdim, Fdim, T_future, Hlog, Wlog = factored_logits.shape
    if H is None or W is None:
        H, W = Hlog, Wlog
    assert Vdim == factored_vocab_size, f"V={Vdim} != {factored_vocab_size}"
    assert Fdim == num_factored_vocabs, f"F={Fdim} != {num_factored_vocabs}"
    assert T_future == (T_total - future_start), f"T_future={T_future} != {T_total - future_start}"
    assert (Hlog, Wlog) == (H, W), f"H/W mismatch: logits {(Hlog,Wlog)} vs args {(H,W)}"

    # 라벨 복원 및 미래만 선택
    if labels.dim() == 2:
        B2 = labels.size(0); assert B2 == B
        labels_THW = labels.view(B, T_total, H, W).to(factored_logits.device)
    elif labels.dim() == 4:
        labels_THW = labels.to(factored_logits.device)
        assert labels_THW.shape == (B, T_total, H, W)
    else:
        raise ValueError(f"Unsupported labels shape: {tuple(labels.shape)}")
    labels_future = labels_THW[:, future_start:]  # [B, T_future, H, W]

    # (옵션) 위치 무시 마스크
    valid_mask = (labels_future != ignore_index)  # [B, T_future, H, W]
    num_valid = valid_mask.sum()

    # 타깃 팩터화 → [B, F, T_future, H, W] (구현에 따라 마지막축 F일 수 있어 보정)
    factored_labels = factorize_labels(
        labels_future,
        num_factored_vocabs=num_factored_vocabs,
        factored_vocab_size=factored_vocab_size,
    )
    if factored_labels.shape[1] != Fdim and factored_labels.shape[-1] == Fdim:
        factored_labels = factored_labels.permute(0, 4, 1, 2, 3).contiguous()

    assert factored_labels.shape[:3] == (B, Fdim, T_future), \
        f"targets must be [B,F,T,H,W], got {tuple(factored_labels.shape)}"

    # CE: 클래스축(V)에서 계산 → [B,F,T,H,W]; 이후 F 합산 → [B,T,H,W]
    ce_FTHW = F.cross_entropy(factored_logits, factored_labels, reduction="none", ignore_index=ignore_index)
    loss_THW = ce_FTHW.sum(dim=1)

    if num_valid == 0:
        return loss_THW.sum() * 0.0  # 모두 무시면 0
    if reduction == "mean":
        return (loss_THW * valid_mask).sum() / num_valid
    if reduction == "sum":
        return (loss_THW * valid_mask).sum()
    if reduction == "none":
        return loss_THW * valid_mask  # [B,T,H,W]
    raise ValueError(f"Invalid reduction: {reduction}")


# ---------------------------
# LPIPS 등 픽셀/프레임 지표 헬퍼
# ---------------------------
@torch.no_grad()
def compute_lpips(
    frames_a: torch.ByteTensor,
    frames_b: torch.ByteTensor,
    lpips_func: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
) -> List[float]:
    """
    LPIPS 측정 헬퍼 (외부 LPIPS 네트워크 주입형).
    - 입력 텐서는 [B, T, C, H, W] 또는 [B*T, C, H, W] 형식을 허용한다.
    - uint8 [0..255] → float [0..1]로 내부 정규화.

    Returns:
        per-sample LPIPS list (len == B*T)
    """
    assert frames_a.dtype == torch.uint8 and frames_b.dtype == torch.uint8, "expect uint8 inputs"
    if frames_a.dim() == 5:
        B, T, C, H, W = frames_a.shape
        flat_a = frames_a.view(B * T, C, H, W)
        flat_b = frames_b.view(B * T, C, H, W)
    elif frames_a.dim() == 4:
        flat_a = frames_a
        flat_b = frames_b
    else:
        raise ValueError(f"Unsupported frame shape: {tuple(frames_a.shape)}")

    a = flat_a.float() / 255.0
    b = flat_b.float() / 255.0
    vals = lpips_func(a, b).flatten().tolist()
    return vals


# ---------------------------
# (선택) 디코드/시각화용 보조
# ---------------------------
@torch.no_grad()
def tokens_to_frames_placeholder(
    tokens_THW: torch.LongTensor,
    decode_fn: Callable[[torch.LongTensor], torch.ByteTensor],
) -> torch.ByteTensor:
    """
    토큰 → 프레임 디코더가 주입될 때 사용할 보조 함수.
    - tokens_THW: [B, T, H, W] (단일 토큰 id 공간)
    - decode_fn : callable([B*T, H, W]) → [B*T, 3, H_rgb, W_rgb] (uint8)
    """
    if tokens_THW.dim() != 4:
        raise ValueError("tokens_THW must be [B, T, H, W]")
    B, T, H, W = tokens_THW.shape
    flat = tokens_THW.view(B * T, H, W)
    frames = decode_fn(flat)  # [B*T, 3, H_rgb, W_rgb], uint8
    return frames.view(B, T, *frames.shape[1:])
