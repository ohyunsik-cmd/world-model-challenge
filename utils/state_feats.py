# ===== FILE: utils/state_feats.py =====
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["fast_state_summary", "get_state_feat_dim"]

# v2.0 계약:
# states_future: [B, 17, 25]
#   - angle channels:   [0..20] (21개, 라디안 가정)
#   - gripper channels: [21..22] (2개)
#   - velocity channels:[23..24] (2개)
#
# 출력 차원 D_s = 138 = (각도 21 × (sin,cos 2) × (mean,last,last-mean 3)) + (그리퍼 2×3) + (속도 2×3)


def get_state_feat_dim() -> int:
    """Return the fixed feature dimension D_s for fast_state_summary."""
    return 138


def _time_stats(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    x: [B, T, C]
    returns (mean:[B,C], last:[B,C], last-mean:[B,C]) in x.dtype on x.device
    """
    # mean over time
    mean = x.mean(dim=1)

    # last time step (t = -1)
    last = x[:, -1]

    # difference
    last_minus_mean = last - mean
    return mean, last, last_minus_mean


@torch.no_grad()
def fast_state_summary(
    states_future: Tensor,
    *,
    include_delta_mean: bool = False,  # 각도 wrap 평균 등 확장용(기본 비활성)
) -> Tensor:
    """
    Build compact state summary features for prefix conditioning.

    Args:
        states_future: Tensor[B, 17, 25], float dtype, radians for angle channels.
        include_delta_mean: (옵션) 확장 플래그. 현재 기본 False로 비활성.

    Returns:
        Tensor[B, 138] on the same device/dtype as input.
    """
    if states_future.ndim != 3 or states_future.size(1) != 17 or states_future.size(2) != 25:
        raise ValueError(f"states_future must be [B,17,25], got {tuple(states_future.shape)}")

    B = states_future.size(0)
    dev = states_future.device
    dt = states_future.dtype

    # split channels
    angles_RT = states_future[:, :, 0:21]    # [B,17,21]
    grip_RT   = states_future[:, :, 21:23]   # [B,17, 2]
    vel_RT    = states_future[:, :, 23:25]   # [B,17, 2]

    # angle -> sin, cos (라디안 가정)
    sin_RT = torch.sin(angles_RT)
    cos_RT = torch.cos(angles_RT)

    # time statistics
    sin_mean, sin_last, sin_lmm = _time_stats(sin_RT)   # 각 [B,21]
    cos_mean, cos_last, cos_lmm = _time_stats(cos_RT)   # 각 [B,21]

    grip_mean, grip_last, grip_lmm = _time_stats(grip_RT)  # 각 [B,2]
    vel_mean,  vel_last,  vel_lmm  = _time_stats(vel_RT)   # 각 [B,2]

    # concat in a fixed, documented order:
    # [sin_mean, sin_last, sin_last-mean, cos_mean, cos_last, cos_last-mean,
    #  grip_mean, grip_last, grip_last-mean,
    #  vel_mean,  vel_last,  vel_last-mean]
    feats = torch.cat(
        [
            sin_mean, sin_last, sin_lmm,
            cos_mean, cos_last, cos_lmm,
            grip_mean, grip_last, grip_lmm,
            vel_mean,  vel_last,  vel_lmm,
        ],
        dim=1,
    )  # [B, 21*6 + 2*3 + 2*3] = [B, 126 + 6 + 6] = [B, 138]

    # numerical safety
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ensure dtype/device match input (they already should)
    feats = feats.to(device=dev, dtype=dt, copy=False)

    if include_delta_mean:
        # 자리만 잡아둔 확장 포인트 (각도 wrap-mean 등 필요한 경우 여기에 추가)
        # 현재 스펙에서는 비활성. 향후 활성화 시 feats = torch.cat([feats, extra], dim=1)
        pass

    # sanity check
    expected = get_state_feat_dim()
    if feats.size(1) != expected:
        raise RuntimeError(f"state feature dim mismatch: got {feats.size(1)}, expect {expected}")

    return feats


if __name__ == "__main__":
    # lightweight self-test
    B = 4
    x = torch.randn(B, 17, 25, dtype=torch.float32)
    # make angles look like radians (-pi..pi)
    x[:, :, 0:21] = (x[:, :, 0:21].tanh()) * torch.pi

    y = fast_state_summary(x)
    assert y.shape == (B, get_state_feat_dim())
    print("fast_state_summary OK:", y.shape)
