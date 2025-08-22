# ===== FILE: models/prefix_adapter.py =====
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatePrefixAdapter(nn.Module):
    """
    Convert summarized robot state features into prefix tokens for conditioning.

    Args:
        d_s: dimension of state summary feature (e.g. 138)
        d_model: transformer embedding dimension
        num_prefix: number of prefix tokens to generate
        id_dim: dimension of slot-ID embedding
        cond_drop_p: probability of condition dropout during training
        future_start: prefix tokens are only active for t >= future_start
    """

    def __init__(
        self,
        d_s: int,
        d_model: int,
        num_prefix: int = 8,
        id_dim: int = 32,
        cond_drop_p: float = 0.1,
        future_start: int = 3,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_model = d_model
        self.num_prefix = num_prefix
        self.future_start = future_start
        self.cond_drop_p = cond_drop_p

        # project state feature to model dim
        self.proj = nn.Linear(d_s, d_model)

        # slot embeddings for num_prefix tokens
        self.slot_ids = nn.Parameter(torch.randn(num_prefix, id_dim))
        self.slot_proj = nn.Linear(id_dim, d_model)

        # gating
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, s_feat: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            s_feat: [B, d_s]
            T: total sequence length (e.g. 6)
        Returns:
            prefix tokens: [B, T, num_prefix, d_model]
        """
        B = s_feat.size(0)

        # --- cond-drop ---
        if self.training and self.cond_drop_p > 0.0:
            mask = (torch.rand(B, 1, device=s_feat.device) > self.cond_drop_p).float()
            s_feat = s_feat * mask

        # project state
        h = self.proj(s_feat)  # [B, d_model]

        # expand to prefix slots
        slot_emb = self.slot_proj(self.slot_ids)  # [num_prefix, d_model]
        slot_emb = slot_emb.unsqueeze(0).expand(B, -1, -1)  # [B, num_prefix, d_model]

        # broadcast h to num_prefix
        h = h.unsqueeze(1).expand(-1, self.num_prefix, -1)  # [B, num_prefix, d_model]

        # combine
        prefix = h + slot_emb  # [B, num_prefix, d_model]

        # gate
        g = torch.sigmoid(self.gate(prefix))
        prefix = prefix * g

        # repeat over time steps
        prefix = prefix.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, num_prefix, d_model]

        # zero out before future_start
        if self.future_start > 0:
            mask = torch.arange(T, device=s_feat.device).unsqueeze(0) >= self.future_start
            mask = mask.unsqueeze(-1).unsqueeze(-1).float()  # [1, T, 1, 1]
            prefix = prefix * mask

        return prefix
