# ===== FILE: config.py =====
import json
from dataclasses import dataclass
from typing import Optional

from genie.factorization_utils import nth_root


@dataclass
class GenieConfig:
    # --- Model core ---
    num_layers: int
    num_heads: int
    d_model: int

    # --- Sequence shape (v2.0) ---
    # T = 6 (past3 + future3), S = 32*32 (latent 32×32)
    T: int = 6
    S: int = 32 * 32

    # --- Vocabulary (Cosmos DV 8×8×8) ---
    #  image_vocab_size = 64_000 = 40^3
    image_vocab_size: int = 64_000
    num_factored_vocabs: int = 3
    factored_vocab_size: Optional[int] = 40  # 40 × 3 vocabs → output channels 120

    # Special/mask tokens (MaskGIT-style); by 규약 mask_token_id == image_vocab_size
    mask_token_id: Optional[int] = None

    # muP / Attention / MLP
    use_mup: bool = False
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    qk_norm: bool = True
    mlp_ratio: float = 4.0
    mlp_drop: float = 0.0
    mlp_bias: bool = True

    # --- (Legacy collator용 하이퍼; v2.0 데이터 경로에서는 보통 미사용이지만 남겨둠) ---
    # 비활성 또는 최소 사용이 권장됨.
    max_corrupt_rate: float = 0.0       # v2.0에서는 데이터셋에서 이미 마스크 구성
    non_mlm_ratio: float = 0.0          # v2.0에서는 teacher-forced time만 사용
    num_prompt_frames: int = 3          # past3

    # --- Prefix conditioning (v2.0 기본 On) ---
    use_prefix_condition: bool = True
    num_prefix: int = 8
    cond_drop_p: float = 0.1
    future_start: int = 3               # t >= 3 (미래 구간)에만 prefix 활성

    # ---------- I/O ----------
    def save_pretrained(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(vars(self), f)

    @classmethod
    def from_pretrained(cls, json_path: str):
        with open(json_path, "r") as f:
            cfg = json.load(f)
        return cls(**cfg)

    def shallow_copy(self):
        return GenieConfig(**vars(self))

    # ---------- Post init checks / derived ----------
    def __post_init__(self):
        # factored_vocab_size 유도/검증
        if self.factored_vocab_size is None:
            # image_vocab_size = (factored_vocab_size)^(num_factored_vocabs)
            self.factored_vocab_size = nth_root(self.image_vocab_size, self.num_factored_vocabs)
        else:
            # 주어진 조합이 일관적인지 확인 (Cosmos: 64_000 == 40^3)
            assert self.image_vocab_size == self.factored_vocab_size ** self.num_factored_vocabs, \
                f"image_vocab_size({self.image_vocab_size}) != " \
                f"{self.factored_vocab_size}^{self.num_factored_vocabs}"

        # v2.0 규약: 출력 채널 = factored_vocab_size * num_factored_vocabs = 120
        assert self.factored_vocab_size * self.num_factored_vocabs == 120, \
            f"Expected 120 output channels, got {self.factored_vocab_size * self.num_factored_vocabs}"

        # Mask token id = image_vocab_size
        if self.mask_token_id is None:
            self.mask_token_id = self.image_vocab_size
