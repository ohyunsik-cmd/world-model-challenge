# ===== FILE: genie/st_mask_git.py =====
import math
from typing import Optional, Tuple

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm
from transformers.utils import ModelOutput

from genie.factorization_utils import (
    FactorizedEmbedding,
    factorize_labels,
    unfactorize_token_ids,
)
from genie.config import GenieConfig
from genie.st_transformer import STTransformerDecoder

# 새로 추가된 상태 프리픽스 조건부
try:
    # 프로젝트 구조에 맞춰 경로를 조정하세요.
    from models.prefix_adapter import StatePrefixAdapter
    from utils.state_feats import fast_state_summary
except Exception:
    # 경로가 다르면 로컬 패키지 기준으로 재시도 (예: genie.models / genie.utils)
    from genie.models.prefix_adapter import StatePrefixAdapter  # type: ignore
    from genie.utils.state_feats import fast_state_summary       # type: ignore


def cosine_schedule(u):
    """ u in [0, 1] """
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")
    return cls.cos(u * cls.pi / 2)


class STMaskGIT(nn.Module, PyTorchModelHubMixin):
    """
    Spatial-Temporal MaskGIT decoder with optional robot-state prefix conditioning.

    - 입력 토큰: [B,T,H,W] (unfactorized)
    - 출력 로짓: [B, C=V*num_vocabs, T, H, W]
    - (옵션) 상태 프리픽스: 각 시간 t별로 N개의 prefix 토큰을 S축 앞에 주입 후, 디코더 출력에서 prefix 채널 제거
    """

    def __init__(self, config: GenieConfig):
        super().__init__()
        self.config = config

        # 공간 해상도
        self.h = self.w = math.isqrt(config.S)
        assert self.h * self.w == config.S, "Expected S to be a perfect square"

        # 디코더
        self.decoder = STTransformerDecoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            qk_norm=config.qk_norm,
            use_mup=config.use_mup,
            attn_drop=config.attn_drop,
            mlp_ratio=config.mlp_ratio,
            mlp_bias=config.mlp_bias,
            mlp_drop=config.mlp_drop,
        )

        # 위치 임베딩 (시간/공간)
        self.pos_embed_TSC = nn.Parameter(torch.zeros(1, config.T, config.S, config.d_model))

        # 마스크 토큰 id (이미지 vocab의 크기와 동일하게 사용)
        self.mask_token_id = config.image_vocab_size

        # 팩터 임베딩(팩터 수가 1이면 일반 임베딩과 동일)
        self.token_embed = FactorizedEmbedding(
            factored_vocab_size=config.factored_vocab_size,
            num_factored_vocabs=config.num_factored_vocabs,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )

        # 출력 projection: d_model → (factored_vocab_size * num_factored_vocabs)
        Readout = FixedMuReadout if config.use_mup else nn.Linear
        self.out_x_proj = Readout(config.d_model, config.factored_vocab_size * config.num_factored_vocabs)

        # -------- 상태 프리픽스 어댑터 (옵션) --------
        self.use_prefix = bool(getattr(config, "use_prefix_condition", False))
        if self.use_prefix:
            d_s = getattr(config, "d_s", 138)  # 기본 138 (sin/cos 126 + gripper 6 + vel 6)
            num_prefix = getattr(config, "num_prefix", 8)
            cond_drop_p = getattr(config, "cond_drop_p", 0.1)
            future_start = getattr(config, "future_start", 3)

            self.prefix_adapter = StatePrefixAdapter(
                d_s=d_s,
                d_model=config.d_model,
                num_prefix=num_prefix,
                id_dim=32,
                cond_drop_p=cond_drop_p,
                future_start=future_start,
            )
            self.num_prefix = num_prefix
            self.future_start = future_start
        else:
            self.prefix_adapter = None
            self.num_prefix = 0
            self.future_start = 0

    # ----------------- Generation (기존 유지) -----------------
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        max_new_tokens: int,
        min_new_tokens: Optional[int] = None,
        return_logits: bool = False,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
    ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Returns: `(sample_THW_flat, factored_logits)` if `return_logits` else `sample_THW_flat`
          - sample_THW_flat: (B, (T_prompt+T_gen)*H*W)
          - factored_logits: (B, V, F, T_gen, H, W)
        """
        assert min_new_tokens in (None, max_new_tokens), \
            "Expecting `min_new_tokens`, if specified, to equal `max_new_tokens`."
        assert max_new_tokens % self.config.S == 0, "max_new_tokens must be a multiple of S"

        num_new_frames = max_new_tokens // self.config.S

        inputs_THW = rearrange(input_ids.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w)
        inputs_masked_THW = torch.cat(
            [
                inputs_THW,
                torch.full((input_ids.size(0), num_new_frames, self.h, self.w),
                           self.mask_token_id, dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )

        all_factored_logits = []
        for t in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            sample_HW, factored_logits = self.maskgit_generate(
                inputs_masked_THW,
                out_t=t,
                maskgit_steps=maskgit_steps,
                temperature=temperature,
            )
            inputs_masked_THW[:, t] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W -> B (T H W)")
        if return_logits:
            return predicted_tokens, torch.stack(all_factored_logits, dim=3)
        else:
            return predicted_tokens, None

    @staticmethod
    def init_mask(prompt_THW: torch.LongTensor) -> torch.BoolTensor:
        # 한 프레임(H*W)만 마스크 스케줄링
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(prompt_THW.size(0), H * W, dtype=torch.bool, device=prompt_THW.device)
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.LongTensor,  # [B,T,H,W]
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
        states_future: Optional[torch.Tensor] = None,  # (옵션) prefix용
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        MaskGIT-style inference to predict frame `out_t`.
        Returns: (sample_HW, factored_logits_CHW) where logits: [B, V, F, H, W]
        """
        assert out_t > 0, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), \
            f"When generating z{out_t}, frames >= out_t must be fully masked."

        bs, t, h, w = prompt_THW.shape
        unmasked = self.init_mask(prompt_THW)

        # (중요) 상태 프리픽스는 미래 타임만 활성화 → inference에서는 필요하면 states_future 전달
        logits_CTHW = self.compute_logits(prompt_THW, states_future=states_future)
        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = logits_CHW.clone()

        for step in tqdm(range(maskgit_steps)):
            if step > 0:
                logits_CHW = self.compute_logits(prompt_THW, states_future=states_future)[:, :, out_t]

            # (C=V*F) → (B, V, F, H, W)
            factored_logits = rearrange(
                logits_CTHW,
                "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
            )
            factored_probs = F.softmax(factored_logits, dim=1)

            samples_HW = torch.zeros((bs, h, w), dtype=torch.long, device=prompt_THW.device)
            confidences_HW = torch.ones((bs, h, w), dtype=torch.float, device=prompt_THW.device)

            # 팩터 독립 샘플링 (뒤 팩터부터 곱셈)
            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:
                    sample = probs.argmax(dim=1)
                else:
                    dist = torch.distributions.categorical.Categorical(
                        probs=rearrange(probs, "b v ... -> b ... v") / temperature
                    )
                    sample = dist.sample()
                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")
            samples_flat = samples_HW.view(bs, self.config.S)

            if step != maskgit_steps - 1:
                # 코사인 스케줄로 다음 step에서 재가릴 토큰 수 n
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * self.config.S)
                if unmask_mode == "greedy":
                    confidences_flat = confidences_HW.view(bs, self.config.S)
                elif unmask_mode == "random":
                    confidences_flat = torch.rand_like(confidences_HW).view(bs, self.config.S)
                else:
                    raise NotImplementedError

                confidences_flat[unmasked] = torch.inf
                least_conf = torch.argsort(confidences_flat, dim=1)
                # 가장 자신있는 토큰 (S - n)개는 유지 → 나머지 n개는 다시 mask
                unmasked.scatter_(1, least_conf[:, n:], True)
                samples_flat.scatter_(1, least_conf[:, :n], self.mask_token_id)

            # 이전에 확정된(unmasked) 토큰은 유지
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.view(-1, h, w)
            prompt_THW[:, out_t] = samples_HW

        # 원본 step-0 로짓 반환
        return samples_HW, rearrange(
            orig_logits_CHW,
            "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
            vocab_size=self.config.factored_vocab_size,
            num_vocabs=self.config.num_factored_vocabs,
        )

    # ----------------- Loss / Logits -----------------
    def compute_loss_and_acc(self, logits_CTHW, targets_THW, relevant_mask_THW):
        """
        logits_CTHW: [B, C=V*F, T, H, W]
        targets_THW: [B, T, H, W]
        relevant_mask_THW: [B, T-1, H, W] (미래 프레임만)
        """
        # 첫 프레임은 항상 주어짐 → t>=1만 학습
        logits_CTHW, targets_THW = logits_CTHW[:, :, 1:], targets_THW[:, 1:]

        factored_logits = rearrange(
            logits_CTHW,
            "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
            vocab_size=self.config.factored_vocab_size,
            num_vocabs=self.config.num_factored_vocabs,
        )
        factored_targets = factorize_labels(
            targets_THW,
            num_factored_vocabs=self.config.num_factored_vocabs,
            factored_vocab_size=self.config.factored_vocab_size,
        )

        if not getattr(self, "_shape_logged", False):
            print("debug logits:", factored_logits.shape, "targets:", factored_targets.shape)
            self._shape_logged = True

        # CE: 팩터별 합 → (B,T,H,W)
        loss_THW = F.cross_entropy(factored_logits, factored_targets, reduction="none").sum(dim=1)
        acc_THW = (factored_logits.argmax(dim=1) == factored_targets).all(dim=1)

        # 마스크된 위치(미래)만 평균
        num_masked = torch.sum(relevant_mask_THW)
        relevant_loss = torch.sum(loss_THW * relevant_mask_THW) / num_masked
        relevant_acc = torch.sum(acc_THW * relevant_mask_THW).float() / num_masked
        return relevant_loss, relevant_acc

    def compute_logits(self, x_THW: torch.LongTensor, states_future: Optional[torch.Tensor] = None):
        """
        x_THW: [B,T,H,W] (unfactorized token ids; t>=1 미래 프레임은 일부/전부 mask id일 수 있음)
        states_future: [B,17,25] or [17,25] (옵션) — 프리픽스 on일 때만 사용
        """
        B, T, H, W = x_THW.shape
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")        # [B,T,S]
        x_TSC = self.token_embed(x_TS)                         # [B,T,S,C]

        # ----- 상태 프리픽스 -----
        if self.use_prefix and (states_future is not None):
            # states_future: [B,17,25] (데이터로더에서 배치 단위 제공)
            if states_future.dim() == 2:
                states_future = states_future.unsqueeze(0)  # [1,17,25] → [B,17,25] 브로드캐스트 대비
            s_feat = fast_state_summary(states_future)      # [B, d_s]
            # P: [B,T,N,C]  (미래 t>=future_start만 활성)
            P_TNC = self.prefix_adapter(s_feat, T=T)
            # [B,T,N,C] → [B,T,N,C] (그대로) / 비디오 토큰과 concat하려면 S축 앞에 붙임
            # x_TSC: [B,T,S,C] → concat N개 prefix → [B,T,S+N,C]
            x_TSC = torch.cat([P_TNC, x_TSC], dim=2)
            S_all = x_TSC.size(2)
        else:
            S_all = x_TSC.size(2)

        # 위치 임베딩 더하기 (prefix가 있으면 pos_embed를 S축 앞부분에 브로드캐스트)
        if S_all != self.config.S:
            # pos_embed_TSC: [1,T,S,C] → prefix 길이만큼 제로 pos를 앞에 붙여 정렬
            pad_prefix = torch.zeros((1, T, S_all - self.config.S, self.config.d_model),
                                     dtype=self.pos_embed_TSC.dtype, device=x_TSC.device)
            pos = torch.cat([pad_prefix, self.pos_embed_TSC.to(x_TSC.device)], dim=2)
        else:
            pos = self.pos_embed_TSC.to(x_TSC.device)

        # 디코더 통과
        h_TSC = self.decoder(x_TSC + pos)            # [B,T,S(+N),C]

        # 프리픽스 채널 제거 후 projection
        if self.use_prefix and (S_all != self.config.S):
            h_TSC = h_TSC[:, :, self.num_prefix:, :]  # [B,T,S,C]

        x_next_TSC = self.out_x_proj(h_TSC)           # [B,T,S,C_out]
        logits_CTHW = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=self.h, W=self.w)
        return logits_CTHW

    # ----------------- Forward -----------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        states_future: Optional[torch.Tensor] = None,
    ):
        T, H, W = self.config.T, self.h, self.w
        x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)

        logits_CTHW = self.compute_logits(x_THW, states_future=states_future)

        labels_THW = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # 미래 3프레임만 CE 집계: 입력에서 mask 토큰인 위치만 사용
        relevant_mask_THW = (x_THW[:, 1:] == self.mask_token_id)  # [B,T-1,H,W]
        loss, acc = self.compute_loss_and_acc(logits_CTHW, labels_THW, relevant_mask_THW)

        # (선택) 예측 토큰을 unfactorize해 모니터링할 때는 F축을 마지막으로 보낸다.
        # preds_idx: [B,F,T,H,W] → [B,T,H,W,F] → unfactorize
        with torch.no_grad():
            factored_logits = rearrange(
                logits_CTHW[:, :, 1:],
                "b (f v) t h w -> b v f t h w",
                f=self.config.num_factored_vocabs,
                v=self.config.factored_vocab_size,
            )
            preds_idx = factored_logits.argmax(dim=1)                # [B,F,T-1,H,W]
            preds_idx_last = preds_idx.permute(0, 2, 3, 4, 1).contiguous()  # [B,T-1,H,W,F]
            _ = unfactorize_token_ids(
                preds_idx_last,
                num_factored_vocabs=self.config.num_factored_vocabs,
                factored_vocab_size=self.config.factored_vocab_size,
            )

        return ModelOutput(loss=loss, acc=acc, logits=logits_CTHW)

    # ----------------- Init / muP -----------------
    def init_weights(self):
        """ Works with and without muP. """
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256
        base_model = STMaskGIT(base_config)
        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        if model.config.use_mup:
            model.set_mup_shapes(rescale_params=False)
        return model


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        # torch.compile에서 width_mult 중복 나눗셈 이슈 회피
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())
