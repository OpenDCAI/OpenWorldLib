from __future__ import annotations

import torch

from ...base_synthesis import BaseSynthesis
from ....synthesis.vla_generation.giga_brain_0.giga_brain_0_policy import GigaBrain0Policy


class GigaBrain0Synthesis(BaseSynthesis):
    """Lightweight synthesis wrapper around GigaBrain0Policy."""

    def __init__(self, policy: GigaBrain0Policy, device: str | torch.device = 'cpu'):
        super().__init__()
        self.device = device
        self.policy = policy.to(device)
        self.policy.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str | torch.device | None = None, **kwargs) -> "GigaBrain0Synthesis":
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        policy = GigaBrain0Policy.from_pretrained(pretrained_model_path, **kwargs)
        return cls(policy=policy, device=device)

    def to(self, device: str | torch.device):
        self.device = device
        self.policy.to(device)
        return self

    @torch.no_grad()
    def predict(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        emb_ids: torch.Tensor,
        enable_2d_traj_output: bool = False,
    ):
        """Forward to policy.sample_actions with provided embeddings/tokens."""
        return self.policy.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            emb_ids=emb_ids,
            enable_2d_traj_output=enable_2d_traj_output,
        )
