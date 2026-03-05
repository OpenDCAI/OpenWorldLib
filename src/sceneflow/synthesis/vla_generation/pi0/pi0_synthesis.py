import torch

from ...base_synthesis import BaseSynthesis
from .pi0.modeling_pi0 import PI0Policy


class PI0Synthesis(BaseSynthesis):
    """Lightweight synthesis wrapper around PI0Policy."""

    def __init__(self, policy: PI0Policy, device: str | torch.device = 'cpu'):
        super().__init__()
        self.device = device
        self.policy = policy.to(device)
        self.policy.eval()

    @property
    def max_action_dim(self) -> int:
        """Return the maximum action dimension from the policy."""
        return self.policy.max_action_dim

    @property
    def max_state_dim(self) -> int:
        """Return the maximum state dimension from the policy."""
        return self.policy.max_state_dim

    @property
    def n_action_steps(self) -> int:
        """Return the number of action steps from the policy."""
        return self.policy.n_action_steps

    @property
    def pi05_enabled(self) -> bool:
        """Return whether pi05 mode is enabled."""
        return self.policy.pi05_enabled

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str | torch.device | None = None, **kwargs) -> "PI0Synthesis":
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        policy = PI0Policy.from_pretrained(pretrained_model_path, **kwargs)
        return cls(policy=policy, device=device)

    def to(self, device: str | torch.device):
        self.device = device
        self.policy.to(device)
        return self

    def compile(self, **kwargs):
        """Compile sample_actions for speed."""
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)
        return self

    def quantize(self) -> None:
        """Apply dynamic float8 quantization to the Paligemma blocks only."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        layers = self.policy.paligemma_with_expert.layers
        for i in range(len(layers)):
            quantize_(layers[i].mlps[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.q_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.k_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.v_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.o_proj[0], Float8DynamicActivationFloat8WeightConfig())







    @torch.no_grad()
    def predict(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ):
        """Forward to policy.sample_actions with provided embeddings/tokens."""
        return self.policy.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )



