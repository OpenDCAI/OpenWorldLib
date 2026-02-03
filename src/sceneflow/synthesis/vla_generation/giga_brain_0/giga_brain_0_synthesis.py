import torch

from ...base_synthesis import BaseSynthesis
from .giga_brain_0.modeling_giga_brain_0 import GigaBrain0Policy



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

    @property
    def vision_in_channels(self) -> int:
        return self.policy.vision_in_channels

    @property
    def max_action_dim(self) -> int:
        return self.policy.max_action_dim

    @property
    def n_action_steps(self) -> int:
        return self.policy.n_action_steps

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

    @torch.no_grad()
    def init_lang_generation(self, images, img_masks, lang_tokens, lang_masks):
        return self.policy.init_lang_generation(images, img_masks, lang_tokens, lang_masks)

    @torch.no_grad()
    def next_lang_logits(self, state: dict, input_token: torch.Tensor):
        return self.policy.next_lang_logits(state, input_token)

    @property
    def inner_policy(self) -> GigaBrain0Policy:
        return self.policy
