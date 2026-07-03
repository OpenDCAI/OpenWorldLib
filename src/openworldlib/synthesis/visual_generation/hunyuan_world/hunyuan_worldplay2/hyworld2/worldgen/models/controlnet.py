import torch
import torch.nn as nn
from diffusers.models import ModelMixin

from .attention import SingleAttentionBlock, SingleAttentionBlockRef, WanAttnProcessorSparseSpatialSP

try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log
from diffusers.utils.torch_utils import maybe_allow_in_graph
from typing import Tuple, Optional
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention import FeedForward, AttentionModuleMixin
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class WanXControlNet(ModelMixin):
    def __init__(self, controlnet_cfg):
        super().__init__()

        self.controlnet_cfg = controlnet_cfg
        if controlnet_cfg.conv_out_dim != controlnet_cfg.dim:
            self.proj_in = nn.Linear(controlnet_cfg.conv_out_dim, controlnet_cfg.dim)
        else:
            self.proj_in = nn.Identity()

        self.controlnet_blocks = nn.ModuleList(
            [
                SingleAttentionBlock(
                    dim=controlnet_cfg.dim,
                    ffn_dim=controlnet_cfg.ffn_dim,
                    num_heads=controlnet_cfg.num_heads,
                    time_embed_dim=controlnet_cfg.time_embed_dim,
                    qk_norm="rms_norm_across_heads",
                )
                for _ in range(controlnet_cfg.num_layers)
            ]
        )
        out_ch = 5120 if "5B" not in self.controlnet_cfg.get("base_model", "") else 3072
        self.proj_out = nn.ModuleList(
            [
                zero_module(nn.Linear(controlnet_cfg.dim, out_ch))
                for _ in range(controlnet_cfg.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb, rotary_emb, **kwargs):
        hidden_states = self.proj_in(hidden_states)
        controlnet_states = []
        for i, block in enumerate(self.controlnet_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    rotary_emb,
                    kwargs["extrinsics"],
                    kwargs["intrinsics"],
                    kwargs["patches_x"],
                    kwargs["patches_y"],
                    kwargs["image_width"],
                    kwargs["image_height"],
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    rotary_emb=rotary_emb,
                    extrinsics=kwargs["extrinsics"],
                    intrinsics=kwargs["intrinsics"],
                    patches_x=kwargs["patches_x"],
                    patches_y=kwargs["patches_y"],
                    image_width=kwargs["image_width"],
                    image_height=kwargs["image_height"],
                )

            controlnet_states.append(self.proj_out[i](hidden_states))

        return controlnet_states


class WanXControlNetRef(ModelMixin):
    def __init__(self, controlnet_cfg):
        super().__init__()

        self.controlnet_cfg = controlnet_cfg
        rank0_log(f"Ref ControlNet, Update Ref: {controlnet_cfg.update_ref}.")
        if controlnet_cfg.conv_out_dim != controlnet_cfg.dim:
            self.proj_in = nn.Linear(controlnet_cfg.conv_out_dim, controlnet_cfg.dim)
        else:
            self.proj_in = nn.Identity()

        self.controlnet_blocks = nn.ModuleList(
            [
                SingleAttentionBlockRef(
                    dim=controlnet_cfg.dim,
                    ffn_dim=controlnet_cfg.ffn_dim,
                    num_heads=controlnet_cfg.num_heads,
                    time_embed_dim=controlnet_cfg.time_embed_dim,
                    qk_norm="rms_norm_across_heads",
                )
                for _ in range(controlnet_cfg.num_layers)
            ]
        )
        out_ch = 5120 if "5B" not in self.controlnet_cfg.get("base_model", "") else 3072
        self.proj_out = nn.ModuleList(
            [
                zero_module(nn.Linear(controlnet_cfg.dim, out_ch))
                for _ in range(controlnet_cfg.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states, ref_states, temb, rotary_emb_hidden, rotary_emb_ref, f, h, w):
        hidden_states = self.proj_in(hidden_states)
        ref_states_in = self.proj_in(ref_states)
        controlnet_states = []
        controlnet_ref_states = []
        ref_states = ref_states_in.clone()
        for i, block in enumerate(self.controlnet_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, ref_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    ref_states if self.controlnet_cfg.update_ref else ref_states_in,
                    temb,
                    rotary_emb_hidden,
                    rotary_emb_ref,
                    f,
                    h,
                    w
                )
            else:
                hidden_states, ref_states = block(
                    hidden_states=hidden_states,
                    ref_states=ref_states if self.controlnet_cfg.update_ref else ref_states_in,
                    temb=temb,
                    rotary_emb_hidden=rotary_emb_hidden,
                    rotary_emb_ref=rotary_emb_ref,
                    f=f,
                    h=h,
                    w=w
                )
            controlnet_states.append(self.proj_out[i](hidden_states))
            controlnet_ref_states.append(self.proj_out[i](ref_states))

        return controlnet_states, controlnet_ref_states


class WanAttentionSparseSpatial(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessorSparseSpatialSP
    _available_processors = [WanAttnProcessorSparseSpatialSP, WanAttnProcessorSparseSpatialSP]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        if processor is None:
            processor = WanAttnProcessorSparseSpatialSP()
        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_states: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_emb_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        f = None,
        h = None,
        w = None,
        ref_index = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, ref_states, encoder_hidden_states, attention_mask, rotary_emb, rotary_emb_ref, f, h, w, ref_index, **kwargs)



@maybe_allow_in_graph
class WanTransformerSparseSpatialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttentionSparseSpatial(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
        )

        # 2. Cross-attention
        # TODO: added_kv_proj_dim
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        rotary_emb_ref: torch.Tensor,
        f = None,
        h = None,
        w = None,
        ref_index = None,
    ) -> [torch.Tensor, torch.Tensor]:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        norm_ref_states = (self.norm1(ref_states.float()) * (1 + scale_msa) + shift_msa).type_as(ref_states)
        attn_output, ref_attn_output = self.attn1(norm_hidden_states, norm_ref_states, encoder_hidden_states, None, rotary_emb, rotary_emb_ref, f, h, w, ref_index)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        ref_states = (ref_states.float() + ref_attn_output * gate_msa).type_as(ref_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        norm_ref_states = (self.norm3(ref_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(ref_states)
        ff_ref = self.ffn(norm_ref_states)
        ref_states = (ref_states.float() + ff_ref.float() * c_gate_msa).type_as(ref_states)

        return hidden_states, ref_states