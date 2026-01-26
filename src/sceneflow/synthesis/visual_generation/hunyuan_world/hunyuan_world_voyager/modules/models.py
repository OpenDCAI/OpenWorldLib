from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import torch.nn as nn
from einops import rearrange

from diffusers.configuration_utils import register_to_config

# 从基础模块导入原始类
from ......base_models.diffusion_model.video.hunyuan_video.modules.models import (
    HYVideoDiffusionTransformer as BaseHYVideoDiffusionTransformer,
    MMDoubleStreamBlock as BaseMMDoubleStreamBlock,
    MMSingleStreamBlock as BaseMMSingleStreamBlock,
)
from ......base_models.diffusion_model.video.hunyuan_video.modules.activation_layers import get_activation_layer
from ......base_models.diffusion_model.video.hunyuan_video.modules.embed_layers import PatchEmbed
from ......base_models.diffusion_model.video.hunyuan_video.modules.attenion import get_cu_seqlens
from ......base_models.diffusion_model.video.hunyuan_video.modules.posemb_layers import apply_rotary_emb
from ......base_models.diffusion_model.video.hunyuan_video.modules.modulate_layers import modulate, apply_gate, ckpt_wrapper


class MMDoubleStreamBlock(BaseMMDoubleStreamBlock):
    """
    Extended MMDoubleStreamBlock with token replacement conditioning support.
    
    Inherits from BaseMMDoubleStreamBlock and adds token_replace condition handling
    for image-to-video generation tasks.
    """

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If no token_replace condition, use parent's forward
        if condition_type != "token_replace":
            return super().forward(
                img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, freqs_cis
            )
        
        # Token replace specific logic
        img_mod1, token_replace_img_mod1 = self.img_mod(
            vec, condition_type=condition_type, token_replace_vec=token_replace_vec
        )
        (img_mod1_shift, img_mod1_scale, img_mod1_gate,
         img_mod2_shift, img_mod2_scale, img_mod2_gate) = img_mod1.chunk(6, dim=-1)
        (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate,
         tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)

        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention with token replace modulation
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale,
            condition_type=condition_type,
            tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
            frist_frame_token_num=frist_frame_token_num
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run attention
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        from ......base_models.diffusion_model.video.hunyuan_video.modules.attenion import attention, parallel_attention
        
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn, q, k, v,
                img_q_len=img_q.shape[1], img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
            )

        img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

        # Calculate img blocks with token replace
        img = img + apply_gate(
            self.img_attn_proj(img_attn), gate=img_mod1_gate,
            condition_type=condition_type, tr_gate=tr_img_mod1_gate,
            frist_frame_token_num=frist_frame_token_num
        )
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale,
                    condition_type=condition_type, tr_shift=tr_img_mod2_shift,
                    tr_scale=tr_img_mod2_scale, frist_frame_token_num=frist_frame_token_num
                )
            ),
            gate=img_mod2_gate, condition_type=condition_type,
            tr_gate=tr_img_mod2_gate, frist_frame_token_num=frist_frame_token_num
        )

        # Calculate txt blocks (unchanged)
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(BaseMMSingleStreamBlock):
    """
    Extended MMSingleStreamBlock with token replacement conditioning support.
    """

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        # If no token_replace condition, use parent's forward
        if condition_type != "token_replace":
            return super().forward(
                x, vec, txt_len, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, freqs_cis
            )

        # Token replace specific logic
        mod, tr_mod = self.modulation(
            vec, condition_type=condition_type, token_replace_vec=token_replace_vec
        )
        mod_shift, mod_scale, mod_gate = mod.chunk(3, dim=-1)
        tr_mod_shift, tr_mod_scale, tr_mod_gate = tr_mod.chunk(3, dim=-1)

        x_mod = modulate(
            self.pre_norm(x), shift=mod_shift, scale=mod_scale,
            condition_type=condition_type,
            tr_shift=tr_mod_shift, tr_scale=tr_mod_scale,
            frist_frame_token_num=frist_frame_token_num
        )
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        from ......base_models.diffusion_model.video.hunyuan_video.modules.attenion import attention, parallel_attention

        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn, q, k, v,
                img_q_len=img_q.shape[1], img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
            )

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(
            output, gate=mod_gate, condition_type=condition_type,
            tr_gate=tr_mod_gate, frist_frame_token_num=frist_frame_token_num
        )


class HYVideoDiffusionTransformer(BaseHYVideoDiffusionTransformer):
    """
    Extended HunyuanVideo Transformer with Image-to-Video conditioning support.
    
    This class extends the base transformer with:
    - Token replacement conditioning for I2V generation
    - Context block for additional conditioning
    - Gradient checkpointing support
    
    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        # Initialize parent class
        super().__init__(
            args=args,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            heads_num=heads_num,
            mlp_width_ratio=mlp_width_ratio,
            mlp_act_type=mlp_act_type,
            mm_double_blocks_depth=mm_double_blocks_depth,
            mm_single_blocks_depth=mm_single_blocks_depth,
            rope_dim_list=rope_dim_list,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            guidance_embed=guidance_embed,
            text_projection=text_projection,
            use_attention_mask=use_attention_mask,
            dtype=dtype,
            device=device,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        
        # I2V specific attributes
        self.i2v_condition_type = getattr(args, 'i2v_condition_type', None)
        self.gradient_checkpoint = getattr(args, 'gradient_checkpoint', False)
        self.gradient_checkpoint_layers = getattr(args, 'gradient_checkpoint_layers', 0)
        
        if self.gradient_checkpoint:
            total_depth = mm_double_blocks_depth + mm_single_blocks_depth
            assert self.gradient_checkpoint_layers <= total_depth, \
                f"Gradient checkpoint layers must be <= depth. " \
                f"Got {self.gradient_checkpoint_layers} and depth={total_depth}."

        # Replace blocks with token replace versions
        self._replace_blocks_with_extended_versions(
            mm_double_blocks_depth, mm_single_blocks_depth,
            mlp_width_ratio, mlp_act_type, qk_norm, qk_norm_type, qkv_bias,
            factory_kwargs
        )

        # Context block for additional conditioning
        self.use_context_block = getattr(args, 'use_context_block', False)
        if self.use_context_block:
            self._init_context_blocks(
                mlp_width_ratio, mlp_act_type, qk_norm, qk_norm_type, qkv_bias,
                factory_kwargs
            )

    def _replace_blocks_with_extended_versions(
        self, mm_double_blocks_depth, mm_single_blocks_depth,
        mlp_width_ratio, mlp_act_type, qk_norm, qk_norm_type, qkv_bias,
        factory_kwargs
    ):
        """Replace base blocks with extended versions supporting token replacement."""
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size, self.heads_num,
                mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type,
                qk_norm=qk_norm, qk_norm_type=qk_norm_type, qkv_bias=qkv_bias,
                **factory_kwargs,
            )
            for _ in range(mm_double_blocks_depth)
        ])

        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size, self.heads_num,
                mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type,
                qk_norm=qk_norm, qk_norm_type=qk_norm_type,
                **factory_kwargs,
            )
            for _ in range(mm_single_blocks_depth)
        ])

    def _init_context_blocks(
        self, mlp_width_ratio, mlp_act_type, qk_norm, qk_norm_type, qkv_bias,
        factory_kwargs
    ):
        """Initialize context blocks for additional conditioning."""
        self.condition_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        self.context_block1 = MMDoubleStreamBlock(
            self.hidden_size, self.heads_num,
            mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type,
            qk_norm=qk_norm, qk_norm_type=qk_norm_type, qkv_bias=qkv_bias,
            **factory_kwargs,
        )

        self.context_block2 = MMSingleStreamBlock(
            self.hidden_size, self.heads_num,
            mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type,
            qk_norm=qk_norm, qk_norm_type=qk_norm_type,
            **factory_kwargs,
        )

        self.zero_linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.zero_linear2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors
        vec = self.time_in(t)

        # Token replace conditioning
        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        # Text modulation
        vec_2 = self.vector_in(text_states_2)
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

        # Guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        # Context block processing
        condition1, condition2 = None, None
        if self.use_context_block:
            condition1, condition2 = self._process_context_blocks(
                img, txt, vec, text_mask, freqs_cos_cond, freqs_sin_cond,
                token_replace_vec, frist_frame_token_num
            )

        # Embed image and text
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_seqlens for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Pass through double blocks
        for layer_num, block in enumerate(self.double_blocks):
            double_block_args = [
                img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, freqs_cis,
                self.i2v_condition_type, token_replace_vec, frist_frame_token_num,
            ]

            if self._should_checkpoint(layer_num):
                img, txt = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(block), *double_block_args, use_reentrant=False
                )
            else:
                img, txt = block(*double_block_args)

            if condition1 is not None:
                img = img + condition1

        # Merge and pass through single blocks
        x = torch.cat((img, txt), 1)

        for layer_num, block in enumerate(self.single_blocks):
            single_block_args = [
                x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin),
                self.i2v_condition_type, token_replace_vec, frist_frame_token_num,
            ]

            global_layer_num = layer_num + len(self.double_blocks)
            if self._should_checkpoint(global_layer_num):
                x = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(block), *single_block_args, use_reentrant=False
                )
            else:
                x = block(*single_block_args)

            if condition2 is not None:
                x = x + condition2

        img = x[:, :img_seq_len, ...]

        # Final layer
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)

        if return_dict:
            out["x"] = img
            return out
        return img

    def _should_checkpoint(self, layer_num: int) -> bool:
        """Determine if gradient checkpointing should be applied for this layer."""
        if not self.training or not self.gradient_checkpoint:
            return False
        return self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers

    def _process_context_blocks(
        self, img, txt, vec, text_mask, freqs_cos_cond, freqs_sin_cond,
        token_replace_vec, frist_frame_token_num
    ):
        """Process context blocks for additional conditioning."""
        condition = img.clone()
        height = (condition.shape[-2] - 2) // 2
        condition = condition[..., -height:, :]
        condition = self.condition_in(condition)

        # Temporarily embed txt for context blocks
        if self.text_projection == "single_refiner":
            txt_for_ctx = self.txt_in(
                txt, torch.zeros_like(vec[:, 0]),
                text_mask if self.use_attention_mask else None
            )
        else:
            txt_for_ctx = self.txt_in(txt)

        cond_seq_len = condition.shape[1]
        txt_seq_len = txt_for_ctx.shape[1]
        cu_seqlens_q_cond = get_cu_seqlens(text_mask, cond_seq_len)
        max_seqlen_q_cond = cond_seq_len + txt_seq_len

        # Context block 1
        condition1, txt1 = self.context_block1(
            condition, txt_for_ctx, vec,
            cu_seqlens_q_cond, cu_seqlens_q_cond,
            max_seqlen_q_cond, max_seqlen_q_cond,
            (freqs_cos_cond, freqs_sin_cond),
            self.i2v_condition_type, token_replace_vec, frist_frame_token_num,
        )

        # Context block 2
        condition2 = torch.cat((condition1, txt1), 1)
        condition2 = self.context_block2(
            condition2, vec, txt_seq_len,
            cu_seqlens_q_cond, cu_seqlens_q_cond,
            max_seqlen_q_cond, max_seqlen_q_cond,
            (freqs_cos_cond, freqs_sin_cond),
            self.i2v_condition_type, token_replace_vec, frist_frame_token_num,
        )

        # Apply zero linear and pad
        condition1 = self.zero_linear1(condition1)
        condition2 = self.zero_linear2(condition2)

        # Get img_seq_len from img_in output size
        img_embedded = self.img_in(img)
        img_seq_len = img_embedded.shape[1]

        condition1 = torch.cat(
            (torch.zeros(condition1.shape[0], img_seq_len - condition1.shape[1], 
                        condition1.shape[2], device=condition1.device, dtype=condition1.dtype),
             condition1), dim=1
        )
        condition2 = torch.cat(
            (torch.zeros(condition2.shape[0], img_seq_len - condition2.shape[1] + txt_seq_len,
                        condition2.shape[2], device=condition2.device, dtype=condition2.dtype),
             condition2), dim=1
        )

        return condition1, condition2

    def set_input_tensor(self, input_tensor):
        """Compatibility method for distributed training."""
        pass


# Config remains the same
HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-S/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
}
