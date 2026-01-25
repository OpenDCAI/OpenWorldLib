import torch
import torch.nn as nn
import os
from .models.wan_video_dit_moe import WanModelMoe
from .configs.model_config import model_loader_configs
from .models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE
# =========================================================================
# 来自 infer_demo.py 的模型动态修改工具函数 (原封不动)
# =========================================================================

def replace_dit_model_in_manager():
    """Replace DiT model class with MoE version"""    


    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)
                    print(f"Replaced model class: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)


def add_framepack_components(dit_model):
    """Add FramePack related components"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                if scale == "1x":
                    x = x.to(self.proj.weight.dtype)
                    return self.proj(x)
                elif scale == "2x":
                    x = x.to(self.proj_2x.weight.dtype)
                    return self.proj_2x(x)
                elif scale == "4x":
                    x = x.to(self.proj_4x.weight.dtype)
                    return self.proj_4x(x)
                else:
                    raise ValueError(f"Unsupported scale: {scale}")
        
        dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("Added FramePack clean_x_embedder component")


def add_moe_components(dit_model, moe_config):
    """Add MoE related components - corrected version"""
    if not hasattr(dit_model, 'moe_config'):
        dit_model.moe_config = moe_config
        print("Added MoE config to model")
    dit_model.top_k = moe_config.get("top_k", 1)

    # Dynamically add MoE components for each block
    dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
    unified_dim = moe_config.get("unified_dim", 25)
    num_experts = moe_config.get("num_experts", 4)

    dit_model.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
    dit_model.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
    dit_model.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX uses 13-dim input, similar to sekai but handled independently
    dit_model.global_router = nn.Linear(unified_dim, num_experts)


    for i, block in enumerate(dit_model.blocks):
        # MoE network - input unified_dim, output dim
        block.moe = MultiModalMoE(
            unified_dim=unified_dim,
            output_dim=dim,  # Output dimension matches transformer block dim
            num_experts=moe_config.get("num_experts", 4),
            top_k=moe_config.get("top_k", 2)
        )
        
        print(f"Block {i} added MoE component (unified_dim: {unified_dim}, experts: {moe_config.get('num_experts', 4)})")