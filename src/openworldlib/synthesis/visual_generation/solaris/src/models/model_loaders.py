from flax import nnx

from src.models.clip import CLIPModel as JaxCLIP
from src.models.wan_vae import WanVAE_


def get_jax_clip_model():
    config = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        vision_pre_norm=True,
        vision_post_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
    )
    jax_model = JaxCLIP(**config)

    return jax_model


def get_vae_model():
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )

    vae_jax = WanVAE_(
        rngs=nnx.Rngs(0),
        dim=cfg["dim"],
        z_dim=cfg["z_dim"],
        dim_mult=cfg["dim_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_scales=cfg["attn_scales"],
        temperal_downsample=cfg["temperal_downsample"],
        dropout=cfg["dropout"],
    )
    return vae_jax
