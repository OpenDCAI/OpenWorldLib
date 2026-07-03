from typing import Optional

import torch
import torch.nn.functional as F

from .sp_utils.communications import all_gather
from .sp_utils.parallel_states import get_parallel_state

# Cache for compiled VAE encode function
_compiled_vae_encode_cache = {}

def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



def keyframe_vae_encode(vae, images, rescale=True, use_compile=False, compile_mode="default"): # supports multi-GPU parallelism
    """
    Encode images to latents using VAE.

    Args:
        vae: VAE model
        images: Input images tensor (b, c, f, h, w)
        rescale: Whether to rescale latents
        use_compile: Whether to use torch.compile for acceleration
        compile_mode: torch.compile mode, options: "default", "reduce-overhead", "max-autotune"

    Returns:
        frame_latents: Encoded latents tensor (b, c, f, h, w)
    """
    global _compiled_vae_encode_cache
    vae_config = vae.config
    # images: (b, c, f, h, w)
    assert images.dim() == 5
    frame_latents = []
    input_f = images.shape[2]

    parallel_dims = get_parallel_state()
    sp_size = parallel_dims.world_size
    if sp_size > 1:
        if input_f % sp_size != 0:
            images = F.pad(images, (0, 0, 0, 0, 0, sp_size - input_f % sp_size), mode="constant", value=0)
        images = images.chunk(sp_size, dim=2)[parallel_dims.sp_rank]

    b, c, f, h, w = images.shape  # value: -1~1
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1).to(images.device, images.dtype)
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1).to(images.device, images.dtype)

    # Get encode function (compiled or original)
    if use_compile:
        cache_key = (id(vae), compile_mode)
        if cache_key not in _compiled_vae_encode_cache:
            _compiled_vae_encode_cache[cache_key] = torch.compile(vae.encode, mode=compile_mode)
        encode_fn = _compiled_vae_encode_cache[cache_key]
    else:
        encode_fn = vae.encode

    for i in range(f):
        # Mark step begin for CUDA Graphs to prevent tensor overwriting
        if use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        frame_latent = retrieve_latents(encode_fn(images[:, :, i:i + 1]), sample_mode="argmax")  # (b, c, 1, h, w)
        # Clone the tensor to prevent CUDA Graphs from overwriting it in subsequent runs
        if use_compile:
            frame_latent = frame_latent.clone()
        frame_latents.append(frame_latent)
    frame_latents = torch.cat(frame_latents, dim=2)  # (b, c, f, h, w)

    if rescale:
        frame_latents = (frame_latents - latents_mean) * latents_std

    if sp_size > 1:
        frame_latents = all_gather(frame_latents, dim=2, group=parallel_dims.sp_group)
        frame_latents = frame_latents[:, :, :input_f]

    return frame_latents


def keyframe_vae_decode(vae, frame_latents, rescale=True):
    vae_config = vae.config
    # frame_latents: (b, c, f, h, w)
    assert frame_latents.dim() == 5
    frames = []
    input_f = frame_latents.shape[2]

    parallel_dims = get_parallel_state()
    sp_size = parallel_dims.world_size
    if sp_size > 1:
        if input_f % sp_size != 0:
            frame_latents = F.pad(frame_latents, (0, 0, 0, 0, 0, sp_size - input_f % sp_size), mode="constant", value=0)
        frame_latents = frame_latents.chunk(sp_size, dim=2)[parallel_dims.sp_rank]

    b, c, f, h, w = frame_latents.shape
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1).to(frame_latents.device, frame_latents.dtype)
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1).to(frame_latents.device, frame_latents.dtype)

    if rescale:
        frame_latents = frame_latents / latents_std + latents_mean

    for i in range(f):
        decoded_frame = vae.decode(frame_latents[:, :, i:i + 1], return_dict=False)[0]  # (b, c, 1, h, w)
        frames.append(decoded_frame)
    frames = torch.cat(frames, dim=2)  # (b, c, f, h, w)
    frames = torch.clip(frames, -1, 1)

    if sp_size > 1:
        frames = all_gather(frames, dim=2, group=parallel_dims.sp_group)
        frames = frames[:, :, :input_f]

    return frames
