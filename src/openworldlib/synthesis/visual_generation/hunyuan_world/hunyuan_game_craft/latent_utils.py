"""Utilities for initializing Hunyuan GameCraft diffusion latents."""

from typing import Optional, Sequence, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


def prepare_initial_latents(
    shape: Sequence[int],
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    generator=None,
    latents: Optional[torch.Tensor] = None,
    init_noise_sigma=None,
) -> torch.Tensor:
    """Return the initial diffusion state for a requested latent shape.

    A caller-provided latent is a pre-noised diffusion state and must be kept
    intact (apart from moving it to the requested device and dtype).  When no
    latent is supplied, exactly one generator-aware noise sample is created.
    ``init_noise_sigma`` is applied after either path to match diffusers'
    scheduler contract.

    ``gt_latents`` are intentionally not accepted here: GameCraft uses those
    latents as a separate image-conditioning stream, rather than as the
    diffusion process' initial state.
    """
    expected_shape = tuple(int(dim) for dim in shape)
    device = torch.device(device)

    if latents is None:
        initial_latents = randn_tensor(
            expected_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                "`latents` must be a torch.Tensor when provided, "
                f"but got {type(latents).__name__}."
            )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                "Unexpected latent shape: "
                f"expected {expected_shape}, got {tuple(latents.shape)}."
            )
        initial_latents = latents.to(device=device, dtype=dtype)

    if init_noise_sigma is not None:
        initial_latents = initial_latents * init_noise_sigma

    return initial_latents
