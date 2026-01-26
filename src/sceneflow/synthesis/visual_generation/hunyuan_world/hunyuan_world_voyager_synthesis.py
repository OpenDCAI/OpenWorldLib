from typing import Union, Optional
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import os
import time
import random
import functools
import json
from typing import List, Optional, Tuple, Union

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from .hunyuan_world_voyager.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE, NEGATIVE_PROMPT_I2V
from ....base_models.diffusion_model.video.hunyuan_video.vae import load_vae
from .hunyuan_world_voyager.modules import load_model
from .hunyuan_world_voyager.text_encoder import TextEncoder
from .hunyuan_world_voyager.utils.lora_utils import load_lora_for_pipeline
from .hunyuan_world_voyager.utils.geometry import get_plucker_coordinates
from .hunyuan_world_voyager.utils.train_utils import load_state_dict
from ....base_models.diffusion_model.video.hunyuan_video.modules.posemb_layers import get_nd_rotary_pos_embed
from ....base_models.diffusion_model.video.hunyuan_video.modules.fp8_optimization import convert_fp8_linear
from ....base_models.diffusion_model.video.hunyuan_video.diffusion.schedulers import FlowMatchDiscreteScheduler
from .hunyuan_world_voyager.diffusion.pipelines import HunyuanVideoPipeline
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from safetensors.torch import load_file
import cv2
import pyexr
import torchvision.transforms as T

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
        initialize_model_parallel,
        init_distributed_environment
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None
    initialize_model_parallel = None
    init_distributed_environment = None


def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=pos.device)
                  [: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(
            freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def load_models(args, device, logger, pretrained_model_path):
    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    if args.i2v_mode and args.i2v_condition_type == "latent_concat":
        in_channels = args.latent_channels * 3 + 2
        image_embed_interleave = 2
    elif args.i2v_mode and args.i2v_condition_type == "token_replace":
        in_channels = args.latent_channels
        image_embed_interleave = 4
    else:
        in_channels = args.latent_channels
        image_embed_interleave = 1
    out_channels = args.latent_channels

    model = load_model(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        factor_kwargs=factor_kwargs,
    )
    
    model = model.to(device)
    model = load_state_dict(args, model, logger, f"{pretrained_model_path}/Voyager/transformers/mp_rank_00_model_states.pt")
    model.eval()

    # VAE
    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        args.vae_precision,
        logger=logger,
        vae_path=f"{pretrained_model_path}/hunyuan-video-i2v-720p/vae",
        device=device if not args.use_cpu_offload else "cpu",
    )
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
    vae = vae.to(device)

    text_encoder_paths = {
        "clipL": f"{pretrained_model_path}/text_encoder_2",
        "llm": f"{pretrained_model_path}/text_encoder",
        "llm-i2v": f"{pretrained_model_path}/text_encoder_i2v",
    }

    if args.i2v_mode:
        args.text_encoder = "llm-i2v"
        args.tokenizer = "llm-i2v"
        args.prompt_template = "dit-llm-encode-i2v"
        args.prompt_template_video = "dit-llm-encode-video-i2v"

    if args.prompt_template_video is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
            "crop_start", 0)
    elif args.prompt_template is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template].get(
            "crop_start", 0)
    else:
        crop_start = 0
    max_length = args.text_len + crop_start

    prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None
    prompt_template_video = PROMPT_TEMPLATE[
        args.prompt_template_video] if args.prompt_template_video is not None else None

    # Text encoder
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder,
        max_length=max_length,
        text_encoder_path=text_encoder_paths[args.text_encoder],
        text_encoder_precision=args.text_encoder_precision,
        tokenizer_type=args.tokenizer,
        tokenizer_path=text_encoder_paths[args.tokenizer],
        i2v_mode=args.i2v_mode,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=args.hidden_state_skip_layer,
        apply_final_norm=args.apply_final_norm,
        reproduce=args.reproduce,
        logger=logger,
        device=device if not args.use_cpu_offload else "cpu",
        image_embed_interleave=image_embed_interleave
    ).to(device)

    text_encoder_2 = None
    if args.text_encoder_2 is not None:
        text_encoder_2 = TextEncoder(
            text_encoder_type=args.text_encoder_2,
            max_length=args.text_len_2,
            text_encoder_path=text_encoder_paths[args.text_encoder_2],
            text_encoder_precision=args.text_encoder_precision_2,
            tokenizer_type=args.tokenizer_2,
            tokenizer_path=text_encoder_paths[args.tokenizer_2],
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        ).to(device)

    return model, vae, text_encoder, text_encoder_2, vae_kwargs


# need to set register strategy
class HunyuanWorldVoyagerSynthesis(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload
        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        if args.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
            if args.use_lora:
                self.pipeline = load_lora_for_pipeline(
                    self.pipeline, args.lora_path, LORA_PREFIX_TRANSFORMER="Hunyuan_video_I2V_lora",
                    alpha=args.lora_scale, device=self.device,
                    is_parallel=(self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1))
                logger.info(
                f"load lora {args.lora_path} into pipeline, lora scale is {args.lora_scale}.")
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT

        if self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1:
            parallelize_transformer(self.pipeline)
    
    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
    ):
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, \
                text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is None.
        """
        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        # ========================================================================
        # Initialize Distributed Environment
        # ========================================================================
        # pftq: Modified to extract rank and world_size early for sequential loading
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            assert xfuser is not None, "Ulysses Attention and Ring Attention requires xfuser package."
            assert args.use_cpu_offload is False, "Cannot enable use_cpu_offload in the distributed environment."
            # pftq: Set local rank and device explicitly for NCCL
            local_rank = int(os.environ['LOCAL_RANK'])
            device = torch.device(f"cuda:{local_rank}")
            # pftq: Set CUDA device explicitly
            torch.cuda.set_device(local_rank)
            # pftq: Removed device_id, rely on set_device
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == args.ring_degree * args.ulysses_degree, \
                "number of GPUs should be equal to ring_degree * ulysses_degree."
            init_distributed_environment(rank=rank, world_size=world_size)
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=args.ring_degree,
                ulysses_degree=args.ulysses_degree,
            )
        else:
            rank = 0  # pftq: Default rank for single GPU
            world_size = 1  # pftq: Default world_size for single GPU
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        parallel_args = {"ulysses_degree": args.ulysses_degree,
                         "ring_degree": args.ring_degree}
        torch.set_grad_enabled(False)

        # ========================================================================
        # Build main model, VAE, and text encoder sequentially on rank 0
        # ========================================================================
        # pftq: Load models only on rank 0, then broadcast
        if rank == 0:
            logger.info("Building model...")
            model, vae, text_encoder, text_encoder_2, vae_kwargs = \
                load_models(args, device, logger, pretrained_model_path)
        else:
            # pftq: Initialize as None on non-zero ranks
            model = None
            vae = None
            vae_kwargs = None
            text_encoder = None
            text_encoder_2 = None

        # pftq: Broadcast models to all ranks
        if world_size > 1:
            logger.info(f"Rank {rank}: Starting broadcast synchronization")
            dist.barrier()  # Ensure rank 0 finishes loading before broadcasting
            if rank != 0:
                # Reconstruct model skeleton on non-zero ranks
                model, vae, text_encoder, text_encoder_2, vae_kwargs = \
                    load_models(args, device, logger, pretrained_model_path)

            # Broadcast model parameters with logging
            logger.info(f"Rank {rank}: Broadcasting model parameters")
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            model.eval()
            logger.info(f"Rank {rank}: Broadcasting VAE parameters")
            for param in vae.parameters():
                dist.broadcast(param.data, src=0)
            # pftq: Use broadcast_object_list for vae_kwargs
            logger.info(f"Rank {rank}: Broadcasting vae_kwargs")
            vae_kwargs_list = [vae_kwargs] if rank == 0 else [None]
            dist.broadcast_object_list(vae_kwargs_list, src=0)
            vae_kwargs = vae_kwargs_list[0]
            logger.info(f"Rank {rank}: Broadcasting text_encoder parameters")
            for param in text_encoder.parameters():
                dist.broadcast(param.data, src=0)
            if text_encoder_2 is not None:
                logger.info(
                    f"Rank {rank}: Broadcasting text_encoder_2 parameters")
                for param in text_encoder_2.parameters():
                    dist.broadcast(param.data, src=0)

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )
    
    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W

        # Compute latent sizes based on VAE type
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 +
                            1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 +
                            1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        # Compute rope sizes
        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(
                s % self.model.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size[idx]
                          for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)
                                ) + rope_sizes  # Pad time axis

        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly

        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list or [
            head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(
            rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"

        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")

            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(
                size, device=self.device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]

            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")

            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")

        return freqs_cos, freqs_sin

    def process(self, pil_img):
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = image[:, :, :3]
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5], inplace=True)(image)
        return image
    
    def create_hunyuan_video_input(self, 
                                   render_list, 
                                   mask_list, 
                                   depth_list,
                                   ref_image=None, 
                                   ref_depth=None,
                                   Width=512, 
                                   Height=512,
                                   min_percentile=2, 
                                   max_percentile=98
                                   ):
        """
        直接返回处理好的数据,不保存文件
        """
        # 处理 depth 值
        processed_render_list = []
        processed_mask_list = []
        processed_depth_list = []
        if ref_image is None:
            ref_image = render_list[0].copy()
        if ref_depth is None:
            ref_depth = depth_list[0].copy()
        
        for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):
            # 处理 mask
            mask = mask > 0
            
            # 处理 depth
            depth[mask] = 1 / (depth[mask] + 1e-6)
            depth_values = depth[mask]
            
            min_val = np.percentile(depth_values, min_percentile)
            max_val = np.percentile(depth_values, max_percentile)
            
            depth[mask] = (depth[mask] - min_val) / (max_val - min_val)
            depth[~mask] = depth[mask].min()
            
            # Resize
            render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), 
                            (Width, Height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)
            
            processed_render_list.append(render)
            processed_mask_list.append(mask)
            processed_depth_list.append(depth)
        
        # 处理参考图像的 depth
        if ref_depth is not None:
            mask_ref = mask_list[0] > 0
            ref_depth[mask_ref] = 1 / (ref_depth[mask_ref] + 1e-6)
            depth_values_ref = ref_depth[mask_ref]
            
            min_val_ref = np.percentile(depth_values_ref, min_percentile)
            max_val_ref = np.percentile(depth_values_ref, max_percentile)
            
            ref_depth[mask_ref] = (ref_depth[mask_ref] - min_val_ref) / (max_val_ref - min_val_ref)
            ref_depth[~mask_ref] = ref_depth[mask_ref].min()
            ref_depth = cv2.resize(ref_depth, (Width, Height), interpolation=cv2.INTER_LINEAR)
        
        return {
            'render_list': processed_render_list,
            'mask_list': processed_mask_list,
            'depth_list': processed_depth_list,
            'ref_image': ref_image,
            'ref_depth': ref_depth
        }

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        i2v_mode=False,
        i2v_resolution="720p",
        i2v_image_path=None,
        i2v_condition_type=None,
        i2v_stability=True,
        ulysses_degree=1,
        ring_degree=1,
        ref_image=None,
        ref_depth=None,
        render_list=None,
        depth_list=None,
        mask_list=None,
        **kwargs,
    ):
        out_dict = dict()

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(
            self.device).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, \
                    got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = height  # align_to(height, 16)
        target_height = target_height * 2 + 16
        target_width = width  # align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if guidance_scale == 1.0:
            negative_prompt = ""
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # prepare reference images
        closest_size = (height, width)   # prepare the suitable size for reference image and partial cond/mask
        ref_rgb = self.process(Image.fromarray(ref_image).resize((closest_size[1], closest_size[0])))
        ref_depth = torch.from_numpy(cv2.resize(ref_depth, (closest_size[1], closest_size[0]), 
                                                interpolation=cv2.INTER_LINEAR)).float()
        ref_depth = ref_depth.unsqueeze(0).repeat(3, 1, 1)
        ref_depth = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(ref_depth)

        # 拼接 RGB 和 depth
        ref_images_pixel_values = torch.cat([ref_rgb, torch.ones_like(ref_rgb)[..., :16, :], ref_depth], dim=1)
        ref_images_pixel_values = ref_images_pixel_values.unsqueeze(0).unsqueeze(2).to(self.device)

        # 转换回 PIL Image 用于可视化
        ref_images = [Image.fromarray(((torch.clamp(ref_images_pixel_values[0, :3, 0].permute(
            1, 2, 0), min=-1, max=1).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))]

        # prepare partial_cond (49 frames)
        partial_cond_list = []
        for j in range(49):
            # deal with RGB image
            rgb = self.process(Image.fromarray(render_list[j]).resize((closest_size[1], closest_size[0])))
            
            # deal with depth image
            depth = torch.from_numpy(cv2.resize(depth_list[j], (closest_size[1], closest_size[0]), 
                                                interpolation=cv2.INTER_LINEAR)).float()
            depth = depth.unsqueeze(0).repeat(3, 1, 1)
            depth = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(depth)
            
            # concatation
            combined = torch.cat([rgb, torch.ones_like(rgb)[..., :16, :], depth], dim=1)
            partial_cond_list.append(combined)

        partial_cond = torch.stack(partial_cond_list, dim=1).unsqueeze(0).to(self.device)

        # prepare partial_mask (49 frames)
        partial_mask_list = []
        for j in range(49):
            # mask 同时用于 RGB 和 depth
            mask_img = Image.fromarray(mask_list[j]).resize((closest_size[1], closest_size[0]))
            mask = self.process(mask_img)
            
            # 拼接两份mask (对应RGB和depth部分)
            combined_mask = torch.cat([mask, torch.ones_like(mask)[..., :16, :], mask], dim=1)
            partial_mask_list.append(combined_mask)

        partial_mask = torch.stack(partial_mask_list, dim=1).unsqueeze(0).to(self.device)

        # Use automatic mixed precision for memory efficiency during encoding
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            # Enable VAE tiling for processing large images/videos efficiently
            self.pipeline.vae.enable_tiling()

            # Encode reference images to latent space representation
            # This creates a compressed representation that guides the video generation
            ref_latents = self.pipeline.vae.encode(
                ref_images_pixel_values).latent_dist.sample()  # B, C, F, H, W
            ref_latents.mul_(self.pipeline.vae.config.scaling_factor)

            # Encode partial condition images to latent space
            # These latents provide temporal guidance for the video sequence
            partial_cond = self.pipeline.vae.encode(
                partial_cond).latent_dist.sample()
            partial_cond.mul_(self.pipeline.vae.config.scaling_factor)

            # Process mask frames for controlling video generation
            # Invert the mask so that 1 indicates regions to be generated
            mask_frames = 1 - partial_mask
            first_mask = mask_frames[:, :, 0:1]  # Extract the first mask frame
            
            # Prepend 3 copies of the first mask to create temporal consistency
            # This ensures the initial frames have consistent masking
            mask_frames = torch.cat(
                [first_mask, first_mask, first_mask, mask_frames], dim=2)
            
            # Apply 3D max pooling to downsample masks to match latent space dimensions
            # Reduces temporal dimension by 4, spatial dimensions by 8
            mask_frames = torch.nn.functional.max_pool3d(
                mask_frames,  # Input: [1, C, F, H, W]
                kernel_size=(4, 8, 8),  # Reduce F by 4, H and W by 8
                stride=(4, 8, 8)
            )  # Output: [C, F//4, H//8, W//8]
            
            # Invert the mask again so that 1 indicates regions to preserve
            mask_frames = 1 - mask_frames
            partial_mask = mask_frames[:, 0:1]

        # Generate rotary position embeddings for the target video dimensions
        # These embeddings provide positional information to the transformer model
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        
        # Generate rotary position embeddings for conditional frames
        # Adjusted dimensions account for the conditional frame structure
        freqs_cos_cond, freqs_sin_cond = self.get_rotary_pos_embed(
            target_video_length, (target_height - 16) // 2, target_width
        )
        
        # Calculate the total number of tokens for the transformer model
        # This determines the sequence length for attention mechanisms
        n_tokens = freqs_cos.shape[0]

        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            freqs_cis_cond=(freqs_cos_cond, freqs_sin_cond),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=ref_latents,
            semantic_images=ref_images,
            partial_cond=partial_cond,
            partial_mask=partial_mask
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict


def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        # Text embedding for modulation.
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        # Guidance for modulation, should be cfg_scale x 1000.
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ):
        if x.shape[-2] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(
            f"Cannot split video sequence into ulysses_degree x ring_degree \
            ({get_sequence_parallel_world_size()}) parts evenly")

        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2
        h_cond = h // 2

        x = torch.chunk(x, get_sequence_parallel_world_size(), dim=split_dim)[
            get_sequence_parallel_rank()]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)

        dim_thw_cond = freqs_cos_cond.shape[-1]
        freqs_cos_cond = freqs_cos_cond.reshape(temporal_size, h_cond, w, dim_thw_cond)
        freqs_cos_cond = torch.chunk(freqs_cos_cond, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos_cond = freqs_cos_cond.reshape(-1, dim_thw_cond)
        dim_thw_cond = freqs_sin_cond.shape[-1]
        freqs_sin_cond = freqs_sin_cond.reshape(temporal_size, h_cond, w, dim_thw_cond)
        freqs_sin_cond = torch.chunk(freqs_sin_cond, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin_cond = freqs_sin_cond.reshape(-1, dim_thw_cond)

        from xfuser.core.long_ctx_attention import xFuserLongContextAttention

        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            x,
            t,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            freqs_cos_cond,
            freqs_sin_cond,
            guidance,
            return_dict,
        )

        return_dict = not isinstance(output, tuple)
        sample = output["x"]
        sample = get_sp_group().all_gather(sample, dim=split_dim)
        output["x"] = sample
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
