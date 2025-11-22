import os
import torch
from omegaconf import OmegaConf
from ...base_synthesis import BaseSynthesis
from .matrix_game_2.pipeline import CausalInferencePipeline, CausalInferenceStreamingPipeline
from .matrix_game_2.wan.vae.wanx_vae import get_wanx_vae_wrapper
from .matrix_game_2.demo_utils.vae_block3 import VAEDecoderWrapper
from .matrix_game_2.utils.visualize import process_video
from .matrix_game_2.utils.misc import set_seed
from .matrix_game_2.utils.conditions import *
from .matrix_game_2.utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file


class MatrixGame2Synthesis(BaseSynthesis):
    def __init__(self, pipeline, vae, mode="universal", device="cuda"):
        """
        the mode including "gta_drive", "templerun", "universal"
        """
        super(MatrixGame2Synthesis, self).__init__()
        self.pipeline = pipeline
        self.vae = vae
        self.device = device
        self.mode = mode

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        mode="universal",
                        device=None,
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        if mode not in ['universal', 'gta_drive', 'templerun']:
            raise NotImplementedError("mode should be one of ['universal', 'gta_drive', 'templerun']")
        if mode == 'universal':
            config_path = f"./configs/inference_yaml/inference_universal.yaml"
        elif mode == 'gta_drive':
            config_path = f"./configs/inference_yaml/inference_gta_drive.yaml"
        elif mode == 'templerun':
            config_path = f"./configs/inference_yaml/inference_templerun.yaml"
        
        config = OmegaConf.load(config_path)

        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=current_vae_decoder)

        checkpoint_path = os.path.join(pretrained_model_path, "base_distilled_model/base_distill.safetensors")
        if checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        pipeline = pipeline.to(device=device, dtype=weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        return cls(pipeline=pipeline, vae=vae, mode=mode, device=device)

    @torch.no_grad()
    def predict(self, input_image, operator_condition):
        pass
