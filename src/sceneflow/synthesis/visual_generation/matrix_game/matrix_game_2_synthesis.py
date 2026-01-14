import os
import torch
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from huggingface_hub import snapshot_download, hf_hub_download
from ...base_synthesis import BaseSynthesis
from .matrix_game_2.pipeline import CausalInferencePipeline
from .matrix_game_2.extension_modules.wanx_vae import get_wanx_vae_wrapper
from .matrix_game_2.demo_utils.vae_block3 import VAEDecoderWrapper
from .matrix_game_2.utils.visualize import process_video
from .matrix_game_2.utils.misc import set_seed
from .matrix_game_2.utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file


class MatrixGame2Synthesis(BaseSynthesis):
    def __init__(self,
                 pipeline,
                 vae,
                 weight_dtype = torch.bfloat16,
                 mode="universal",
                 device="cuda"):
        """
        the mode including "gta_drive", "templerun", "universal"
        """
        super(MatrixGame2Synthesis, self).__init__()
        self.pipeline = pipeline
        self.vae = vae
        self.weight_dtype = weight_dtype
        self.device = device
        self.mode = mode

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        mode="universal",
                        device=None,
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if mode not in ['universal', 'gta_drive', 'templerun']:
            raise NotImplementedError("mode should be one of ['universal', 'gta_drive', 'templerun']")
        if mode == 'universal':
            config_path = os.path.join(script_dir, f"./matrix_game_2/configs/inference_yaml/inference_universal.yaml")
        elif mode == 'gta_drive':
            config_path = os.path.join(script_dir, f"./matrix_game_2/configs/inference_yaml/inference_gta_drive.yaml")
        elif mode == 'templerun':
            config_path = os.path.join(script_dir, f"./matrix_game_2/configs/inference_yaml/inference_templerun.yaml")
        
        config = OmegaConf.load(config_path)
        config["model_kwargs"]['model_config'] = os.path.join(os.path.join(script_dir, "./matrix_game_2/"), 
                                                              config["model_kwargs"]['model_config'])

        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            # download from HuggingFace repo_id
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")

        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(model_root, "Wan2.1_VAE.pth"), map_location="cpu")
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

        checkpoint_path = os.path.join(model_root, "base_distilled_model/base_distill.safetensors")
        if checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        pipeline = pipeline.to(device=device, dtype=weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(model_root, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        return cls(pipeline=pipeline, vae=vae, mode=mode, device=device)

    @torch.no_grad()
    def predict(self,
                cond_concat,
                visual_context,
                operator_condition,
                num_output_frames,
                operation_visualization=True,
                ):
        sampled_noise = torch.randn(
            [1, 16, num_output_frames, cond_concat.size(-2), cond_concat.size(-1)], device=self.device, dtype=self.weight_dtype
        )

        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        if 'mouse_condition' in operator_condition:
            mouse_condition = operator_condition['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        if 'keyboard_condition' not in operator_condition:
            raise ValueError("keyboard_condition must be provided in operator_condition")
        keyboard_condition = operator_condition['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition

        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=self.mode,
                profile=False,
            )
        
        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)

        # mouse_icon = 'file path to mouse.png'
        mouse_icon = None
        if self.mode != 'templerun':
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy()
            )
        else:
            config = (
                keyboard_condition[0].float().cpu().numpy()
            )
        output_video = process_video(video.astype(np.uint8),
                                    config, mouse_icon, mouse_scale=0.1,
                                    process_icon=operation_visualization,
                                    mode=self.mode)
        return output_video
