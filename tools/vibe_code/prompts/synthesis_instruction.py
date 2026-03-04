synthesis_prompt = """
The world model requires the implementation of multimodal generation, such as video and audio generation. Our framework needs to possess multimodal generation capabilities; therefore, a Synthesis class must be defined.

The Synthesis class is invoked within the Pipeline class. It accepts processing results from the Operator or other classes and outputs multimodal generation results.
It should follow the structure below:
```python
class BaseSynthesis(object):
    def __init__(self):
        ## Initialize the model used by the Synthesis class

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        ## Load the model weights required by the Synthesis class
    
    def api_init(self, api_key, endpoint):
        ## If calling an online model, initialize the API key or API URL

    @torch.no_grad()
    def predict(self):
        ## Accept external inputs and output the corresponding multimodal results
```
"""

example_synthesis_code = """
Here are the organized code results for matrix-game-2: https://github.com/SkyworkAI/Matrix-Game".
The Operator implementation is as follows:
```python
from .base_operator import BaseOperator
import torch
from torchvision.transforms import v2
import random

class MatrixGame2Operator(BaseOperator):
    def __init__(self, operation_types=[], mode="universal", interaction_template=[]):
        super().__init__(operation_types=operation_types)
        self.mode = mode
        if mode == 'universal':
            interaction_template = ["forward", "left", "right", "forward_left", "forward_right",
                                    "camera_l", "camera_r"]
        elif mode == 'gta_drive':
            interaction_template = ["forward", "back", "camera_l", "camera_r"]
        elif mode == 'templerun':
            interaction_template = ["jump","slide","leftside","rightside",
                                    "turnleft","turnright","nomove"]
        self.interaction_template = interaction_template
        self.interaction_template_init()
        self.current_interaction = []
        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f\"\{interaction\} not in template\")
        return True

    def get_interaction(self, interaction_list):
        for act in interaction_list:
            self.check_interaction(act)
        self.current_interaction.append(interaction_list)

    def _build_sequence(self, num_frames, frames_per_action=4):
        if len(self.current_interaction) == 0:
            raise RuntimeError("No interaction registered")
        cur_interaction = self.current_interaction[-1]
        total_actions = len(cur_interaction)
        available_frames = num_frames
        frames_per_action = max(frames_per_action, available_frames // total_actions)
        if frames_per_action < 1:
            frames_per_action = 1
        padded_actions = []
        for action in cur_interaction:
            padded_actions.extend([action] * frames_per_action)
        while len(padded_actions) < num_frames:
            padded_actions.append(padded_actions[-1])
        padded_actions = padded_actions[:num_frames]
        keyboard_list = []
        mouse_list = []
        mouse_enabled = (self.mode != "templerun")
        for action in padded_actions:
            kb, ms = encode_actions([action], self.mode)
            keyboard_list.append(kb)
            if mouse_enabled:
                mouse_list.append(ms)
        keyboard_tensor = torch.stack(keyboard_list)
        if mouse_enabled:
            mouse_tensor = torch.stack(mouse_list)
            return {
                "keyboard_condition": keyboard_tensor,
                "mouse_condition": mouse_tensor
            }
        return {"keyboard_condition": keyboard_tensor}

    def process_action_universal(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_gta_drive(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_templerun(self, num_frames):
        return self._build_sequence(num_frames)
    
    def process_interaction(self, num_frames):
        if self.mode == "universal":
            return self.process_action_universal(num_frames)
        elif self.mode == "gta_drive":
            return self.process_action_gta_drive(num_frames)
        elif self.mode == "templerun":
            return self.process_action_templerun(num_frames)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def process_perception(self,
                           input_image,
                           num_output_frames,
                           resize_H=352,
                           resize_W=640,
                           device: str = "cuda",
                           weight_dtype = torch.bfloat16,):
        image = resizecrop(input_image, resize_H, resize_W)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=weight_dtype, device=device)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [resize_H//8, resize_W//8], "tile_stride": [resize_H//16+1, resize_W//16-2]}
        return {
            "image": image,
            "img_cond": img_cond,
            "tiler_kwargs": tiler_kwargs
        }
```

The Pipeline implementation is as follows:
```python
import torch
import numpy as np
import cv2
import os
from PIL import Image
from typing import Optional, Any, List, Union
from torchvision.transforms import v2
from ...operators.matrix_game_2_operator import MatrixGame2Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis
from ...memories.visual_synthesis.matrix_game.matrix_game_2_memory import MatrixGame2Memory

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    last_frame = (tensor * 255).astype(np.uint8)
    pil_image = Image.fromarray(last_frame)
    return pil_image

class MatrixGame2Pipeline:
    def __init__(self,
                 operators: Optional[MatrixGame2Operator] = None,
                 synthesis_model: Optional[MatrixGame2Synthesis] = None,
                 memory_module: Optional[Any] = None,
                 device: str = "cuda",
                 weight_dtype = torch.bfloat16,
                 ):
        self.synthesis_model = synthesis_model 
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.current_image = None

    @classmethod
    def from_pretrained(cls,
                        synthesis_model_path: Optional[str] = None,
                        mode = "universal",
                        weight_dtype = torch.bfloat16,
                        device: str = "cuda",
                        **kwargs) -> "MatrixGame2Pipeline":
        if synthesis_model_path is None:
            synthesis_model_path = "Skywork/Matrix-Game-2.0"
        
        print(f"Loading MatrixGame2 synthesis model from {synthesis_model_path}...")
        synthesis_model = MatrixGame2Synthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            mode=mode,
            weight_dtype=weight_dtype,
            **kwargs
        )
        operators = MatrixGame2Operator(mode=mode)
        memory_module = MatrixGame2Memory()

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline
    
    def process(self,
                input_image,
                num_output_frames,
                resize_H=352,
                resize_W=640,
                interaction_signal=["forward", "left", "right",
                                    "forward_left", "forward_right",
                                    "camera_l", "camera_r"]):
        ### the input_image is PIL image
        preception_dict = self.operators.process_perception(input_image, num_output_frames, resize_H, resize_W,
                                                            device=self.device, weight_dtype=self.weight_dtype)
        img_cond = self.synthesis_model.vae.encode(preception_dict["img_cond"], device=self.device,
                                                   **preception_dict["tiler_kwargs"]).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.synthesis_model.vae.clip.encode_video(preception_dict["image"])
        output_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context
        }
        # define the interaction
        self.operators.get_interaction(interaction_signal)
        num_frames = (num_output_frames - 1) * 4 + 1
        operator_condition = self.operators.process_interaction(num_frames=num_frames)
        output_dict['operator_condition'] = operator_condition
        self.operators.delete_last_interaction()
        return output_dict

    def __call__(self,
                 input_image,
                 num_output_frames,
                 resize_H=352,
                 resize_W=640,
                 interaction_signal=["forward", "left", "right",
                                     "forward_left", "forward_right",
                                     "camera_l", "camera_r"],
                 operation_visualization=True,
                 **kwds):
        output_dict = self.process(
            input_image=input_image,
            num_output_frames=num_output_frames,
            resize_H=resize_H,
            resize_W=resize_W,
            interaction_signal=interaction_signal
        )
        output_video = self.synthesis_model.predict(
            cond_concat=output_dict['cond_concat'],
            visual_context=output_dict['visual_context'],
            operator_condition=output_dict['operator_condition'],
            num_output_frames=num_output_frames,
            operation_visualization=operation_visualization,
            **kwds
        )
        return output_video
    
    def stream(self,
               interaction_signal: List[str],
               initial_image: Optional[Image.Image] = None,
               num_output_frames: int = 15,
               resize_H: int = 352,
               resize_W: int = 640,
               operation_visualization: bool = False,
               **kwds) -> torch.Tensor:
        if initial_image is not None:
            print("--- Stream Started ---")
            self.memory_module.record(initial_image)
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'initial_image' first.")
        video_output = self.__call__(
            input_image=current_image,
            num_output_frames=num_output_frames,
            interaction_signal=interaction_signal,
            resize_H=resize_H,
            resize_W=resize_W,
            operation_visualization=operation_visualization,
            **kwds
        )
        self.memory_module.record(video_output)
        return video_output
```

The Synthesis class implementation is as follows:
```python
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
        ### the mode including "gta_drive", "templerun", "universal"
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
```
"""
