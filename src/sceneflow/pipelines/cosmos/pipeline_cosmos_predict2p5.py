import math
import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from ...operators.cosmos_predict2p5_operator import CosmosPredict2p5Operator
from ...synthesis.visual_generation.cosmos.cosmos_predict2p5_synthesis import CosmosPredict2p5Synthesis
from ...memories.visual_synthesis.cosmos.cosmos_predict2p5_memory import CosmosPredict2p5Memory


class CosmosPredict2p5Pipeline:

    def __init__(
        self,
        operator: Optional[CosmosPredict2p5Operator] = None,
        synthesis_model: Optional[CosmosPredict2p5Synthesis] = None,
        memory_module: Optional[CosmosPredict2p5Memory] = None,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        self.synthesis_model = synthesis_model
        self.operator = operator
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.current_image = None

    @classmethod
    def from_pretrained(
        cls, 
        model_path: Optional[str] = None,
        required_components: Optional[Dict] = None,
        token: Optional[str] = None,
        mode: str = 'img2world',
        device: str = "cuda",
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
        **kwargs,
    ) -> "CosmosPredict2p5Pipeline":
        if model_path is None:
            model_path = "nvidia/Cosmos-Predict2.5"
        if required_components is None:
            required_components = {
                "text_encoder_model_path": "nvidia/Cosmos-Reason1-7B",
                "vae_model_path": "Wan-AI/Wan2.1-T2V-1.3B",
            }

        synthesis_model = CosmosPredict2p5Synthesis.from_pretrained(
            mode=mode,
            transformer_model_path=model_path,
            text_encoder_model_path=required_components["text_encoder_model_path"],
            vae_model_path=required_components["vae_model_path"],
            token=token,
            device=torch.device(device),
            weight_dtype=weight_dtype,
        )
        operator = CosmosPredict2p5Operator()
        memory_module = CosmosPredict2p5Memory()

        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline

    def set_negative_prompt(self, neg_prompt: Optional[str] = None):
        if neg_prompt is not None:
            self.negative_prompt = neg_prompt
        else:
            self.negative_prompt = (
                'The video captures a series of frames showing ugly scenes, static with no motion, '
                'motion blur, over-saturation, shaky footage, low resolution, grainy texture,'
                ' pixelated images, poorly lit areas, underexposed and overexposed scenes, '
                'poor color balance, washed out colors, choppy sequences, jerky movements, '
                'low frame rate, artifacting, color banding, unnatural transitions, '
                'outdated special effects, fake elements, unconvincing visuals, '
                'poorly edited content, jump cuts, visual noise, and flickering. '
                'Overall, the video is of poor quality.'
            )

    def process(
        self,
        prompt: str,
        images: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        height: int = 704,
        width: int = 1280,
    ) -> Dict[str, Any]:
        input_for_perception = images if images is not None else image_path
        perception = self.operator.process_perception(
            input_path=input_for_perception,
            height=height,
            width=width,
        )
        
        image = perception["input_image"]
        height = perception["height"]
        width = perception["width"]

        self.operator.get_interaction(prompt)
        interaction = self.operator.process_interaction()
        
        prompt = interaction["input_prompt"]

        return {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
        }

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        images: Any = None,
        image_path: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        fps: int = 28,
        num_frames: int = 93,
        height: int = 704,
        width: int = 1280,
        action_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_scale: Optional[Union[float, Dict[str, float]]] = 1.0,
        cond_timestep: float = 0,
        timestep_scale: float = 0.001,
        seed: int = -1,
        use_kerras_sigma: bool = True,
        pad_mode: str = 'repeat',
        output_type: Optional[str] = 'pt',
    ) -> Any:
        if prompt is None:
            raise ValueError("prompt must be provided either in initialization or call().")
        if negative_prompt is None:
            negative_prompt = getattr(self, "negative_prompt", None)

        processed_input = self.process(
            prompt=prompt,
            images=images,
            image_path=image_path,
            height=height,
            width=width,
        )

        video = self.synthesis_model.predict(
            **processed_input,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            action_latents=action_latents,
            control_video=control_video,
            control_scale=control_scale,
            cond_timestep=cond_timestep,
            timestep_scale=timestep_scale,
            seed=seed,
            use_kerras_sigma=use_kerras_sigma,
            pad_mode=pad_mode,
            output_type=output_type,
        )

        return video

    def stream(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        images: Any = None,
        image_path: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        fps: int = 28,
        num_frames: int = 93,
        height: int = 704,
        width: int = 1280,
        action_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_scale: Optional[Union[float, Dict[str, float]]] = 1.0,
        cond_timestep: float = 0,
        timestep_scale: float = 0.001,
        seed: int = -1,
        use_kerras_sigma: bool = True,
        pad_mode: str = 'repeat',
        output_type: Optional[str] = 'pt',
    ) -> Any:
        video = self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=images,
            image_path=image_path,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            action_latents=action_latents,
            control_video=control_video,
            control_scale=control_scale,
            cond_timestep=cond_timestep,
            timestep_scale=timestep_scale,
            seed=seed,
            use_kerras_sigma=use_kerras_sigma,
            pad_mode=pad_mode,
            output_type=output_type,
        )

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"[CosmosPredict2p5Pipeline.stream] Expected torch.Tensor from predict, got {type(video)}"
            )

        video = video.squeeze(0)
        self.memory_module.record(video)
        print(
            f"[CosmosPredict2p5Pipeline.stream] Recorded segment. "
            f"Total frames in memory: {len(getattr(self.memory_module, 'all_frames', []))}"
        )

        return video