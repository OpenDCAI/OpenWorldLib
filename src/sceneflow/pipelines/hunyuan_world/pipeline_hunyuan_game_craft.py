import inspect
import torch
from PIL import Image
from typing import Optional, Any, List

from ...operators.hunyuan_game_craft_operator import HunyuanGameCraftOperator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft_synthesis import HunyuanGameCraftSynthesis
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.config import parse_args
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.modules.parallel_states import initialize_distributed
from ...memories.visual_synthesis.hunyuan_world.hunyuan_game_craft_memory import HunyuanGameCraftMemory


class HunyuanGameCraftPipeline:
    def __init__(
        self,
        operators: Optional[HunyuanGameCraftOperator] = None,
        synthesis_model: Optional[HunyuanGameCraftSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
    ):
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype

        self._predict_sig = None
        if self.synthesis_model is not None and hasattr(self.synthesis_model, "predict"):
            try:
                self._predict_sig = inspect.signature(self.synthesis_model.predict)
            except Exception:
                self._predict_sig = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        cpu_offload: bool = False,
        seed: int = 250160,
        **kwargs,
    ) -> "HunyuanGameCraftPipeline":
        args = parse_args(args=[])
        args.cpu_offload = cpu_offload
        args.seed = seed

        initialize_distributed(args.seed)

        if model_path is not None:
            synthesis_model_path = model_path
        else:
            synthesis_model_path = "tencent/Hunyuan-GameCraft-1.0"

        synthesis_model = HunyuanGameCraftSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device if not args.cpu_offload else torch.device("cpu"),
            weight_dtype=weight_dtype,
            args=args,
            **kwargs,
        )
        operators = HunyuanGameCraftOperator()
        memory_module = HunyuanGameCraftMemory()

        return cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype,
        )

    def _dist_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _has_predict_arg(self, name: str) -> bool:
        return self._predict_sig is not None and name in self._predict_sig.parameters

    def _predict_call(self, **kwargs):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model is None")

        if self._predict_sig is None:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            return self.synthesis_model.predict(**clean)

        filtered = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in self._predict_sig.parameters:
                filtered[k] = v
        return self.synthesis_model.predict(**filtered)

    def process(self,
                input_image,
                output_H,
                output_W,
                interaction_signal):
        """
        the input_image is PIL image
        """

        visual_context = self.operators.process_perception(image=input_image, output_H=output_H, output_W=output_W, process_model=self.synthesis_model)
        
        # define the interaction
        self.operators.get_interaction(interaction_signal)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        output_dict = {
            "visual_context": visual_context,
            "operator_condition": operator_condition,
        }
        
        return output_dict

    def __call__(self,
                # default condition
                images,     # PIL image
                prompt="",
                interactions=["forward", "left", "right", "right", "camera_l", "camera_r", "camera_up", "camera_down"],
                size=(704, 1216),
                num_frames=129,
                # other generation condition
                interaction_speed=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                interaction_positive_prompt="Realistic, High-quality.",
                interaction_negative_prompt="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
                cfg_scale=2.0,
                infer_steps=50,
                flow_shift_eval_video=5.0,
                **kwds):
        input_image = images
        output_H, output_W = size
        output_dict = self.process(
            input_image=input_image,
            output_H=output_H,
            output_W=output_W,
            interaction_signal=interactions
        )
        output_video = self.synthesis_model.predict(
            # condition
            ref_images=output_dict["visual_context"]['ref_images'],
            last_latents=output_dict["visual_context"]['last_latents'],
            ref_latents=output_dict["visual_context"]['ref_latents'],
            action_list=output_dict['operator_condition'],
            action_speed_list=interaction_speed,
            prompt=prompt,
            negative_prompt=interaction_negative_prompt,
            # generation config
            size=(output_H, output_W),
            video_length=num_frames,
            guidance_scale=cfg_scale,
            infer_steps=infer_steps,
            flow_shift=flow_shift_eval_video,
            **kwds
        )
        return output_video

    def stream(
        self,
        interactions: List[str],
        interaction_speed: List[float],
        images=None,
        prompt: str = "",
        size=(704, 1216),
        interaction_positive_prompt: str = "Realistic, High-quality.",
        interaction_negative_prompt: str = (
            "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, "
            "bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
        ),
        num_frames: int = 129,
        cfg_scale: float = 2.0,
        infer_steps: int = 50,
        flow_shift_eval_video: float = 5.0,
        **kwds,
    ):
        if self.memory_module is None:
            raise ValueError("memory_module is None")

        rank = self._dist_rank()
        output_H, output_W = size

        if images is not None:
            visual_context = self.operators.process_perception(
                image=images,
                output_H=output_H,
                output_W=output_W,
                process_model=self.synthesis_model,
            )
            self.memory_module.record(images, visual_context=visual_context, record_frames=False)

        ctx = self.memory_module.select_context()
        if ctx is None:
            raise ValueError("No context in memory. Provide 'images' in the first stream() call.")

        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        if len(operator_condition) != len(interaction_speed):
            raise ValueError(
                f"interaction_speed length mismatch: {len(interaction_speed)} vs actions {len(operator_condition)}"
            )

        first_is_image = (getattr(self.memory_module, "n_generated_segments", 0) == 0)

        prompt = prompt or ""
        positive_prompt = interaction_positive_prompt or ""
        if not self._has_predict_arg("positive_prompt"):
            if positive_prompt.strip():
                prompt = (prompt + " " + positive_prompt).strip()

        out = self._predict_call(
            ref_images=ctx.get("ref_images"),
            last_latents=ctx.get("last_latents"),
            ref_latents=ctx.get("ref_latents"),
            action_list=operator_condition,
            action_speed_list=interaction_speed,
            prompt=prompt,
            negative_prompt=interaction_negative_prompt,
            positive_prompt=positive_prompt if self._has_predict_arg("positive_prompt") else None,
            size=(output_H, output_W),
            video_length=num_frames,
            guidance_scale=cfg_scale,
            infer_steps=infer_steps,
            flow_shift=flow_shift_eval_video,
            first_is_image=first_is_image if self._has_predict_arg("first_is_image") else None,
            return_latents=True if self._has_predict_arg("return_latents") else None,
            **kwds,
        )

        if isinstance(out, dict):
            video_frames = out.get("video", None)
            last_latents = out.get("last_latents", None)
            ref_latents = out.get("ref_latents", None)
        else:
            video_frames = out
            last_latents = None
            ref_latents = None

        self.memory_module.record(
            video_frames if video_frames is not None else [],
            last_latents=last_latents,
            ref_latents=ref_latents,
            record_frames=(rank == 0),
        )

        return video_frames 
