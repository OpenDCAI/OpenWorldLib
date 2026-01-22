import torch
from typing import Optional, Any, List
from torchvision.transforms import v2
from ...operators.hunyuan_game_craft_operator import HunyuanGameCraftOperator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft_synthesis import HunyuanGameCraftSynthesis
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.config import parse_args
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.modules.parallel_states import initialize_distributed
from ...memories.visual_synthesis.hunyuan_world.hunyuan_game_craft_memory import HunyuanGameCraftMemory

class HunyuanGameCraftPipeline:
    def __init__(self,
                operators: Optional[HunyuanGameCraftOperator] = None,
                synthesis_model: Optional[HunyuanGameCraftSynthesis] = None,
                memory_module: Optional[Any] = None,
                device: str = "cuda",
                weight_dtype = torch.bfloat16
                ):
        self.synthesis_model = synthesis_model 
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(cls,
                        synthesis_model_path: Optional[str] = None,
                        weight_dtype=torch.bfloat16,
                        device: str = "cuda",
                        cpu_offload: bool = False,
                        seed: int = 250160,
                        **kwargs) -> "HunyuanGameCraftPipeline":

        args = parse_args()
        args.cpu_offload = cpu_offload
        args.seed = seed

        initialize_distributed(args.seed)

        if synthesis_model_path is None:
            synthesis_model_path = "tencent/Hunyuan-GameCraft-1.0"

        print(f"Loading HunyuanGameCraft synthesis model from {synthesis_model_path}...")

        synthesis_model = HunyuanGameCraftSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device if not args.cpu_offload else torch.device("cpu"),
            weight_dtype=weight_dtype,
            args=args,
            **kwargs
        )
        operators = HunyuanGameCraftOperator()

        # NEW: memory module for streaming
        from ...memories.visual_synthesis.hunyuan_world.hunyuan_game_craft_memory import HunyuanGameCraftMemory
        memory_module = HunyuanGameCraftMemory()

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,   # NEW
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline


    def __call__(self,
                input_image,
                **kwargs):
        output_video = self.synthesis_model.predict(
            **kwargs
        )
        return output_video

    def _dist_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def stream(self,
               interaction_signal: List[str],
               interaction_speed: List[float],
               initial_image=None,
               # prompts
               interaction_text_prompt: str = "",
               interaction_positive_prompt: str = "Realistic, High-quality.",
               interaction_negative_prompt: str = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
               # generation config
               output_H: int = 704,
               output_W: int = 1216,
               num_output_frames: int = 129,
               cfg_scale: float = 2.0,
               infer_steps: int = 50,
               flow_shift_eval_video: float = 5.0,
               **kwds):
        """
        Multi-turn streaming API:
          - First call must provide initial_image (PIL.Image) to bootstrap latents
          - Later calls omit initial_image to continue from memory (last_latents/ref_latents)
        """
        if self.memory_module is None:
            raise ValueError("memory_module is None. Please instantiate HunyuanGameCraftMemory in from_pretrained().")

        rank = self._dist_rank()

        # 1) Bootstrap memory (first turn)
        if initial_image is not None:
            visual_context = self.operators.process_perception(
                image=initial_image,
                output_H=output_H,
                output_W=output_W,
                process_model=self.synthesis_model
            )

            self.memory_module.record(initial_image, visual_context=visual_context, record_frames=False)

        ctx = self.memory_module.select_context()
        if ctx is None:
            raise ValueError("No context in memory. Provide 'initial_image' in the first stream() call.")

        # 2) Build operator condition (action list)
        self.operators.get_interaction(interaction_signal)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        if len(operator_condition) != len(interaction_speed):
            raise ValueError(f"interaction_speed length mismatch: {len(interaction_speed)} vs actions {len(operator_condition)}")

        # 3) Decide whether the first segment is image-mode
        #    - only the very first generated segment uses first_is_image=True
        first_is_image = (self.memory_module.n_generated_segments == 0)

        # 4) Call synthesis, request latents back to continue across turns
        out = self.synthesis_model.predict(
            # condition
            ref_images=ctx["ref_images"],
            last_latents=ctx["last_latents"],
            ref_latents=ctx["ref_latents"],
            action_list=operator_condition,
            action_speed_list=interaction_speed,
            prompt=interaction_text_prompt,
            negative_prompt=interaction_negative_prompt,
            # generation config
            size=(output_H, output_W),
            video_length=num_output_frames,
            guidance_scale=cfg_scale,
            infer_steps=infer_steps,
            flow_shift=flow_shift_eval_video,
            first_is_image=first_is_image,
            return_latents=True,
            **kwds
        )

        # out: dict(video=..., last_latents=..., ref_latents=...)
        video_frames = out.get("video", None)
        last_latents = out.get("last_latents", None)
        ref_latents = out.get("ref_latents", None)

        self.memory_module.record(
            video_frames if video_frames is not None else [],
            last_latents=last_latents,
            ref_latents=ref_latents,
            record_frames=(rank == 0)
        )

        return video_frames if rank == 0 else None