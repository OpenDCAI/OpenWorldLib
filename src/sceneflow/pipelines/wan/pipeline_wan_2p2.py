from typing import Any, Dict, Optional
import random
import sys
import torch

from ...base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS, SUPPORTED_SIZES
from PIL import Image

from ...operators.wan_2p2_operator import Wan2p2Operator
from ...synthesis.visual_generation.wan.wan2p2_synthesis import Wan2p2Synthesis
from ...memories.visual_synthesis.wan.wan_2p2_memeory import Wan2p2Memory

EXAMPLE_PROMPT = {
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    }
}



class Wan2p2Pipeline:

    def __init__(
        self,
        *,
        operator: Wan2p2Operator,
        synthesis_model: Wan2p2Synthesis,
        memory_module: Optional[Wan2p2Memory] = None,
        task: str = "ti2v-5B",
        ulysses_size: int = 1,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        prompt: Optional[str] = None,
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "local_qwen",
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: str = "zh",
        image: Optional[str] = None,
        sample_solver: str = "unipc",
        sample_steps: Optional[int] = None,
        sample_shift: Optional[float] = None,
        sample_guide_scale: Optional[float] = None,
        convert_model_dtype: bool = False,
        base_seed: int = -1,
    ) -> None:
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module if memory_module else Wan2p2Memory()
        
        # Store parameters
        self.task = task
        self.ulysses_size = ulysses_size
        self.t5_fsdp = t5_fsdp
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self.prompt = prompt
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_method = prompt_extend_method
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.image = image
        self.convert_model_dtype = convert_model_dtype
        
        # Set default sampling parameters from config
        cfg = WAN_CONFIGS[task]
        self.sample_solver = sample_solver
        self.sample_steps = sample_steps if sample_steps is not None else cfg.sample_steps
        self.sample_shift = sample_shift if sample_shift is not None else cfg.sample_shift
        self.sample_guide_scale = sample_guide_scale if sample_guide_scale is not None else cfg.sample_guide_scale
        self.base_seed = base_seed if base_seed >= 0 else random.randint(0, sys.maxsize)


    @classmethod
    def from_pretrained(
        cls,
        synthesis_model_path: str,
        task: str = "ti2v-5B",
        ulysses_size: int = 1,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        convert_model_dtype: bool = False,
        device_id: int = 0,
        rank: int = 0,
        **kwargs
    ) -> "Wan2p2Pipeline":
        """
        Load a pretrained Wan2p2Pipeline.
        
        Args:
            synthesis_model_path: Path to the pretrained model
            task: Task type (e.g., "ti2v-5B")
            device_id: GPU device ID
            rank: Distributed training rank
            ... (other model loading parameters)
        
        Note:
            Generation parameters like `prompt`, `image`, `size`, `frame_num`,
            `offload_model`, sampling parameters, and prompt extension parameters
            should be passed to `__call__()` method instead of here.
        """
        # Validate task
        assert task in WAN_CONFIGS, f"Unsupport task: {task}"
        assert task in EXAMPLE_PROMPT, f"Unsupport task: {task}"

        operator = Wan2p2Operator()
        memory_module = Wan2p2Memory()
        synthesis_model = Wan2p2Synthesis.from_pretrained(
            task=task,
            ckpt_dir=synthesis_model_path,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            ulysses_size=ulysses_size,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype
        )

        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            task=task,
            ulysses_size=ulysses_size,
            t5_fsdp=t5_fsdp,
            t5_cpu=t5_cpu,
            dit_fsdp=dit_fsdp,
            convert_model_dtype=convert_model_dtype,
        )


    def process(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        use_prompt_extend: Optional[bool] = None,
        prompt_extend_method: Optional[str] = None,
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: Optional[str] = None,
        base_seed: Optional[int] = None,
    ) -> Dict[str, Any]:

        # 优先使用内存中的 image，其次才是 image_path
        # 如果 image_path 是空字符串，视为 None（ti2v 任务允许没有参考图像）
        if image is not None:
            input_for_perception = image
        elif image_path and image_path.strip():
            input_for_perception = image_path
        else:
            input_for_perception = None
        
        perception = self.operator.process_perception(input_path=input_for_perception)
        img = perception["input_image"]

        self.operator.get_interaction(prompt)
        interaction = self.operator.process_interaction(
            task=self.task,
            image=img,
            use_prompt_extend=use_prompt_extend if use_prompt_extend is not None else self.use_prompt_extend,
            prompt_extend_method=prompt_extend_method if prompt_extend_method is not None else self.prompt_extend_method,
            prompt_extend_model=prompt_extend_model if prompt_extend_model is not None else self.prompt_extend_model,
            prompt_extend_target_lang=prompt_extend_target_lang if prompt_extend_target_lang is not None else self.prompt_extend_target_lang,
            base_seed=base_seed if base_seed is not None else self.base_seed,
        )

        return {
            "prompt": interaction["processed_prompt"],
            "image": img,
            "paths": {
                "image_path": image_path,
            },
            "meta": {
                "task": self.task,
            },
        }

    def __call__(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        size: Optional[str] = None,
        frame_num: Optional[int] = None,
        sample_solver: Optional[str] = None,
        sample_steps: Optional[int] = None,
        sample_shift: Optional[float] = None,
        sample_guide_scale: Optional[float] = None,
        base_seed: Optional[int] = None,
        offload_model: Optional[bool] = None,
        use_prompt_extend: Optional[bool] = None,
        prompt_extend_method: Optional[str] = None,
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: Optional[str] = None,
    ) -> Any:
        """
        Generate video from prompt and optional image.
        
        Args:
            prompt: Text prompt for video generation (required)
            image_path: Path to input image (optional)
            image: PIL Image object (optional, takes precedence over image_path)
            size: Output video size (optional, defaults to config value, e.g., "1280*720")
            frame_num: Number of frames (optional, defaults to config value)
            sample_solver: Override sampling solver (optional, defaults to "unipc")
            sample_steps: Override sampling steps (optional, defaults to config value)
            sample_shift: Override sample shift (optional, defaults to config value)
            sample_guide_scale: Override guidance scale (optional, defaults to config value)
            base_seed: Override random seed (optional)
            offload_model: Whether to offload model to CPU during generation (optional)
            use_prompt_extend: Enable prompt extension (optional, defaults to False)
            prompt_extend_method: Prompt extension method (optional, defaults to "local_qwen")
            prompt_extend_model: Model for prompt extension (optional)
            prompt_extend_target_lang: Target language for prompt extension (optional, defaults to "zh")
        
        Returns:
            Generated video tensor
        """
        cfg = WAN_CONFIGS[self.task]
        
        # Set default size from config if not provided
        if size is None:
            # Use a reasonable default, typically "1280*720"
            size = "1280*720"
        
        # Validate size
        if 's2v' not in self.task:
            assert size in SUPPORTED_SIZES[self.task], \
                f"Unsupport size {size} for task {self.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[self.task])}"
        
        # Set default frame_num from config if not provided
        if frame_num is None:
            frame_num = cfg.frame_num
        
        # Use provided parameters or fall back to instance defaults (from config)
        video_sample_solver = sample_solver if sample_solver is not None else self.sample_solver
        video_sample_steps = sample_steps if sample_steps is not None else self.sample_steps
        video_sample_shift = sample_shift if sample_shift is not None else self.sample_shift
        video_sample_guide_scale = sample_guide_scale if sample_guide_scale is not None else self.sample_guide_scale
        video_base_seed = base_seed if base_seed is not None else self.base_seed

        processed = self.process(
            prompt=prompt,
            image_path=image_path,
            image=image,
            use_prompt_extend=use_prompt_extend,
            prompt_extend_method=prompt_extend_method,
            prompt_extend_model=prompt_extend_model,
            prompt_extend_target_lang=prompt_extend_target_lang,
            base_seed=video_base_seed,
        )

        # Create a dict with all the synthesis parameters
        synthesis_params = {
            "task": self.task,
            "size": size,
            "frame_num": frame_num,
            "sample_solver": video_sample_solver,
            "sample_steps": video_sample_steps,
            "sample_shift": video_sample_shift,
            "sample_guide_scale": video_sample_guide_scale,
            "base_seed": video_base_seed,
            "offload_model": offload_model,
        }

        video = self.synthesis_model.predict(
            processed_inputs=processed,
            **synthesis_params,
        )

        return video


    def stream(
        self,
        *,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        use_prompt_extend: Optional[bool] = None,
        prompt_extend_method: Optional[str] = None,
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: Optional[str] = None,
    ) -> Any:
        """
        - 每次调用都会复用 __call__ 完整生成一段视频；
        - 始终将该段视频记录到 memory_module（拆帧追加到 all_frames）；
        - 返回本轮生成的视频张量。
        """
        # Use provided prompt or fall back to instance default
        if prompt is None:
            if self.prompt is None:
                raise ValueError("prompt must be provided either in initialization or stream().")
            prompt = self.prompt
        
        video = self.__call__(
            prompt=prompt,
            image_path=image_path,
            image=image,
            use_prompt_extend=use_prompt_extend,
            prompt_extend_method=prompt_extend_method,
            prompt_extend_model=prompt_extend_model,
            prompt_extend_target_lang=prompt_extend_target_lang,
        )

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"[Wan2p2Pipeline.stream] Expected torch.Tensor from predict, got {type(video)}"
            )
        self.memory_module.record(video)
        print(
            f"[Wan2p2Pipeline.stream] Recorded segment. "
            f"Total frames in memory: {len(getattr(self.memory_module, 'all_frames', []))}"
        )

        return video