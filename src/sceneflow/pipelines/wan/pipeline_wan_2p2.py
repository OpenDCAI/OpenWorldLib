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
        size: str = "1280*720",
        frame_num: Optional[int] = None,
        offload_model: Optional[bool] = None,
        ulysses_size: int = 1,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        prompt: Optional[str] = None,
        save_file: Optional[str] = None,
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
        self.size = size
        self.frame_num = frame_num
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.t5_fsdp = t5_fsdp
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self.prompt = prompt
        self.save_file = save_file
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_method = prompt_extend_method
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.image = image
        self.sample_solver = sample_solver
        self.sample_steps = sample_steps
        self.sample_shift = sample_shift
        self.sample_guide_scale = sample_guide_scale
        self.convert_model_dtype = convert_model_dtype
        self.base_seed = base_seed


    @classmethod
    def from_pretrained(
        cls,
        synthesis_model_path: str,
        task: str = "ti2v-5B",
        size: str = "1280*720",
        frame_num: Optional[int] = None,
        offload_model: Optional[bool] = None,
        ulysses_size: int = 1,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        prompt: Optional[str] = None,
        save_file: Optional[str] = None,
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
        device_id: int = 0,
        rank: int = 0,
        **kwargs
    ) -> "Wan2p2Pipeline":
        
        # Validate task
        assert task in WAN_CONFIGS, f"Unsupport task: {task}"
        assert task in EXAMPLE_PROMPT, f"Unsupport task: {task}"
        
        # Set default prompt if None
        if prompt is None:
            prompt = EXAMPLE_PROMPT[task]["prompt"]
        
        cfg = WAN_CONFIGS[task]
        
        # Set default values from config
        if sample_steps is None:
            sample_steps = cfg.sample_steps
        if sample_shift is None:
            sample_shift = cfg.sample_shift
        if sample_guide_scale is None:
            sample_guide_scale = cfg.sample_guide_scale
        if frame_num is None:
            frame_num = cfg.frame_num
        
        # Set random seed
        if base_seed < 0:
            base_seed = random.randint(0, sys.maxsize)
        
        # Size check
        if 's2v' not in task:
            assert size in SUPPORTED_SIZES[task], \
                f"Unsupport size {size} for task {task}, supported sizes are: {', '.join(SUPPORTED_SIZES[task])}"

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
            size=size,
            frame_num=frame_num,
            offload_model=offload_model,
            ulysses_size=ulysses_size,
            t5_fsdp=t5_fsdp,
            t5_cpu=t5_cpu,
            dit_fsdp=dit_fsdp,
            prompt=prompt,
            save_file=save_file,
            use_prompt_extend=use_prompt_extend,
            prompt_extend_method=prompt_extend_method,
            prompt_extend_model=prompt_extend_model,
            prompt_extend_target_lang=prompt_extend_target_lang,
            image=image,
            sample_solver=sample_solver,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            convert_model_dtype=convert_model_dtype,
            base_seed=base_seed,
        )


    def process(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:

        # 优先使用内存中的 image，其次才是 image_path
        input_for_perception = image if image is not None else image_path
        perception = self.operator.process_perception(input_path=input_for_perception)
        img = perception["input_image"]

        self.operator.get_interaction(prompt)
        interaction = self.operator.process_interaction(
            task=self.task,
            image=img,
            use_prompt_extend=self.use_prompt_extend,
            prompt_extend_method=self.prompt_extend_method,
            prompt_extend_model=self.prompt_extend_model,
            prompt_extend_target_lang=self.prompt_extend_target_lang,
            base_seed=self.base_seed,
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
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        save: bool = False,
    ) -> Any:
        if prompt is None:
            if self.prompt is None:
                raise ValueError("prompt must be provided either in initialization or call().")
            prompt = self.prompt

        if image is None and image_path is None:
            image_path = self.image

        processed = self.process(
            prompt=prompt,
            image_path=image_path,
            image=image,
        )

        # Create a dict with all the synthesis parameters
        synthesis_params = {
            "task": self.task,
            "size": self.size,
            "frame_num": self.frame_num,
            "sample_solver": self.sample_solver,
            "sample_steps": self.sample_steps,
            "sample_shift": self.sample_shift,
            "sample_guide_scale": self.sample_guide_scale,
            "base_seed": self.base_seed,
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
    ) -> Any:
        """
        - 每次调用都会复用 __call__ 完整生成一段视频；
        - 始终将该段视频记录到 memory_module（拆帧追加到 all_frames）；
        - 返回本轮生成的视频张量。
        """
        video = self.__call__(
            prompt=prompt,
            image_path=image_path,
            image=image,
            save=False,
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