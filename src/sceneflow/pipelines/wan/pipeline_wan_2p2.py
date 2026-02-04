from typing import Any, Dict, Optional
import random
import sys
import torch

from dataclasses import dataclass

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

def _validate_args(args):
    # Basic check
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # 仅 ti2v 任务：补默认 prompt
    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

@dataclass
class Wan2p2Args:
    """
    完整版参数类：一一对应 Wan2.2 原始 generate.py 中的 argparse 参数。
    这样可以在不丢失任何能力的前提下复用同一套配置。
    """

    task: str = "ti2v-5B"
    size: str = "1280*720"
    frame_num: Optional[int] = None

    # --- 分布式 / 并行 ---
    offload_model: Optional[bool] = None
    ulysses_size: int = 1
    t5_fsdp: bool = False
    t5_cpu: bool = False
    dit_fsdp: bool = False

    # --- 输出 & 文本输入 ---
    prompt: Optional[str] = None
    save_file: Optional[str] = None

    # --- Prompt 扩写相关 ---
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"  # ["dashscope", "local_qwen"]
    prompt_extend_model: Optional[str] = None
    prompt_extend_target_lang: str = "zh"  # ["zh", "en"]

    # --- 图像 / 基本采样控制 ---
    image: Optional[str] = None
    sample_solver: str = "unipc"  # ["unipc", "dpm++"]
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    sample_guide_scale: Optional[float] = None
    convert_model_dtype: bool = False

    # --- 随机种子 ---
    base_seed: int = -1



class Wan2p2Pipeline:

    def __init__(
        self,
        *,
        operator: Wan2p2Operator,
        synthesis_model: Wan2p2Synthesis,
        args: Wan2p2Args,
        memory_module: Optional[Wan2p2Memory] = None,
    ) -> None:
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.args = args
        self.memory_module = memory_module if memory_module else Wan2p2Memory()


    @classmethod
    def from_pretrained(
        cls,
        *,
        synthesis_model_path: str,
        args: Wan2p2Args,
        device_id: int = 0,
        rank: int = 0,
    ) -> "Wan2p2Pipeline":

        _validate_args(args)


        operator = Wan2p2Operator()
        memory_module = Wan2p2Memory()
        synthesis_model = Wan2p2Synthesis.from_pretrained(
            task=args.task,
            ckpt_dir=synthesis_model_path,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            ulysses_size=args.ulysses_size,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype
        )

        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            args=args,
            memory_module=memory_module,
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
            task=self.args.task,
            image=img,
            use_prompt_extend=self.args.use_prompt_extend,
            prompt_extend_method=self.args.prompt_extend_method,
            prompt_extend_model=self.args.prompt_extend_model,
            prompt_extend_target_lang=self.args.prompt_extend_target_lang,
            base_seed=self.args.base_seed,
        )

        return {
            "prompt": interaction["processed_prompt"],
            "image": img,
            "paths": {
                "image_path": image_path,
            },
            "meta": {
                "task": self.args.task,
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
            if self.args.prompt is None:
                raise ValueError("prompt must be provided either in args or call().")
            prompt = self.args.prompt

        if image is None and image_path is None:
            image_path = self.args.image

        processed = self.process(
            prompt=prompt,
            image_path=image_path,
            image=image,
        )

        video = self.synthesis_model.predict(
            processed_inputs=processed,
            args=self.args,
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