from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

from PIL import Image

from .base_operator import BaseOperator
from ..base_models.diffusion_model.video.wan_2p2.utils.prompt_extend import (
    DashScopePromptExpander,
    QwenPromptExpander,
)


def _load_input_image(input_path: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    img = Image.open(p).convert("RGB")
    return img


class Wan2p2Operator(BaseOperator):
    """
    Wan2.2 ti2v 数据处理 Operator

    - process_interaction: 处理文本（含可选的 prompt 扩写）
    - process_perception: 处理图像
    - 不创建模型、不做推理
    """

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Wan2p2Operator, self).__init__(operation_types=operation_types)

        self.interaction_template = ["text_prompt", "image_prompt"]
        self.interaction_template_init()


    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(
        self,
        *,
        mode: str,
        images: Optional[Image.Image] = None,
        # prompt 扩写相关参数（只在 ti2v 任务下生效）
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "local_qwen",
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: str = "zh",
        base_seed: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:


        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]

        # 记录交互历史
        self.interaction_history.append(prompt)

        # 仅在 ti2v 任务下，且打开 use_prompt_extend 时做扩写
        if "ti2v" in mode and use_prompt_extend:
            logging.info("Extending prompt ...")
            if prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model,
                    mode=mode,
                    is_vl=images is not None,
                )
            elif prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    mode=mode,
                    is_vl=images is not None,
                    device=0,
                )
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {prompt_extend_method}"
                )

            prompt_output = prompt_expander(
                prompt,
                image=images,
                tar_lang=prompt_extend_target_lang,
                seed=base_seed,
            )
            if prompt_output.status is False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}"
                )
                logging.info("Falling back to original prompt.")
            else:
                prompt = prompt_output.prompt

            logging.info(f"Extended prompt: {prompt}")

        return {
            "processed_prompt": prompt,
        }


    def process_perception(
        self,
        *,
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        处理图像输入：
        - ti2v 允许没有参考图像，此时返回的 input_image 为 None
        - 如果提供了 input_path，则加载为 PIL.Image
        """
        if input_path is None:
            input_image = None
        else:
            input_image = _load_input_image(input_path)

        return {
            "input_image": input_image,
        }
