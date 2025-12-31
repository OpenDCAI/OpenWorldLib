from PIL import Image
from typing import Optional, Any, Union, Dict
from pathlib import Path
from ...operators.wow_operator import WoWOperator
from ...synthesis.visual_generation.wow.wow_synthesis import WoWSynthesis

class WoWArgs:
    """
    WoW Pipeline 参数类
    """
    def __init__(
        self,
        gpu: int = 0,
        steps: int = 50,
        seed: int = 42,
        num_frames: int = 41,
        no_tiled: bool = False,
        enable_vram_management: bool = True,
        no_vram_management: bool = False,
        persistent_param_gb: int = 70
    ):

        self.gpu = gpu
        self.steps = steps
        self.seed = seed
        self.num_frames = num_frames
        self.no_tiled = no_tiled
        self.enable_vram_management = enable_vram_management
        self.no_vram_management = no_vram_management
        self.persistent_param_gb = persistent_param_gb


class WoWPipeline:

    
    def __init__(
        self, 
        operator: Optional[WoWOperator] = None, 
        synthesis_model: Optional[WoWSynthesis] = None, 
        synthesis_args: Optional[WoWArgs] = None,
        device: str = "cuda"):

        self.operator = operator
        self.synthesis_model = synthesis_model
        self.synthesis_args = synthesis_args or WoWArgs()
        self.device = device

    @classmethod
    def from_pretrained(
        cls, 
        synthesis_model_path: str = 'WoW-world-model/WoW-1-Wan-1.3B-2M', 
        synthesis_args: Optional[WoWArgs] = None,
        device: Optional[str] = None,
        **kwargs):
        """
        从预训练模型路径创建 WoWPipeline
        
        Args:
            synthesis_model_path: 模型路径（本地路径或HuggingFace repo_id）
                                 如果是 HuggingFace repo_id，模型将下载到当前工作目录
            synthesis_args: WoWArgs 参数对象
            device: 设备（如 'cuda' 或 'cpu'）
            **kwargs: 其他参数
            
        Returns:
            WoWPipeline 实例
        """
        if synthesis_args is None:
            synthesis_args = WoWArgs()

        # 若未显式指定 device，则使用 WoWArgs 中的 gpu 选择 GPU
        if device is None:
            device = f"cuda:{synthesis_args.gpu}"
        
        synthesis_model = WoWSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            synthesis_args=synthesis_args,
            device=device,
            **kwargs
        )

        operator = WoWOperator()

        pipeline = cls(
            operator=operator, 
            synthesis_model=synthesis_model, 
            synthesis_args=synthesis_args,
            device=device,
            **kwargs
        )

        return pipeline

    def process(self, 
                input_path: Union[str, Path, Image.Image], 
                text_prompt: str) -> Dict[str, Any]:

        # 处理感知输入
        processed_perception = self.operator.process_perception(input_path)
        input_image = processed_perception['input_image']
        
        # 处理交互输入
        self.operator.get_interaction(text_prompt)
        processed_interaction = self.operator.process_interaction()
        processed_prompt = processed_interaction['processed_prompt']
        
        return {
            'input_image': input_image,
            'interaction': processed_prompt
        }

    def __call__(self, 
                 input_path: Union[str, Path, Image.Image], 
                 text_prompt: str, 
                 args: Optional[WoWArgs] = None,
                 **kwargs) -> Any:

        if args is None:
            args = self.synthesis_args
        
        processed_data = self.process(input_path, text_prompt)

        output_video = self.synthesis_model.predict(
            input_image=processed_data['input_image'],
            text_prompt=processed_data['interaction'],
            synthesis_args=args,
            **kwargs
        )

        return output_video
