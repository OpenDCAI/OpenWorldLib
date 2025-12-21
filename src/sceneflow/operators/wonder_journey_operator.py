import torch
import numpy as np
import random
import sys
import os

# 动态修复导入路径，确保能找到 util
try:
    from util.utils import load_example_yaml
except ImportError:
    pass

class WonderJourneyOperator:
    def __init__(self, config):
        self.config = config
        # 1. 修复 device 问题：从 config 中安全获取字符串，默认 'cuda'
        self.device = config.get("device", "cuda")
        
        self.interaction_template = ["move_forward", "rotate", "text_control", "idle"]
        self.current_interaction = []
        
        # 从 config 中读取 rotation_path (village.yaml 里定义的 list)
        self.rotation_path = config.get('rotation_path', [])
        
        # 初始化 yaml_data 为 None，用于稍后懒加载
        self.yaml_data = None
        
    def check_interaction(self, interaction):
        # 简单检查逻辑
        if interaction not in self.interaction_template and not isinstance(interaction, (int, float, str)):
             pass 
        return True

    def get_interaction(self, interaction):
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames=None):
        """
        核心逻辑：解析 YAML 配置，返回 pipeline 所需的所有参数字典
        """
        # 1. 确保 yaml_data 已加载 (从 examples/examples.yaml 读取 prompt 等信息)
        if self.yaml_data is None:
            self._ensure_util_path()
            from util.utils import load_example_yaml
            # 获取 example_name (如 'village')
            example_name = self.config.get("example_name", "village")
            # 加载对应的 prompt 配置
            self.yaml_data = load_example_yaml(example_name, 'examples/wonder_journey/examples.yaml')
        
        # 2. 提取 Prompt 数据
        content_prompt = self.yaml_data['content_prompt']
        style_prompt = self.yaml_data['style_prompt']
        adaptive_negative_prompt = self.yaml_data['negative_prompt']
        background_prompt = self.yaml_data.get('background', None)
        control_text = self.yaml_data.get('control_text', None)
        
        # 处理 negative prompt
        if adaptive_negative_prompt != "":
            adaptive_negative_prompt += ", "
        
        # 处理 scene_name 和 entities (复刻 run.py 逻辑)
        content_list = content_prompt.split(',')
        scene_name = content_list[0]
        entities = content_list[1:]
        
        # 处理 control_text 列表逻辑
        if isinstance(control_text, list):
            self.config['num_scenes'] = len(control_text)
            
        # 3. 【关键】构建并返回字典
        # Pipeline 会通过 interaction_data["key"] 来访问这些值
        return {
            "content_prompt": content_prompt,
            "style_prompt": style_prompt,
            "adaptive_negative_prompt": adaptive_negative_prompt,
            "background_prompt": background_prompt,
            "control_text": control_text,
            "scene_name": scene_name,
            "entities": entities,
            "rotation_path": self.rotation_path,
            "inpainting_prompt": style_prompt + ', ' + content_prompt
        }

    def process_perception(self, multimodal_input=None):
        """
        加载初始视觉信号 (Start Image)
        """
        print(f"[Operator] Processing perception signals...")
        
        start_keyframe = None

        # 1. 优先处理传入的信号 (如果有)
        if multimodal_input is not None:
            if "visual" in multimodal_input:
                image = multimodal_input["visual"]
                width = 512
                # 处理 PIL Image
                if hasattr(image, "size"): 
                    width = image.size[0]
                # 处理 Tensor / Numpy
                elif hasattr(image, "shape"): 
                    width = image.shape[-1]
                
                self.config["init_focal_length"] = width
                print(f"  - Visual: Set focal length to {width}")
                start_keyframe = image

            if "text" in multimodal_input:
                prompt = multimodal_input["text"].lower()
                if "fast" in prompt: self.config["forward_speed_multiplier"] = 0.1
                elif "slow" in prompt: self.config["forward_speed_multiplier"] = 0.02

        # 2. 兜底逻辑：从 YAML 加载 (Pipeline 初始化时通常走这里)
        if start_keyframe is None:
            print("  - No visual input provided, loading from YAML config...")
            if self.yaml_data is None:
                self._ensure_util_path()
                from util.utils import load_example_yaml
                example_name = self.config.get("example_name", "village")
                self.yaml_data = load_example_yaml(example_name, 'examples/wonder_journey/examples.yaml')
            
            from PIL import Image
            # 读取图片路径 (例如 data/village.jpg)
            image_path = self.yaml_data['image_filepath']
            print(f"  - Loading image from: {image_path}")
            start_keyframe = Image.open(image_path).convert('RGB').resize((512, 512))
            
            # 默认 focal length
            self.config["init_focal_length"] = 512

        return start_keyframe

    def delete_last_interaction(self):
        self.current_interaction = self.current_interaction[:-1]

    def seeding(self, seed):
        if seed == -1:
            seed = np.random.randint(2 ** 32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"running with seed: {seed}.")

    def _ensure_util_path(self):
        """
        辅助函数：确保 util 文件夹在 sys.path 中
        防止 ModuleNotFoundError: No module named 'util'
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 定位到 src/sceneflow/representations/models/wonder_journey/wonder_journey
        target_dir = os.path.abspath(os.path.join(
            current_dir, 
            "../representations/models/wonder_journey/wonder_journey"
        ))
        if target_dir not in sys.path:
            sys.path.append(target_dir)
        
        # 备用方案：项目根目录
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
        if project_root not in sys.path:
            sys.path.append(project_root)