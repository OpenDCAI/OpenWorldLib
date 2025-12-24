import torch
import numpy as np
import random
import sys
import os
from PIL import Image

class WonderJourneyOperator:
    def __init__(self, args):
        # 接收 args 
        self.args = args
        
        self.device = args.device
        
        self.interaction_template = ["move_forward", "rotate", "text_control", "idle"]
        self.current_interaction = []

        self.rotation_path = args.rotation_path
        
    def check_interaction(self, interaction):
        if interaction not in self.interaction_template and not isinstance(interaction, (int, float, str)):
             pass 
        return True

    def get_interaction(self, interaction):
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames=None):
        """
        核心逻辑：直接从 self.args 中获取 Prompt 信息，不再读取 YAML 文件
        """
        # 1. 从 args 获取 Prompt
        content_prompt = self.args.content_prompt
        style_prompt = self.args.style_prompt
        background_prompt = self.args.background_prompt
        control_text = self.args.control_text
        
        # 处理 negative prompt (拼接 adaptive 和固定 negative)
        # 注意：这里假设 args 里只给了negative_inpainting_prompt，
        adaptive_negative_prompt = "" 
        
        # 解析 Scene 和 Entities 
        # content_prompt 格式示例: "SceneName, entity1, entity2"
        if content_prompt:
            content_list = content_prompt.split(',')
            scene_name = content_list[0].strip()
            entities = [e.strip() for e in content_list[1:]]
        else:
            scene_name = "Unknown"
            entities = []
        
        # 处理 control_text 列表逻辑
        if isinstance(control_text, list):
            self.args.num_scenes = len(control_text)
            
        # 返回 Pipeline 所需的字典
        return {
            "content_prompt": content_prompt,
            "style_prompt": style_prompt,
            "adaptive_negative_prompt": adaptive_negative_prompt,
            "background_prompt": background_prompt,
            "control_text": control_text,
            "scene_name": scene_name,
            "entities": entities,
            "rotation_path": self.rotation_path,
            "inpainting_prompt": f"{style_prompt}, {content_prompt}"
        }

    def process_perception(self, multimodal_input=None):
        """
        加载初始视觉信号 (Start Image)
        """
        print(f"[Operator] Processing perception signals...")
        
        start_keyframe = None

        # 1. 优先处理传入的动态信号 (如果有)
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
                
                self.args.init_focal_length = width
                print(f"  - Visual: Set focal length to {width}")
                start_keyframe = image

            if "text" in multimodal_input:
                prompt = multimodal_input["text"].lower()
                if "fast" in prompt: self.args.forward_speed_multiplier = 0.1
                elif "slow" in prompt: self.args.forward_speed_multiplier = 0.02

        # 2. 兜底逻辑：直接从 args 读取图片路径 
        if start_keyframe is None:
            image_path = self.args.image_filepath
            
            if not image_path:
                raise ValueError("No image_filepath provided in args, and no dynamic input received.")
                
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at: {image_path}")
            
            print(f"  - Loading image from: {image_path}")
            start_keyframe = Image.open(image_path).convert('RGB').resize((512, 512))
            
            # 设置默认焦距
            self.args.init_focal_length = 512

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