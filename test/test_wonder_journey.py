from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import sys

# 获取 test.py 所在的目录
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 sceneflow-main)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))

# 将项目根目录加入 sys.path，确保能 import src
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f"Project Root: {PROJECT_ROOT}")

from src.sceneflow.pipelines.wonder_journey.pipeline_wonder_journey import WonderJourneyPipeline, WonderJourneyArgs

def make_abs(path):
    if os.path.isabs(path): return path
    return os.path.join(PROJECT_ROOT, path)

def main():
    parser = ArgumentParser()
    # 只需要路径参数了，config 参数可以去掉了（或者留着不用）
    parser.add_argument("--oneformer_path", default="oneformer_chk")
    parser.add_argument("--sd_path", default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--depth_model_path", default="dpt_beit_large_512.pt")
    # 额外加一个图片路径参数，方便测试
    parser.add_argument("--image_path", default="data/test_case1/ref_image.png")
    
    args_cmd = parser.parse_args()
    
    # 1. 初始化 Args
    args = WonderJourneyArgs()

    # 2. 【关键】在这里填入原本 YAML 里的数据
    args.example_name = "village"
    args.runs_dir = make_abs(f"data/test_wonder_journey/56_{args.example_name}")
    
    # ======== 🔴 这里是原本 examples.yaml 的内容 ========
    args.content_prompt = "Mountain Pass Entrance, rocky path, wooden signpost, pine trees"
    args.style_prompt = "Style: DSLR 35mm landscape"
    args.background_prompt = "Passing beyond the quaint village, a winding path leads travelers towards the foot of the mountains."
    
    # 绝对路径图片
    args.image_filepath = make_abs(args_cmd.image_path)
    # 下面这个放到operator里面input的地方：
    # input_image = Image.open(image_path).convert('RGB').resize((512, 512))
    # =================================================
    
    # 场景参数
    args.num_scenes = 1
    args.num_keyframes = 2
    args.rotation_path = [0, 0, 0, 1, 1, 0, 0, 0]
    args.rotation_range = 0.45
    args.camera_speed_multiplier_rotation = 0.2
    
    # 模型路径
    args.oneformer_path = make_abs(args_cmd.oneformer_path)
    args.sd_path = make_abs(args_cmd.sd_path)
    args.depth_model_path = make_abs(args_cmd.depth_model_path)
    
    # API & GPT
    args.api_key = "sk-Wnv0VFqre5WleXvBGVmr7UqtwBBvuI5p5ZT8SujVTtldvUsZ"
    args.api_base = "https://sg.uiuiapi.com/v1"
    args.use_gpt = False  # 暂时关闭 GPT
    
    # 生成参数
    args.frames = 10
    args.inpainting_resolution_gen = 1024
    args.finetune_decoder_interp = False
    args.seed = -1
    
    print(f"Running Example: {args.example_name}")
    print(f"Image Path: {args.image_filepath}")

    # 3. 初始化 & 运行
    pipeline = WonderJourneyPipeline.from_pretrained(args)
    operator = pipeline.create_operator(args)
    pipeline(operator)

if __name__ == "__main__":
    main()