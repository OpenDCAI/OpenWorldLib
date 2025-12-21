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

from sceneflow.pipelines.wonder_journey.pipeline_wonder_journey import WonderJourneyPipeline

def make_abs(path):
    """
    辅助函数：如果路径是相对的，将其转换为基于项目根目录的绝对路径
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="examples/wonder_journey/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="examples/wonder_journey/village.yaml",
        help="Example config path e.g. config/village.yaml"
    )
    # 增加参数以支持 HuggingFace 模型路径输入
    parser.add_argument(
        "--oneformer_path",
        default="oneformer_chk",
        help="Path to OneFormer model (local or HF repo_id)"
    )
    parser.add_argument(
        "--sd_path",
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to Stable Diffusion model (local or HF repo_id)"
    )
    parser.add_argument(
        "--depth_model_path",
        default="dpt_beit_large_512.pt",
        help="Path to depth model checkpoint"
    )
    
    args = parser.parse_args()
    
    # 1. 转换所有路径为绝对路径 (这是解决报错的关键！！！)
    abs_base_config = make_abs(args.base_config)
    abs_example_config = make_abs(args.example_config)
    abs_oneformer_path = make_abs(args.oneformer_path)
    abs_sd_path = make_abs(args.sd_path)
    abs_depth_model_path = make_abs(args.depth_model_path)

    print(f"Loading OneFormer from: {abs_oneformer_path}")
    print(f"Loading SD from: {abs_sd_path}")

    # 2. 加载配置
    base_config = OmegaConf.load(abs_base_config)
    example_config = OmegaConf.load(abs_example_config)
    config = OmegaConf.merge(base_config, example_config)
# ======== 🔴 新增修复代码 开始 ========
    # 从 example_config 的路径中提取文件名作为 example_name
    # # 例如：config/village.yaml -> village
    # import os
    # example_filename = os.path.basename(args.example_config) # 得到 "village.yaml"
    # example_name = os.path.splitext(example_filename)[0]     # 得到 "village"
    
    # # 允许修改 config 结构（OmegaConf 默认可能是锁定的）
    # OmegaConf.set_struct(config, False)
    
    # # 将名字注入配置中
    # config["example_name"] = example_name
    # print(f"Set example_name to: {example_name}")
    # ======== 🟢 新增修复代码 结束 ========
    # 3. 传入绝对路径初始化 Pipeline
    pipeline = WonderJourneyPipeline.from_pretrained(
        config=config,
        oneformer_path=abs_oneformer_path,
        sd_path=abs_sd_path,
        depth_model_path=abs_depth_model_path
    )

    operator = pipeline.create_operator(config)
    
    pipeline(operator)

if __name__ == "__main__":
    main()