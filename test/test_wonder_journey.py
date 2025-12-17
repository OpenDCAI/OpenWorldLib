from argparse import ArgumentParser
from omegaconf import OmegaConf

from pipeline import WonderJourneyPipeline

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="../../../../examples/wonder_journey/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="../../../../examples/wonder_journey/village.yaml",
        help="Example config path e.g. config/village.yaml"
    )
    # 增加参数以支持 HuggingFace 模型路径输入
    parser.add_argument(
        "--oneformer_path",
        default="./oneformer_chk",
        help="Path to OneFormer model (local or HF repo_id)"
    )
    parser.add_argument(
        "--sd_path",
        default="./stabilityai/stable-diffusion-2-inpainting",
        help="Path to Stable Diffusion model (local or HF repo_id)"
    )
    parser.add_argument(
        "--depth_model_path",
        default="dpt_beit_large_512.pt",
        help="Path to depth model checkpoint"
    )
    
    args = parser.parse_args()
    
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    pipeline = WonderJourneyPipeline.from_pretrained(
        config=config,
        oneformer_path=args.oneformer_path,
        sd_path=args.sd_path,
        depth_model_path=args.depth_model_path
    )

    operator = pipeline.create_operator(config)
    
    pipeline(operator)

if __name__ == "__main__":
    main()