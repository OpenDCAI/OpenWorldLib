import os
import torch
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline
from sceneflow.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.parallel_states import initialize_parallel_state
from sceneflow.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.infer_state import initialize_infer_state
from sceneflow.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.generate import save_video

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parallel_dims = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

image_path = "./data/test_case/test_image_seq_case1/image_0001.jpg"
output_path = "./outputs/"
os.makedirs(output_path, exist_ok=True)

if not os.path.exists(image_path):
    print(f"❌ 图片文件不存在: {image_path}")
    exit(1)

class Args:
    def __init__(self):
        self.pose = "w-10, right-10, d-11"
        self.prompt = 'A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone. The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
        self.negative_prompt = ""
        self.resolution = "480p"
        self.model_path = "/home/tiger/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
        self.action_ckpt = "/home/tiger/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/228d059b54d59a028ccc611d4d5efa06541523de/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
        self.aspect_ratio = "16:9"
        self.num_inference_steps = 4
        self.video_length = 125
        self.sr = False
        self.save_pre_sr_video = False
        self.rewrite = False
        self.offloading = True
        self.group_offloading = None
        self.dtype = "bf16"
        self.seed = 1
        self.image_path = image_path
        self.output_path = output_path
        self.enable_torch_compile = False
        self.few_step = True
        self.model_type = "ar"
        self.height = 480
        self.width = 832
        self.with_ui = False
        self.use_sageattn = False
        self.sage_blocks_range = "0-53"
        self.use_vae_parallel = False
        self.use_fp8_gemm = False
        self.quant_type = "fp8-per-block"
        self.include_patterns = "double_blocks"

args = Args()

initialize_infer_state(args)

task = "i2v" if args.image_path else "t2v"
transformer_version = f"{args.resolution}_{task}"
assert transformer_version == "480p_i2v"

if args.dtype == "bf16":
    transformer_dtype = torch.bfloat16
elif args.dtype == "fp32":
    transformer_dtype = torch.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}")

print("🚀 正在加载模型...")
pipeline = HunyuanWorldPlayPipeline.from_pretrained(
    synthesis_path=args.model_path,
    transformer_version=transformer_version,
    enable_offloading=args.offloading,
    enable_group_offloading=args.group_offloading,
    create_sr_pipeline=args.sr,
    force_sparse_attn=False,
    transformer_dtype=transformer_dtype,
    action_ckpt=args.action_ckpt,
)

print("🎬 正在生成视频...")
out = pipeline(
    prompt=args.prompt,
    reference_image=args.image_path,
    pose=args.pose,
    aspect_ratio=args.aspect_ratio,
    video_length=args.video_length,
    num_inference_steps=args.num_inference_steps,
    negative_prompt=args.negative_prompt,
    seed=args.seed,
    output_type="pt",
    prompt_rewrite=args.rewrite,
    enable_sr=args.sr,
    sr_num_inference_steps=None,
    return_pre_sr_video=args.save_pre_sr_video,
    few_step=args.few_step,
    chunk_latent_frames=4 if args.model_type == "ar" else 16,
    model_type=args.model_type,
    user_height=args.height,
    user_width=args.width,
)

print("💾 正在保存视频...")
save_video_path = os.path.join(args.output_path, "gen.mp4")
save_video(out.videos, save_video_path)

if args.sr and hasattr(out, "sr_videos") and out.sr_videos is not None:
    save_video_sr_path = os.path.join(args.output_path, "gen_sr.mp4")
    save_video(out.sr_videos, save_video_sr_path)
    print(f"超分辨率视频已保存到: {save_video_sr_path}")

print("✅ 视频生成完成！")
print(f"结果已保存到: {save_video_path}")
