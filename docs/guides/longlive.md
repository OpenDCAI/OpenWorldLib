# LongLive 使用指南

本文档说明 LongLive 在 OpenWorldLib 中的代码位置、手动安装依赖、手动下载模型，以及基本使用方式。下面列出的安装和下载命令需要由使用者自行执行，OpenWorldLib 不会自动安装 Python 包或下载模型。

## 代码位置

LongLive 原始推理运行时已经迁移到 OpenWorldLib 内部：

- `src/openworldlib/synthesis/visual_generation/longlive/runtime/`

OpenWorldLib 对外使用的适配层包括：

- `src/openworldlib/operators/longlive_operator.py`
- `src/openworldlib/memories/visual_synthesis/longlive/longlive_memory.py`
- `src/openworldlib/synthesis/visual_generation/longlive/longlive_synthesis.py`
- `src/openworldlib/pipelines/longlive/pipeline_longlive.py`

最终的 `LongLivePipeline` 只依赖 OpenWorldLib 包内路径，不会从仓库根目录的 `LongLive/` 目录导入代码。

## Python 包依赖

LongLive 上游测试环境为 Python 3.10、CUDA 12.4、PyTorch 2.5.0、`flash-attn==2.7.4.post1`。请根据本机 CUDA 和 PyTorch 版本选择匹配的安装命令。

手动安装命令：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers==0.31.0" "transformers>=4.49.0" "tokenizers>=0.20.3" "accelerate>=1.1.1"
pip install omegaconf einops easydict ftfy regex imageio imageio-ffmpeg av==13.1.0
pip install open_clip_torch peft "huggingface_hub[cli]" sentencepiece datasets lmdb
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

以下依赖主要用于 LongLive 上游训练或 prompt-extension 工具。OpenWorldLib 的 LongLive 推理路径默认不需要这些包，只有使用相关上游工具时才需要安装：

```bash
pip install dashscope wandb pycocotools matplotlib scikit-image dominate flask flask-socketio torchao
pip install nvidia-pyindex nvidia-tensorrt pycuda onnx onnxruntime onnxscript onnxconverter_common
pip install git+https://github.com/openai/CLIP.git
```

## 模型权重

LongLive 适配会直接复用 OpenWorldLib 已有的 Wan2.1 基座模型目录：

- `checkpoints/Wan2.1-T2V-1.3B`

不要再把 Wan2.1 基座模型下载到 `checkpoints/LongLive/Wan2.1-T2V-1.3B`。`checkpoints/LongLive/` 只用于存放 LongLive 自己的 base/LoRA 权重和 prompts。

手动下载命令：

```bash
mkdir -p checkpoints/LongLive
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --local-dir checkpoints/LongLive
```

推荐目录结构：

```text
checkpoints/Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model*.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/

checkpoints/LongLive/
  models/
    longlive_base.pt
    lora.pt
  prompts/
    interactive_example.jsonl
```

如果 Hugging Face snapshot 保留了上游嵌套目录，加载器也兼容下面这种结构：

```text
checkpoints/LongLive/
  longlive_models/models/longlive_base.pt
  longlive_models/models/lora.pt
```

如果你的模型文件放在其他位置，可以通过 `required_components` 显式传入：

```python
pipe = LongLivePipeline.from_pretrained(
    model_path="checkpoints/LongLive",
    required_components={
        "wan_model_path": "checkpoints/Wan2.1-T2V-1.3B",
        "generator_ckpt": "checkpoints/LongLive/models/longlive_base.pt",
        "lora_ckpt": "checkpoints/LongLive/models/lora.pt",
    },
    device="cuda",
)
```

默认情况下，pipeline 会按照 LongLive 推理配置启用 LoRA。如果本地 snapshot 中没有 `lora.pt`，可以将 LoRA 权重放到上述路径之一，或者加载时关闭 LoRA：

```python
pipe = LongLivePipeline.from_pretrained(
    model_path="checkpoints/LongLive",
    device="cuda",
    use_lora=False,
)
```

## 使用方式

单 prompt 生成：

```python
from openworldlib.pipelines.longlive.pipeline_longlive import LongLivePipeline

pipe = LongLivePipeline.from_pretrained(
    model_path="checkpoints/LongLive",
    device="cuda",
)
video = pipe(
    prompt="A cinematic shot of a robot walking through a rainy neon street.",
    num_frames=120,
    seed=0,
)
```

多 prompt 交互式切换：

```python
video = pipe(
    prompts=[
        "A calm forest path at sunrise.",
        "The same path slowly becomes covered in snow.",
        "The scene transitions into a futuristic neon city.",
    ],
    switch_frame_indices=[40, 80],
    num_frames=120,
    seed=1,
)
```

输出结果是 `torch.uint8` tensor，形状为 `[B, T, H, W, C]`。如需导出 mp4，可以使用 `examples.pipeline_infer_mapping.infer_longlive_pipeline` 中已经注册的辅助函数。

## 说明

- LongLive 使用 causal attention、KV cache，以及 prompt 切换时的 KV recache。OpenWorldLib 的 memory 模块会记录每次 stream 调用生成的视频，并可选记录 latents。
- 默认 latent shape 为 `16 x 60 x 104`，对应上游 Wan2.1-T2V-1.3B 的 480p 设置。
- 上游说明 LongLive 推理主要面向 NVIDIA A100/H100 级别 GPU，建议显存至少 40 GB。

## 参考链接

- LongLive 模型、安装说明、许可证和 `models/longlive_base.pt`：https://huggingface.co/Efficient-Large-Model/LongLive-1.3B
- Wan2.1-T2V-1.3B 基座模型：https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
