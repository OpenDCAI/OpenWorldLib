# Rolling Forcing OpenWorldLib 迁移说明

本文档说明 Rolling Forcing 在 OpenWorldLib 中的迁移范围、手动安装依赖、手动下载模型和基本使用方式。下面所有安装和下载命令都需要由使用者自行执行；本次迁移代码不会自动安装 Python 包，也不会自动下载模型。

## 迁移范围

Rolling Forcing 的推理运行时代码已经迁移到 OpenWorldLib 包内：

- `src/openworldlib/synthesis/visual_generation/rolling_forcing/runtime/`

OpenWorldLib 对外使用的适配层包括：

- `src/openworldlib/operators/rolling_forcing_operator.py`
- `src/openworldlib/memories/visual_synthesis/rolling_forcing/rolling_forcing_memory.py`
- `src/openworldlib/synthesis/visual_generation/rolling_forcing/rolling_forcing_synthesis.py`
- `src/openworldlib/pipelines/rolling_forcing/pipeline_rolling_forcing.py`

最终的 `RollingForcingPipeline` 只依赖 OpenWorldLib 包内路径，不会从仓库根目录的 `RollingForcing/` 目录导入代码。

上游许可证原文保留在：

- `src/openworldlib/synthesis/visual_generation/rolling_forcing/runtime/LICENSE`

## Python 包依赖

上游 Rolling Forcing 推理环境使用 Python 3.10、PyTorch 2.5.1、CUDA 版本匹配的 `flash-attn`。请按本机 CUDA 驱动和 PyTorch 版本选择兼容命令。

基础推理依赖：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers==0.31.0" "transformers>=4.49.0" "tokenizers>=0.20.3" "accelerate>=1.1.1"
pip install omegaconf einops easydict ftfy regex imageio imageio-ffmpeg av==13.1.0
pip install opencv-python open_clip_torch sentencepiece "huggingface_hub[cli]" safetensors tqdm
pip install flash-attn --no-build-isolation
```

以下依赖主要用于上游训练、数据集或 Web Demo。OpenWorldLib 的 Rolling Forcing 推理路径默认不需要这些包，只有使用相关上游工具时才安装：

```bash
pip install lmdb wandb matplotlib scikit-image pycocotools dominate gradio>=4.44.0
pip install dashscope flask flask-socketio torchao tensorboard ninja packaging
pip install nvidia-pyindex nvidia-tensorrt pycuda onnx onnxruntime onnxscript onnxconverter_common
```

## 模型权重

所有模型都放在仓库根目录的 `checkpoints/` 下。默认路径如下：

- Wan2.1 基座模型：`checkpoints/Wan2.1-T2V-1.3B`
- Rolling Forcing DMD 权重：`checkpoints/RollingForcing/checkpoints/rolling_forcing_dmd.pt`

手动下载命令：

```bash
mkdir -p checkpoints/Wan2.1-T2V-1.3B checkpoints/RollingForcing

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir checkpoints/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False

huggingface-cli download TencentARC/RollingForcing \
  --include "checkpoints/rolling_forcing_dmd.pt" \
  --local-dir checkpoints/RollingForcing \
  --local-dir-use-symlinks False
```

推荐目录结构：

```text
checkpoints/Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model*.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/

checkpoints/RollingForcing/
  checkpoints/
    rolling_forcing_dmd.pt
```

如果权重放在其他路径，可以通过 `required_components` 显式指定：

```python
from openworldlib.pipelines.rolling_forcing.pipeline_rolling_forcing import RollingForcingPipeline

pipe = RollingForcingPipeline.from_pretrained(
    model_path="checkpoints/RollingForcing",
    required_components={
        "wan_model_path": "checkpoints/Wan2.1-T2V-1.3B",
        "generator_ckpt": "checkpoints/RollingForcing/checkpoints/rolling_forcing_dmd.pt",
    },
    device="cuda",
)
```

## 使用方式

单次文本生成：

```python
from openworldlib.pipelines.rolling_forcing.pipeline_rolling_forcing import RollingForcingPipeline

pipe = RollingForcingPipeline.from_pretrained(
    model_path="checkpoints/RollingForcing",
    required_components={
        "wan_model_path": "checkpoints/Wan2.1-T2V-1.3B",
    },
    device="cuda",
)

video = pipe(
    prompt="A cinematic tracking shot of a futuristic city street at sunrise.",
    num_frames=126,
    seed=0,
)
```

流式续写会把上一次生成的 latent 存入 `RollingForcingMemory`，下一次 `stream()` 会用最近的 latent 作为上下文继续生成：

```python
first = pipe.stream(
    prompt="A silver train crosses a desert at sunrise.",
    num_frames=126,
    seed=0,
    reset=True,
)

second = pipe.stream(
    prompt="The same train continues into a glowing futuristic city at dusk.",
    num_frames=126,
    seed=1,
)
```

输出结果是 `torch.uint8` tensor，形状为 `[B, T, H, W, C]`。默认帧率按上游示例使用 16 FPS。

## 测试入口

手动下载模型并安装依赖后，可以运行：

```bash
bash scripts/test_inference/test_nav_video_gen.sh rolling-forcing
bash scripts/test_stream/test_nav_video_gen.sh rolling-forcing
```

也可以直接运行：

```bash
python test/test_rolling_forcing.py
python test_stream/test_rolling_forcing_stream.py
```

如果要从分段 JSON 生成长视频，可以运行下面的示例。该脚本会把所有 JSON 片段合并为一个长视频 prompt，并只调用一次原生 Rolling Forcing 推理，让 rolling-window denoising、KV cache 和 attention sink 都由迁移出的 runtime 自己管理：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_rolling_forcing_from_json.py \
  --json_path path/to/case.json \
  --output_dir_name rolling_forcing_case
```

## 说明

- 默认迁移的是上游 `rolling_forcing_dmd.yaml` 对应的文本到视频推理路径。
- `RollingForcingMemory` 默认保留最近 24 个 latent context frames，用于后续 `stream()` 续写。这个值可通过 `RollingForcingPipeline.from_pretrained(max_latent_context_frames=...)` 调整。
- 默认配置的 context latent 数量需要是 `num_frame_per_block=3` 的整数倍；普通文本生成和基于 memory 的续写已经按这个约束处理。
- `examples/run_rolling_forcing_from_json.py` 不使用 OpenWorldLib 的跨调用 latent memory 拼接，而是使用一次原生 Rolling Forcing 长视频推理，以保持与上游 rolling forcing memory 机制一致。
- 默认 DMD 配置是文本到视频路径，不支持单帧图片作为初始 latent。`RollingForcingPipeline` 保留了 `images` 参数，但只有在使用支持单帧 context 的配置时才可用。
- 当前适配不会自动调用 Hugging Face 下载，也不会从原始 `RollingForcing/` 目录读取运行时代码。

## 参考链接

- Rolling Forcing 模型与上游说明：https://huggingface.co/TencentARC/RollingForcing
- Wan2.1-T2V-1.3B 基座模型：https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
