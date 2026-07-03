[English](README.md) | [简体中文](README_zh_CN.md)

## 🤗 快速开始

### 环境安装

推荐使用 CUDA 12.8 进行安装。

```bash
# 1. 克隆仓库
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0
cd HY-World-2.0

# 2. 创建 conda 环境
conda create -n hyworld2-pano python=3.10
conda activate hyworld2-pano

# 3. 安装 PyTorch（CUDA 11.8）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
pip install -r requirements.txt
```

## 代码使用 — HY-Pano 2.0

HY-Pano 2.0 提供两种后端。使用 **HunyuanImage-3**（`pipeline.py`）可获得完整的推理流程（含思维链重描述），使用 **Qwen-Image-Edit**（`pipeline_with_qwen_image.py`）则是基于 diffusers 的轻量后端。

### 后端一 — HunyuanImage-3

#### Python API

```python
from pipeline import HunyuanPanoPipeline

# 从 HuggingFace 下载（默认）
pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-World-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

**自定义 prompt 和随机种子：**

```python
pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-World-2.0')
output = pipeline(
    'input.png',
    prompt='Expand this image to a 360-degree equirectangular panorama. Sunny day.',
    seed=42,
)
output.save('output_panorama.png')
```

**从本地路径加载：**

```python
pipeline = HunyuanPanoPipeline.from_pretrained('/path/to/HY-Pano-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

#### 命令行

```bash
# 基础全景图生成
python pipeline.py --image input.png

# 指定 prompt 和输出路径
python pipeline.py --image input.png \
    --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \
    --save output_panorama.png

# 自定义推理步数和任务类型
python pipeline.py --image input.png \
    --diff-infer-steps 50 --bot-task think_recaption --use-system-prompt en_unified

# 固定随机种子以复现结果
python pipeline.py --image input.png --seed 42 --reproduce

# 使用 Taylor Cache 加速采样
python pipeline.py --image input.png \
    --use-taylor-cache --taylor-cache-interval 5 --taylor-cache-order 2
```

---

### 后端二 — Qwen-Image-Edit

#### Python API

```python
from pipeline_with_qwen_image import HunyuanPanoPipeline

# 从 HuggingFace 下载（默认）
pipeline = HunyuanPanoPipeline.from_pretrained(
    lora_path='tencent/HY-World-2.0', lora_subfolder='HY-Pano-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

**自定义 prompt 和随机种子：**

```python
pipeline = HunyuanPanoPipeline.from_pretrained(
    lora_path='tencent/HY-World-2.0', lora_subfolder='HY-Pano-2.0')
output = pipeline(
    'input.png',
    prompt='A sunny outdoor scene.',
    seed=42,
)
output.save('output_panorama.png')
```

**从本地路径加载自定义 LoRA：**

```python
pipeline = HunyuanPanoPipeline.from_pretrained(
    '/path/to/base_model',
    lora_path='/path/to/lora',
    lora_subfolder='',
)
output = pipeline('input.png')
output.save('output_panorama.png')
```

#### 命令行

```bash
# 基础全景图生成
python pipeline_with_qwen_image.py --image input.png

# 指定 prompt、随机种子和输出路径
python pipeline_with_qwen_image.py --image input.png \
    --prompt "A sunny outdoor scene." --seed 42 --save output_panorama.png
```
