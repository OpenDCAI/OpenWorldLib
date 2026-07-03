[English](README.md) | [简体中文](README_zh.md)

## 🤗 Get Started

### Install Requirements

We recommend CUDA 12.8 for installation.

```bash
# 1. Clone the repository
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0
cd HY-World-2.0

# 2. Create conda environment
conda create -n hyworld2-pano python=3.10
conda activate hyworld2-pano

# 3. Install PyTorch (CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt
```

## Code Usage — HY-Pano 2.0

HY-Pano 2.0 offers two backends. Use **HunyuanImage-3** (`pipeline.py`) for the full reasoning pipeline with chain-of-thought recaptioning, or **Qwen-Image-Edit** (`pipeline_with_qwen_image.py`) for a lighter diffusers-based backend.

### Backend 1 — HunyuanImage-3

#### Python API

```python
from pipeline import HunyuanPanoPipeline

# Download from HuggingFace (default)
pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-World-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

**With custom prompt and seed:**

```python
pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-World-2.0')
output = pipeline(
    'input.png',
    prompt='Expand this image to a 360-degree equirectangular panorama. Sunny day.',
    seed=42,
)
output.save('output_panorama.png')
```

**From a local path:**

```python
pipeline = HunyuanPanoPipeline.from_pretrained('/path/to/HY-Pano-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

#### CLI

```bash
# Basic panorama generation
python pipeline.py --image input.png

# Specify prompt and output path
python pipeline.py --image input.png \
    --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \
    --save output_panorama.png

# Customize inference steps and task type
python pipeline.py --image input.png \
    --diff-infer-steps 50 --bot-task think_recaption --use-system-prompt en_unified

# Reproducible generation with a fixed seed
python pipeline.py --image input.png --seed 42 --reproduce

# Use Taylor Cache to speed up sampling
python pipeline.py --image input.png \
    --use-taylor-cache --taylor-cache-interval 5 --taylor-cache-order 2
```

---

### Backend 2 — Qwen-Image-Edit

#### Python API

```python
from pipeline_with_qwen_image import HunyuanPanoPipeline

# Download from HuggingFace (default)
pipeline = HunyuanPanoPipeline.from_pretrained(
    lora_path='tencent/HY-World-2.0', lora_subfolder='HY-Pano-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

**With custom prompt and seed:**

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

**From a local path with a custom LoRA:**

```python
pipeline = HunyuanPanoPipeline.from_pretrained(
    '/path/to/base_model',
    lora_path='/path/to/lora',
    lora_subfolder='',
)
output = pipeline('input.png')
output.save('output_panorama.png')
```

#### CLI

```bash
# Basic panorama generation
python pipeline_with_qwen_image.py --image input.png

# Specify prompt, seed and output path
python pipeline_with_qwen_image.py --image input.png \
    --prompt "A sunny outdoor scene." --seed 42 --save output_panorama.png
```
