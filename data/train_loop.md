# Multimodal Training Code

> 仅保留**训练代码设计中真正会反复用到的抽象结构**：
>
> * Dataset 实际返回什么
> * forward / loss 在约束什么
> * iteration 的最小语义单位
> * world model vs reasoning / alignment 的本质差异
>
> 并按 **模态 / 世界类型** 统一组织

---

## Part I · 3D Scene World Models（空间世界建模）

> **核心特征**：
>
> * 有显式 world state
> * 多视角 / 几何一致性
> * forward = observation / rendering

---

### 1. [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) / [Gaussian Object](https://github.com/GaussianObject/GaussianObject)

**模态定位**：3D Scene · Static World Model（Point-based）

---

#### 1.1 State / Condition

```text
State     = Global Gaussian Scene
            (xyz, scale, rotation, opacity, SH)
Condition = Camera
```

---

#### 1.2 Dataset 返回（≈ `__getitem__`）

Dataset ≈ **Camera Pool**，并不返回 image batch：

```python
{
  image: GT RGB,
  mask / depth: optional,
  pose: (R, T),
  intrinsics: fx, fy, cx, cy,
  image_size: (H, W)
}
```

---

#### 1.3 Training Loop（最小语义单位）

```text
sample camera
→ render(camera, gaussians)
→ image loss
→ backward
→ densify / prune
→ optimizer.step
```

* 无 batch / 无 epoch
* **1 iteration = 1 camera view**

---

#### 1.4 forward / loss

```text
Camera + Gaussian Scene
→ Rasterization
→ RGB / depth / alpha
```

```text
L = L1 + (1 - SSIM) (+ depth / mask)
```

---

### 2. NeRF / [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

**模态定位**：3D Scene · Static World Model（Ray-based）

---

#### 2.1 State / Condition

```text
State     = Neural Radiance Field (MLP)
Condition = Camera + Rays
```

---

#### 2.2 Dataset / DataManager 返回

Dataset 提供 images + cameras，但真正送入模型的是：

```python
RayBundle = {
  origins: (N, 3),
  directions: (N, 3),
  pixel_area: (N, 1),
  camera_indices: (N, 1)
}

gt_rgb: (N, 3)
```

---

#### 2.3 Training Loop

```text
sample camera
→ sample rays
→ query MLP (σ, c)
→ volume rendering
→ RGB loss
→ optimizer.step
```

* 有 batch（ray batch）
* 无 epoch
* **1 iteration = 1 ray batch**

---

#### 2.4 forward / loss

```text
RayBundle → MLP → Volume Rendering → RGB / Depth
```

```text
L = L_rgb + L_regularization
```

---

#### 2.5 Gaussian vs NeRF（抽象对比）

| 维度 | Gaussian      | NeRF            |
| ---- | ------------- | --------------- |
| 表示 | 离散点        | 连续函数        |
| 渲染 | Rasterization | Ray Integration |
| 采样 | Camera        | RayBundle       |

---

## Part II · Vision → Language Reasoning（视觉理解）

> **核心特征**：
>
> * 无显式 world state
> * 视觉作为 condition
> * loss = token-level CE

---

### 3. [LLaVA](https://github.com/haotian-liu/LLaVA)

**模态定位**：Image Reasoning · Vision-conditioned LLM

---

#### 3.1 Dataset 返回

```python
{
  input_ids: [T],
  labels: [T],        # mask user
  image: [3, H, W]
}
```

---

#### 3.2 Training Loop（最小单位）

```text
image + text
→ multimodal forward
→ token CE
→ optimizer.step
```

* 有 batch / 有 epoch
* **1 iteration = 1 text-image batch**

---

#### 3.3 forward / loss

```text
image → vision encoder → projector
→ image tokens + text tokens
→ LLM causal forward
```

```text
Loss = CrossEntropy (assistant tokens only)
```

---

### 4. [Video-LLaVA](https://github.com/tkhan11/Video-LLaVA)

**模态定位**：Video Reasoning · Video-conditioned LLM

---

#### 4.1 Dataset 返回

```python
{
  prompt,
  video: (T, H, W, 3)
}
```

---

#### 4.2 forward / loss

```text
video frames → vision encoder
→ video tokens + text tokens
→ LLM causal forward
→ token CE
```

---

### 5. [QwenVL](https://github.com/QwenLM/Qwen-VL)

**模态定位**：Text / Multimodal Dialogue Reasoning

---

#### 5.1 Dataset 返回

```python
{
  input_ids,
  attention_mask,
  labels   # mask system / user
}
```

---

#### 5.2 loss

```text
Loss = CrossEntropy (assistant tokens)
```

---

## Part III · Audio World & Cross-Modal Perception（声音）

---

### 6. [ThinkSound](https://github.com/liuhuadai/ThinkSound)

**模态定位**：Audio-Centric Dynamic World Model

---

#### 6.1 State / Observation

```text
State       = audio latent
Observation = text / video / sync signal
Dynamics    = diffusion
```

---

#### 6.2 Dataset 返回

```python
(audio_or_latent, info)

info = {
  timestamps,
  padding_mask,
  metaclip_features,
  video_exist
}
```

---

#### 6.3 forward / loss

```text
(audio_latent_t, condition)
→ diffusion
→ predict x_0 / x_{t-1}
```

---

### 7. [CLAP / Audio–Text Contrastive](https://github.com/LAION-AI/CLAP)

**模态定位**：Audio–Text Perception Alignment

---

#### 7.1 Dataset 返回

```python
{
  waveform,
  mel_fusion,
  text
}
```

---

#### 7.2 forward / loss

```text
audio → encoder → z_a
text  → encoder → z_t
```

```text
Loss = InfoNCE
```

---

## Part IV · Self-Supervised Perception Models（MAE 系）

---

### 8. [AudioMAE](https://github.com/facebookresearch/AudioMAE)

**模态定位**：Audio Representation · Self-Supervised

---

#### 8.1 State / Condition

```text
State     = Audio Encoder Weights
Condition = Spectrogram (fbank)
```

---

#### 8.2 Dataset 返回

```python
{
  fbank: (1, F, T),
  label_indices,
  wav_filename
}
```

---

#### 8.3 Training Loop

```text
fbank batch
→ masking
→ encoder
→ reconstruction / classification
→ loss
→ optimizer.step
```

* 有 batch / 有 epoch

---

#### 8.4 forward / loss

**预训练（MAE）**

```text
masked fbank → encoder → reconstruct masked patches
Loss = MSE / L1
```

**微调（分类）**

```text
fbank → encoder → cls head
Loss = CE / BCE
```

---

### 9. [VideoMAE](https://github.com/MCG-NJU/VideoMAE)

**模态定位**：Video Representation · Self-Supervised

---

#### 9.1 State / Condition

```text
State     = VideoMAE Weights
Condition = Masked Video Clips
```

---

#### 9.2 Dataset 返回

```python
{
  videos: (C, T, H, W),
  bool_masked_pos
}
```

---

#### 9.3 Training Loop

```text
video batch
→ tube masking
→ encoder (visible patches)
→ decoder (masked patches)
→ MSE loss
→ optimizer.step
```

---

#### 9.4 forward / loss

```text
Video → Patch → Mask → Encoder → Decoder
→ Predicted Patches
```

```text
L = MSE(predicted, gt_masked_patches)
```

---

## Part V · 总体抽象对比

| 系统        | 模态定位        | State          | Dataset 核心 | Loss        | World Model |
| ----------- | --------------- | -------------- | ------------ | ----------- | ----------- |
| Gaussian    | 3D Scene        | Gaussians      | Camera       | pixel       | ✅          |
| NeRF        | 3D Scene        | Radiance Field | RayBundle    | pixel       | ✅          |
| LLaVA       | Image Reasoning | ❌             | image+text   | token CE    | ❌          |
| Video-LLaVA | Video Reasoning | ❌             | video+text   | token CE    | ❌          |
| Qwen SFT    | Text Reasoning  | ❌             | text         | token CE    | ❌          |
| ThinkSound  | Audio World     | audio latent   | latent+info  | diffusion   | ✅          |
| CLAP        | Audio–Text     | ❌             | audio+text   | contrastive | ❌          |
| AudioMAE    | Audio Rep.      | encoder        | fbank        | recon / cls | ❌          |
| VideoMAE    | Video Rep.      | encoder        | video+mask   | recon       | ❌          |

---
