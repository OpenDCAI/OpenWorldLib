# ✨ **World Model 核心模态体系与高质量项目清单**

这份清单系统性地梳理了构建下一代 World Model 所需的 **11 个核心模态方向**及其对应的 **SOTA/高价值可训练项目**。

## ✅ **Part I: 核心模态体系 (11 大类)**

| 类别 | 模态方向 | 核心任务与技术 |
| :---: | :--- | :--- |
| **1** | 🌍 **3D / 4D Scene** | 3D 重建、神经渲染 (NeRF/GS)、动态场景建模 |
| **2** | 🎬 **Video Generation / Reasoning** | 视频生成、时序自监督学习、未来帧预测 |
| **3** | 🖼 **Image Reasoning / VLM** | 视觉语言模型 (VLM)、多图推理、通用图像生成 (Diffusion) |
| **4** | 🎧 **Audio / Speech** | 音频表示学习、音视频 (AV) 联合表示与融合 |
| **5** | 🕹 **Embodied / Robotics** | **具身世界模型 (World Model Policy)**、Model-Based 强化学习 |
| **6** | 📡 **Sensor Fusion / BEV** | LiDAR/Camera 融合、鸟瞰图 (BEV) 世界感知与建模 |
| **7** | 🧭 **Multimodal Representation** | 联合嵌入预测架构 (JEPA)、通用自监督 Representation |
| **8** | 🧍‍♂️ **Dynamic 3D/4D Human** | 动态人体姿态、形状与运动的 4D 重建/建模 |
| **9** | ⚙️ **Physics / Interaction** | 物理规律预测、因果推理、神经物理引擎 (NPE) |
| **10** | 👥 **Multi-Agent** | 多智能体环境建模、社会交互与策略学习 |
| **11** | 🌀 **Motion / Trajectory** | 人体运动生成、高维轨迹预测与建模 |

---

## ✅ **Part II: 高质量可训练项目 (Project Showcase)**

### 🌍 A. 3D / 4D Scene (核心世界表征)

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **Instant-NGP** | [https://github.com/NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | `train.py` | 🚀 基于 HashGrid 的**超快速 NeRF 3D 重建** |
| **Gaussian Splatting (3DGS)** | [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `train.py` | ✨ **高质量实时 3D 场景重建**与渲染 |
| **GaussianObject** | [https://github.com/GaussianObject/GaussianObject](https://github.com/GaussianObject/GaussianObject) | `train.py` | ⏳ **全流程动态 4D 场景与物体建模** |
| **Nerfstudio** | [https://github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | `ns-train` | 🛠 **最完善的 NeRF/GS 统一训练框架** |

### 🎬 B. Video Generation / Video Reasoning

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **VideoGPT** | [https://github.com/wilson1yan/VideoGPT](https://github.com/wilson1yan/VideoGPT) | `train.py` | 📼 基于 VQ-VAE 和 Transformer 的**视频生成** |
| **VideoMAE** | [https://github.com/MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) | `run_mae_pretraining.sh` | ⚙️ 基于 MAE 的**视频时空自监督学习** |
| **PredRNN / PredRNN++** | [https://github.com/thuml/predrnn-pytorch](https://github.com/thuml/predrnn-pytorch) | `train.py` | 🔭 **时序预测 SOTA：视频未来帧预测** |
| **Video-LLaVA** | [https://github.com/tkhan11/Video-LLaVA](https://github.com/tkhan11/Video-LLaVA) | `scripts/train/` | 💬 **视频 VLM：长时序 QA 与推理**能力 |

### 🖼 C. Image Reasoning / VLM

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **LLaVA** | [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | `train_mem.py` | 🔬 **VLM 指令微调与多模态全流程推理** |
| **Qwen-VL** | [https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL) | `scripts/finetune.sh` | 🇨🇳 强大的**视觉语言模型，中文支持突出** |
| **BLIP / BLIP2 (LAVIS)** | [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS) | `train_vqa.py` | 🧱 基于 Q-Former 的**多任务 VLM 训练框架** |

### 🎧 D. Audio / Audio-Visual

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **AudioMAE** | [https://github.com/facebookresearch/AudioMAE](https://github.com/facebookresearch/AudioMAE) | `train_clip.py` | 👂 基于 MAE 的**音频自监督表示学习** |
| **AV-HuBERT** | [https://github.com/facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) | `train.py` | 🔗 **大规模音视频 (AV) 联合自监督** |
| **AudioSet Tagger** | [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | `train.py` | 🔊 大规模**声音事件分类与识别** |

### 🕹 E. Embodied / RL World Model

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **DreamerV2** | [https://github.com/danijar/dreamerv2](https://github.com/danijar/dreamerv2) | `train.py` | 💡 基于 RSSM 的**隐状态世界模型与想象训练** |
| **PlaNet** | [https://github.com/danijar/dreamer](https://github.com/danijar/dreamer) | `scripts/train.sh` | 🗺 **Model-Based RL** 的开创性**规划与预测** |
| **D4RL Offline RL** | [https://github.com/rail-berkeley/d4rl](https://github.com/rail-berkeley/d4rl) | `SAC/BEAR 脚本` | 📊 大规模 **Offline RL 基准数据集** |

### 📡 F. Sensor Fusion / BEV

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **BEVFusion** | [https://github.com/mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | `tools/train.py` | 🚗 **LiDAR/Camera 多传感器 BEV 融合感知** |
| **OpenPCDet** | [https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | `tools/train.py` | 基础 **Lidar 3D 目标检测与感知基准** |
| **UniAD** | [https://github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) | `tools/train.py` | 🎯 **端到端：感知-预测-规划统一框架** |


### 🧭 G. Multimodal Representation

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **V-JEPA** | [https://github.com/facebookresearch/jepa](https://github.com/facebookresearch/jepa) | `launch_training.py` | 🧠 **非生成式**的**联合嵌入预测架构 (JEPA)** |
| **SEEM/SAM** | [https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) | `train.py` | ✂️ **提示驱动的通用图像分割 (SAM/SEEM)** |

### 🧍‍♂️ H. Dynamic 3D / 4D Human

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **Human4D** | [https://github.com/shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | `train.py` | 🕺 基于多视角数据的 **4D 人体动态重建** |

### ⚙️ I. Physics / Interaction

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **PhyDNet** | [https://github.com/vincent-leguen/PhyDNet](https://github.com/vincent-leguen/PhyDNet) | `train.py` | 🔭 **物理学启发的时序预测双分支网络** |
| **NPE** | [https://github.com/mbchang/dynamics](https://github.com/mbchang/dynamics) | `train.py` | 🔬 **神经物理引擎 (NPE)**：可控物理模拟学习 |

### 👥 J. Multi-Agent World Model

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **MineRL** | [https://github.com/minerllabs/minerl](https://github.com/minerllabs/minerl) | `train_bc.py` | 🕹 **开放世界探索**与模仿学习基准 |
| **MeltingPot** | [https://github.com/deepmind/meltingpot](https://github.com/deepmind/meltingpot) | `python/run_training.py` | 🤝 **多智能体社会交互**与合作/竞争学习 |

### 🌀 K. Motion / Trajectory Modeling

| 项目 | 代码库链接 | 训练脚本 | **World Model 核心亮点 (精炼)** |
| :--- | :--- | :--- | :--- |
| **MotionGPT** | [https://github.com/OpenMotionLab/MotionGPT](https://github.com/OpenMotionLab/MotionGPT) | `scripts/train.sh` | 🚶‍♂️ 基于 Transformer 的**文本驱动动作生成** |
| **MotionDiffuse** | [https://github.com/mingyuan-zhang/MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse) | `src/train.py` | 💫 基于扩散模型的**高质量人体运动生成 SOTA** |