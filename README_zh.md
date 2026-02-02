<div align="center" markdown="1">

<img src="assets/logo.png" alt="sceneflow_logo" width="80%" />

### A Unified Framework for Advanced World Models

<a href="https://github.com/OpenDCAI/SceneFlow"><img alt="Build" src="https://img.shields.io/github/stars/OpenDCAI/SceneFlow"></a>

</div>

### Table of Contents

- [A Unified Framework for Advanced World Models](#a-unified-framework-for-advanced-world-models)
- [Table of Contents](#table-of-contents)
- [Features](#features)
  - [仓库目标](#仓库目标)
  - [支持任务](#支持任务)
- [Quick Start](#quick-start)
  - [Install](#install)
  - [Inference](#inference)
- [Structure](#structure)
- [Planning](#planning)
- [For Developers](#for-developers)
- [Citation](#citation)


### Features
#### 仓库目标
SceneFlow 的主要目标包括以下内容：

- 建立一套统一且规范的 **world model framework**，使现有世界模型相关代码的调用更加规范与一致；
- 集成开源的世界模型相关研究成果，并系统整理收录相关论文，供研究者参考与使用。

#### 支持任务
SceneFlow 涵盖以下与 **World Model** 相关的研究方向：

| 任务类别 | 子任务 | 代表性方法/模型 |
| :--- | :--- | :--- |
| **Video Generation** | Navigation Generation | lingbot, matrixgame, hunyuan-worldplay, genie3 等 |
| | Long Video Generation | sora-2, veo-3, wan 等 |
| **3D Scene Generation** | 3D 场景生成 | flash-world, vggt 等 |
| **Reasoning** | VQA（视觉问答） | spatialVLM, omnivinci 等具备世界理解的 VLM 模型 |
| | VLA（视觉语言行动） | pi-0, pi-0.5, giga-brain 等方法 |

### Quick Start
#### Install
首先安装torch以及flash_attn
```bash
pip install torch==2.5.1 torchvision torchaudio
pip install "flash-attn==2.5.9.post1" --no-build-isolation
```
接着安装不同版本的sceneflow，进入`./SceneFlow`，下面分别是高版本transformers安装`[transformers_high]`以及低版本transformers安装`[transformers_low]`，推荐安装高版本transformers:
```bash
pip install -e ".[transformers_high]"
```
下面是低版本transformers环境安装
```bash
pip install -e ".[transformers_low]"
```

#### Inference

### Structure


### Planning
- 我们在[awesome_world_models.md](docs/awesome_world_model.md)中记录了最前沿的world models相关的研究，同时我们欢迎大家在这里提供一些有价值的研究。
- 我们在[planning.md](docs/planning.md)中记录了我们后续的**训练**以及**优化**计划。

### For Developers
我们欢迎各位开发者共同参与，帮助完善 **SceneFlow** 这一统一世界模型仓库。推荐采用 **Vibe Coding** 方式进行代码提交，相关提示词可参考 `tools/vibe_code/prompts` 目录下的内容。期待你的贡献！


### Citation
