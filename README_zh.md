<div align="center" markdown="1">

<img src="https://github.com/user-attachments/assets/1935c89a-76cb-4edc-a6ac-0c3658d347f6" alt="openworldlib_logo" width="90%" />

#### 欢迎加入我们的开源世界模型项目！ <!-- omit in toc -->
---

<a href="https://github.com/OpenDCAI/OpenWorldLib"><img alt="Build" src="https://img.shields.io/github/stars/OpenDCAI/OpenWorldLib"></a> <!-- License --> <a href="https://github.com/OpenDCAI/OpenWorldLib/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/OpenDCAI/OpenWorldLib"></a> <!-- GitHub Issues --> <a href="https://github.com/OpenDCAI/OpenWorldLib/issues"><img alt="Issues" src="https://img.shields.io/github/issues/OpenDCAI/OpenWorldLib"></a>
<img alt="Report" src="https://img.shields.io/badge/📄 Technical Report-Coming Soon-lightgrey">
<!-- <a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b?logo=arxiv&logoColor=white"></a> -->

[English](README.md) | [中文](README_zh.md)

扩展仓库：[[三维生成]](https://github.com/zengbohan0217/OpenWorldLib-extension-3D) | [[VLA]]() | [[仿真环境]](https://github.com/YF0224/OpenWorldLib-extension-Simulator)

<!-- Extension code link: -->

</div>

我们将世界模型定义为：**一种以感知为核心、具备交互与长期记忆能力的模型或框架，用于理解和预测复杂世界。** 因此，🎓 *多模态理解*、🤖 *视觉动作预测* 和 🖼️ *视觉生成* 都是世界模型需要完成的子任务。

我们热烈欢迎研究者在 Issues 区分享对该框架的看法或对世界模型的思考。同时也希望您能通过 Pull Request 将有价值的世界模型相关方法提交到我们的框架中，或将其记录并提交到 [[awesome_world_models]](docs/awesome_world_model.md)。欢迎给我们的仓库点个 🌟 Star，以关注 OpenWorldLib 的最新进展！

<!-- the demonstration demo insert here -->

### 重要文档 <!-- omit in toc -->
以下三份文档对本项目至关重要（点击可跳转）：

- [docs/planning.md](docs/planning.md)：该文档跟踪 OpenWorldLib 的短期优化目标和未来发展计划。
- [docs/awesome_world_models.md](docs/awesome_world_model.md)：该文档记录了世界模型相关的前沿研究、综述论文和开源项目。
- [docs/installation.md](docs/installation.md)：该文档提供了 OpenWorldLib 中不同方法的安装说明。


### 目录 <!-- omit in toc -->
- [特性](#特性)
  - [项目目标](#项目目标)
  - [支持的任务](#支持的任务)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [快速体验](#快速体验)
- [项目结构](#项目结构)
- [规划](#规划)
- [开发者指南](#开发者指南)
- [致谢](#致谢)
- [引用](#引用)


### 特性
#### 项目目标
OpenWorldLib 的主要目标包括：
- 建立一个统一、规范的**世界模型框架**，使现有世界模型相关代码的调用更加一致和结构化；
- 整合开源世界模型研究成果，并系统性地整理相关论文，供研究者参考和使用。

#### 支持的任务
OpenWorldLib 涵盖以下与**世界模型**相关的研究方向：

| 任务类别 | 子任务 | 代表性方法/模型 |
| :--- | :--- | :--- |
| **视频生成** | 导航生成 | lingbot, matrix-game, hunyuan-worldplay, genie3 等 |
| | 长视频生成 | sora-2, veo-3, wan 等 |
| **3D 场景生成** | 3D 场景生成 | flash-world, vggt 等 |
| **推理** | VQA（视觉问答） | spatialVLM, omnivinci 及其他具备世界理解能力的 VLM |
| | VLA（视觉-语言-动作） | pi-0, pi-0.5, giga-brain 等 |


### 快速开始
#### 安装
首先，创建一个 conda 环境：
```bash
conda create -n "openworldlib" python=3.10 -y
conda activate "openworldlib"
```
接着可以利用已有的安装脚本进行安装
```bash
cd OpenWorldLib
bash scripts/setup/default_install.sh
```
一些方法有特殊的安装需求，所有安装脚本在 `./scripts/setup`
> 📖 完整安装指南请参阅 [docs/installation.md](docs/installation.md)


#### 快速体验
在安装过基础环境后，可以通过下面的两个指令分别测试 matrix-game-2 生成以及多轮交互效果：
```bash
cd OpenWorldLib
bash scripts/test_inference/test_nav_video_gen.sh matrix-game-2
bash scripts/test_stream/test_nav_video_gen.sh matrix-game-2
```
其他方法的运行脚本可在 `scripts/test_inference` 以及 `scripts/test_stream` 路径下进行查看


### 项目结构
为了让开发者以及用户们更好地了解我们的 OpenWorldLib，我们在这里对我们代码中的细节进行介绍，首先我们的框架结构如下：
```txt
OpenWorldLib
├─ assets
├─ data                                # 测试数据
│  ├─ benchmarks
│  │  └─ reasoning
│  ├─ test_case
│  └─ ...
├─ docs                                # 相关文档
├─ examples                            # 运行benchmark测例
├─ scripts                             # 所有关键测试脚本
├─ src
│  └─ openworldlib                        # 主路径
│     ├─ base_models                   # 基础模型
│     │  ├─ diffusion_model
│     │  │  ├─ image
│     │  │  ├─ video
│     │  │  └─ ...
│     │  ├─ llm_mllm_core
│     │  │  ├─ llm
│     │  │  ├─ mllm
│     │  │  └─ ...
│     │  ├─ perception_core
│     │  │  ├─ detection
│     │  │  ├─ general_perception
│     │  │  └─ ...
│     │  └─ three_dimensions
│     │     ├─ depth
│     │     ├─ general_3d
│     │     └─ ...
│     ├─ memories                      # 记忆模块
│     │  ├─ reasoning
│     │  └─ visual_synthesis
│     ├─ operators                     # 输入、交互信号处理
│     ├─ pipelines                     # 所有运行管线
│     ├─ reasoning                     # 推理模块
│     │  ├─ audio_reasoning
│     │  ├─ general_reasoning
│     │  └─ spatial_reasoning
│     ├─ representations               # 表征模块
│     │  ├─ point_clouds_generation
│     │  └─ simulation_environment
│     └─ synthesis                     # 生成模块
│        ├─ audio_generation
│        ├─ visual_generation
│        └─ vla_generation
├─ submodules                          # diff-gaussian-raster 等附属安装
├─ test                                # 所有测试代码
├─ test_stream                         # 所有交互测试代码
└─ tools                               # 相关工具
   ├─ installing
   └─ vibe_code
```
在使用 OpenWorldLib 时通常直接调用 **pipeline** 类，而 pipeline 类中，需要完成权重加载，环境初始化等任务，同时用户与 **operator** 类进行交互，并且利用 **synthesis**、**reasoning**、**representation** 等类完成生成。在多轮交互中，使用 **memory** 类对运行流程进行记忆。


### 规划
- 我们在 [docs/awesome_world_models.md](docs/awesome_world_model.md) 中记录了最前沿的 world models 相关的研究，同时我们欢迎大家在这里提供一些有价值的研究。
- 我们在 [docs/planning.md](docs/planning.md) 中记录了我们后续的**训练**以及**优化**计划。


### 开发者指南
我们欢迎各位开发者共同参与，帮助完善 **OpenWorldLib** 这一统一世界模型仓库。推荐采用 **Vibe Coding** 方式进行快捷的代码提交，相关提示词可参考 `tools/vibe_code/prompts` 目录下的内容。同时也可以向 [docs/planning.md](docs/planning.md) 以及 [docs/awesome_world_models.md](docs/awesome_world_model.md) 补充高质量的world model相关工作。期待你的贡献！

相关文档可以查看：[[工程整体目标]](https://wcny4qa9krto.feishu.cn/wiki/NAy4wGbGrilzZ6kVInKcqkB9nUe)；[[开发流程指南]](https://wcny4qa9krto.feishu.cn/wiki/XF2Ew583ziTT8LkNiAQcyZtTn2N)；[[开发分工细节]](https://wcny4qa9krto.feishu.cn/wiki/NLFdw1NpBickgCkb2WQcGJrInUe)；[[代码提交规范]](https://wcny4qa9krto.feishu.cn/wiki/Zs8cwPWqMi45FSknzLPc1WlhnAg)


### 致谢
本项目为 [DataFlow](https://github.com/OpenDCAI/DataFlow) 、[DataFlow-MM](https://github.com/OpenDCAI/DataFlow-MM) 在世界模型任务上的拓展。同时我们与 [RayOrch](https://github.com/OpenDCAI/RayOrch) 、[Paper2Any](https://github.com/OpenDCAI/Paper2Any) 等工作积极联动中。


### 引用
如果我们的 **OpenWorldLib** 为你带来了帮助，欢迎给我们的repo一个star🌟，并考虑引用相关论文：
```bibtex
@misc{dataflow-team-openworldlib,
  author = {{OpenDCAI}},
  title = {OpenWorldLib: A Unified Codebase for Advanced World Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDCAI/OpenWorldLib}}
}

@article{zeng2026research,
  title={Research on World Models Is Not Merely Injecting World Knowledge into Specific Tasks},
  author={Zeng, Bohan and Zhu, Kaixin and Hua, Daili and Li, Bozhou and Tong, Chengzhuo and Wang, Yuran and Huang, Xinyi and Dai, Yifan and Zhang, Zixiang and Yang, Yifan and others},
  journal={arXiv preprint arXiv:2602.01630},
  year={2026}
}
```
为了后续更佳具体地说明我们框架的设计思路，以及对于世界模型的理解，我们会发布 OpenWorldLib 的报告。希望我们的工作能为您带来帮助！
