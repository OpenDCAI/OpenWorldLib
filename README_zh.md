<div align="center" markdown="1">

<img src="https://github.com/user-attachments/assets/0521e467-681f-42bd-853c-bee16a309e9d" alt="openworldlib_logo" width="90%" />

#### 欢迎加入我们的开源世界模型项目！ <!-- omit in toc -->
---

<a href="https://github.com/OpenDCAI/OpenWorldLib"><img alt="Build" src="https://img.shields.io/github/stars/OpenDCAI/OpenWorldLib"></a> <!-- License --> <a href="https://github.com/OpenDCAI/OpenWorldLib/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/OpenDCAI/OpenWorldLib"></a> <!-- GitHub Issues --> <a href="https://github.com/OpenDCAI/OpenWorldLib/issues"><img alt="Issues" src="https://img.shields.io/github/issues/OpenDCAI/OpenWorldLib"></a>
<a href="https://github.com/user-attachments/assets/35d48c4f-adb3-4f10-b30f-e7f4a245ab48"><img alt="Add me on WeChat" src="https://img.shields.io/badge/Connect_on-WeChat-07C160?style=flat-square&logo=wechat&logoColor=white"></a> <a href="https://arxiv.org/abs/2604.04707"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2604.04707-b31b1b?logo=arxiv&logoColor=white"></a>
<!-- <img alt="Report" src="https://img.shields.io/badge/📄 Technical Report-Coming Soon-lightgrey"> -->

[English](README.md) | [中文](README_zh.md)

扩展仓库：[[三维生成]](https://github.com/zengbohan0217/OpenWorldLib-extension-3D) | [[VLA]](https://github.com/yfanDai/OpenWorldLib-extension-VLA) | [[仿真环境]](https://github.com/YF0224/OpenWorldLib-extension-Simulator)

</div>


<!-- add demo -->
<div align="center">
<table>
<tr>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/1c328d17-ef94-4d53-8dc2-f77ec3964a74" width="218" height="120"><br/>
    <b>Matrix-Game-2</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/2fac45f5-d365-4794-bf38-45aee58f3d45" width="218" height="120"><br/>
    <b>Hunyuan-GameCraft</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/c594ac11-1fc7-45da-ab2e-426e66e927f6" width="218" height="120"><br/>
    <b>Hunyuan-Worldplay</b>
  </td>
</tr>
<tr>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/e07456a3-869a-408c-9755-abe896f51c0e" width="218" height="120"><br/>
    <b>Lingbot-World</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/ae38060b-f82f-48b1-89b5-6344e85e8354" width="218" height="120"><br/>
    <b>YUME-1.5</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/c6b176fd-1ebe-4760-ab2f-6f2ba20445ca" width="218" height="120"><br/>
    <b>FlashWorld</b>
  </td>
</tr>
<tr>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/c4fa5245-e837-4e47-a4e4-585a61358f91" width="218" height="120"><br/>
    <b>Wan-2.2-IT2V</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/a47b2e0e-b62c-45fb-bb99-6377f53e3f5e" width="218" height="120"><br/>
    <b>WoW</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/c8ec425d-3815-4652-b2af-7e3c06e76b72" width="218" height="120"><br/>
    <b>Cosmos-Predict-2.5</b>
  </td>
</tr>
<tr>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/68c59c52-35c5-400c-a5a0-94c8eb802a79" width="218" height="120"><br/>
    <b>Pi3</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/dab9c489-fcb8-412d-99af-9fdad4e76e0e" width="120" height="120"><br/>
    <b>Libero</b>
  </td>
  <td align="center">
    <img src="https://github.com/user-attachments/assets/f8b4253b-9172-4afa-bd0e-ae0bc7ea8b25" width="120" height="120"><br/>
    <b>Ai2-THOR</b>
  </td>
</tr>
</table>
</div>



我们将世界模型定义为：**一种以感知为核心、具备交互与长期记忆能力的模型或框架，用于理解和预测复杂世界。** 因此，🎓 *多模态理解*、🤖 *视觉动作预测* 和 🖼️ *视觉生成* 都是世界模型需要完成的子任务。

我们热烈欢迎研究者在 Issues 区分享对该框架的看法或对世界模型的思考。同时也希望您能通过 Pull Request 将有价值的世界模型相关方法提交到我们的框架中，或将其记录并提交到 [[awesome_world_model]](docs/awesome_world_model.md)。欢迎给我们的仓库点个 🌟 Star，以关注 OpenWorldLib 的最新进展！


### 重要文档 <!-- omit in toc -->
以下四份文档对本项目至关重要（点击可跳转）：

- [docs/planning.md](docs/planning.md)：该文档跟踪 OpenWorldLib 的短期优化目标和未来发展计划。
- [docs/awesome_world_model.md](docs/awesome_world_model.md)：该文档记录了世界模型相关的前沿研究、综述论文和开源项目。
- [docs/installation.md](docs/installation.md)：该文档提供了 OpenWorldLib 中不同方法的安装说明。
- [开发流程指南](https://wcny4qa9krto.feishu.cn/wiki/XtPJwf5XQipP7RkeVv0ckyWlnNd?from=from_copylink): 该文档提供了 OpenWorldLib 的框架模板，旨在为世界模型社区的开发者提供代码合并的参考。


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
OpenWorldLib 涵盖以下与**世界模型**相关的研究方向，**我们衷心感谢所有纳入本框架的优秀方法，它们为世界模型研究做出了巨大贡献**：

| 任务类别 | 子任务 | 代表性方法/模型 |
| :--- | :--- | :--- |
| **视频生成** | 导航生成 | lingbot, matrix-game, hunyuan-worldplay, genie3 等 |
| | 长视频生成 | sora-2, veo-3, wan 等 |
| **3D 场景生成** | 3D 场景生成 | flash-world, vggt 等 |
| **推理** | VQA（视觉问答） | spatialVLM, omnivinci 及其他具备世界理解能力的 VLM |
| | VLA（视觉-语言-动作） | pi-0, pi-0.5, giga-brain 等 |
> 常用推理框架包括：[diffusers](https://github.com/huggingface/diffusers), [DiffSynth](https://github.com/modelscope/DiffSynth-Studio), [LightX2V](https://github.com/ModelTC/LightX2V)


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
其他方法的运行脚本可在 `scripts/test_inference` 以及 `scripts/test_stream` 路径下进行查看，目前我们主要使用 **80GB** 和 **141GB** 显存的显卡进行测试，后续我们会测试更多型号，并在 `./docs/installation.md` 文件中更新。


### 项目结构
为了让开发者以及用户们更好地了解我们的 OpenWorldLib，我们在这里对我们代码中的细节进行介绍，首先我们的框架结构如下：
```txt
OpenWorldLib
├─ assets
├─ data                                # 测试数据
│  ├─ benchmarks
│  ├─ test_case
│  └─ ...
├─ docs                                # 相关文档
├─ examples                            # 运行benchmark测例
├─ scripts                             # 所有关键测试脚本
├─ src
│  └─ openworldlib                     # 主路径
│     ├─ base_models                   # 基础模型，为其他部分提供基础模块
│     │  ├─ diffusion_model
│     │  ├─ llm_mllm_core
│     │  ├─ perception_core
│     │  └─ three_dimensions
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
   └─ vibe_code
```
在使用 OpenWorldLib 时通常直接调用 **pipeline** 类，而 pipeline 类中，需要完成权重加载，环境初始化等任务，同时用户与 **operator** 类进行交互，并且利用 **synthesis**、**reasoning**、**representation** 等类完成生成。在多轮交互中，使用 **memory** 类对运行流程进行记忆。
```txt
User Input
    │
    ▼
┌─────────────┐
│  Pipeline   │──── from_pretrained(): load models & initialize modules
│             │
│  __call__() │──┬── process() ──► Operator (validate & preprocess)
│      │      │  │
│      │      │  ├── ► Synthesis.predict()           → multimodel outputs
│      │      │  ├── ► Reasoning.inference()         → text outputs
│      │      │  └── ► Representation.get_repr..()   → 3D outputs
│      │      │
│  stream()   │──┬── memory.select()  → retrieve context
│      │      │  ├── __call__()       → generate current turn
│      │      │  └── memory.record()  → store results
└─────────────┘
```


### 规划
- 我们在 [docs/awesome_world_model.md](docs/awesome_world_model.md) 中记录了最前沿的 world models 相关的研究，同时我们欢迎大家在这里提供一些有价值的研究。
- 我们在 [docs/planning.md](docs/planning.md) 中记录了我们后续的**训练**以及**优化**计划。


### 开发者指南
我们欢迎各位开发者共同参与，帮助完善 **OpenWorldLib** 这一统一世界模型仓库。推荐采用 **Vibe Coding** 方式进行快捷的代码提交，相关提示词可参考 `tools/vibe_code/prompts` 目录下的内容。同时也可以向 [docs/planning.md](docs/planning.md) 以及 [docs/awesome_world_model.md](docs/awesome_world_model.md) 补充高质量的world model相关工作。期待你的贡献！

相关文档可以查看：[[开发流程指南]](https://wcny4qa9krto.feishu.cn/wiki/XtPJwf5XQipP7RkeVv0ckyWlnNd?from=from_copylink)


### 致谢
本项目为 [DataFlow](https://github.com/OpenDCAI/DataFlow) 、[DataFlow-MM](https://github.com/OpenDCAI/DataFlow-MM) 在世界模型任务上的拓展。同时我们与 [RayOrch](https://github.com/OpenDCAI/RayOrch) 、[Paper2Any](https://github.com/OpenDCAI/Paper2Any) 等工作积极联动中。


### 引用
如果我们的 **OpenWorldLib** 为你带来了帮助，欢迎给我们的repo一个star🌟，并考虑引用相关论文：
<!-- @misc{dataflow-team-openworldlib,
  author = {{OpenDCAI}},
  title = {OpenWorldLib: A Unified Codebase for Advanced World Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDCAI/OpenWorldLib}}
} -->
```bibtex
@article{team2026openworldlib,
  title={OpenWorldLib: A Unified Codebase and Definition of Advanced World Models},
  author={Team, DataFlow and Zeng, Bohan and Hua, Daili and Zhu, Kaixin and Dai, Yifan and Li, Bozhou and Wang, Yuran and Tong, Chengzhuo and Yang, Yifan and Chang, Mingkun and others},
  journal={arXiv preprint arXiv:2604.04707},
  year={2026}
}

@article{zeng2026research,
  title={Research on World Models Is Not Merely Injecting World Knowledge into Specific Tasks},
  author={Zeng, Bohan and Zhu, Kaixin and Hua, Daili and Li, Bozhou and Tong, Chengzhuo and Wang, Yuran and Huang, Xinyi and Dai, Yifan and Zhang, Zixiang and Yang, Yifan and others},
  journal={arXiv preprint arXiv:2602.01630},
  year={2026}
}
```
为了后续更佳具体地说明我们框架的设计思路，以及对于世界模型的理解，我们会发布 OpenWorldLib 的报告。希望我们的工作能为您带来帮助！
