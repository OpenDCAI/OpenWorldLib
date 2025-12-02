# MLLM Benchmark 技术能力分类框架

多模态大模型（MLLM）在世界建模、视觉理解与认知推理中涉及多层次能力。
从视觉生成到行动执行，现有基准可系统划分为五大能力维度：

| 能力类别                 | 层级                   | 核心问题                                           |
| ------------------------ | ---------------------- | -------------------------------------------------- |
| **Generation**     | 输出行为层             | 能否生成时空一致、可控、语义合理的视频/4D内容？    |
| **Representation** | 感知层                 | 能否表示与解析时序结构、动作、事件、状态变化？     |
| **Reasoning**      | 高级认知层             | 能否执行因果推断、未来预测、反事实推理、常识推理？ |
| **Specialized**    | 世界知识层             | 是否具备空间智能与物理规律建模能力？               |
| **Embodied**       | 感知–认知–执行闭环层 | 是否能规划行动并在环境中执行？                     |

---

## 1️ 生成能力（Generation Capability）

<details>
<summary>Benchmarks</summary>

* **[WorldScore](https://arxiv.org/pdf/2504.00983)** · [[Code]](https://github.com/haoyi-duan/WorldScore)核心评测重点：综合评估生成质量、可控性、动态一致性；不涉及物理推演合理

</details>

---

## 2️ 表征与理解能力（Representation & Understanding）

<details>
<summary>Benchmarks</summary>

* **[MLVU](https://arxiv.org/pdf/2406.04264)** · [[Code]](https://github.com/JUNJIE99/MLVU)核心评测重点：长视频理解、显式记忆、跨片段整合、细粒度时序推理
* **[MotionBench](https://arxiv.org/pdf/2501.02955)** · [[Code]](https://github.com/zai-org/MotionBench)核心评测重点：高密度运动理解：微动作、摄像机运动、动作顺序与频率
* **[VinoGround](https://arxiv.org/pdf/2410.02763)** · [[Code]](https://github.com/Vinoground/Vinoground)核心评测重点：状态变化 + 时间顺序 + 因果一致性（短视频方向）
* **[FAVOR-Bench](https://arxiv.org/html/2503.14935)** · [[Code]](https://github.com/FAVOR-Bench/FAVOR-Bench)核心评测重点：动态动作辨识与时序逻辑，不涉及真实物理因果机制
* **[ShortVid-Bench](https://www.emergentmind.com/topics/shortvid-bench)** · [[Code]](https://github.com/TencentARC/ARC-Hunyuan-Video-7B)核心评测重点：多模态叙事推理：视觉 + 音频 + ASR，多段结构化理解

</details>

---

## 3️ 推理与认知能力（Reasoning & Cognitive Capability）

<details>
<summary>Benchmarks</summary>

* **[VideoVerse](https://arxiv.org/pdf/2510.08398)** · [[Code]](https://github.com/Zeqing-Wang/VideoVerse)核心评测重点：面向10维世界模型能力的因果、物理、自然常识与时间结构推理
* **[MMWorld](https://arxiv.org/pdf/2406.08407)** · [[Code]](https://github.com/eric-ai-lab/MMWorld)核心评测重点：跨领域（科学/医学/机器人/商业）的视频推理、反事实与未来预测

</details>

---

## 4️ 专项认知能力（Specialized Cognitive Competencies）

<details>
<summary>Benchmarks</summary>

* **[EASI](https://arxiv.org/pdf/2508.13142v3)** · [[Code]](https://github.com/EvolvingLMMs-Lab/EASI/)核心评测重点：空间智能六维能力，用于映射不同 benchmarks 测试维度
* **[PHYX](https://arxiv.org/pdf/2505.15929)** · [[Code]](https://github.com/killthefullmoon/PhyX)核心评测重点：静态图像中的物理常识与材料属性推理
* **[SeePHYX](https://arxiv.org/pdf/2505.19099)** · [[Code]](https://github.com/AI4Phys/SeePhys)核心评测重点：视频情境下的动态物理直觉与因果过程预测

</details>

---

## 5️ 具身智能与执行能力（Embodied Intelligence & Action Execution）

<details>
<summary>Benchmarks</summary>

* **[WoWbench](https://arxiv.org/pdf/2509.22642)** · [[Code]](https://github.com/wow-world-model/wow-world-model)核心评测重点：行动生成 × 物理推理 × 执行验证，覆盖 VLA 全流程

</details>

---

## 6 4D benchmark

<details>
<summary>Benchmarks</summary>

* **[4d-bench](https://arxiv.org/pdf/2503.17827)** · [[Code]](https://github.com/WenxuanZhu1103/4D-Bench)核心评测重点：评估多模态大语言模型（MLLMs）在4D对象理解方面的能力
* **[4dworldbench](https://arxiv.org/pdf/2511.19836v1)** · [[Code]](#)核心评测重点：评估生成的 3D/4D 世界的真实性、动态性、物理一致性和指令控制能力

</details>

---

## 小结

本工作将现有 MLLM Benchmarks 沿能力维度划分为五层：生成、表征、推理、世界知识与具身执行，呈现从视觉内容生成到真实世界行动执行的递进范式。该框架揭示了从“可见的视觉一致性”到“可执行的世界推演能力”之间的能力鸿沟，为后续系统评估与模型设计提供统一视角。
