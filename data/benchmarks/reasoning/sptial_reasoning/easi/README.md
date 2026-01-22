# **EASI — Benchmark Record**

## **1. Meta**

* **Name**: **EASI** — Holistic Evaluation of Multimodal LLMs on Spatial Intelligence

* **Task**: 综合空间智能评测（覆盖六大核心能力：
  **MM** 多模态理解、**MR** 空间关系推理、**SR** 空间检索、**PT** 时空追踪、**DA** 动态感知、**CR** 因果推理）

* **Paper**: *Holistic Evaluation of Multimodal LLMs on Spatial Intelligence*
  [[PDF]](https://arxiv.org/pdf/2508.13142v3)

* **Code**: [[GitHub]](https://github.com/EvolvingLMMs-Lab/EASI/)

* **Benchmark Code Path**: `EASI/VLMEvalKit`

* **Dataset Path**:

  **EASI 本身不是一个单独的数据集，而是一个 meta-benchmark**，统一收录多个外部子基准。
  所有子数据集均放置在：

  ```
  EASI/VLMEvalKit/vlmeval/dataset/
  ```

  **已集成的 8 个主要 benchmark：**

  * VSI-Bench
  * SITE
  * MMSI
  * OmniSpatial
  * MindCube
  * STARE
  * CoreCognition
  * SpatialViz

* **Task Type**:

  * 多选题（MCQ）
  * 数值题（Numeric Answers）
  * VQA-style 问答
  * 统一的 **EASI Protocol**（CoT + Tag 格式）

---

## **2. Evaluation**

EASI 的执行依赖 **VLMEvalKit** 的评测框架。基本流程如下：

---

### **(1) 环境配置**

在 `VLMEvalKit/.env` 填写所需的 API Key，或配置你的本地 judge（LMDeploy / vLLM 等）。

---

### **(2) 注册你的模型**

在：

```
vlmeval/config.py
```

中将你的模型加入 `supported_VLM` 列表。（也要在 `vlmeval/vlm` 载入）

可用以下命令验证模型是否能正常跑：

```bash
vlmutil check <MODEL_NAME>
```

---

### **(3) 运行评测**

**图像类任务示例：**

```bash
python run.py --data MMBench_DEV_EN MME --model idefics_80b_instruct
```

**视频类任务示例（多卡）：**

```bash
torchrun --nproc-per-node=8 run.py \
    --data MMBench_Video_8frame_nopack \
    --model idefics2_8
```

---

### **(4) 评测结果输出**

所有结果会保存在：

```
$WORK_DIR/<model_name>/
```

其中包含：

* 任务级别指标（CSV）
* 完整 Excel 评分
* 模型答案记录

---

### **(5) 更多可选项**

* **仅做推理（不评分）**：

```bash
python run.py --mode infer
```

* **提升 API 调用速度（并发）**：

```bash
--api-nproc <NUM>
```

* **使用本地 Judge 模型**：
  将 `.env` 的 `OPENAI_API_BASE` 指向 LMDeploy/vLLM 所暴露的接口即可。
