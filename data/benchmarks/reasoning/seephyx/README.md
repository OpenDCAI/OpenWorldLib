# **SEEPHYS — Benchmark Record**

## **1. Meta**

**Name**: SEEPHYS

**Task**: Visual-based Physics Reasoning（基于图示的物理推理）

**Paper**: [https://arxiv.org/pdf/2505.19099](https://arxiv.org/pdf/2505.19099)

**Code Links**: [https://github.com/AI4Phys/SeePhys](https://github.com/AI4Phys/SeePhys)

**Benchmark Code Path**: `seephys/vlmeval`

**Dataset Path**: `[download from huggingface](https://huggingface.co/datasets/SeePhys/SeePhys)`

**Task Type**: Multimodal Reasoning / Physics QA
（Open-ended, Text–Image Multimodal Reasoning）

---

## **2. Dataset Structure**

SeePhys 是一个 **QA + 图示** 的物理题库，每个样本包含题目文本、参考答案、推理步骤、图像列表以及题目属性标签。数据在 HuggingFace 上以 **Parquet** 存储，字段与类型如下：

```
$DATA/SeePhys/
├── train-*.parquet          # 训练集
├── dev-*.parquet            # 开发集
└── images/                  # 存放所有图示文件
    ├── seephys_00001_1.png
    ├── seephys_00001_2.png
    └── ...
```

### **Features (字段说明)**

| 字段名           | 类型        | 说明                                       |
| ---------------- | ----------- | ------------------------------------------ |
| index            | int64       | 样本索引                                   |
| question         | string      | 题目文本                                   |
| answer           | string      | 标准答案                                   |
| reasoning        | string      | 推理过程 / 解题步骤                        |
| images           | list[image] | 对应图示路径                               |
| sig_figs         | string      | 有效数字要求                               |
| level            | int64       | 难度等级（1=初中 ... 7=博士）              |
| subject          | string      | 学科分类，如 EM、Thermodynamics            |
| language         | string      | 题目语言（EN/其他）                        |
| img_category     | string      | 图示类型，如 circuit_diagram、optical_path |
| vision_relevance | string      | 是否必需视觉信息，optional/essential       |
| caption          | string      | 图像文字说明或题注                         |

---

### **Sample JSON Entry**

```json
{
  "index": 1,
  "question": "As shown in the figure, after the switch is moved to position B, what is the time rate of change of current through R?",
  "answer": "-10^4 A/s",
  "reasoning": "As -L di_L/dt = i_L R, we have di_L/dt|_{t=0} = -i_L(0) R/L = -10^4 A/s",
  "images": ["images/seephys_00001_1.png"],
  "sig_figs": "1",
  "level": 7,
  "subject": "Electromagnetism",
  "language": "English",
  "img_category": "circuit_diagram",
  "vision_relevance": "essential",
  "caption": "Circuit diagram containing a 1Ω resistor, a 1V battery, a switch S, a 10^4Ω resistor labeled R, and an inductor labeled L=1H."
}
```

---

## **3. IO Specification**

SeePhys 是一个 **推理型多模态 benchmark**，模型输入为文本 + 图像，输出为开链推理和最终答案。

### **Input Format**

```json
{
  "question": "As shown in the figure, after the switch is moved to position B, what is the time rate of change of current through R?",
  "images": ["images/seephys_00001_1.png"],
  "sig_figs": 1
}
```

### **Expected Model Output**

```
<step-by-step chain-of-thought reasoning...>

Final Answer: -10^4 A/s
```

---

## **4. Metrics Specification**

### **VLM Evaluation Metrics Overview**

| 指标                        | 描述               | 适用子集 / 条件                    | 说明                                           |
| --------------------------- | ------------------ | ---------------------------------- | ---------------------------------------------- |
| Accuracy                    | 正确率             | MCQ / YN / MT / Video / Text       | 默认主要指标，字符串精确匹配或 LLM judge       |
| Exact Match                 | 精确匹配           | 文本题 / MCQ                       | 对文本或 MCQ 选项严格匹配                      |
| Symbolic / Numeric Accuracy | 符号/数值精度      | 数学/物理/量化题                   | 使用 SymPy 或 judge LLM 处理数值/公式          |
| BLEU / ROUGE / CIDEr        | 文本生成质量       | Caption / Text Generation          | 可选，script未显式但 judge LLM 可计算          |
| Vision Compliance           | 是否利用视觉信息   | Vision-Essential / Vision-Optional | VE题目模型忽略图像计为错误                     |
| Multi-Turn Consistency      | 多轮一致性         | Multi-Turn (MT) Dataset            | MT题目评估模型对多轮对话的逻辑一致性           |
| Retry / API Success Rate    | API调用成功率      | 所有API模型                        | judge_kwargs 中 retry 参数控制，统计调用可靠性 |
| Inference Time / Throughput | 推理时间 / 吞吐量  | 所有模型                           | 可通过 verbose 或 API 并行数记录               |
| Submission Compliance       | 官方提交格式正确率 | MMMU / MMT-Bench                   | 对应官方评测要求的格式校验                     |

### **说明**

* `dataset.evaluate(result_file, **judge_kwargs)` 是核心评测接口，不同 dataset 会返回不同指标，表格列出最常见。
* 数值题 / 公式题：使用 judge_kwargs 中的 `model`（如 gpt-4o-mini）或 exact_matching 判定数值/符号正确性。
* 视频 / 图像题：VE/VO 子集通过 Vision Compliance 检查模型是否使用图像信息。
* 多轮/MT题目：通过 Multi-Turn Consistency 评估连续问答的逻辑一致性。

---

## **5. Evaluation**

### **1. 环境准备**

* **API Key 设置**（适用于 OpenAI / DeepSeek 等需要 API 调用的模型）：

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="YOUR_API_BASE_URL"  # 例如：https://api.openai.com/v1
# DeepSeek Judge 可能需要设置对应 API Key
# export DEEPSEEK_API_KEY="..."
```

* **数据路径设置**（可选，根据实际存储位置）：

```bash
export LMUData="/LMUData"  # 示例路径
```

---

### **2. 评估脚本与参数**

#### **分布式运行示例**

```bash
#!/bin/bash
# 自动检测 GPU 数量
export GPU=$(nvidia-smi --list-gpus | wc -l) 

# 分布式运行脚本，使用所有 GPU
torchrun --nproc-per-node=${GPU} run.py \
    --model Qwen2.5-VL-7B-Instruct \
    --data SeePhys \
    --api-nproc 32 \
    --work-dir /work_dir \
    --judge deepseek \
    --judge-args '{"valid_type": "LLM"}' \
    --reuse
```

#### **关键参数说明**

| 参数                 | 说明                             | 示例                             |
| :------------------- | :------------------------------- | :------------------------------- |
| `--model`          | 待评估的多模态模型名称           | `Qwen2.5-VL-7B-Instruct`       |
| `--data`           | 评估数据集名称                   | `SeePhys`                      |
| `--work-dir`       | 推理结果、日志和评估报告输出路径 | `/work_dir`                    |
| `--judge`          | 裁判 LLM 模型                    | `deepseek`                     |
| `--judge-args`     | 裁判参数（启用 LLM 裁判模式）    | `'{"valid_type": "LLM"}'`      |
| `--api-nproc`      | 裁判模型 API 并行调用数          | `32`                           |
| `--reuse`          | 启用断点续传                     | `--reuse`                      |
| `--nproc-per-node` | GPU 数量（由 torchrun 自动设置） | `8`（若机器有 8 张 GPU）<br /> |

---

### **3. 评估后处理**

* **结果文件**：评估完成后，生成 JSON/CSV 文件，存放在 `--work-dir` 路径下。
* **指标查看**：结果文件包含 **Symbolic/Numeric Accuracy**、**Vision Compliance**、**Multi-Turn Consistency** 等评估指标，可用于进一步分析模型性能。

### **4. 自定义model**

可以在 `\PhyX\vlmeval\vlm`定义自己的model
