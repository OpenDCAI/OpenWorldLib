# **PhyX — Benchmark Record**

## **1. Meta**

* **Name**: **PhyX** *(Does Your Model Have the “Wits” for Physical Reasoning?)*
* **Task**: 评估多模态大语言模型（MLLMs）在**视觉场景下的物理推理能力**
* **Paper**: [https://arxiv.org/pdf/2505.15929v2](https://arxiv.org/pdf/2505.15929v2)
* **Code**: [https://github.com/killthefullmoon/PhyX](https://github.com/killthefullmoon/PhyX)
* **Benchmark Code Path**: `PhyX`
* **Dataset Path**: [download from huggingface](https://huggingface.co/datasets/Cloudriver/PhyX)
* **Task Type**: Multimodal Physical Reasoning, Question Answering (QA)，Open-Ended (OE)，Multiple-Choice (MC)

---

## **2. Dataset Structure**

PhyX 包含适配多种评估场景的配置（configs），涵盖 **3,000（test）+ 1,000（test_mini）** 道大学物理题。
数据以 **TSV / Parquet** 形式存储，并附带所有题目的物理示意图。

### **2.1 Dataset Directory Layout**

```
$DATA_PATH/PhyX/
├── data_llms_eval/               # 适用于 LLM / MLLM 的 Parquet 格式
│   ├── PhyX_MC.parquet
│   ├── PhyX_mini_MC.parquet
│   ├── PhyX_OE.parquet
│   ├── PhyX_mini_OE.parquet
├── dataset_card/                 # 默认配置（带图像）
│   ├── test-00000-of-00001.parquet
│   ├── test_mini-00000-of-00001.parquet
├── multilingual/                 # 多语言扩展
├── with_steps/                   # 带推理步骤版本
└── images/                       # 所有题目的图像文件
```

---

### **2.2 Dataset Fields (Default Config)**

| Field                               | Type         | Description                   |
| ----------------------------------- | ------------ | ----------------------------- |
| `id`                              | string       | 样本唯一 ID                   |
| `question`                        | string       | 问题文本                      |
| `question_description`            | string       | 原题完整描述                  |
| `question_description_simplified` | string       | 去冗余后的简化描述            |
| `options`                         | list(string) | MC 题目选项（仅 MC）          |
| `answer`                          | string       | 标准答案                      |
| `image`                           | image        | 题目图像                      |
| `image_caption`                   | string       | 图像内容文字描述              |
| `category`                        | string       | 物理大类（如 Optics）         |
| `subfield`                        | string       | 子类（如 Geometrical Optics） |
| `reasoning_type`                  | list(string) | 推理类型标签                  |

---

### **2.3 Sample JSON Entry**

```json
{
  "id": "235",
  "question": "Find the angle of refraction θ₂ if the second medium is water.",
  "question_description": "A ray of light travels from air ... θ₁ = 45° ...",
  "question_description_simplified": "Light ray from air enters water, θ₁=45°.",
  "options": [
    "A: 22.7°",
    "B: 31.4°",
    "C: 32.0°",
    "D: 33.5°"
  ],
  "answer": "C",
  "image": "images/phyx_0235.png",
  "image_caption": "Diagram of light refraction across air-water interface.",
  "category": "Optics",
  "subfield": "Geometrical Optics",
  "reasoning_type": [
    "Physical Model Grounding Reasoning",
    "Spatial Relation Reasoning"
  ]
}
```

---

## **3. IO Specification**

### **3.1 Input Format**

每个样本为一个多模态物理推理任务，一般包含：**图像 + 文本描述 + 问题（+ 选项）**。

| Input                    | Description                                |
| ------------------------ | ------------------------------------------ |
| **Image**          | 题目图像（物理装置 / 光路图 / 实验示意图） |
| **Description**    | 物理背景描述                               |
| **Question**       | 具体要求解的问题                           |
| **Options (MC)**   | 多项选择题的候选项                         |
| **Reasoning Type** | 标签，不用于模型输入，仅用于子集分析       |

#### **输入模式（Variants）**

| Variant                | 内容                     | 用途                   |
| ---------------------- | ------------------------ | ---------------------- |
| Full-Text              | 完整文本 + 图像          | 标准评测               |
| Text-DeRedundancy      | 去除图像可直接推断的信息 | 强制使用视觉信息       |
| Text-Minimal           | 最小化文本 + 图像        | 视觉主导任务           |
| Image-Only（内部实验） | 仅图像                   | 测试模型纯视觉理解能力 |

---

### **3.2 Output Format**

模型输出包括：

1. **可选**：Chain-of-Thought reasoning
2. **必需**：最终答案（Final Answer）

**Open-ended 示例**

```
Step 1: Apply Snell's Law...
Step 2: Compute sin(theta2)...

Final Answer: 32°
```

**Multiple-choice 示例**

```
The correct option is: C
```

---

## **4. Metrics Specification**

| **Task**                | **Dataset.TYPE** | **Metrics**                    | **说明**               |
| ----------------------------- | ---------------------- | ------------------------------------ | ---------------------------- |
| Multiple Choice (MCQ)         | `MCQ`                | Accuracy                             | 字符精确匹配                 |
| Yes/No                        | `Y/N`                | Accuracy                             | 直接比对                     |
| Open-ended (LLM 判断)         | default                | LLM-Score / GPT-Judged Score         | DeepSeek-V3 / GPT-4 系列判分 |
| Video QA                      | `VIDEO`              | Accuracy / LLM-Score                 | 同原数据集规则               |
| Multi-turn Conversation       | `MT`                 | Per-turn Accuracy / Consistency      | 多轮逐轮评测                 |
| Math Tasks                    | 自动触发               | Exact Match / Numeric Accuracy / LLM | 数值解析更严格               |
| MMVet / LLaVA-Bench           | 自动匹配               | LLM-Score                            | 统一 GPT-Judge               |
| No GT Splits (DocVQA_TEST 等) | NA                     | Only Inference                       | 不提供本地评分               |

---

## **5. Evaluation**

### **工具与环境**

| 工具包               | 描述                                | 示例命令      |
| -------------------- | ----------------------------------- | ------------- |
| **VLMEvalKit** | 官方推荐，支持规则匹配与 LLM 判分   | `run.py`    |
| **lmms-eval**  | 官方 LLM/MLLM 统一评测框架          | `lmms_eval` |
| **本仓库脚本** | 深度集成 VLMEvalKit，支持更多自定义 | 见 repo       |

---

### **评估核心参数**

#### 任务类型（`--data` / `--tasks`）

* `phyx_mc`, `phyx_mini_mc`：多项选择
* `phyx_oe`, `phyx_mini_oe`：开放式

`mini` = 1000 samples
full = 3000 samples

#### 评分方式（`valid_type`）

* `STR` — **规则匹配（免费）**
* `LLM` — **大模型评判（DeepSeek-V3 或 GPT 系列）**

---

### **VLMEvalKit：标准评测流程**

#### **1. 环境配置**

```bash
pip install -r requirements.txt
export SiliconFlow_API_KEY=YOUR_KEY  # 若使用 DeepSeek 判分
```

#### **2. 运行评测**

**规则评判（STR）**

```bash
python -u run.py --data PhyX_mini_OE \
    --model GPT4o_20241120 \
    --judge-args '{"valid_type": "STR"}'
```

**LLM 判分（DeepSeek-V3, SiliconFlow）**

```bash
python -u run.py --data PhyX_mini_OE \
    --model GPT4o_20241120 \
    --judge deepseek-v3-si \
    --judge-args '{"valid_type": "LLM"}'
```

---

### **LLM（纯文本）评测**

```bash
export PHYX_TEXT_ONLY=true
python -u run.py --data PhyX_mini_OE --model YOUR_LLM
```

---

### **结果查看**

所有结果将自动保存于：

```
outputs/
```

### **自定义model**
可以在`\PhyX\vlmeval\vlm`定义自己的model