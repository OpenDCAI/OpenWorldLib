# **MMWorld — Benchmark Record**

## **1. Meta**

* **Name**: MMWorld
* **Task**: 多学科、多维度视频理解与世界模型评估（覆盖解释、反事实推理、未来预测、专业知识、时间理解等）
* **Paper**: [MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos](https://arxiv.org/abs/2406.08407)
* **Code**: [https://mmworld-bench.github.io/](https://mmworld-bench.github.io/)
* **Benchmark Code Path**: `mmworld/evaluation`
* **Dataset**: [download from huggingface](https://huggingface.co/datasets/Xuehai/MMWorld/tree/main)
* **Task Type**: 多项选择与自由格式结合的视频理解问答（Video-based Multidisciplinary QA）

---

## **2. Dataset Structure**

MMWorld 数据集包含视频及其对应的多学科问题标注，目录结构如下：

```
mmworld/
├── xxx_1
│   └── xxx.mp4
├── xxx_2
│   └── xxx.mp4
├── README.md
└── mmworld.json
```

其中 `mmworld.json` 包含所有视频的元数据、字幕与问题集合。

### **Sample JSON Entry**

```json
{
  "video_id": "eng_vid1",
  "video_url": "https://youtu.be/-e1_QhJ1EhQ",
  "discipline": "Tech & Engineering",
  "subdiscipline": "Robotics",
  "captions": [
    "The humanoid robot Atlas interacts with objects and modifies the course to reach its goal."
  ],
  "questions": [
    {
      "type": "Explanation",
      "question": "Why is the engineer included at the beginning of the video?",
      "options": {
        "a": "...",
        "b": "...",
        "c": "...",
        "d": "..."
      },
      "answer": "...",
      "requires_domain_knowledge": false,
      "requires_audio": false,
      "requires_visual": true,
      "question_only": false,
      "correct_answer_label": "a"
    }
  ]
}
```

每条样本都包含多种维度的信息，支持跨学科、多模态推理。

---

## **3. IO Specification**

### **Input (Instance)**

每个测试样本包含：

* **video**：完整视频（10–60 秒不等）
* **discipline / subdiscipline**：所属学科标签
* **captions（可选）**：人工提供的视频摘要
* **question & options**
* **question_type**：

  * Explanation
  * Counterfactual Thinking
  * Future Prediction
  * Domain Expertise
  * Temporal Understanding
  * Attribution Understanding
  * Procedure Understanding

这些类型覆盖了视频理解中的主要世界模型能力。

### **Output Specification**

模型需输出一段自然语言回答：

| 项目               | 说明                                                   |
| ------------------ | ------------------------------------------------------ |
| **问答形式** | 自由格式回答（Free-form QA），模型可用自然语言直接作答 |
| **评估方式** | 由 GPT 裁判（GPT Referee）根据语义一致性判定是否正确   |
| **最终指标** | 二值化准确率（正确=1，错误=0）                         |
| **统计方式** | 按学科、问题类型、多模态需求等分组统计正确率           |

模型无需输出选项编号，而是输出自然语言即可。

---

## **4. Metrics Specification**

| Major Dimension                               | Level-1 Metric             | Code Variable                                              | Code Logic                     |
| --------------------------------------------- | -------------------------- | ---------------------------------------------------------- | ------------------------------ |
| **I. Overall Performance**              | Overall Accuracy           | `overall_accuracy = correct_answers / total_questions`   | Main loop aggregation          |
| **II. Discipline Coverage**             | Accuracy per Subject       | `results_by_subject[subject]["correct"] / total`         | results_by_subject aggregation |
| **III. Multi-faceted Reasoning**        | Accuracy per Question Type | `accuracy_per_question_type[type]["correct"] / total`    | per-type aggregation           |
|                                               | Explanation                | `question_data["type"] == Explanation`                   | —                             |
|                                               | Future Prediction          | —                                                         | —                             |
|                                               | Counterfactual Thinking    | —                                                         | —                             |
|                                               | Domain Expertise           | —                                                         | —                             |
|                                               | Temporal Understanding     | —                                                         | —                             |
|                                               | Attribution Understanding  | —                                                         | —                             |
|                                               | Procedure Understanding    | —                                                         | —                             |
| **IV. Modality / Knowledge Dependency** | Accuracy per Annotation    | `accuracy_per_annotation[annotation]["correct"] / total` | annotation aggregation         |
|                                               | Audio Reliance             | `requires_audio`                                         | —                             |
|                                               | Visual Reliance            | `requires_visual`                                        | —                             |
|                                               | Domain Knowledge Reliance  | `requires_domain_knowledge`                              | —                             |
|                                               | Question Only              | `question_only`                                          | —                             |

这些指标覆盖从整体能力到细粒度世界知识依赖的全路径评测。

---

## **5. Evaluation**

MMWorld 使用自由格式回答并配合 GPT 裁判进行多维度评估。您可以选择本地运行评测脚本，或直接上传模型输出文件至 EvalAI。

### **1. Configure GPT Referee**

评测依赖 GPT-4 / GPT-Omni 作为裁判。
请在 `eval.py` 中补全 API 初始化（约第 387 行）：

```python
answer_evaluator = AzureOpenAI(
    azure_endpoint="xx",
    api_key="xx",
    api_version="2023-12-01-preview"
)
```

如需，可替换为 OpenAI 官方 API。

### **2. Add Your Model to the Pipeline**

要接入新模型，需要在 `eval.py` 中实现两个函数：

#### **(1) Model Initialization（约 357 行）**

```python
def modelname_init():
    model = ...
    return model
```

用于加载模型、权重和推理环境。

#### **(2) Model Answer Function（约 226 行）**

```python
def modelname_answer(model, video_path, question):
    # run inference on video + question
    return answer_str
```

该函数需返回自然语言回答，不要求返回选项字母。

### **3. Run Local Evaluation**

实现上述函数后，即可运行评测：

```bash
python evaluate.py --modelname <model_name>
```

如果模型只使用文本，不需要视频：

```bash
python evaluate.py --modelname <model_name> --textonly
```

脚本会自动：

1. 遍历所有视频问题
2. 调用模型生成回答
3. 使用 GPT Referee 判分
4. 输出多维度准确率统计结果

### **4. Submit to EvalAI（可选）**

如不想本地运行完整评测，可直接生成以下格式的结果文件：

```json
{
    "detailed_results": [
        {
            "video_id": "eng_vid1",
            "model_answer": "the robot is shown interacting to demonstrate commercial applicability."
        }
    ]
}
```

上传到 EvalAI 后即可自动完成评测。

---
