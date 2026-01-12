# **MMAU — Benchmark Record**

## **1. Meta**

* **Name** : MMAU (Massive Multi-Task Audio Understanding and Reasoning Benchmark)
* **Task** : Advanced Audio Reasoning & Information Extraction
* **Paper** : [https://arxiv.org/abs/2410.19168](https://arxiv.org/abs/2410.19168)
* **Code** : [https://github.com/Sakshi113/MMAU](https://github.com/Sakshi113/MMAU)
* **Benchmark Code Path** : `AudioBench/` (or standalone MMAU scripts)
* **Task Type** : Multi-choice Question Answering (MCQ) / Multimodal Reasoning

---

## **2. Dataset Structure**

MMAU 包含 10k 个音频片段，覆盖三大领域（Speech, Sound, Music）和 27 种特定技能。

```
$DATA_PATH/MMAU/
├── audios/
│   └── XXX
├── mmau_test_mini.json  # 1k samples (for hyperparameter tuning)
└── mmau_test.json       # 9k samples (main test set)
```

**Sample JSON Entry**

**JSON**

```

{
  "id": "3fe64f3d-282c-4bc8-a753-68f8f6c35652",
  "audio_id": "./test-mini-audios/3fe64f3d-282c-4bc8-a753-68f8f6c35652.wav",
  "question": "Based on the given audio, identify the source of the speaking voice.",
  "choices": [
    "Man",
    "Woman",
    "Child",
    "Robot"
   ],
  "answer": "Man",
  "dataset": "AudioSet",
  "task": "sound",
  "split": "test-mini",
  "category": "Reasoning",
  "sub-category": "Acoustic Source Inference",
  "difficulty": "medium"
}
```

---

## **3. IO Specification**

### **Input (Evaluation Script)**

评测脚本 `evaluation.py` 接收一个命令行参数 `--input`，指向一个 JSON 文件。该文件**必须**包含原始数据以及模型生成的预测结果。

* **Argument**: `--input` (str): Path to input JSON file to be evaluated.

### **Input File Format (JSON Structure)**

推理阶段（Inference）生成的 JSON 文件需包含以下字段，其中 `model_output` 是必需的预测字段。

```json
[
  {
    "id": "3fe64f3d-282c-4bc8-a753-68f8f6c35652",
    "audio_id": "./test-mini-audios/3fe64f3d-282c-4bc8-a753-68f8f6c35652.wav",
    "question": "Based on the given audio, identify the source of the speaking voice.",
    "choices": ["Man", "Woman", "Child", "Robot"],
    "answer": "Man",
    "task": "sound",
    "difficulty": "medium",
    "sub-category": "Acoustic Source Inference",
    "dataset": "AudioSet",
    "split": "test-mini",
    "category": "Reasoning",
    "model_output": "The source of the speaking voice is a Man."  // 模型生成的原始文本
  },
  ...
]

```

### **Output (Console Log)**

脚本直接在标准输出（stdout）打印统计结果，不生成额外的文件。

```text
******************************
Task-wise Accuracy:
sound : 85.20% over 250 samples
music : 70.10% over 250 samples
speech : 65.50% over 250 samples
******************************
Difficulty-wise Accuracy:
easy : 80.00% over 300 samples
hard : 55.00% over 200 samples
medium : 72.00% over 250 samples
******************************
Sub-category-wise Accuracy:
Acoustic Source Inference : 82.00% over 50 samples
...
******************************
Total Accuracy: 73.60% over 750 samples
******************************
No prediction count: 0

```

---

## **4. Metrics Specification**

根据 `evaluation.py` 中的 `string_match` 函数，MMAU 使用的是基于 **Token 集合** 的严格匹配逻辑，而非简单的正则提取。

| Major Dimension | Level-1 Metric | Code-Level Metric | Description |
| --- | --- | --- | --- |
| **Accuracy** | Micro-averaged Accuracy | `string_match` | **Token Set Inclusion & Exclusion**<br>

<br>判定逻辑需同时满足两个条件：<br>

<br>1. **Inclusion**: `Answer` 的所有 Token 必须出现在 `Prediction` 中。<br>

<br>2. **Exclusion**: `Prediction` 不能包含其他**错误选项**的 Token（排除掉 Answer 中已有的共用词）。 |
| **Task Analysis** | Task-wise Accuracy | `task_metrics` | 按照 `sound`, `music`, `speech` 三大领域分别统计准确率。 |
| **Difficulty Analysis** | Difficulty-wise Accuracy | `diff_metrics` | 按照 `easy`, `medium`, `hard` 三个难度等级分别统计准确率。 |
| **Skill Analysis** | Sub-category Accuracy | `subcat_metrics` | 针对数据中 `sub-category` 字段（对应 27 种 Skill）分别统计准确率。 |


---

## **5. Evaluation**

### **1. Preparation**

* 下载音频数据：
* [test-mini audios](https://drive.google.com/file/d/1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA/view?usp=sharing)
* [test audios](https://drive.google.com/file/d/1XqkRupC723zAeyDn4dYniqNv4uO-8rEg/view?usp=sharing)


* 准备原始 JSON 文件（如 `mmau_test.json`）。

### **2. Inference (Generate Predictions)**

使用你的模型遍历 JSON 数据，将预测结果写入 `model_output` 字段，并保存为新的 JSON 文件（例如 `mmau_results.json`）。

```python
# 伪代码示例
results = []
for sample in original_data:
    audio = load_audio(sample['audio_id'])
    response = model.generate(audio, sample['question'], sample['choices'])
    sample['model_output'] = response # 关键步骤：写入预测
    results.append(sample)

save_json(results, 'mmau_results.json')

```

### **3. Run Scoring Script**

使用提供的 `evaluation.py` 计算分数。

```bash
python evaluation.py --input mmau_results.json

```