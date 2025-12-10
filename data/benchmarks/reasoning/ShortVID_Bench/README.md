# **ShortVID-Bench — Benchmark Record**

## **1. Meta**

* **Name**: ShortVID-Bench
* **Task**: High-level Understanding & Reasoning over Short Videos
* **Paper**: [https://www.emergentmind.com/topics/shortvid-bench](https://www.emergentmind.com/topics/shortvid-bench)
* **Code**: *N/A（官方仅发布数据集，暂无推理与评测脚本）*
* **Benchmark Code Path**: *N/A*
* **Dataset Path**: [download from huggingface](https://huggingface.co/datasets/TencentARC/ShortVid-Bench)
* **Task Types**:

  * Multiple-choice Video Reasoning
  * Creator Intent Classification
  * Affective Intent Classification
  * Temporal Reasoning / Localization

---

## **2. Dataset Structure**

```
ShortVID-Bench/
├── videos/
│   ├── xxx1.mp4
│   ├── xxx2.mp4
│   ...
└── annotations.json     # 全部题目的标注
```

### **annotations.json 示例结构**

```json
{
  "video": "26TM44a-6a0.mp4",
  "question": "What is the ultimate purpose of the creator making this video?",
  "candidates": [
    "A. ...",
    "B. ...",
    ...
  ],
  "answer": "C",
  "problem_type": "Creator Intent Taxonomy",
  "data_type": "video"
}
```

---

## **3. IO Specification**

由于无官方推理格式，本 benchmark 建议采用统一的 MCQ 输出规范。

### **Model Input**

* `video`: 原始 mp4 视频
* `question`: string
* `options`: list[str]

### **Model Output Format**

```json
{
  "video": "xxx.mp4",
  "question": "...?",
  "model_choice": "C",
  "problem_type": "Creator Intent Taxonomy"
}
```

推理结果存储路径示例：

```
./outputs/<model_name>/results.jsonl
```

每行一个样本。

---

## **4. Metrics Specification**

ShortVID-Bench **没有官方评测脚本或评价指标**，以下为通用推荐方案。

### **4.1 Overall Accuracy**

```
Accuracy = correct_predictions / total_samples
```

### **4.2 Category-wise Accuracy**

按 `problem_type` 分组：

* Creator Intent Accuracy
* Affective Intent Accuracy
* Temporal Reasoning Accuracy

### **Evaluation Implementation**

自行编写脚本读取：

* `annotations.json`
* `results.jsonl`

进行匹配与统计即可。

---

## **5. Evaluation Procedure**

### **(1) Run model inference**

推理全部视频样本，生成：

```
./outputs/<model_name>/results.jsonl
```

格式示例：

```jsonl
{"video": "26TM44a-6a0.mp4", "model_choice": "C"}
{"video": "Hre27BhZvM8.mp4", "model_choice": "C"}
{"video": "a_wxf-SGdHU.mp4", "model_choice": "E"}
```

视频名称需要与 `annotations.json` 中保持一致。

### **(2) Run evaluation**

```
python evaluate_shortvid.py \
    --anno ./ShortVID-Bench/annotations.json \
    --pred ./outputs/qwen2.5-vl/results.jsonl
```

输出包括：

* overall accuracy
* category-wise accuracy
* per-video accuracy（可选）
* exporting Excel/CSV（可选）
