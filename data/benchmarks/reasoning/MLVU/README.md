# **MLVU — Benchmark Record（Final Version）**

## **1. Meta**

**Name**: **MLVU — Multi-task Long Video Understanding Benchmark**
**Task**: Long Video Understanding（长视频理解，多类型推理）
**Paper**: [https://arxiv.org/pdf/2406.04264](https://arxiv.org/pdf/2406.04264)
**Code**: [https://github.com/JUNJIE99/MLVU](https://github.com/JUNJIE99/MLVU)

**Benchmark Code Path**: `MLVU/evaluation`
**Dataset Path**: `https://huggingface.co/datasets/MLVU/MVLU`download from huggingface
**Task Types**:

##### **A. Generation Tasks（生成类任务）**

* **Video Summarization（长视频摘要生成）**
* **Sub-plot Description（剧情片段描述）**

---

##### **B. Multiple-Choice Tasks（选择题任务，共 7 类）**

1. **Topic Reasoning** — 整体主题理解与高级推理
2. **PlotQA** — 剧情内容问答
3. **FindNeedle** — 长视频检索关键事件
4. **Ego** — 主观视角推理
5. **Count** — 数量统计与场景计数
6. **Order** — 时序顺序判断
7. **Anomaly Recognition** — 异常事件识别

---

## **2. Dataset Structure**

MLVU 数据采用 **视频文件 + JSON 标注文件** 的结构。

```
$DATA_PATH/mlvu/
├── videos/
│   ├── vid_xxx/
│   │   ├── xxx.mp4
│   │   └── ...
│   ...
└── json/
    ├── summary.json
    ├── subplot.json
    ├── topic_reasoning.json
    ├── plotQA.json
    ├── findNeedle.json
    ├── ego.json
    ├── count.json
    ├── order.json
    ├── anomaly_reco.json
```

---

## **3. Sample JSON Entry**

### **A. Generation Tasks**

#### **1）Summary**

```json
{
  "video": "217.mp4",
  "duration": 480.0,
  "question": "Please summarize this video, including its main content.",
  "answer": "The video starts with waves lapping against the rocks...",
  "question_type": "summary"
}
```

#### **2）Sub-Plot**

```json
{
  "video": "subPlot_new_all_126.mp4",
  "duration": 5632.83,
  "question": "Please describe the scene when the man in the green plaid shirt...",
  "answer": "The man in the green plaid shirt, wearing sunglasses...",
  "question_type": "subPlot",
  "scoring_points": [
    "The man leads the football players with a swagger",
    "A man in a suit runs to the three people",
    "The man in the suit introduces the man in the green plaid shirt"
  ]
}
```

---

### **B. Multiple-Choice Tasks（7 类）**

#### **1）topic_reasoning**

```json
{
  "video": "AWA-6.mp4",
  "duration": 450.0,
  "question": "What is the main background of the video?",
  "candidates": ["Grassland", "Lake", "Ocean", "Desert"],
  "answer": "Grassland",
  "question_type": "topic_reasoning"
}
```

---

#### **2）plotQA**

```json
{
  "video": "movie101_66.mp4",
  "duration": 246,
  "question": "What color is the main male character in the video?",
  "candidates": ["Yellow", "Red", "Green", "Blue"],
  "answer": "Yellow",
  "question_type": "plotQA"
}
```

---

#### **3）findNeedle**

```json
{
  "video": "needle_32.mp4",
  "duration": 467.98,
  "question": "What does the hand coming out of the computer do?",
  "candidates": [
    "Delivers a product",
    "Shakes the woman's hand",
    "Takes the woman's credit card",
    "Points at something on the screen"
  ],
  "answer": "Delivers a product",
  "question_type": "findNeedle"
}
```

---

#### **4）ego**

```json
{
  "video": "ego_35.mp4",
  "duration": 408.63,
  "question": "What did I put in the orange trashcan?",
  "candidates": [
    "a lemon green sponge",
    "a blue pen",
    "a red apple",
    "a yellow banana"
  ],
  "answer": "a lemon green sponge",
  "question_type": "ego"
}
```

---

#### **5）count**

```json
{
  "video": "count_126.mp4",
  "duration": 572.86,
  "question": "Throughout this video, what is the total count of occurrences for the scene featuring the 'playing trombone' action?",
  "candidates": ["2", "1", "5", "4"],
  "answer": "1",
  "question_type": "count"
}
```

---

#### **6）order**

```json
{
  "video": "order_126.mp4",
  "duration": 665.34,
  "question": "Arrange the following events in correct order: (1) Tape hands; (2) Starts boxing; (3) Sit ups; (4) Bikini photos.",
  "candidates": [
    "2->1->3->4",
    "3->2->1->4",
    "4->3->2->1",
    "1->2->3->4"
  ],
  "answer": "1->2->3->4",
  "question_type": "order"
}
```

---

#### **7）anomaly_reco**

```json
{
  "video": "surveil_20.mp4",
  "duration": 485.17,
  "question": "Does this surveillance footage contain any anomalies? If yes, what kind?",
  "candidates": ["RoadAccidents", "Shooting", "Shoplifting", "Assault"],
  "answer": "Shoplifting",
  "question_type": "anomaly_reco"
}
```

---

## **4. IO Specification**

### **Input（按任务类型）**

#### **Generation Tasks**

##### Summary

```json
{ "video": "path/video.mp4", "question": "Summarize the video." }
```

##### Sub-Plot

```json
{ "video": "path/video.mp4", "question": "Describe the scene when ..." }
```

---

#### **Multiple-Choice Tasks**

通用格式：

```json
{
  "video": "path/video.mp4",
  "question": "...",
  "candidates": ["A", "B", "C", "D"]
}
```

---

### **Model Output**

#### **Generation**

```
<free-form natural language>
```

#### **Multiple-Choice**

```
"B"
```

---

## **5. Metrics Specification**

### **Generation Tasks（LLM Judge）**

#### Summary

| Metric       | Range | Meaning                    |
| ------------ | ----- | -------------------------- |
| completeness | 1–5  | 重要内容覆盖               |
| reliability  | 1–5  | 事实准确性                 |
| total        | 2–10 | completeness + reliability |

---

#### Sub-Plot

| Metric    | Range | Meaning              |
| --------- | ----- | -------------------- |
| accuracy  | 1–5  | 覆盖 scoring_points  |
| relevance | 1–5  | 与问题的贴合度       |
| total     | 2–10 | accuracy + relevance |

---

### **Multiple-Choice Tasks（7 类）**

统一使用：

| Metric             | Meaning              |
| ------------------ | -------------------- |
| **Accuracy** | prediction == answer |

---

## **6. Evaluation**

测评前需要调用model，修改 `evaluation\multiple_choice_evaluation\choice_bench.py` 和 `evaluation\generation_evaluation\open_bench.py` 中的模型加载

```python
dataset = MLVU(data_dir, data_list)
'''
load your model
'''
correct = 0
total = 0
```

### **A. Multiple-Choice（7 类统一）**

```
python multiple_choice_evaluation/choice_bench.py
```

输出：overall accuracy

### **B. Generation Tasks**

#### **Step 1 — 模型生成**

```
python generation_evaluation/open_bench.py
```

生成：

* `summary_all.json`
* `subplot_all.json`

#### **Step 2 — Sub-Plot 评分**

```
python evaluate_ssc.py --pred_path <subplot_all.json> --output_dir <out> --output_json <res.json>
python calculate.py --path <out>
```

#### **Step 3 — Summary 评分**

```
python evaluate_summary.py --pred_path <summary_all.json> --output_dir <out> --output_json <res.json>
python calculate_sum.py --path <out>
```