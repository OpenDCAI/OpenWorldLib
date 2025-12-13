# **FAVOR-Bench — Benchmark Record**

## **1. Meta**

* **Name**: FAVOR-Bench
* **Task**: Fine-Grained Video Motion Understanding
* **Paper**: [https://arxiv.org/abs/2503.14935](https://arxiv.org/abs/2503.14935)
* **Code**: [https://github.com/FAVOR-Bench/FAVOR-Bench](https://github.com/FAVOR-Bench/FAVOR-Bench)
* **Benchmark Code Path**: `FAVOR_Bench`
* **Dataset Path**: [download from huggingface](https://huggingface.co/datasets/zl2048/FAVOR)
* **Task Types**:

  * **Close-ended**：六类多选题（AS / HAC / SAD / MAD / CM / NSM）
  * **Open-ended**：视频描述（caption）+ 双路线评估（LLM-based / LLM-free）

---

## **2. Dataset Structure**

```
$DATA_PATH/favor-bench/
├── videos/                         # Raw clips
│   ├── B7DNX.mp4
│   ├── 00X3U.mp4
│   ├── VR_h_z4JBQY_35.mp4
│   ├── NJMVwGXDKjU_60.mp4
│   ...
│
├── question_perspective.json       # Close-ended QA annotations
├── video_perspective.json          # Video-level structured motion info
└── train.json                      # FAVOR-Train (~17k videos), for supervised finetuning
```

---

## **3. Sample JSON**

### **3.1 `video_perspective.json`（视频级结构化信息）**

```json
{
  "video_name": "B7DNX.mp4",
  "questions": [
    {
      "question": "Based on holistic movement observation, what key stages constitute the core behavioral chain of the man in the black jacket?",
      "options": [
        "Sleep adjustment and clothing organization phase",
        "Dietary intake and hydration replenishment phase",
        "Multi-stage behavioral transition and articulation process",
        "Utensil manipulation and food distribution phase",
        "Postural changes and spatial relocation phase"
      ],
      "correct_answer": "Multi-stage behavioral transition and articulation process",
      "task_type": "HAC"
    }
  ],
  "caption": "A man in a black coat sleeps, then turns over and sits up on the sofa...",
  "camera_motion": "Stationary",
  "subject_attributes": "Subject 1: Man [black coat, gray pants, black hair]",
  "motion_list": "Subject 1: ...",
  "chronological_motion_list": "sleeps → turns over → sits up ... → walks to the right"
}
```

---

### **3.2 `question_perspective.json`（Close-ended QA）**

```json
{
  "question_key": "00X3U_1",
  "video_name": "00X3U.mp4",
  "question": "In the video, what action did the blonde woman perform during and immediately after the upward camera movement?",
  "options": [
    "Walked in from the right side and turned her head to the right",
    "Turned to the right and raised her hand",
    "Leaned forward and picked up a blanket",
    "Turned her head to the right and then immediately turned it to the left",
    "Closed the door and ran to the right"
  ],
  "correct_answer": "Turned her head to the right and then immediately turned it to the left",
  "task_type": "AS"
}
```

---

## **4. IO Specification**

### **Input**

#### **A. Video clips**

放置在：

```
./test_videos/*.mp4
```

#### **B. QA Metadata**

驱动 close-ended 推理的文件：

```
video_perspective.json
```

其格式：

```json
{
  "video_name": "B7DNX",
  "questions": [
    {
      "question": "...",
      "options": ["A", "B", "C", "D", "E"],
      "correct_answer": "C",
      "task_type": "HAC"
    }
  ]
}
```

---

### **Output（模型推理）**

Close-ended 推理会生成：

```
./output_qa/<model_name>.jsonl
```

示例行：

```json
{
  "B7DNX": [
    {
      "task_type": "HAC",
      "correct_answer": "C",
      "output": "The man ... Option C",
      "judge": true
    },
    {
      "task_type": "AS",
      "correct_answer": "A",
      "output": "I think A",
      "judge": false
    }
  ]
}
```

字段含义：

| 字段               | 含义       |
| ---------------- | -------- |
| `task_type`      | 六大任务类型之一 |
| `correct_answer` | 标准答案     |
| `output`         | 模型原始生成文本 |
| `judge`          | 自动判定是否正确 |

---

## **5. Metrics Specification**

FAVOR-Bench 分为两类指标：
**宏观（任务级别）** 与 **微观（描述质量级别）**。

### **I. Macro-level Accuracy（任务级精度）**

| 指标      | 依赖字段                                    | 适用任务      | 说明       |
| ------- | --------------------------------------- | --------- | -------- |
| **ALL** | —                                       | 全部 8184 题 | 简单正确率汇总  |
| **AS**  | chronological_motion_list               | 2637 题    | 强顺序动作识别  |
| **HAC** | motion_list                             | 1541 题    | 整体动作概括   |
| **SAD** | motion_list                             | 1662 题    | 单动作细节    |
| **MAD** | motion_list + chronological_motion_list | 1205 题    | 多动作、并行动作 |
| **CM**  | camera_motion                           | 1075 题    | 镜头运动识别   |
| **NSM** | 环境运动字段                                  | 64 题      | 非主体运动识别  |

---

### **II. Micro-level Scores（细粒度描述能力）**

| 指标                       | 依赖                        | 任务             | 说明           |
| ------------------------ | ------------------------- | -------------- | ------------ |
| Subject Precision/Recall | motion_list               | AS / SAD / MAD | 主体动作的准确性与覆盖度 |
| Subject Order Score      | motion_list               | AS / MAD       | 单主体动作顺序是否正确  |
| Chrono Precision/Recall  | chronological_motion_list | AS / MAD       | 跨主体时间线准确性    |
| Chrono Order Score       | chronological_motion_list | AS / MAD       | 全局动作顺序正确性    |
| Camera Motion Score      | camera_motion             | CM             | 运镜描述语义匹配     |
| Camera Order Score       | camera_motion             | CM             | 运镜阶段顺序       |
| **Final Score**          | 全部                        | ALL            | 综合加权得分       |

---

## **6. Evaluation Pipeline**

你如果想测评自己的模型，只需要改 `inference_qa_qwen.py` 中的 **模型加载部分**。

---

### **Close-ended Evaluation**

1. 下载视频：
   [https://huggingface.co/datasets/zl2048/FAVOR](https://huggingface.co/datasets/zl2048/FAVOR)
2. 安装依赖并准备模型（参考官方 repo）
3. 运行推理：

```
python inference_qa_qwen.py
```

输出结果位于 `./output_qa/`，并自动打印分数。

---

### **Open-ended Evaluation（LLM-free）**

caption 质量评估完全不依赖 LLM，流程如下：

#### **Step 1 — 提取结构信息**

```
cd LLM-free
run LLM-free_step1_extract.ipynb
```

输出提取后的动作序列、交互、摄像机运动等中间文件。

#### **Step 2 — 与 Ground Truth 对比**

```
python LLM-free_step2_compare.py
```

内部使用：

* **SequenceMatcher**（顺序匹配）
* **VideoActionEvaluator**（动作覆盖）
* **Embedding similarity**（语义相似度）

生成 open-ended caption 的最终多维评分。