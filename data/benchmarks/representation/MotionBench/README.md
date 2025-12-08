# **MotionBench — Benchmark Record（Final Version）**

## **1. Meta**

***Name***: MotionBench
***Task***: Video-based Motion Reasoning QA Benchmark
***Paper***: [https://arxiv.org/pdf/2501.02955](https://arxiv.org/pdf/2501.02955)
***Code***: [https://github.com/zai-org/MotionBench](https://github.com/zai-org/MotionBench)
***Benchmark Code Path***: `motionbench/metrics/`
***Dataset Path***: [https://huggingface.co/datasets/zai-org/MotionBench](https://huggingface.co/datasets/zai-org/MotionBench)
***Task Type***: Video Motion Understanding / Motion Reasoning（MCQ QA）

> MotionBench 并非用于评估光流、物理一致性等传统 motion 任务，而是专注于视频中的 **动作逻辑、顺序关系、重复次数、运动物体、位置变化、镜头运动** 等更高层次的推理能力。

---

## **2. Dataset Structure**

MotionBench 的数据由 **视频、motion caption、多项选择 QA** 三部分构成。

```
$DATA_PATH/motionbench/
├── videos/
│   ├── P9xLXmJgGNT6MRIN.mp4
│   ├── A6q3Vhs9aEjOwMKe.mp4
│   ...
├── motionbench.json          # detailed motion captions
└── video_info.meta.jsonl     # QA metadata
```

### **Sample — motionbench.json**

```json
{
    "key": "P9xLXmJgGNT6MRIN",
    "motion_caption": "The woman in the center of the screen...",
    "video_path": "P9xLXmJgGNT6MRIN.mp4",
    "duration": 23.0833
}
```

### **Sample — video_info.meta.jsonl**

```json
{
    "video_id": "P9xLXmJgGNT6MRIN",
    "question_type": "Action Order",
    "qa": [
        {
            "uid": "123456",
            "question": "...",
            "options": ["A ...", "B ...", "C ...", "D ..."],
            "answer": "B"
        }
    ]
}
```

---

## **3. IO Specification**

### **Input: Model Predictions**

模型需要输出一个字典，以 `uid → 选项（A/B/C/D）` 的形式保存：

```json
{
    "123456": "A",
    "123457": "D",
    "123458": "B"
}
```

文件名可自定，例如 `model_answers.json`。

### **Output: Evaluation Results**

评测脚本会生成整体 accuracy 与各类别 accuracy，例如：

```json
{
    "Action Order": 0.63,
    "Camera Motion": 0.57,
    "Location-related Motion": 0.54,
    "Motion Recognition": 0.67,
    "Motion-related Objects": 0.62,
    "Repetition Count": 0.59,
    "acc": 0.61,
    "answered_acc": 0.61,
    "total_qa_num": 6052,
    "total_answered_num": 6052,
    "right_num": 3672
}
```

---

## **4. Metrics Specification**

（完全基于 `motionbench/metrics/compute_accuracy.py`）

MotionBench 的评测逻辑非常直接：核心就是多选题的**标准准确率计算**。

### **Question Types（6 类）**

| Category Name                     |
| --------------------------------- |
| **Action Order**            |
| **Camera Motion**           |
| **Location-related Motion** |
| **Motion Recognition**      |
| **Motion-related Objects**  |
| **Repetition Count**        |

## **4.1 Evaluation Metrics**

| 指标名                       | 描述                            | 范围 | 计算公式                                  |
| ---------------------------- | ------------------------------- | ---- | ----------------------------------------- |
| **acc**                | 全部 QA 的总体正确率            | 0–1 | `right_num / total_qa_num`              |
| **answered_acc**       | 模型实际回答的样本中的正确率    | 0–1 | `right_num / total_answered_num`        |
| **category_acc**       | 六类 question_type 的分类正确率 | 0–1 | `category_right[c] / category_total[c]` |
| **total_qa_num**       | QA 总数量                       | ≥0  | 计数                                      |
| **total_answered_num** | 模型实际回答的数量              | ≥0  | 匹配到 uid 的数量                         |
| **right_num**          | 答对的总数                      | ≥0  | 计数                                      |

---

## **5. Evaluation**

### **1. Generate Model Predictions**

为每条 QA 样本输出一个选项（A/B/C/D），并保存为 JSON 文件：

```json
{
  "uid_0001": "A",
  "uid_0002": "D",
  "uid_0003": "B"
}
```

文件名可自定，如：

```
model_answers.json
```

**`uid` 必须与 `video_info.meta.jsonl` 保持一致，否则视为“未作答”。**

### **2. Run Evaluation Script**

运行官方评测脚本：

```bash
python motionbench/metrics/compute_accuracy.py \
    --pred model_answers.json \
    --meta data/video_info.meta.jsonl
```

将输出：

* **Overall Accuracy**
* **Per-Category Accuracy（6 类）**
* **Answered Accuracy**
* **统计计数（answered / correct / total）**

评测过程清晰透明，无额外参数或隐藏逻辑。

### **3. Quick Start（Random Baseline）**

想快速验证环境，可直接运行随机预测脚本：

```bash
cd scripts
python construct_random_answers.py
python test_acc.py
```

运行后会得到：

```
scripts/random_answers.json
```

适合用于测试评测流程和 leaderboard 提交流程。

### **4. Submit to Leaderboard**

将模型在 **TEST 集** 上的预测 JSON 上传到官方 leaderboard：

[https://huggingface.co/spaces/THUDM/MotionBench](https://huggingface.co/spaces/THUDM/MotionBench)

即可查看最终排名与成绩表现。
