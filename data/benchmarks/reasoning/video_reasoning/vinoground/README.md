# **VinoGround — Benchmark Record（Revised）**

## **1. Meta**

* **Name**: VinoGround
* **Task**: Temporal Counterfactual Video Grounding
* **Paper**: [https://arxiv.org/pdf/2410.02763](https://arxiv.org/pdf/2410.02763)
* **Code**: [https://github.com/Vinoground/Vinoground](https://github.com/Vinoground/Vinoground)
* **Benchmark Code Path**: `vinoground/eval/*.py`
* **Dataset Path**: `https://huggingface.co/datasets/HanSolo9682/Vinoground`(download from huggingface)
* **Task Type**: *Temporal Video–Text Matching / Event Order Reasoning*

> **核心思想**：每条数据都有 *正事件顺序*（pos）和 *反事实颠倒顺序*（neg）。
> 模型必须判断 **事件 A 是否真的发生在事件 B 之前**。

---

## **2. Dataset Structure**

```
$DATA_PATH/vinoground/
│
├── vinoground.csv                        # 原始标注（pos/neg + major/minor）
├── vinoground_qa.json                    # QA 格式，供模型输入
│
├── vinoground_videos/
│   ├── 0_pos.mp4
│   ├── 0_neg.mp4
│   ├── 1_pos.mp4
│   ├── 1_neg.mp4
│   ...
│
└── vinoground_videos_concated/
    ├── 0.mp4           # segmentA + black + segmentB
    ├── 0_reverse.mp4   # segmentB + black + segmentA
    ├── 1.mp4
    ├── 1_reverse.mp4
```

## **2.1 CSV Annotation Format**

| 字段名            | 描述                                                                |
| ----------------- | ------------------------------------------------------------------- |
| `index`         | 样本序号（每条 index 对应一个 pos/neg 样本对）                      |
| `major`         | 主类别：`action / object / viewpoint`                             |
| `minor`         | 细类：`interaction / spatial / cyclical / contextual`（可多标签） |
| `pos_vid`       | 正样本视频 ID                                                       |
| `pos_start,end` | 正样本时间片段                                                      |
| `pos_cap`       | 正事件顺序的 caption                                                |
| `neg_vid`       | 负样本视频 ID                                                       |
| `neg_start,end` | 反事实（顺序颠倒）的时间片段                                        |
| `neg_cap`       | 负 caption（顺序相反）                                              |

示例：

```
index major minor pos_vid pos_start pos_end pos_cap                                       neg_vid neg_start neg_end neg_cap
0     action       QINQHWlQIzU 5   15  a toddler plays...                                 QINQHWlQIzU 10  20  a toddler picks up...
1     action       pVD1fx2Hb0c 0   10  the person begins...                               MX1hcxfiltU 32  42  the fishing pole...
2     viewpoint    tT1NpFX14LE 25  30  the camera shows cockpit then outside...           tT1NpFX14LE 35  38  ...outside then cockpit
...
```

## **2.2 QA Format（vinoground_qa.json）**

JSON 格式包含两类任务：

### **Type A — Single Video + Two Captions（二选一）**

```json
{
    "video_name": "vinoground_videos/0_pos.mp4",
    "question": "Which caption best describes this video?\nA. ...\nB. ...\nAnswer with the option's letter.",
    "GT": "B",
    "idx": "0_pos"
}
```

特征：

* 输入：一个视频
* 输出：选 A 或 B
* `GT` 由 pos/neg 定义（pos 的正确 caption 是顺序正确的那句）


### **Type B — Concatenated Video（Segment A vs B）**

```json
{
    "video_name": "vinoground_videos_concated/0.mp4",
    "question": "Which video segment matches this caption?\nA. First segment\nB. Second segment",
    "GT": "A",
    "idx": "0_pos"
}
```

特征：

* 视频由 **两段真实时间片段** 拼接 + 中间 2 秒黑屏
* 输入 caption
* 模型判断 caption 描述的事件顺序属于前半段还是后半段

---

## **3. I/O Specification（给模型的任务）**

### **Input**

模型需要处理的输入来自 QA JSON：

### 已定义的两种任务：

1. **Caption Matching（视频 → 选项 A/B caption）**
2. **Segment Matching（拼接视频 → 选项 A/B segment）**

所有任务统一为：

> “只需要输出 A / B。”

### **Output**

只需要输出一个简单的 JSON 文件：

```json
{
  "0_pos": "B",
  "0_neg": "B",
  "1_pos": "A",
  "1_neg": "A",
  ...
}
```

**key = `idx`**
**value = `"A"` / `"B"`**

和 `vinoground_qa.json` 一一对应。

---

## **4. Metrics Specification（和 CSV/QA 对齐）**

| Metric                    | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| Text Score (%)            | Accuracy on text preference pairs (pos_cap vs neg_cap)         |
| Video Score (%)           | Accuracy on video preference pairs (pos_video vs neg_video)    |
| Group Score (%)           | Requires both text & video predictions of a pair to be correct |
| Category-wise Text Score  | Same as Text Score but computed per category                   |
| Category-wise Video Score | Same as Video Score but computed per category                  |
| Category-wise Group Score | Same as Group Score but computed per category                  |

---

下面是更流畅、结构更清晰、读起来更像正式 benchmark 文档的版本，顺便保持一点轻松语气：

---

## **5. Evaluation**

评测分为两步：**先生成预测，再汇总评分**。

### **Step 1 — 运行模型，生成预测 JSONL**

模型需要分别对 **text score** 与 **video score** 两种任务进行推理，并将结果写入：

```
./outputs/<model_name>/textscore-response.jsonl
./outputs/<model_name>/videoscore-response.jsonl
```

预测格式如下：

#### textscore-response.jsonl

```jsonl
{"idx": "0_pos", "response": "B"}
{"idx": "0_neg", "response": "A"}
{"idx": "1_pos", "response": "A"}
...
```

#### videoscore-response.jsonl

```jsonl
{"idx": "0_pos", "response": "A"}
{"idx": "0_neg", "response": "B"}
...
```

其中：

* `idx` 对应样本编号
* `response` 为模型选择的 `"A"` 或 `"B"`

完成这一步后，模型预测就准备好进入评分流程。

### **Step 2 — 运行官方统计脚本，生成最终评测表**

当预测文件就绪后，只需执行：

```bash
python evaluate_all_models.py \
    --data ./Vinoground \
    --results ./outputs
```

脚本会自动：

* 加载 ground truth
* 对比你的预测
* 计算 Text Score / Video Score / Group Score
* 生成各类别细粒度指标
* 并最终导出一个 `vinoground_evaluation_results.xlsx`

获得全面评分。

