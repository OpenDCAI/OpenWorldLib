# **4D-Bench — Benchmark Record**

## **1. Meta**

**Name:** 4D-Bench
**Task:** 4D Object Understanding（QA + Captioning）
**Paper:** [https://arxiv.org/pdf/2503.17827](https://arxiv.org/pdf/2503.17827)
**Code:** [https://github.com/WenxuanZhu1103/4D-Bench](https://github.com/WenxuanZhu1103/4D-Bench)

**Benchmark Code Path**

| Module             | Path                                              |
| ------------------ | ------------------------------------------------- |
| QA Inference       | `4D_Object_Question_Answering/eval_code_example/` |
| Caption Generation | `4D_Object_Captioning/code/mllm_gen_caption/`     |
| Caption Evaluation | `4D_Object_Captioning/code/eval_metrics/`         |

**Dataset:**
[download from huggingface](https://huggingface.co/datasets/vxuanz/4D-Bench)

**Task Type:**
Multiview + Temporal 4D Object Understanding
（6 视角视频 × 时序动态，覆盖 Appearance / Action / Motion / Relations）

---

## **2. Dataset Structure**

```
$DATA_PATH/4D-Bench/
│
├── 4D_Object_Question_Answering
│   ├── data/
│   │   ├── 4d_qa.json
│   │   └── 4d_object_multi_view_videos/
│   └── eval_code_example/*.py
│
└── 4D_Object_Captioning
    ├── data/
    │   ├── human_annotations.csv
    │   └── 4d_object_multi_view_videos/
    └── code/
        ├── mllm_gen_caption/*.py
        └── eval_metrics/
```

**Example Metadata (官方样例)**

```json
{
  "f7c1e3ade...e6be2d187": {
    "fileIdentifier": "tree.glb",
    "source": "github",
    "metadata": {
      "animation_count": 23,
      "material_count": 6,
      "poly_count": 11521,
      "vert_count": 13213,
      "scene_size": {
        "bbox_max": [78.57, 63.90, 157.64],
        "bbox_min": [-83.04, -46.96, 3.49]
      }
    }
  }
}
```

---

## **3. IO Specification**

### **Input**

| Task       | Input 内容                              |
| ---------- | ------------------------------------- |
| QA         | 6-view 视频序列 + 单个 Question + 4 options |
| Captioning | 6-view 视频序列                           |

### **Output**

| Task       | Output 格式                                |
| ---------- | ---------------------------------------- |
| QA         | 单个字母：`A` / `B` / `C` / `D`               |
| Captioning | 一段自然语言描述（需涵盖 appearance + action/motion） |

---

## **4. Metrics Specification**

# **4.1 QA Metrics（真实官方分类）**

评测由以下脚本执行：

```
4D_Object_Question_Answering/eval_code_example/*.py
```

脚本会将预测与 `4d_qa.json` 对齐，并根据 `type` 字段自动计算分类 Accuracy。

### **QA Metrics Table**

| Metric                         | 描述                  |
| ------------------------------ | ------------------- |
| **Overall Accuracy**           | 所有样本整体正确率           |
| **Appearance Accuracy**        | 外观属性（颜色/形状/材质等）相关问题 |
| **Action Accuracy**            | 动作 / 行为理解           |
| **Counting Accuracy**          | 物体数量判断              |
| **Spatial Relation Accuracy**  | 空间位置关系（上/下/前/后/左右）  |
| **Temporal Relation Accuracy** | 时间顺序、动态变化理解         |

---

# **4.2 Captioning Metrics（完全基于官方 eval_metrics 实现）**

Captioning 共有两类指标：

---

## **(A) GPT-based LLM Score**

路径：

```
4D_Object_Captioning/code/eval_metrics/llm_score/
│── score_appearance.py
│── score_action.py
```

LLM 会比较：

**你的 caption** vs **多条人工 reference captions**

并给出可解释评分。

### **LLM Score Metrics**

| Metric                   | 描述                 |
| ------------------------ | ------------------ |
| **GPT-Appearance Score** | Caption 对外观描述的准确性  |
| **GPT-Action Score**     | Caption 对动作/动态的准确性 |

输出结果：

```
appearance_score.json
action_score.json
```

可同时评测多个 caption 字段（如 `gpt4o_caption` / `my_vlm_caption`）。

---

## **(B) 传统 Caption Metrics（真实脚本实现）**

路径：

```
4D_Object_Captioning/code/eval_metrics/other_metrics/get_other_metrics_scores.py
```

### **Classical Metrics Table**

| Metric         | 描述                     |
| -------------- | ---------------------- |
| **BLEU-1~4**   | N-gram 重叠度             |
| **METEOR**     | Recall-oriented + 对齐惩罚 |
| **ROUGE-L**    | 最长公共子序列匹配              |
| **CIDEr**      | TF-IDF 加权 n-gram       |
| **BERTScore**  | 语义相似度（BERT）            |
| **SBERTScore** | 语义相似度（Sentence-BERT）   |

最终输出：

```
other_metrics_scores.json
```

---

# **5. Evaluation**

---

## **5.1 QA Evaluation**

模板代码位置：

```
4D_Object_Question_Answering/eval_code_example/
```

### **如何添加自己的模型**

以 `qwen2_vl_7b_exp.py` 为模板，复制为：

```
my_vlm_exp.py
```

只需修改 3 个部分：

1. 模型加载方式
2. 推理接口（处理多视角视频 → 生成答案）
3. 将结果写入 JSON 的部分

### **运行示例**

```bash
python my_vlm_exp.py \
    --save_path ./results/my_vlm.json \
    --vqa_file_path ./data/4d_qa.json \
    --video_data_path ./data/4d_object_multi_view_videos \
    --cache_dir <your_model_path>
```

预测文件格式：

```json
{
  "000001": {"pred": "B"},
  "000002": {"pred": "D"}
}
```

---

## **5.2 Captioning Evaluation**

推理模板：

```
4D_Object_Captioning/code/mllm_gen_caption/
```

复制模板为：

```
my_vlm_caption_exp.py
```

并修改模型加载与 forward。

### **运行推理**

```bash
python my_vlm_caption_exp.py \
    --video_data_path ./data/4d_object_multi_view_videos \
    --results_save_path ./caption_results/my_vlm/
```

输出：

```json
{
  "000001": {
    "caption": "A rotating tree with green leaves..."
  }
}
```

---

## **5.3 Captioning Metrics**

### **(A) LLM-based Metrics**

```bash
python eval_metrics/llm_score/gpt_appearance_action_metrics.py \
    --results_save_path ./caption_results/my_vlm
```

### **(B) Classical Metrics**

```bash
python eval_metrics/other_metrics/eval_metrics.py \
    --results_save_path ./caption_results/my_vlm
```

最终会生成：

* `metrics.json`
* `scores.csv`