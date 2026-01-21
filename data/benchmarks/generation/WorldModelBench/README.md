# **WorldModelBench — Benchmark Record**

## **1. Meta**

* **Name**: WorldModelBench
* **Task**: Judging video generation models as world models (VLM-judged)
* **Paper**: [https://arxiv.org/pdf/2502.20694](https://arxiv.org/pdf/2502.20694)
* **Code**: [https://github.com/WorldModelBench-Team/WorldModelBench](https://github.com/WorldModelBench-Team/WorldModelBench)
* **Benchmark Code Path**: `worldmodelbench/`
* **Dataset Path**:

  * Raw: `worldmodelbench/`（包含 `worldmodelbench.json` + `images/` first frames）
  * Runtime CLI Args:

    ```sh
    --video_dir /path/to/generated_videos
    --judge /path/to/judge_ckpt
    ```
* **Task Type**: Video Generation / World Modeling / VLM-as-Judge Evaluation

---

## **2. Dataset Structure**

```
worldmodelbench/
├── images/                     # first frames of videos (*.jpg)
├── worldmodelbench.json        # benchmark instances (350)
├── evaluation.py               # evaluator script (this file)
└── ...
```

### **Dataset Overview**

* **Domains**: 7（Robotics / Driving / Industry / Human / Gaming / Animation / Natural）
* **Subdomains**: 56
* **Total Instances**: 350

---

### **Sample JSON Entry**

```json
{
    "domain": "autonomous vehicle",
    "subdomain": "Stopping",
    "text_first_frame": "The autonomous vehicle approaches a traffic light on a bridge surrounded by tall buildings. Construction barriers line the sides of the bridge with a yellow traffic light visible ahead.",
    "text_instruction": "The autonomous vehicle stops at the traffic light on the bridge.",
    "first_frame": "images/69620089860948e38a4921dd4869d24f.jpg"
}
```

---

## **3. IO Specification**

### **Input (instance)**

Evaluator 从 `./worldmodelbench.json` 读入一条 instance，并在推理阶段使用 `first_frame` 的 **stem** 来定位视频文件：

```python
video_name = Path(instance["first_frame"]).stem
video_path = Path(video_dir) / f"{video_name}.mp4"
```

---

### **Model Prompt Construction（对齐 evaluator 实现）**

WorldModelBench 的 evaluator 里实际跑了 **3 类评测**（对应 `EvaluationType`）：

#### (1) Instruction Following（score 0–3）

* prompt template：`PROMPT_TEMPLATES["instruction"]`
* 输入变量：`{instruction}`（来自 `instance["text_instruction"]`）

```python
prompt = PROMPT_TEMPLATES["instruction"].format(
    instruction=instance["text_instruction"]
)
```

Judge 期望输出形如：`Score: X`，evaluator 会用：

```python
score = float(pred.split(":")[-1].strip(" ."))
```

解析分数（解析失败则记 0）。

#### (2) Physical Laws（5 个 Yes/No 子问题）

* prompt template：`PROMPT_TEMPLATES["physical_laws"]`
* question pool：5 条（Newton / mass&solid / fluid / penetration / gravity）
* evaluator 会把问题填入 `{physical_laws}`（注意：代码里使用 `.lower()`）

```python
prompt = PROMPT_TEMPLATES["physical_laws"].format(
    physical_laws=question.lower()
)
```

判分逻辑（代码实现）：

```python
acc = ("no" in pred.lower())   # "No" 表示“不违反”，记为 True
```

#### (3) Common Sense（2 个 Yes/No 子问题）

* prompt template：`PROMPT_TEMPLATES["common_sense"]`
* question pool：2 条（Poor Aesthetics / Temporal Inconsistency）
* evaluator 同样填 `{common_sense}`

```python
prompt = PROMPT_TEMPLATES["common_sense"].format(
    common_sense=question.lower()
)
```

判分同 Physical Laws：`"no" in pred.lower()` 记为 True。

---

### **Model Output Format (Required)**

WorldModelBench evaluator **只读取 mp4**，不需要帧目录或元信息：

```
<video_dir>/
├── 69620089860948e38a4921dd4869d24f.mp4
├── ...
```

⚠️ **命名约束（由 evaluator 强制）**

* 取 `Path(instance["first_frame"]).stem` 作为 `video_name`
* 在 `<video_dir>` 下查找 `f"{video_name}.mp4"`
* 找不到则 warning 并跳过该样本

---

### **Output (Evaluation Results)**

#### (1) Raw results（可选保存）

默认会保存（除非 `--no-save`）：

```text
<save_name>.json
# or <save_name>_cot.json (if --cot)
```

保存内容结构（由 evaluator 写入）：

```json
{
  "model_name": "TESTED_MODEL",
  "preds": { "...": { "instruction": [...], "physical_laws": [...], "common_sense": [...] } },
  "accs": { "instruction": [...], "physical_laws": [...], "common_sense": [...] }
}
```

#### (2) Final printed summary（由 `process_results` 汇总）

`process_results()` 会对 `accs` 做均值 / 分组汇总并打印。其内部的分组规则是：

* `instruction`: 每个样本 1 个 score（0–3），overall = mean
* `common_sense`: 每样本 2 个子项（framewise / temporal），overall = sum(两个均值)
* `physical_laws`: 每样本 5 个子项（newton / mass / fluid / penetration / gravity），overall = sum(五个均值)

---

## **4. Metrics Specification（以 evaluator 为准）**

> * Instruction：0–3 的打分
> * Physical Laws：5 个“是否违反物理规律”的二分类（以 “No” 为正确）
> * Common Sense：2 个二分类（以 “No” 为正确）

| Major Dimension  | Level-1 Metric           | Code-Level Metric (questions / scoring)             | Script Path     |
| ---------------- | ------------------------ | --------------------------------------------------- | --------------- |
| **Instruction**  | Instruction Following    | Score in {0,1,2,3} parsed from `Score: X`           | `evaluation.py` |
| **Physical**     | Physical Laws Compliance | 5 Yes/No questions; correct if output contains “No” | `evaluation.py` |
| **Common Sense** | Common Sense Compliance  | 2 Yes/No questions; correct if output contains “No” | `evaluation.py` |

### **Question Pools（代码内置）**

**Physical Laws（5）**

* Newton’s Law violation
* Conservation of Mass / Solid constitutive law violation
* Fluid constitutive law violation
* Non-physical penetration
* Gravity violation

**Common Sense（2）**

* Poor aesthetics
* Temporal inconsistency (flickering / abrupt changes)

---

### **Judge Model（代码视角）**

* evaluator 通过 `llava.load(judge_path)` 加载 judge（变量名 `self.judge`）
* `evaluate_video()` 调用：

  ```python
  self.judge.generate_content([video, prompt])
  ```
* `--cot` 会保留 “Let’s think step-by-step …” 提示词；不加则把 CoT prompt 改成 “Answer with …”

---

## **5. Evaluation（对齐 evaluator CLI）**

### **1. Prepare Files**

确保目录结构：

```
worldmodelbench/
├── worldmodelbench.json
├── evaluation.py
└── images/
```

并准备 `<video_dir>` 目录，里面是生成视频 `.mp4`（命名规则同上）。

---

### **2. Run Evaluation**

```sh
python evaluation.py \
    --model_name <model_name> \
    --video_dir <path_to_generated_videos> \
    --judge <path_to_judge_ckpt>
```

可选参数：

```sh
--cot                 # enable CoT prompt
--save_name xxx       # default: worldmodelbench_results
--no-save             # do not dump results json
```

---

## **6. Key Differences vs WorldScore**

| Aspect                  | WorldScore               | WorldModelBench                                        |
| ----------------------- | ------------------------ | ------------------------------------------------------ |
| Core Goal               | 视频生成质量 & 可控性             | 世界建模能力（VLM judge）                                      |
| Input                   | Image + prompt (+ masks) | first_frame stem 对应的视频 + judge prompts                 |
| Output required by eval | Frames + metadata        | Video only (.mp4)                                      |
| Metrics                 | Flow/CLIP/geometry       | Instruction score + physical/common-sense Yes/No pools |
| Dynamics                | Explicit motion metrics  | Implicit reasoning via judge                           |
| Horizon                 | Short autoregressive     | More task/goal driven                                  |