# **AudioBench — Benchmark Record**

## **1. Meta**

* **Name**: AudioBench
* **Task**: Audio Large Language Model (AudioLLM) Evaluation
* **Paper**: [https://arxiv.org/abs/2406.16020](https://arxiv.org/abs/2406.16020)
* **Code**: [https://github.com/AudioLLMs/AudioBench](https://github.com/AudioLLMs/AudioBench)
* **Benchmark Code Path**: `AudioBench/`
* **Task Type**: Audio Understanding / Multimodal Instruction Following / Speech Translation

---

## **2. Dataset Structure**

AudioBench 通过 `dataset_name` 参数来指定数据集，支持超过 50 个数据集。数据通常由 `src/dataset.py` 动态加载。

**Supported Dataset Keys (Partial List):**

* **ASR**: `librispeech_test_clean`, `aishell_asr_zh_test`, `gigaspeech_test` ...
* **Speech Translation**: `covost2_en_zh_test`, `covost2_zh_en_test` ...
* **QA & Reasoning**: `cn_college_listen_mcq_test`, `slue_p2_sqa5_test`, `mmau_mini` ...
* **Audio Scene**: `clotho_aqa_test`, `wavcaps_test` ...
* **Paralinguistics**: `iemocap_emotion_test`, `voxceleb_gender_test` ...

**Sample Data Instance (Internal Representation)**

```python
# 被脚本转换后的格式
{
    "audio": "/path/to/audio/sample_01.wav",
    "instruction": "Please help me transcribe the speech into text.", 
    "reference": "Hello world",
    "task_type": "ASR"
}
```

---

## **3. IO Specification**

### **Input (Script Arguments)**

评测脚本 `main.py` 接收以下核心参数：

* `dataset_name`: (str) e.g., `librispeech_test_clean`
* `model_name`: (str) e.g., `Qwen2-Audio-7B-Instruct`
* `metrics`: (str) e.g., `wer`, `bleu`, `llama3_70b_judge`
* `number_of_samples`: (int) `-1` for all samples

### **Output (Log Structure)**

根据 `main.py` 中的 `file_save_folder = 'log_for_all_models'`，输出文件结构如下：

```
AudioBench/
└── log_for_all_models/
    └── <model_name>/
        └── <dataset_name>_<metrics>_score.json # 评分报告 (Scores)

```

*** Score File (`<dataset_name>_<metrics>_score.json`)**

```json
{
    "wer": 3.20,       // or "bleu", "accuracy", "judge_score"
    "details": [       // 前 20 个样本的详细打分情况
        { "prediction": "..." ...},
        ...
    ]
}

```

---

## **4. Metrics Specification**

| 函数名                        | 评测裁判 (Judge)                | 评测方法 (Methodology)          | 评分维度 (Dimensions)                                                                     | 输出分值范围                 | 适用任务                               |
| ----------------------------- | ------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- |
| gpt4o_as_judge                | GPT-4o (Azure)                  | Model-as-Judge（0–5 分打分制） | 质量评分（准确性、相关性、细节）                                                          | 0–100（原始 0–5 分 × 20） | 开放式问答（Open-ended QA）            |
| gpt4o_as_judge_binary         | GPT-4o (Azure)                  | Model-as-Judge（二元分类）      | 准确率（Accuracy）（0：错误/拒绝，1：正确）                                               | 0%–100%（Pass Rate）        | 事实性问答（Fact-based QA）            |
| llama3_70b_as_judge           | Llama-3-70B（Instruct / Quant） | Model-as-Judge（0–5 分打分制） | 质量评分（与参考答案的对齐度）                                                            | 0–100（原始 0–5 分 × 20） | 开放式问答（Open-ended QA）            |
| llama3_70b_as_judge_binary    | Llama-3-70B + Python Rules      | Hybrid Judge（混合评测）        | 综合成功率（Success Rate），包含：1）内容正确性（LLM 判断）；2）格式遵循（正则/代码检查） | 0%–100%（及各维度细分统计） | 复杂指令遵循（Constrained Generation） |
| llama3_as_judge（含 8B 版本） | Llama-3（8B）                   | Model-as-Judge（0–5 分打分制） | 质量评分（准确性、相关性）                                                                | 0–5（未归一化）             | 开放式问答（Open-ended QA）            |
| prometheus2_as_judge          | Prometheus 2（vLLM）            | Rubric-based Judge（基于量表）  | 量表评分（严格遵循 1–5 分定义）                                                          | 1–5（原始量表分）           | 通用生成质量评估                       |
| mmau_string_match             | 无（Python Script）             | Rule-based（Token Set Match）   | 选择题准确率（MCQ Accuracy），需包含正确答案且排除错误干扰项                              | 0%–100%                     | 多项选择题（MMAU 数据集）              |

---

## **5. Evaluation**


### **1. Register Your Model (Optional)**

If evaluating a new model, add a class in `src/model.py`:

```python
class MyNewModel(Model):
    def generate(self, input_data):
        # Implement inference logic
        return prediction

```

### **2. Start Judge Server (Crucial for Open-ended Tasks)**

对于需要 `llama3_70b_judge` 的任务，必须先启动 vLLM 服务。

```sh
# GPU 0: Hosting the Judge Model
bash vllm_model_judge_llama_3_70b.sh
# Check if port 5000 is active before proceeding

```

### **3. Run Evaluation**

使用 `main.py` (或封装好的 `eval.sh`) 进行推理和评测。

**Command Template:**

```sh
python main.py \
    --dataset_name <DATASET> \
    --model_name <MODEL_NAME> \
    --batch_size 1 \
    --overwrite True \
    --metrics <METRIC> \
    --number_of_samples -1

```
