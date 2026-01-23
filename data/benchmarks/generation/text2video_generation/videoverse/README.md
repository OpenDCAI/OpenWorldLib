# **VideoVerse — Benchmark Record（Refined Version）**

## **1. Meta**

* **Name**: VideoVerse
* **Task**: Text-to-Video (T2V) World-Model Evaluation
* **Paper**: [VIDEOVERSE: How Far Is Your T2V Generator From a World Model?](https://arxiv.org/pdf/2510.08398)
* **Code**: [https://github.com/Zeqing-Wang/VideoVerse](https://github.com/Zeqing-Wang/VideoVerse)
* **Benchmark Code Path**: `VIDEOVERSE/scripts`
* **Dataset**: [download from huggingface](https://huggingface.co/datasets/NNaptmn/VideoVerse)
* **Task Type**: Event-centric T2V evaluation / World-model capability assessment

---

## **2. Dataset Structure**

```
$DATA_PATH
│--- Scene_A
│    ├── sample1.mp4
│    ├── sample2.mp4
│    ...
│--- Scene_B
│    ├── sample1.mp4
│    ├── sample2.mp4
│    ...
└── prompts_of_VideoVerse.json
```

---

### **Sample JSON Entry**

```json
"8f348e44-546c-4319-aefa-b860c02d9cbc": {
    "verification_checks": [
        {
            "check_type": "Interaction",
            "question": "Does the ax make contact with the log and cause it to split or chip upon impact?"
        },
        {
            "check_type": "Attribution Correctness",
            "question": "Is the ax golden?"
        },
        {
            "check_type": "Natural Constraints",
            "question": "Does the log appear to be affected by fungal decay?"
        }
    ],
    "t2v_following_prompt": {
        "t2v_prompt": "A man walks through the woods holding a golden ax..."
    },
    "t2v_eval_event_info": {
        "verification_plan": [
            { "event_id": 1, "event_description": "A man walks through the woods holding an ax." },
            { "event_id": 2, "event_description": "The man steps on the log." },
            { "event_id": 3, "event_description": "He swings the ax to chop the log." }
        ]
    }
}
```

---

## **3. IO Specification**

### **Input**

从 `prompts_of_VideoVerse.json` 中读取样本，根据每个样本的 key 匹配对应的视频文件。

### **Output**

评测结果按原数据结构写回，每个条目包含多项 Yes/No 判断和事件排序输出。示例：

```json
"8f348e44-546c-4319-aefa-b860c02d9cbc": {
    "verification_checks": [
        { "check_type": "Interaction", "question": "...", "res": "yes" },
        { "check_type": "Attribution Correctness", "question": "...", "res": "no" },
        { "check_type": "Natural Constraints", "question": "...", "res": "yes" }
    ],
    "t2v_following_prompt": { "t2v_prompt": "A man walks..." },
    "t2v_eval_event_info": {
        "verification_plan": [...],
        "overall_event_res": "A,C,B",
        "overall_event_processed_res": "ACB"
    }
}
```

---

## **4. Metrics Specification**

| Category                               | Metric                            | Description                                             |
| -------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| **Temporal Understanding**       | Event Ordering Accuracy           | 按 A/B/C 正确排序事件的能力。                           |
| **Event-Level Understanding**    | Event Existence Accuracy          | 判断事件是否发生（Yes/No）。                            |
| **Object / Scene Understanding** | Static Question Accuracy          | 识别物体属性（如 ax 是否是 golden）。                   |
| **Interaction Understanding**    | Interaction Verification          | 判断物体/角色是否发生相互作用。                         |
| **Natural Constraints**          | Physics / Constraint Verification | 视频内容是否遵循自然规律。                              |
| **Instruction Following**        | Output Format Robustness          | 是否严格遵守 `<output></output>`、Yes/No 等格式要求。 |

---

## **5. Evaluation Guide**

运行：

```
python scripts/eval_with_other_vlm.py
```

### **评估其他模型：需要修改两个地方**

### **① 替换模型加载部分**

```python
from transformers import YourModelClass, YourProcessorClass

model_path = 'YourModelPath'

model = YourModelClass.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = YourProcessorClass.from_pretrained(model_path, use_fast=True)
```

只要模型能处理多模态输入（尤其是 video + text），这里就能直接换。

### **② 修改 single_request() 的推理流程**

原来的形式如下：

```python
text = processor.apply_chat_template(...)
inputs = processor(...)

generated_ids = model.generate(**inputs)
output_text = processor.batch_decode(...)
```

换其他模型时，只需改成它们自己的推理方式即可
