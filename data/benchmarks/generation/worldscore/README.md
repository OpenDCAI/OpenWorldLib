# **WorldScore — Benchmark Record**

## **1. Meta**

* **Name**: WorldScore
* **Task**: Next-scene video generation evaluation
* **Paper**: [https://arxiv.org/pdf/2504.00983](https://arxiv.org/pdf/2504.00983)
* **Code**: [https://github.com/haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore)
* **Benchmark Code Path**: `worldscore/benchmark/`
* **Dataset Path**:

  * Raw: `$DATA_PATH/WorldScore-Dataset/`
  * Runtime ENV:

    ```sh
    export WORLDSCORE_PATH=/path/to/worldscore
    export MODEL_PATH=/path/to/model
    export DATA_PATH=/path/to/dataset
    ```
* **Task Type**: Video Generation / World Modeling

---

## **2. Dataset Structure**

```
$DATA_PATH/worldscore-dataset/
├── dynamic/
│   ├── photorealistic/
│   │   └── images/…
|   |   └── masks/…
│   ├── stylized/
│   │   └── images/…
|   |   └── masks/…
│   ├── dynamic.json
│   ├── photorealistic.json
│   └── stylized.json
└── static/
    ├── photorealistic/
    │   └── images/…
    |   └── masks/…
    ├── stylized/
    │   └── images/…
    |   └── masks/…
    ├── photorealistic.json
    ├── static.json
    └── stylized.json
```

**Sample JSON Entry**

```json
{
    "visual_movement": "dynamic",
    "visual_style": "photorealistic",
    "motion_type": "articulated",
    "style": "photorealistic",
    "objects": [
        "elephant"
    ],
    "prompt": "The elephant moves slowly across the grassy landscape, swinging its trunk and swaying side to side with a gentle rhythm.",
    "image": "./dynamic/photorealistic/images/articulated/001.jpg",
    "masks": [
        "./dynamic/photorealistic/masks/articulated/001/001-1.png"
    ],
    "camera_path": [
        "fixed"
    ]
}
```

---

## **3. IO Specification**

### **Input (instance)**

输入即来自上述 JSON 的一条记录("visual_movement", "visual_style", "motion_type", "style", "objects", "prompt", "image", "masks", "camera_path")
推理脚本会将其解析为：

```python
data_point = {
    "image_path": image_path,
    "inpainting_prompt_list": inpainting_prompt_list,
}
```

送入需要测评的model后会输出以下格式数据作为输入

```
output_dir/
├── image_data.json     # 元信息
├── time.txt            # 生成时间
├── frames/             # 帧序列（评测使用）
│   ├── 000001.png
│   ├── …
└── videos/             # 生成的视频
    └── output.mp4
```

生成过程（核心片段）：

```python
# world_generators\generate_videos.py
image_path, prompt_list = model_helper.adapt(instance)

for prompt in prompt_list:
    frames = generator.generate_video(prompt=prompt, image_path=image_path)

    # autoregressive update
    image_path = model_helper.save_image(frames[-1], ...)
```

WorldScore 仅需要「帧序列」和「元信息」，评测不会读取你的视频文件

### **output**

生成/<model_name>/worldscore_output/worldscore.json
内容类似下方

```
{
    "camera_control": 54.50,
    "object_control": 49.81,
    "content_alignment": 67.29,
    "3d_consistency": 46.65,
    "photometric_consistency": 73.05,
    "style_consistency": 49.66,
    "subjective_quality": 75.00,
    "motion_accuracy": 37.76,
    "motion_magnitude": 40.32,
    "motion_smoothness": 39.62,
    "WorldScore-Static": 59.42, 
    "WorldScore-Dynamic": 53.68 
}
```

---

## **4. Metrics Specification**

| Major Dimension           | Level-1 Metric (10)     | Code-Level Metric (11 classes)                 | Script Path                                                        |
| ------------------------- | ----------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| **Controllability** | Camera Controllability  | `CameraErrorMetric`                          | `worldscore/benchmark/third_party/camera_error_metrics.py`       |
|                           | Object Controllability  | `ObjectDetectionMetric`                      | `worldscore/benchmark/third_party/object_detection_metrics.py`   |
|                           | Content Alignment       | `CLIPScoreMetric`                            | `worldscore/benchmark/iqa_pytorch/clip_score_metrics.py`         |
| **Quality**         | 3D Consistency          | `ReprojectionErrorMetric`                    | `worldscore/benchmark/third_party/reprojection_error_metrics.py` |
|                           | Photometric Consistency | `OpticalFlowAverageEndPointErrorMetric`      | `worldscore/benchmark/third_party/flow_aepe_metrics.py`          |
|                           | Style Consistency       | `GramMatrixMetric`                           | `worldscore/benchmark/third_party/gram_matrix_metrics.py`        |
|                           | Subjective Quality      | `CLIPImageQualityAssessmentPlusMetric` (IQA) | `worldscore/benchmark/iqa_pytorch/clip_iqa_metrics.py`           |
|                           |                         | `IQACLIPAestheticScoreMetric` (Aesthetic)    | `worldscore/benchmark/iqa_pytorch/clip_aesthetic_metrics.py`     |
| **Dynamics**        | Motion Accuracy         | `MotionAccuracyMetric`                       | `worldscore/benchmark/third_party/motion_accuracy_metrics.py`    |
|                           | Motion Magnitude        | `OpticalFlowMetric`                          | `worldscore/benchmark/third_party/flow_metrics.py`               |
|                           | Motion Smoothness       | `MotionSmoothnessMetric`                     | `worldscore/benchmark/third_party/motion_smoothness_metrics.py`  |

---

## **5. Evaluation**

### **1. Register Your Model**

#### **(1) Create Config**

`config/model_configs/<model_name>.yaml`

```yaml
model: <model_name>
runs_root: /path/to/model
resolution: [W, H]
generate_type: i2v   # or t2v
frames: 16
fps: 8
```

#### **(2) Add Model Name**

`modeltype.py`

```python
"videogen": [
    "cogvideox_5b_i2v",
    "<model_name>",
]
```

#### **(3) Minimal Adapter**

`world_generators/<model_name>.py`

```python
class model_name:
    def __init__(self, model_name, generation_type, **kwargs):
        self.generate = ...

    def generate_video(self, prompt, image_path=None):
        return self.generate(prompt, image_path)
```

---

### **2. Generate Videos**

```sh
python world_generators/generate_videos.py --model_name <model_name>
```

---

### **3. Download Evaluation Checkpoints**（后续补充脚本）

```sh
bash scripts/download_metrics_ckpts.sh
```

---

### **4. Run Evaluation**

#### 单卡

```sh
python worldscore/run_evaluate.py --model_name <model_name>
```

#### 多卡（可选）

```sh
python worldscore/run_evaluate.py \
    --model_name <model_name> \
    --use_slurm True \
    --num_jobs <num_gpu>
```
