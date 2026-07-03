import os

from openworldlib.pipelines.solaris.pipeline_solaris import SolarisPipeline


model_path = os.environ.get("SOLARIS_MODEL_DIR", "./models/solaris")
required_components = {
    "dataset_dir": os.environ.get("SOLARIS_DATASET_DIR", "./datasets/solaris-eval-datasets"),
}

eval_datasets = [
    item.strip()
    for item in os.environ.get("SOLARIS_EVAL_DATASETS", "").split(",")
    if item.strip()
] or None
num_frames_eval = int(os.environ["SOLARIS_NUM_FRAMES_EVAL"]) if os.environ.get("SOLARIS_NUM_FRAMES_EVAL") else None

pipeline = SolarisPipeline.from_pretrained(
    model_path=model_path,
    required_components=required_components,
    python_bin=os.environ.get("SOLARIS_PYTHON", "python"),
)

result = pipeline(
    output_dir=os.environ.get("SOLARIS_OUTPUT", "./output/solaris"),
    eval_num_samples=int(os.environ.get("SOLARIS_EVAL_NUM_SAMPLES", "1")),
    eval_datasets=eval_datasets,
    num_frames_eval=num_frames_eval,
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    eval_metrics=os.environ.get("SOLARIS_EVAL_METRICS", ""),
    timeout=None,
)

print("\n".join(result["video_paths"]))
