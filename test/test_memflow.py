import os

from openworldlib.pipelines.memflow.pipeline_memflow import MemFlowPipeline


model_path = os.environ.get("MEMFLOW_CKPT_DIR", "./models/MemFlow")
required_components = {
    "wan_model_path": os.environ.get("MEMFLOW_WAN_MODEL", "./models/Wan2.1-T2V-1.3B"),
}
prompt = os.environ.get(
    "MEMFLOW_PROMPT",
    "A cinematic shot of a red sports car driving along a coastal highway at sunset.",
)

pipeline = MemFlowPipeline.from_pretrained(
    model_path=model_path,
    required_components=required_components,
    python_bin=os.environ.get("MEMFLOW_PYTHON", "python"),
)

result = pipeline(
    prompt=prompt,
    output_dir=os.environ.get("MEMFLOW_OUTPUT", "./output/memflow"),
    num_output_frames=int(os.environ.get("MEMFLOW_NUM_OUTPUT_FRAMES", "12")),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    timeout=None,
)

print(result["video_path"])
