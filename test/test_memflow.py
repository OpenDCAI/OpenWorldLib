from _path_defaults import env_int, env_str, model_path, python_bin
from openworldlib.pipelines.memflow.pipeline_memflow import MemFlowPipeline


def main():
    pipe = MemFlowPipeline.from_pretrained(
        model_path=model_path("MEMFLOW_CKPT_DIR", "MemFlow"),
        wan_model_path=model_path("MEMFLOW_WAN_MODEL", "Wan2.1-T2V-1.3B"),
        python_bin=python_bin("MEMFLOW_PYTHON"),
    )
    result = pipe(
        prompt=env_str("MEMFLOW_PROMPT", "A cinematic shot of a red sports car driving along a coastal highway at sunset."),
        output_dir=env_str("MEMFLOW_OUTPUT", "./output/memflow"),
        num_output_frames=env_int("MEMFLOW_NUM_OUTPUT_FRAMES", 12),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
