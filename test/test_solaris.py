from _path_defaults import dataset_path, env_csv, env_int, env_optional_int, env_str, model_path, python_bin
from openworldlib.pipelines.solaris.pipeline_solaris import SolarisPipeline


def main():
    pipe = SolarisPipeline.from_pretrained(
        model_path=model_path("SOLARIS_MODEL_DIR", "solaris"),
        dataset_dir=dataset_path("SOLARIS_DATASET_DIR", "solaris-eval-datasets"),
        python_bin=python_bin("SOLARIS_PYTHON"),
    )
    result = pipe(
        output_dir=env_str("SOLARIS_OUTPUT", "./output/solaris"),
        eval_num_samples=env_int("SOLARIS_EVAL_NUM_SAMPLES", 1),
        eval_datasets=env_csv("SOLARIS_EVAL_DATASETS"),
        num_frames_eval=env_optional_int("SOLARIS_NUM_FRAMES_EVAL"),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        eval_metrics=env_str("SOLARIS_EVAL_METRICS", ""),
        timeout=None,
    )
    print("\n".join(result["video_paths"]))


if __name__ == "__main__":
    main()
