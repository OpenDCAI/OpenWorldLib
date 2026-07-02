from _path_defaults import env_int, env_str, model_path
from openworldlib.pipelines.cosmos.pipeline_cosmos3 import Cosmos3Pipeline


def main():
    pipe = Cosmos3Pipeline.from_pretrained(
        model_path=model_path("COSMOS3_MODEL_PATH", "Cosmos3-Nano", fallback="nvidia/Cosmos3-Nano"),
    )
    result = pipe(
        prompt=env_str("COSMOS3_PROMPT", "A mobile robot navigates a warehouse aisle and stops at a shelf."),
        output_path=env_str("COSMOS3_OUTPUT", "./output/cosmos3/cosmos3_t2v.mp4"),
        num_frames=env_int("COSMOS3_NUM_FRAMES", 17),
        height=env_int("COSMOS3_HEIGHT", 352),
        width=env_int("COSMOS3_WIDTH", 640),
        fps=env_int("COSMOS3_FPS", 24),
        num_inference_steps=env_int("COSMOS3_STEPS", 1),
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
