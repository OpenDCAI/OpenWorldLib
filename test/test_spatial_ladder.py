
from sceneflow.pipelines.spatial_ladder.pipeline_spatial_ladder import SpatialLadderPipeline


MODEL_PATH = "hongxingli/SpatialLadder-3B"


def test_spatial_ladder_pipeline_image():
    pipe = SpatialLadderPipeline.from_pretrained(MODEL_PATH)
    image_path = "./data/test_case1/ref_image.png"
    instruction = "Describe the scene."
    output = pipe(
        instruction=instruction,
        image_paths=[image_path],
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print(output[0])


def test_spatial_ladder_pipeline_video():
    pipe = SpatialLadderPipeline.from_pretrained(MODEL_PATH)
    video_path = "./data/test_video_case1/talking_man.mp4"
    instruction = "Summarize the video content."
    output = pipe(
        instruction=instruction,
        video_paths=[video_path],
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print(output[0])


if __name__ == "__main__":
    test_spatial_ladder_pipeline_image()
    test_spatial_ladder_pipeline_video()

# Note: 运行有报错：AttributeError: 'Qwen2_5_VLProcessor' object has no attribute 'process_vision_info'
