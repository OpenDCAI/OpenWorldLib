

from sceneflow.pipelines.spatial_reasoner.pipeline_spatial_reasoner import (
    SpatialReasonerPipeline,
)



MODEL_PATH="hongxingli/SpatialReasoner-3B"
def test_spatial_reasoner_pipeline_image(MODEL_PATH):
    pipe = SpatialReasonerPipeline.from_pretrained(MODEL_PATH)
    image_path = "./data/test_case1/ref_image.png"
    instruction = "Describe the scene."
    output = pipe(
        instruction=instruction,
        image_paths=[image_path],
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print(output[0])


def test_spatial_reasoner_pipeline_video(MODEL_PATH):
    pipe = SpatialReasonerPipeline.from_pretrained(MODEL_PATH)
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
    test_spatial_reasoner_pipeline_image(MODEL_PATH=MODEL_PATH)
    test_spatial_reasoner_pipeline_video(MODEL_PATH=MODEL_PATH)
