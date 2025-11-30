from diffusers.utils import export_to_video
from PIL import Image
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline


image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = "Skywork/Matrix-Game-2.0"
pipeline = MatrixGame2Pipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    mode="universal",
    device="cuda"
)

output_video = pipeline(
    input_image=input_image,
    num_output_frames=150,
    interaction_signal=["forward", "left", "right",
                        "forward_left", "forward_right",
                        "camera_l", "camera_r"]
)

export_to_video(output_video, "matrix_game_2_demo.mp4", fps=12)
