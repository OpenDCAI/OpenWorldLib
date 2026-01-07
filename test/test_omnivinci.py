from sceneflow.pipelines.omnivinci.pipeline_omnivinci import OmniVinciPipeline
import soundfile as sf

# Support more than one image/video/audio input
images = "./data/test_case1/ref_image.png" 
videos = None
audios = None
test_prompt = "Describe this image"

# OmniVinci model path (replace with actual model path)
model_path = "nvidia/omnivinci"  

# Initialize pipeline with OmniVinci-specific parameters
pipeline = OmniVinciPipeline.from_pretrained(
    pretrained_model_path=model_path,
    load_audio_in_video=True, 
    num_video_frames=128,      # Number of frames to extract from video
)

# Run inference
text = pipeline(
    text=test_prompt, 
    images=images, 
    videos=videos, 
    audios=audios
)

print(text)
