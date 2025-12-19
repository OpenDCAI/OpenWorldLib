from sceneflow.pipelines.qwen_omni.pipeline_qwen2p5_omni import Qwen2p5_OmniPipeline
import soundfile as sf

#support more than one image/video/audio input
images = "./data/test_case1/ref_image.png" 
videos = "./data/test_video_case1/talking_man.mp4"
audios=None
return_audio=False
test_prompt = "Describe this video"

model_path = "Qwen/Qwen2.5-Omni-7B"
pipeline = Qwen2p5_OmniPipeline.from_pretrained(
    pretrained_model_path=model_path,
    use_audio_in_video=False,
)
if return_audio:
    text,audio = pipeline(text=test_prompt, images=images, videos=None, audios=None,return_audio=return_audio)
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
else:
    text = pipeline(text=test_prompt, images=images, videos=None, audios=None,return_audio=return_audio)
print(text)
