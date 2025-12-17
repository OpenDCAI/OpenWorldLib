import os
import torch
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from huggingface_hub import snapshot_download
from util.utils import prepare_scheduler
from util.segment_utils import create_mask_generator

class BaseSynthesis(object):
    def __init__(self, model):
        """
        初始化synthesis模型
        """
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        output: 对应的Synthesis类
        """
        raise NotImplementedError

    def api_init(self, api_key, endpoint):
        pass

    @torch.no_grad()
    def predict(self, data):
        pass

class VisualSynthesis(BaseSynthesis):
    def __init__(self, inpainter_pipeline, vae, mask_generator):
        self.inpainter_pipeline = inpainter_pipeline
        self.vae = vae
        self.mask_generator = mask_generator
        self.model = inpainter_pipeline # Base class compliance

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        """
        初始化 Stable Diffusion 和 SAM
        """
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            print(f"Downloading SD weights from HuggingFace repo: {pretrained_model_path}")
            try:
                model_root = snapshot_download(pretrained_model_path)
            except:
                model_root = pretrained_model_path
            print(f"Model downloaded to: {model_root}")

        inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_root,
            safety_checker=None,
            torch_dtype=torch.float16,
            # revision="fp16", # Commented out as per user's latest context
        ).to(device)
        
        inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
        inpainter_pipeline.scheduler = prepare_scheduler(inpainter_pipeline.scheduler)
        
        # Load VAE from the same checkpoint subfolder
        vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae").to(device)
        
        # Load SAM (using utility from segment_utils as in run.py)
        mask_generator = create_mask_generator()
        
        return cls(inpainter_pipeline, vae, mask_generator)

    def api_init(self, api_key, endpoint):
        pass

    @torch.no_grad()
    def predict(self, data):
        """
        Placeholder. The actual generation logic is inside KeyframeGen.inpaint
        which calls self.inpainter_pipeline
        """
        pass