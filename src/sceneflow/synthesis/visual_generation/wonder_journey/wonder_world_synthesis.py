import os
import numpy as np
from PIL import Image
import skimage.measure
import torch
from torchvision.transforms import ToTensor, ToPILImage
from huggingface_hub import snapshot_download
from diffusers import AutoPipelineForInpainting
from ...base_synthesis import BaseSynthesis
from ....representations.point_clouds_generation.wonder_journey.wonder_world.utils.utils import functbl


class WonderWorldSynthesis(BaseSynthesis):
    def __init__(self, inpaint_pipeline):
        super().__init__()
        self.inpaint_pipeline = inpaint_pipeline

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        args=None,
                        device=None,
                        **kwargs):
        """
        load the inpaint model for multiview generation
        """
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            # download from HuggingFace repo_id
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")

        inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
                model_root,
                safety_checker=None,
                torch_dtype=torch.bfloat16,
            ).to(device)

        return cls(inpaint_pipeline)

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode='cv2_telea', self_guidance=False, inpainting_prompt=None, negative_prompt=None, mask_strategy=np.min, diffusion_steps=50):
        # Handle resolution padding
        if self.inpainting_resolution > 512 and rendered_image.shape[-1] == 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[:, :, self.border_size:-self.border_size, self.border_size:-self.border_size] = inpaint_mask
            padded_rendered_image = self.border_image.clone()
            padded_rendered_image[:, :, self.border_size:-self.border_size, self.border_size:-self.border_size] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_rendered_image = rendered_image

        # Pre-fill (Telea)
        img = (padded_rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = padded_inpainting_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (padded_inpainting_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img, _ = functbl[fill_mode](img, fill_mask_)

        # Process mask (block reduce strategy)
        mask_block_size = 8
        mask_boundary = mask.shape[0] // 2
        mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)
        
        prompt = inpainting_prompt if inpainting_prompt is not None else self.inpainting_prompt
        neg_prompt = negative_prompt if negative_prompt is not None else (self.adaptive_negative_prompt + self.negative_inpainting_prompt if self.adaptive_negative_prompt else self.negative_inpainting_prompt)

        inpainted_image = self.inpainting_pipeline(
            prompt=prompt, negative_prompt=neg_prompt,
            image=init_image.resize((1024, 1024)), mask_image=mask_image.resize((1024, 1024)),
            num_inference_steps=diffusion_steps, guidance_scale=8.0, height=1024, width=1024, self_guidance=self_guidance
        ).images[0]

        inpainted_image = inpainted_image.resize((self.inpainting_resolution, self.inpainting_resolution))
        inpainted_image = ToTensor()(inpainted_image).to(self.device)
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)[None]

        self.post_mask_latest = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        self.inpaint_input_image_latest = init_image
        self.image_latest = inpainted_image
        return inpainted_image

    def generation_360_data(self, input_image, mask, text_prompt):
        """
        generation sky image
        """
        pass

    @torch.no_grad()
    def predict(self):
        pass
