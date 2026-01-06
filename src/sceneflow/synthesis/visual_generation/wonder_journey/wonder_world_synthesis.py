import os
import cv2
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
    def __init__(self, inpaint_pipeline, device):
        super().__init__()
        self.inpaint_pipeline = inpaint_pipeline
        self.device = device

        self.inpainting_prompt=""
        self.adaptive_negative_prompt=""
        self.negative_inpainting_prompt=""

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

        return cls(inpaint_pipeline, device)

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode='cv2_telea', self_guidance=False, inpainting_prompt=None,
                negative_prompt=None, mask_strategy=np.min, diffusion_steps=50, inpainting_resolution=512, border_mask=None, border_image=None,):
        # Handle resolution padding
        if inpainting_resolution > 512 and rendered_image.shape[-1] == 512:
            border_size=(inpainting_resolution - 512) // 2
            padded_inpainting_mask = border_mask.clone()
            padded_inpainting_mask[:, :, border_size:-border_size, border_size:-border_size] = inpaint_mask
            padded_rendered_image = border_image.clone()
            padded_rendered_image[:, :, border_size:-border_size, border_size:-border_size] = rendered_image
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

        inpainted_image = self.inpaint_pipeline(
            prompt=prompt, negative_prompt=neg_prompt,
            image=init_image.resize((1024, 1024)), mask_image=mask_image.resize((1024, 1024)),
            num_inference_steps=diffusion_steps, guidance_scale=8.0, height=1024, width=1024, self_guidance=self_guidance
        ).images[0]

        inpainted_image = inpainted_image.resize((inpainting_resolution, inpainting_resolution))
        inpainted_image = ToTensor()(inpainted_image).to(self.device)
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)[None]

        self.post_mask_latest = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        self.inpaint_input_image_latest = init_image
        self.image_latest = inpainted_image
        return inpainted_image

    def linear_blend(self, images, overlap=100):
        # create blending field
        alpha = np.linspace(0, 1, overlap).reshape(overlap, 1, 1)
        target_width, target_height = images[0].size

        for i, img in enumerate(images):
            img_new = np.array(img)
            if i != 0:
                overlap_img2 = img_new[512-overlap:, :, :]
                top_img = img_new[:512-overlap, :, :]
                blend_overlap = overlap_img1 * (1 - alpha) + overlap_img2 * alpha

                # combine the image
                blended_image = np.concatenate((top_img, blend_overlap, bottom_img), axis=0)
                img_old = blended_image
            else:
                img_old = img_new

            overlap_img1 = img_old[:overlap, :, :]
            bottom_img = img_old[overlap:, :, :]

        blended_image = (blended_image).astype(np.uint8)
        return Image.fromarray(blended_image).resize((target_width, target_height))

    def generation_360_data(self,
                            input_image, 
                            sky_text_prompt="blue sky",
                            negative_prompt="",
                            width=6144, height=512, overlap=128,
                            num_inference_steps=50, guidance_scale=7.5
                            ):
        """
        generation sky image using SyncDiffusion approach
        Args:
            input_image: PIL Image, the input base image
            sky_text_prompt: str, prompt for sky generation
            negative_prompt: str, negative prompt
            width: int, panorama width (default 6144)
            height: int, panorama height (default 512)
            overlap: int, overlap between blocks (default 128)
            num_inference_steps: int, diffusion steps
            guidance_scale: float, guidance scale
        Returns:
            dict: {
                'layer0': PIL.Image (building layer),
                'layer1': PIL.Image (sky layer)
            }
        """
        # Get input image dimensions
        input_w, input_h = input_image.size
        if input_image.size != (512, 512):
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            input_w, input_h = 512, 512
        
        w_start = 256  # Start position for preserving core region
        layers_panorama = 2
        style = "realistic, high quality"
        
        imgs_result = []
        
        # === Layer 0: Building Layer ===
        print(f"[WonderWorld] Processing Layer 0 (Building Layer)...")
        
        # 1. Create large canvas with tiled input
        init_image_layer0 = input_image.resize((width, height), Image.LANCZOS)
        
        # 2. Create mask - protect core region
        mask_image_layer0 = Image.new("L", (width, height), 255)  # White = inpaint
        keep_mask = Image.new("L", (input_w, input_h), 0)  # Black = keep
        mask_image_layer0.paste(keep_mask, (w_start, 0))
        
        # 3. Generate Layer 0 with SyncDiffusion
        prompt_layer0 = f"horizon, distant hills, {sky_text_prompt}. {style}"
        layer0_image = self._syncdiff_generate(
            prompt=prompt_layer0,
            negative_prompt=negative_prompt or "tree, text, watermark, low quality, human",
            width=width,
            height=height,
            overlap=overlap,
            input_image=init_image_layer0,
            input_mask=mask_image_layer0,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            layer_id=0
        )
        
        imgs_result.append(layer0_image)
        
        # === Layer 1: Sky Layer ===
        print(f"[WonderWorld] Processing Layer 1 (Sky Layer)...")
        
        # 1. Extract top slice from Layer 0
        top_slice_height = 100
        top_slice = layer0_image.crop((0, 0, width, top_slice_height))
        
        # 2. Stretch to full image
        init_image_layer1 = top_slice.resize((width, height), resample=Image.Resampling.LANCZOS)
        
        # 3. Paste back the seam at bottom
        init_image_layer1.paste(top_slice, (0, height - top_slice_height))
        
        # 4. Create mask - protect bottom seam
        mask_image_layer1 = Image.new("L", (width, height), 255)  # White = inpaint
        keep_mask_layer1 = Image.new("L", (width, top_slice_height), 0)  # Black = keep
        mask_image_layer1.paste(keep_mask_layer1, (0, height - top_slice_height))
        
        # 5. Generate Layer 1 with SyncDiffusion
        prompt_layer1 = f"{sky_text_prompt}, {style}"
        layer1_image = self._syncdiff_generate(
            prompt=prompt_layer1,
            negative_prompt=negative_prompt or "tree, text, watermark, low quality, buildings, human",
            width=width,
            height=height,
            overlap=overlap,
            input_image=init_image_layer1,
            input_mask=mask_image_layer1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            layer_id=1
        )

        imgs_result.append(layer1_image)

        print("[WonderWorld] 360 Data Generation Complete!")

        return self.linear_blend(imgs_result)

    def _syncdiff_generate(self, prompt, negative_prompt, width, height, overlap, 
                        input_image, input_mask, num_inference_steps, guidance_scale, layer_id):
        """
        SyncDiffusion block-based generation
        """
        
        def linear_mask_blend(img_np, mask_np, scale_factor=16, overlap=100):
            h, w = img_np.shape[:2]

            small = cv2.resize(img_np, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_AREA)
            low_res = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

            if len(mask_np.shape) == 3:
                mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            else:
                mask_gray = mask_np.copy()
            _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

            dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)

            sdf = dist_inside - dist_outside
            alpha = np.clip(sdf / float(overlap) + 0.5, 0, 1)

            alpha = alpha * alpha * (3 - 2 * alpha)
            alpha = alpha[:, :, np.newaxis]

            img_float = img_np.astype(np.float32)
            low_res_float = low_res.astype(np.float32)
            blended = img_float * (1.0 - alpha) + low_res_float * alpha
            return np.clip(blended, 0, 255).astype(np.uint8)
        
        final_image = input_image.copy()
        final_mask = input_mask.copy()
        
        block_width = 512
        block_height = 512
        step_stride = block_width - overlap
        num_steps = (width - block_width) // step_stride + 2
        current_x = 0
        
        print(f"   [SyncDiffusion] Start: {width}x{height}, Steps: {num_steps}")
        
        for i in range(num_steps):
            if current_x + block_width > width:
                current_x = width - block_width
            
            # Crop regions
            crop_box = (current_x, 0, current_x + block_width, height)
            init_image = final_image.crop(crop_box)
            current_mask_slice = final_mask.crop(crop_box)
            
            # Protect left overlap (except first block)
            # i=1时mask需要修改
            if i > 0:
                mask_np = np.array(current_mask_slice)
                mask_np[:, :overlap] = 0  # Black = keep
                mask_image = Image.fromarray(mask_np)
            else:
                mask_image = current_mask_slice
            
            # Pre-fill with cv2 TELEA for layer 0
            init_image_np = np.array(init_image)
            mask_image_np = np.array(mask_image)
            
            if mask_image_np.max() > 0:
                print(f"   [SyncDiffusion] Block {i+1}: Applying Low resolution...")
                init_image_np = linear_mask_blend(init_image_np, mask_image_np)
                init_image = Image.fromarray(init_image_np)
            
            # Generate with inpainting pipeline
            generated_image = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image.resize((1024, 1024)),
                mask_image=mask_image.resize((1024, 1024)),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
            ).images[0]
            
            # Resize back to block size
            result_block = generated_image.resize((block_width, block_height), Image.LANCZOS)
            
            # Paste into final image
            if i == 0:
                final_image.paste(result_block, (current_x, 0))
            else:
                new_content = result_block.crop((overlap, 0, block_width, block_height))
                paste_x = current_x + overlap
                final_image.paste(new_content, (paste_x, 0))
            
            print(f"   [SyncDiffusion] Block {i+1}/{num_steps} done.")
            
            current_x += step_stride
            if current_x >= width:
                break
        
        return final_image

    @torch.no_grad()
    def predict(self):
        pass
