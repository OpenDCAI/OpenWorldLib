import random

import numpy as np
import torch
from giga_datasets import video_utils
from giga_train import TRANSFORMS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from ..pipeline import load_pipeline
from .action_utils import get_actions_from_states


@TRANSFORMS.register
class Cosmos25Transform:
    """A transformation class for preparing data for the Cosmos2.5 model.

    This class handles video sampling, resizing, cropping, normalization, and the generation of conditioning masks. It can also process actions and
    controlnet conditioning images.
    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        image_cfg: dict,
        action_cfg: dict | None = None,
        transfer_cfg: dict | None = None,
        fps: int = None,
    ):
        """Initializes the transform.

        Args:
            num_frames: The number of frames to sample from the video.
            height: The target height of the frames.
            width: The target width of the frames.
            image_cfg: Configuration for image processing, including the mask generator.
            action_cfg: Configuration for action processing.
            transfer_cfg: Configuration for controlnet/transfer learning processing.
            fps: The target frames per second.
        """
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        # Normalization transform: convert pixel values from [0, 1] to [-1, 1]
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.action_cfg = action_cfg
        self.transfer_cfg = transfer_cfg
        # Load pipelines for controlnet conditioning if specified
        if self.transfer_cfg is not None:
            mode = self.transfer_cfg['mode']
            if mode == 'edge':
                self.pipe = load_pipeline('edge_detection/canny', lazy=True)
            elif mode == 'blur':
                self.pipe = load_pipeline('others/blur', lazy=True)

    def __call__(self, data_dict):
        """Applies the transformations to a data dictionary.

        Args:
            data_dict: A dictionary containing the video and other relevant data.

        Returns:
            A new dictionary with the transformed data.
        """
        video = data_dict['video']
        video_length = len(video)
        # Sample frames from the video
        sample_indexes = np.linspace(0, video_length - 1, self.num_frames, dtype=int)
        input_images_np = video_utils.sample_video(video, sample_indexes, method=2)
        input_images = torch.from_numpy(input_images_np).permute(0, 3, 1, 2).contiguous()

        # --- Resize and Crop ---
        image_height = input_images.shape[2]
        image_width = input_images.shape[3]
        dst_width, dst_height = self.width, self.height
        # Maintain aspect ratio while resizing
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))
        # Random crop
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        # Normalize images
        input_images = input_images / 255.0
        input_images = self.normalize(input_images)

        # Generate conditioning mask
        cond_masks = self.mask_generator.get_mask(input_images.shape[0])
        new_data_dict = dict(
            fps=self.fps if self.fps is not None else video.get_avg_fps(),
            images=input_images,
            cond_masks=cond_masks,
            prompt_embeds=data_dict['prompt_embeds'],
        )

        # Process actions if configured
        if self.action_cfg is not None:
            action_scale = self.action_cfg.get('action_scale', 20.0)
            gripper_scale = self.action_cfg.get('gripper_scale', 1.0)
            states = data_dict['states']
            states = states[sample_indexes]
            actions = get_actions_from_states(states, action_scale=action_scale, gripper_scale=gripper_scale)
            new_data_dict['actions'] = actions

        # Process controlnet/transfer images if configured
        if self.transfer_cfg is not None:
            mode = self.transfer_cfg['mode']
            data_name = self.transfer_cfg.get('data_name', f'{mode}_video')
            if data_name in data_dict:
                # Use pre-computed conditioning video
                cn_video = data_dict[data_name]
                assert len(cn_video) == len(video)
                cn_images = video_utils.sample_video(cn_video, sample_indexes, method=2)
            else:
                # Generate conditioning images on the fly
                if mode == 'edge':
                    cn_images = [self.pipe(image) for image in input_images_np]
                elif mode == 'blur':
                    cn_images = self.pipe(input_images_np)
                else:
                    assert False, f'Unknown transfer mode: {mode}'
                cn_images = np.stack([np.array(cn_image) for cn_image in cn_images])
            # Apply the same resize and crop to the conditioning images
            input_cn_images = torch.from_numpy(cn_images).permute(0, 3, 1, 2).contiguous()
            assert input_cn_images.shape[2] == image_height and input_cn_images.shape[3] == image_width
            input_cn_images = F.resize(input_cn_images, (new_height, new_width), InterpolationMode.BILINEAR)
            input_cn_images = F.crop(input_cn_images, y1, x1, dst_height, dst_width)
            input_cn_images = input_cn_images / 255.0
            input_cn_images = self.normalize(input_cn_images)
            new_data_dict['cn_images'] = input_cn_images
        return new_data_dict


class MaskGenerator:
    """Generates a conditioning mask for the frames.

    The mask determines which frames are used as conditioning and which are predicted.
    """

    def __init__(self, max_cond_frames, factor=4, start=1):
        """Initializes the mask generator.

        Args:
            max_cond_frames: The maximum number of conditioning frames.
            factor: The downsampling factor from frames to latents.
            start: The minimum number of conditioning latents.
        """
        assert max_cond_frames > 0 and (max_cond_frames - 1) % factor == 0
        self.max_cond_frames = max_cond_frames
        self.factor = factor
        self.start = start
        self.max_cond_latents = 1 + (max_cond_frames - 1) // factor
        assert self.start <= self.max_cond_latents

    def get_mask(self, num_frames):
        """Generates a random conditioning mask.

        Args:
            num_frames: The total number of frames in the sequence.

        Returns:
            A tensor representing the mask, with 1s for conditioning frames and 0s for predicted frames.
        """
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_cond_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        # Randomly choose the number of conditioning latents
        num_cond_latents = random.randint(self.start, self.max_cond_latents)
        cond_masks = torch.zeros((1, num_latents, 1, 1), dtype=torch.float32)
        # Set the first `num_cond_latents` to 1
        cond_masks[0, :num_cond_latents] = 1
        return cond_masks
