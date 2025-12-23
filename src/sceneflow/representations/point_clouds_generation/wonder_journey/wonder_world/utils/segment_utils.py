import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from .utils import save_depth_map


def save_sam_anns(anns, save_path="saved_image.png"):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.8]])
        img[m] = color_mask

    # Convert the image from float to uint8 for saving with PIL
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    pil_img.save(save_path)


def refine_disp_with_segments(disparity, segments, keep_threshold=7*0.3):
    """
    Refine disparity values based on provided segmentations.

    Args:
    - disparity (numpy.ndarray): The disparity array of shape [H, W].
    - segments (list): List of segmentation masks represented by dicts.

    Returns:
    - numpy.ndarray: The refined disparity array.
    """

    # Initialize refined_disparity as a copy of disparity
    refined_disparity = disparity.copy()

    # Iterate over each segmentation
    for segment in segments:
        mask = segment['segmentation']  # Extracting the mask

        # 3.a. Query the values from refined_disparity using the mask
        disp_pixels = refined_disparity[mask]
        
        p70 = np.percentile(disparity[mask], 70)  # 20 for garden to reserve flag
        p30 = np.percentile(disparity[mask], 30)
        disparity_range = p70 - p30

        # Check if disparity range is too significant to be a valid object
        if disparity_range > keep_threshold:
            refined_disparity[mask] = disparity[mask]
        else:
            # 3.b. Find the median value of these disp_pixels
            median_val = np.percentile(disp_pixels, 50)

            # 3.c. Set refined_disparity[mask] to the median value
            refined_disparity[mask] = median_val

    return refined_disparity

def refine_disp_with_segments_2(disparity, segments, keep_threshold=10, return_refined_mask=False, no_refine_mask=None,
                                existing_mask=None, existing_disp=None):
    """
    Refine depth values based on provided segmentations.

    Args:
    - disparity: The disparity array of shape [H, W].
    - segments (list): List of segmentation masks represented by dicts. The later segments will have higher priority to overwrite disp values.

    Returns:
    - numpy.ndarray: The refined disparity array.
    """

    refined_disparity = disparity.copy()
    refined_mask = np.zeros_like(disparity)
    # Iterate over each segmentation
    for segment in segments:
        mask = segment['segmentation']  # Extracting the mask

        if no_refine_mask is not None:
            intersection = np.logical_and(mask, no_refine_mask).sum()
            if intersection > 20:  # nontrivial intersection, discard this mask
                continue

        disp_pixels_old = disparity[mask]
        
        # Remove extreme values
        p80 = np.percentile(disp_pixels_old, 80)  # 20 for garden to reserve flag
        p20 = np.percentile(disp_pixels_old, 20)
        valid_px_mask = (disp_pixels_old >= p20) & (disp_pixels_old <= p80)
        valid_disp_px = disp_pixels_old[valid_px_mask]
        disp_std = valid_disp_px.std()

        # Check if disparity range is too significant to be a valid object
        if disp_std > keep_threshold:
            refined_mask[mask] = disp_std
            continue
        else:
            # 3.b. Find the median value of these disp_pixels
            median_val = np.percentile(valid_disp_px, 50)

            if existing_mask is not None:  # When we are doing layer-inpainted image
                if np.logical_and(mask, ~existing_mask).sum() < 5:  # This mask is outside of the inpainted region, we can skip it.
                    continue
                intersection_mask = np.logical_and(mask, existing_mask)
                if intersection_mask.sum() > 20:  # This is a mask that includes regions from the initial outpainted image
                    median_val = np.percentile(existing_disp[intersection_mask], 50)  # Use the existing region disp values

            # 3.c. Set refined_disparity[mask] to the median value
            refined_disparity[mask] = median_val

            refined_mask[mask] = disp_std

    # save_depth_map(refined_mask, 'tmp/refined_mask.png', vmax=keep_threshold)
    # save_depth_map(refined_disparity, 'tmp/refined_disp.png', vmax=1/0.001)
    if return_refined_mask:
        return refined_disparity, refined_mask
    else:
        return refined_disparity


def create_mask_generator():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    # sam_checkpoint = "/viscam/projects/wonderland/segment-anything/sam_vit_h_4b8939.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    return mask_generator

def create_mask_generator_repvit(model_root_path):
    from repvit_sam import SamAutomaticMaskGenerator, sam_model_registry
    sam_checkpoint = os.path.join(model_root_path, "repvit_sam.pt")
    model_type = "repvit"

    repvit_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    repvit_sam = repvit_sam.to(device='cuda')
    repvit_sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=repvit_sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        # min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    return mask_generator