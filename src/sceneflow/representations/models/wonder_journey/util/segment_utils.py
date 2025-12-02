import sys
sys.path.append("/viscam/projects/wonderland/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image


def save_sam_anns(anns, save_path="saved_image.png"):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

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


def create_mask_generator():
    sam_checkpoint = "/viscam/projects/wonderland/segment-anything/sam_vit_h_4b8939.pth"
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