from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import morphology
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from zim_anything import zim_model_registry, ZimPredictor


def build_gd_model(GROUNDING_MODEL, device="cuda"):
    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return processor, grounding_model


# build zim model
def build_zim_model(ZIM_MODEL_CONFIG, ZIM_CHECKPOINT, device="cuda"):
    # build zim-anything from huggingface
    zim_model = zim_model_registry[ZIM_MODEL_CONFIG](checkpoint=ZIM_CHECKPOINT).to(device)
    zim_predictor = DetPredictor(zim_model)

    return zim_predictor


class DetPredictor(ZimPredictor):
    def predict(
            self,
            point_coords: [np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch = None
        labels_torch = None
        box_torch = None

        if point_coords is not None:
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.float, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            multimask_output,
            return_logits=return_logits,
        )
        if not return_logits:
            masks = masks > 0.5

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].squeeze(0).float().detach().cpu().numpy()

        return masks_np, iou_predictions_np, low_res_masks_np


# filter the small bboxes to avoid memory overflow
def filter_small_bboxes(results):
    max_num = 100
    bboxes = results[0]["boxes"]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = [order[i].item() for i in range(min(max_num, order.numel()))]
    return torch.LongTensor(keep)


def get_contours_sky(mask):
    binary = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask

    mask = np.zeros_like(binary)

    cv2.drawContours(mask, contours, -1, 1, -1)

    return mask.astype(np.bool_)


def remove_sky_floaters(mask, min_size=1000):
    mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=2)

    return mask


def get_sky(image, zim_predictor, processor, grounding_model, DEVICE="cuda"):
    text = "sky."
    H, W = image.height, image.width
    zim_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    # get the box prompt for ZIM
    results[0]["boxes"] = results[0]["boxes"]
    # filter the small boxes to avoid memory overflow
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    results[0]["scores"] = results[0]["scores"][filter_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        sky_mask = np.zeros((H, W), dtype=np.bool_)
        return sky_mask

    masks, scores, logits = zim_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    sky_mask = np.zeros((H, W), dtype=np.bool_)

    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.bool_)
        sky_mask[mask] = 1

    # remove the small objects in masks
    min_floater = 500
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = get_contours_sky(sky_mask)
    sky_mask = 1 - sky_mask  # invert the mask to get the sky area
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = remove_sky_floaters(sky_mask, min_size=min_floater)
    sky_mask = get_contours_sky(sky_mask)

    return sky_mask


def get_zim_mask(image, text, box_conf, text_conf, zim_predictor, processor, grounding_model, DEVICE="cuda"):
    H, W = image.height, image.width
    zim_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_conf,
        text_threshold=text_conf,
        target_sizes=[image.size[::-1]]
    )
    # filter the small boxes to avoid memory overflow
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    # results[0]["scores"] = results[0]["scores"][filter_keep]
    # results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        result_mask = np.zeros((H, W), dtype=np.bool_)
        return result_mask

    masks, scores, logits = zim_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    masks = np.clip(np.sum(masks, axis=0, keepdims=True), 0, 1)

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    result_mask = np.zeros((H, W), dtype=np.bool_)

    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.bool_)
        result_mask[mask] = 1

    if type(text) == str and "sky" in text:
        # remove the small objects in masks
        min_floater = 500
        result_mask = result_mask.astype(np.bool_)
        result_mask = get_contours_sky(result_mask)
        result_mask = 1 - result_mask  # invert the mask to get the sky area
        result_mask = result_mask.astype(np.bool_)
        result_mask = remove_sky_floaters(result_mask, min_size=min_floater)
        result_mask = get_contours_sky(result_mask)
    else:
        result_mask = 1 - result_mask
        result_mask = result_mask.astype(np.bool_)

    return result_mask
