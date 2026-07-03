import jax
import jax.numpy as jnp


def crop_and_resize_image(
    image_BFHWC,
    target_height,
    target_width,
):
    B, F, H, W, C = image_BFHWC.shape
    src_aspect = H / W
    tgt_aspect = target_height / target_width
    if src_aspect > tgt_aspect:
        new_w = W
        new_h = int(new_w * tgt_aspect)
    else:
        new_h = H
        new_w = int(new_h / tgt_aspect)
    h_start = (H - new_h) // 2
    w_start = (W - new_w) // 2
    cropped_BFHWC = image_BFHWC[
        :, :, h_start : h_start + new_h, w_start : w_start + new_w, :
    ]
    resized_BFHWC = jax.image.resize(
        cropped_BFHWC,
        (B, F, target_height, target_width, C),
        method=jax.image.ResizeMethod.LINEAR,
    )
    return resized_BFHWC


def normalize_image(
    image_bfhwc,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    mean_array = jnp.array(mean).reshape(1, 1, 1, 1, 3)
    std_array = jnp.array(std).reshape(1, 1, 1, 1, 3)
    return (image_bfhwc - mean_array) / std_array


def wan_image_condition_preprocess(
    image_bfhwc,
    target_height,
    target_width,
):
    image_bfhwc = crop_and_resize_image(image_bfhwc, target_height, target_width)
    image_bfhwc = image_bfhwc / 255.0  # [0, 255] -> [0, 1]
    image_bfhwc = normalize_image(image_bfhwc)  # [0, 1] -> [-1, 1]
    return image_bfhwc
