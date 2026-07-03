
import jax
import jax.numpy as jnp


def crop_and_resize_image(
    image_BPFHWC,
    target_height,
    target_width,
    h_dim=2,
    w_dim=3,
):
    H = image_BPFHWC.shape[h_dim]
    W = image_BPFHWC.shape[w_dim]

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

    cropped_BFHWC = image_BPFHWC[
        :, :, :, h_start : h_start + new_h, w_start : w_start + new_w, :
    ]

    new_shape = list(image_BPFHWC.shape)
    new_shape[h_dim] = target_height
    new_shape[w_dim] = target_width
    new_shape = tuple(new_shape)
    resized_BFHWC = jax.image.resize(
        cropped_BFHWC,
        new_shape,
        method=jax.image.ResizeMethod.LINEAR,
    )
    return resized_BFHWC


def normalize_image(
    image_BPFHWC,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    mean_array = jnp.array(mean).reshape(1, 1, 1, 1, 1, 3)
    std_array = jnp.array(std).reshape(1, 1, 1, 1, 1, 3)
    return (image_BPFHWC - mean_array) / std_array


def wan_image_condition_preprocess(
    image_BPFHWC,
    target_height,
    target_width,
    h_dim=3,
    w_dim=4,
):
    image_BPFHWC = crop_and_resize_image(
        image_BPFHWC, target_height, target_width, h_dim=h_dim, w_dim=w_dim
    )
    image_BPFHWC = image_BPFHWC / 255.0  # [0, 255] -> [0, 1]
    image_BPFHWC = normalize_image(image_BPFHWC)  # [0, 1] -> [-1, 1]
    return image_BPFHWC
