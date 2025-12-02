from typing import Any, Generator, Dict, List, Iterable, Literal, Union
from jaxtyping import Float
from torch import Tensor
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from einops import rearrange
from matplotlib import cm
from einops import pack
from mirror_src.models.models.rasterization import GaussianSplatRenderer
import moviepy.editor as mpy
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from mirror_training.losses.utils import Depth2Normal

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]

VIZ_INTERVAL = 250

def to_uint8_img(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


def add_text_labels_to_grid(
    grid_np: np.ndarray, labels: List[str], row_height: int
) -> np.ndarray:
    """Add text labels to the left side of each row in the grid."""
    img = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, label in enumerate(labels):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x_position = 10
        y_position = i * row_height + 10
        padding = 5

        # Draw background rectangle and text
        draw.rectangle(
            [
                x_position - padding,
                y_position - padding,
                x_position + text_width + padding,
                y_position + text_height + padding,
            ],
            fill=(0, 0, 0, 180),
        )
        draw.text((x_position, y_position), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def process_normals_for_vis(
    normals: torch.Tensor, num_views: int, batch_idx: int = 0
) -> List[torch.Tensor]:
    """Process normals tensor for visualization. Handles both [B, S, H, W, 3] and [B, S, 3, H, W] formats."""
    normals_vis = []
    if normals is not None and normals.shape[1] >= num_views:
        for i in range(num_views):
            # Handle different formats
            if normals.shape[2] == 3 and normals.ndim == 5 and normals.shape[-1] != 3:
                # Format: [B, S, 3, H, W]
                normal = normals[batch_idx, i].permute(1, 2, 0)  # [H, W, 3]
            else:
                # Format: [B, S, H, W, 3]
                normal = normals[batch_idx, i]  # [H, W, 3]
            # Normalize to [0, 1] for visualization
            normal = (normal + 1) / 2
            normals_vis.append(normal.clamp(0, 1))
    return normals_vis


def process_depths_for_vis(
    depths: torch.Tensor, num_views: int, batch_idx: int = 0
) -> List[torch.Tensor]:
    """Process depth tensor for visualization with turbo colormap."""
    depths_vis = []
    if depths is not None and depths.shape[1] >= num_views:
        for i in range(num_views):
            depth = depths[batch_idx, i]  # [H, W] or [H, W, 1]
            if depth.dim() == 3:
                depth = depth.squeeze(-1)

            # Normalize depth
            depth_min = depth[depth > 0].min() if (depth > 0).any() else 0
            depth_max = depth.max()
            if depth_max > depth_min:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = depth

            # Apply colormap
            depth_colored = apply_color_map_to_image(
                depth_normalized.clamp(0, 1), "turbo"
            )
            depths_vis.append(depth_colored.permute(1, 2, 0))  # [H, W, 3]
    return depths_vis


def log_training_input_and_output_images(
    preds: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    logger: TensorBoardLogger,
    global_step: int,
    save_path: str = None,
):
    """
    Log training input/output visualizations: input images, pred/GT normals, pred/GT depth.
    Following the same data format as used in loss computation (NormalLoss, DepthLoss).
    """
    if global_step % VIZ_INTERVAL != 0:
        return

    try:
        # Get data
        batch_idx = 0
        num_views = min(4, inputs["img"].shape[1])
        imgs = inputs["img"][batch_idx, :num_views]
        h, w = imgs.shape[-2:]
        device = imgs.device

        # Process input images
        imgs_vis = [imgs[i].permute(1, 2, 0).clamp(0, 1) for i in range(num_views)]

        # Get GT normals (compute from depth if not available, following NormalLoss)
        gt_normals = inputs.get("normals", None)
        gt_normal_source = "GT"

        if gt_normals is None and inputs.get("depthmap") is not None:
            try:
                B, S = inputs["depthmap"].shape[:2]
                H, W = inputs["depthmap"].shape[-2:]
                depth2normal = Depth2Normal()
                gt_norm, _ = depth2normal(
                    inputs["depthmap"].reshape(B * S, 1, H, W),
                    inputs["camera_intrs"].reshape(B * S, 3, 3),
                    inputs["valid_mask"].reshape(B * S, 1, H, W),
                    scale=1.0,
                )
                gt_normals = gt_norm.reshape(B, S, 3, H, W)
                gt_normal_source = "Depth2Normal"
            except:
                pass

        # Process normals and depths
        normals_vis = process_normals_for_vis(
            preds.get("normals"), num_views, batch_idx
        )
        gt_normals_vis = process_normals_for_vis(gt_normals, num_views, batch_idx)
        depths_vis = process_depths_for_vis(preds.get("depth"), num_views, batch_idx)
        gt_depths_vis = process_depths_for_vis(
            inputs.get("depthmap"), num_views, batch_idx
        )

        # Create placeholders for missing data
        placeholder = torch.zeros((h, w, 3), device=device)
        normals_vis = normals_vis or [placeholder] * num_views
        gt_normals_vis = gt_normals_vis or [placeholder] * num_views
        depths_vis = depths_vis or [placeholder] * num_views
        gt_depths_vis = gt_depths_vis or [placeholder] * num_views

        # Concatenate all rows
        rows = [
            torch.cat(imgs_vis, dim=1),
            torch.cat(normals_vis, dim=1),
            torch.cat(gt_normals_vis, dim=1),
            torch.cat(depths_vis, dim=1),
            torch.cat(gt_depths_vis, dim=1),
        ]
        grid = torch.cat(rows, dim=0)

        # Add text labels
        grid_np = to_uint8_img(grid)
        labels = [
            "Input Images",
            "Pred Normal",
            f"GT Normal ({gt_normal_source})",
            "Pred Depth",
            "GT Depth",
        ]
        grid_np = add_text_labels_to_grid(grid_np, labels, row_height=h)

        # Log to TensorBoard
        if logger is not None:
            try:
                logger.experiment.add_image(
                    "train/input_and_output", grid_np.transpose(2, 0, 1), global_step
                )
            except:
                pass

        # Save to disk
        if save_path is not None:
            save_dir = os.path.join(save_path, "training_vis")
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(grid_np).save(
                os.path.join(save_dir, f"step_{global_step:06d}.png")
            )

    except Exception as e:
        pass  # Don't break training on visualization errors


def save_novel_view_render(
    gs_renderer: GaussianSplatRenderer,
    preds: Dict[str, Dict[str, torch.Tensor]],
    views: Dict[str, torch.Tensor],
    logger: TensorBoardLogger,
    global_step: int,
    save_separately: bool = False,
    save_mask: bool = False,
    save_path: str = "vis_logs",
):
    save_path = os.path.join(save_path, "validation_nvs_vis", f"step_{global_step:06d}")

    def prepare_and_concat_images(gt_img, pred_img, mask_img) -> Image.Image:
        gt_image = Image.fromarray(gt_img).convert("RGB")
        pred_image = Image.fromarray(pred_img).convert("RGB")
        mask_image = Image.fromarray(mask_img).convert("L").convert("RGB")

        total_width = (
            gt_image.width + pred_image.width + mask_image.width
            if save_mask
            else gt_image.width + pred_image.width
        )
        max_height = max(gt_image.height, pred_image.height, mask_image.height)

        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        img_ls = (
            [gt_image, pred_image, mask_image] if save_mask else [gt_image, pred_image]
        )
        for img in img_ls:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return new_img

    def log_save_image(dataset_name, image):
        try:
            logger.experiment.add_image(
                f"val_nvs_{cond_name}_{dataset_name}/image",
                image,
                global_step=global_step,
            )
        except:
            pass

    def save_individual_images(
        gt_img, pred_img, mask_img, dir_path, base_fname, save_mask=False
    ):
        gt_image = Image.fromarray(gt_img).convert("RGB")
        pred_image = Image.fromarray(pred_img).convert("RGB")
        if save_mask:
            mask_image = Image.fromarray(mask_img).convert("L")

        gt_image.save(os.path.join(dir_path, f"{base_fname}_gt.png"))
        pred_image.save(os.path.join(dir_path, f"{base_fname}_pred.png"))
        if save_mask:
            mask_image.save(str(dir_path / f"{base_fname}_mask.png"))
        return pred_image

    def save_concat_image(
        preds: Dict[str, torch.Tensor],
        context_nums: int,
        cond_name: str,
        dataset_name: str,
    ):
        pred_imgs = preds["rendered_colors"][0]
        gt_imgs = preds["gt_colors"][0]
        masks = preds["valid_masks"][0]
        for i in range(pred_imgs.shape[0]):
            pred_img = to_uint8_img(pred_imgs[i])
            gt_img = to_uint8_img(gt_imgs[i])
            mask_img = (masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            if i < context_nums:
                dir_path = os.path.join(
                    save_path, dataset_name, "source", views["label"][0].split("/")[-1]
                )
            else:
                dir_path = os.path.join(
                    save_path, dataset_name, "nvs", views["label"][0].split("/")[-1]
                )
            os.makedirs(dir_path, exist_ok=True)
            base_fname = f"view{i:03}_{cond_name}"

            if save_separately:
                pil_img = save_individual_images(
                    gt_img, pred_img, mask_img, dir_path, base_fname, save_mask
                )
            else:
                pil_img = prepare_and_concat_images(gt_img, pred_img, mask_img)
                fname = f"{base_fname}.png"
                pil_img.save(os.path.join(dir_path, fname))
        img_numpy = np.array(pil_img).transpose(2, 0, 1)
        log_save_image(dataset_name, img_numpy)

    context_nums = (views["is_target"][0] == False).sum().item()
    for cond_name, pred in preds.items():
        save_concat_image(
            pred,
            context_nums=context_nums,
            cond_name=cond_name,
            dataset_name=views["dataset"][0],
        )
    render_video_interpolation_multiview(
        gs_renderer, preds, views, logger, global_step, save_path=save_path
    )


def render_video_interpolation_multiview(
    gs_renderer: GaussianSplatRenderer,
    preds: Dict[str, Dict[str, torch.Tensor]],
    views: Dict[str, torch.Tensor],
    logger: TensorBoardLogger,
    global_step: int,
    t: int = 20,
    loop_reverse: bool = True,
    save_path: str = "vis_logs",
) -> None:
    context_nums = (views["is_target"][0] == False).sum().item()
    splats = preds["nocond"]["splats"]
    pred_extrinsics = preds["nocond"]["rendered_extrinsics"][
        :1, :context_nums
    ]  # B S 4 4
    pred_intrinsics = preds["nocond"]["rendered_intrinsics"][
        :1, :context_nums
    ]  # B S 3 3
    b, v, _, _ = pred_extrinsics.shape
    assert b == 1
    h, w = views["img"].shape[-2:]
    # Interpolate between neighboring frames
    # t: Number of extra views to interpolate between each pair
    interpolated_extrinsics = []
    interpolated_intrinsics = []

    # For each pair of neighboring frame
    for i in range(v - 1):
        # Add the current frame
        interpolated_extrinsics.append(pred_extrinsics[:, i : i + 1])
        interpolated_intrinsics.append(pred_intrinsics[:, i : i + 1])

        # Interpolate between current and next frame
        for j in range(1, t + 1):
            alpha = j / (t + 1)

            # Interpolate extrinsics
            start_extrinsic = pred_extrinsics[:, i]
            end_extrinsic = pred_extrinsics[:, i + 1]

            # Separate rotation and translation
            start_rot = start_extrinsic[:, :3, :3]
            end_rot = end_extrinsic[:, :3, :3]
            start_trans = start_extrinsic[:, :3, 3]
            end_trans = end_extrinsic[:, :3, 3]

            # Interpolate translation (linear)
            interp_trans = (1 - alpha) * start_trans + alpha * end_trans

            # Interpolate rotation (spherical)
            start_rot_flat = start_rot.reshape(b, 9)
            end_rot_flat = end_rot.reshape(b, 9)
            interp_rot_flat = (1 - alpha) * start_rot_flat + alpha * end_rot_flat
            interp_rot = interp_rot_flat.reshape(b, 3, 3)

            # Normalize rotation matrix to ensure it's orthogonal
            u, _, v = torch.svd(interp_rot)
            interp_rot = torch.bmm(u, v.transpose(1, 2))

            # Combine interpolated rotation and translation
            interp_extrinsic = (
                torch.eye(4, device=pred_extrinsics.device).unsqueeze(0).repeat(b, 1, 1)
            )
            interp_extrinsic[:, :3, :3] = interp_rot
            interp_extrinsic[:, :3, 3] = interp_trans

            # Interpolate intrinsics (linear)
            start_intrinsic = pred_intrinsics[:, i]
            end_intrinsic = pred_intrinsics[:, i + 1]
            interp_intrinsic = (1 - alpha) * start_intrinsic + alpha * end_intrinsic

            # Add interpolated frame
            interpolated_extrinsics.append(interp_extrinsic.unsqueeze(1))
            interpolated_intrinsics.append(interp_intrinsic.unsqueeze(1))

    # Concatenate all frames
    pred_all_extrinsic = torch.cat(interpolated_extrinsics, dim=1)[:1]
    pred_all_intrinsic = torch.cat(interpolated_intrinsics, dim=1)[:1]

    # Color-map the result.
    def depth_map(result):
        near = result[result > 0][:16_000_000].quantile(0.01).log()
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        result = result.log()
        result = 1 - (result - near) / (far - near)
        return apply_color_map_to_image(result, "turbo")

    # Render interpolated views
    chunk_size = 20
    b, n, _, _ = pred_all_extrinsic.shape

    rendered_colors_list = []
    rendered_depths_list = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        extrinsic_chunk = pred_all_extrinsic[:, start:end]  # (b, chunk, 4, 4)
        intrinsic_chunk = pred_all_intrinsic[:, start:end]  # (b, chunk, 3, 3)

        rendered_colors_chunk, rendered_depths_chunk, _ = (
            gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1],
                splats["quats"][:1],
                splats["scales"][:1],
                splats["opacities"][:1],
                splats["sh"][:1] if "sh" in splats else splats["colors"][:1],
                extrinsic_chunk.to(torch.float32),
                intrinsic_chunk.to(torch.float32),
                width=w,
                height=h,
                sh_degree=min(gs_renderer.sh_degree, 0) if "sh" in splats else None,
            )
        )

        rendered_colors_list.append(rendered_colors_chunk)  # (b, chunk, h, w, 3)
        rendered_depths_list.append(rendered_depths_chunk)  # (b, chunk, h, w, 1)

    rendered_colors = torch.cat(rendered_colors_list, dim=1)
    rendered_depths = torch.cat(rendered_depths_list, dim=1)
    images_prob = [
        vcat(rgb, depth)
        for rgb, depth in zip(
            rendered_colors[0].permute(0, 3, 1, 2),
            depth_map(rendered_depths[0, ..., 0]),
        )
    ]

    video = torch.stack(images_prob)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]

    clip = mpy.ImageSequenceClip(list(video.transpose(0, 2, 3, 1)), fps=30)
    dir_path = os.path.join(
        save_path,
        views["dataset"][0],
        "video",
        os.path.splitext(views["label"][0].split("/")[-1])[0],
    )
    os.makedirs(dir_path, exist_ok=True)
    clip.write_videofile(
        os.path.join(
            dir_path, f"{os.path.splitext(views['label'][0].split('/')[-1])[0]}.mp4"
        ),
        logger=None,
    )
    try:
        logger.experiment.add_video(
            f"val_nvs_nocond_{views['dataset'][0]}/video",
            video[None],
            global_step=global_step,
            fps=30,
        )
    except:
        pass


def apply_color_map(
    x: torch.Tensor,
    color_map: str = "inferno",
) -> torch.Tensor:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: torch.Tensor,  # Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
):  # -> Float[Tensor, "*batch 3 height with"]
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def vcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def cat(
    main_axis: Axis,
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))
