# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import collections
import collections.abc
import os
from typing import Any

import torch
from einops import rearrange
from megatron.core import parallel_state

from gamma_world._src.imaginaire.lazy_config import instantiate
from gamma_world._src.imaginaire.utils import distributed, log, misc
from gamma_world._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from gamma_world._src.imaginaire.visualize.video import save_img_or_video
from gamma_world._src.gamma_world.utils.misc import sync_timer
from gamma_world._src.gamma_world.utils.model_loader import load_model_from_checkpoint

IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"
VIEW_INDICES_KEY = "view_indices"
NUM_VIDEO_FRAMES_PER_VIEW_KEY = "num_video_frames_per_view"

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."



def to_with_skip_tensor(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
    key: str | None = None,
) -> Any:

    skip_tensor_name = [
        "camera",
        "depth",
        "intrinsics",
        "buffer_depths",
        "buffer_w2cs",
        "target_w2cs",
        "buffer_intrinsics",
        "target_intrinsics",
        "buffer_points",
        "buffer_masks",
        "num_video_frames_per_view",
    ]
    assert device is not None or dtype is not None or memory_format is not None, (
        "at least one of device, dtype, memory_format should be specified"
    )
    if isinstance(data, torch.Tensor):
        if (
            memory_format == torch.channels_last
            and data.dim() != 4
            or memory_format == torch.channels_last_3d
            and data.dim() != 5
        ):
            memory_format = torch.preserve_format
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        if key is not None and key in skip_tensor_name:
            data = data.to(
                device=device,
                dtype=torch.float32,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        else:
            data = data.to(
                device=device,
                dtype=dtype,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        return data
    elif isinstance(data, collections.abc.Mapping):
        converted = {
            key: to_with_skip_tensor(data[key], device=device, dtype=dtype, memory_format=memory_format, key=key)
            for key in data
        }
        return type(data)(converted)
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        converted_list = [
            to_with_skip_tensor(elem, device=device, dtype=dtype, memory_format=memory_format, key=key) for elem in data
        ]
        return type(data)(converted_list)
    else:
        return data


def to_model_input(data_batch: dict, model: torch.nn.Module) -> dict:

    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


def prepare_multiview_data_batch(
    data_batch: dict,
    n_views: int,
    num_video_frames_per_view: int,
    device: torch.device | str = "cuda",
) -> dict:

    batch_size = data_batch["video"].shape[0] if "video" in data_batch else 1


    if NUM_VIDEO_FRAMES_PER_VIEW_KEY not in data_batch:
        data_batch[NUM_VIDEO_FRAMES_PER_VIEW_KEY] = torch.tensor(num_video_frames_per_view, device=device)




    if VIEW_INDICES_KEY not in data_batch:
        view_indices = torch.arange(n_views, device=device).repeat_interleave(num_video_frames_per_view)

        view_indices = view_indices.unsqueeze(0).expand(batch_size, -1)
        data_batch[VIEW_INDICES_KEY] = view_indices

    return data_batch


def save_output(to_show: list[torch.Tensor], vid_save_path: str, fps: int = 16) -> None:

    legancy_to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    video_array = (rearrange(legancy_to_show, "n b c t h w -> t (n h) (b w) c") * 255).to(torch.uint8).cpu().numpy()
    log.info(
        f"video_array.shape: {video_array.shape} value: {video_array.max()}, {video_array.min()}, save to {vid_save_path}"
    )
    save_img_or_video(
        rearrange(legancy_to_show, "n b c t h w -> c t (n h) (b w)"),
        vid_save_path.split(".mp4")[0],
        fps=fps,
    )
    log.info(f"save video to {vid_save_path}", rank0_only=True)


def save_multiview_output(
    to_show: list[torch.Tensor],
    vid_save_path: str,
    n_views: int,
    frames_per_view: int,
    fps: int = 16,
) -> None:


    stacked = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0
    n, b, c, vt, h, w = stacked.shape


    stacked_mv = rearrange(stacked, "n b c (v t) h w -> n b c v t h w", v=n_views, t=frames_per_view)



    video_for_save = rearrange(stacked_mv, "n b c v t h w -> c t (n h) (b v w)")

    log.info(
        f"Saving multiview video with {n_views} views, {frames_per_view} frames each. "
        f"Output shape: {video_for_save.shape}, save to {vid_save_path}"
    )
    save_img_or_video(
        video_for_save,
        vid_save_path.split(".mp4")[0],
        fps=fps,
    )
    log.info(f"Saved multiview video to {vid_save_path}", rank0_only=True)


class I2VInference:


    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        config_file: str = "gamma_world/_src/gamma_world/configs/causal_cosmos2/config.py",
        context_parallel_size: int = 1,
        guidance: float = 5.0,
        shift: float = 5.0,
        num_sampling_steps: int = 35,
        seed: int = 1,
        chunk_duration: int | None = None,
    ):

        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.config_file = config_file
        self.context_parallel_size = context_parallel_size
        self.guidance = guidance
        self.shift = shift
        self.num_sampling_steps = num_sampling_steps
        self.process_group = None

        if "RANK" in os.environ:
            self._init_distributed()

        misc.set_random_seed(seed=seed, by_rank=True)


        self.model, self.config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=self.config_file,
            load_ema_to_reg=False,
            instantiate_ema=False,
            cache_text_encoder=True,
            local_cache_dir=os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire")),
        )
        try:
            for pos_embedder in self.model.net.pos_embedder_options.values():
                pos_embedder.reset_parameters()
            log.info(f"Reset pos_embedder parameters")
        except Exception as e:
            try:
                self.model.net.pos_embedder.reset_parameters()
                log.info(f"Reset pos_embedder parameters")
            except Exception as e:
                log.warning(f"Error resetting pos_embedder parameters: {e}")
            log.warning(f"Error resetting pos_embedder parameters: {e}")


        if chunk_duration is not None:
            if hasattr(self.config, "model") and hasattr(self.config.model, "config"):
                if hasattr(self.config.model.config, "num_frame_per_block"):
                    log.info(
                        f"Overwriting config.model.config.num_frame_per_block from "
                        f"{self.config.model.config.num_frame_per_block} to {chunk_duration}",
                        rank0_only=True,
                    )
                    self.config.model.config.num_frame_per_block = chunk_duration
                    self.model.num_frame_per_block = chunk_duration
                    self.model.config.num_frame_per_block = chunk_duration
                    self.model.net.num_frame_per_block = chunk_duration
                else:
                    log.warning(
                        "chunk_duration specified but config.model.config.num_frame_per_block does not exist",
                        rank0_only=True,
                    )
            else:
                log.warning(
                    "chunk_duration specified but config.model.config does not exist",
                    rank0_only=True,
                )


        self.rank0 = True
        if self.context_parallel_size > 1:
            self.model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model.eval()
        self.model = self.model.to(dtype=torch.bfloat16)

        self.model.config.split_cp_in_model = False
        self.batch_size = 1
        self.generate_cnt = 0
        torch.cuda.empty_cache()

    def _init_distributed(self) -> None:


        distributed.init()


        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )


        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def clear_cache(self) -> None:

        self.model.kv_cache1 = None
        self.model.kv_cache2 = None


    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]) -> None:

        if (
            self.model.config.text_encoder_config is not None
            and self.model.config.text_encoder_config.compute_online
            and self.model.text_encoder is not None
        ):

            is_preprocessed = data_batch.get(IS_PREPROCESSED_KEY, False)
            if is_preprocessed:
                num_video_frames_per_view = int(self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t))
            else:
                num_video_frames_per_view = int(data_batch[NUM_VIDEO_FRAMES_PER_VIEW_KEY].cpu().item())

            num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
            n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
            B = data_batch["video"].shape[0]


            captions = data_batch.get("ai_caption", [[""]])[0]
            self.model.text_encoder.model = self.model.text_encoder.model.to("cuda")

            if len(captions) == 1:

                caption = captions[0] if isinstance(captions, list) else captions
                view0_text_embeddings_B_L_D = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={self.model.input_caption_key: [caption]},
                    input_caption_key=self.model.input_caption_key,
                )

                output_text_embeddings = view0_text_embeddings_B_L_D.repeat_interleave(n_views, dim=1)
            else:

                if len(captions) != n_views:
                    raise ValueError(f"Expected {n_views} captions or 1 caption, got {len(captions)}")

                view_text_embeddings = []
                for caption in captions:
                    view_text_embedding = self.model.text_encoder.compute_text_embeddings_online(
                        data_batch={self.model.input_caption_key: [caption]},
                        input_caption_key=self.model.input_caption_key,
                    )
                    view_text_embeddings.append(view_text_embedding)


                view_text_embeddings_B_V_L_D = torch.stack(view_text_embeddings, dim=1)
                output_text_embeddings = rearrange(view_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")


            if self.model.neg_text_embeddings is not None:
                output_neg_text_embeddings = self.model.neg_text_embeddings.repeat(B, n_views, 1)
            else:
                output_neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={self.model.input_caption_key: [""]},
                    input_caption_key=self.model.input_caption_key,
                )
                output_neg_text_embeddings = output_neg_text_embeddings.repeat_interleave(n_views, dim=1)

            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[debug] PyTorch VRAM usage: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
            self.model.text_encoder.model = self.model.text_encoder.model.to("cpu")
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[debug] PyTorch VRAM usage: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

            dropout_text_embeddings = torch.zeros_like(output_text_embeddings[:, 0:1, :]).repeat(1, n_views, 1)

            data_batch["t5_text_embeddings"] = output_text_embeddings
            data_batch["neg_t5_text_embeddings"] = output_neg_text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                output_text_embeddings.shape[0], output_text_embeddings.shape[1], device="cuda"
            )

    def generate_from_batch(
        self,
        data_batch: dict,
        guidance: float | None = None,
        seed: int = 1,
        num_steps: int | None = None,
        shift: float | None = None,
        use_negative_prompt: bool = True,
        save_output_for_viz: bool = False,
        output_path: str | None = None,
        output_name: str | None = None,
        fps: int = 16,
    ) -> torch.Tensor:

        guidance = guidance if guidance is not None else self.guidance
        num_steps = num_steps if num_steps is not None else self.num_sampling_steps
        shift = shift if shift is not None else self.shift


        if "video" in data_batch:
            data_batch["video"] = data_batch["video"].float()

            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["video"] = data_batch["video"] / 127.5 - 1.0
            data_batch["video"] = torch.clamp(data_batch["video"], -1, 1)


        if "control_input_hdmap_bbox" in data_batch:
            log.info("Preprocess hdmap condition", rank0_only=True)
            data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"].float()

            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"] / 127.5 - 1.0
            data_batch["control_input_hdmap_bbox"] = torch.clamp(data_batch["control_input_hdmap_bbox"], -1, 1)

        data_batch[IS_PREPROCESSED_KEY] = True


        state_t = self.model.config.state_t if hasattr(self.model.config, "state_t") else self.model.state_t
        pixel_frames_per_view = int(self.model.tokenizer.get_pixel_num_frames(state_t))


        if "video" in data_batch:
            total_frames = data_batch["video"].shape[2]
            n_views = total_frames // pixel_frames_per_view
        else:
            n_views = 1


        data_batch = prepare_multiview_data_batch(
            data_batch,
            n_views=n_views,
            num_video_frames_per_view=pixel_frames_per_view,
            device="cuda",
        )

        data_batch = to_with_skip_tensor(data_batch, **self.model.tensor_kwargs)


        self.inplace_compute_text_embeddings_online(data_batch)


        control_input_hdmap_bbox_viz = data_batch.get("control_input_hdmap_bbox")

        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        with torch.no_grad():
            log.info(
                f"Start multiview inference with {n_views} views, {state_t} latent frames per view", rank0_only=True
            )
            with sync_timer("generate_samples_from_batch"):
                sample = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=guidance,
                    shift=shift,
                    state_shape=x0.shape[1:],
                    n_sample=x0.shape[0],
                    seed=seed,
                    num_steps=num_steps,
                    is_negative_prompt=use_negative_prompt,
                    verbose=True,
                )
            with sync_timer("decode"):
                video = self.model.decode(sample)
            log.info("End multiview inference", rank0_only=True)

        if save_output_for_viz and output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if output_name is not None:
                base_fp_wo_ext = os.path.join(output_path, output_name + "_with_hdmap.mp4")
            else:
                base_fp_wo_ext = os.path.join(output_path, f"_Sample_Iter{self.generate_cnt:03d}.mp4")
            self.generate_cnt += 1
            to_show = [video.float().cpu(), raw_data.float().cpu()]

            if control_input_hdmap_bbox_viz is not None:
                to_show.insert(0, control_input_hdmap_bbox_viz.float().cpu())
            if self.context_parallel_size > 1:
                if is_tp_cp_pp_rank0():
                    save_multiview_output(to_show, base_fp_wo_ext, n_views, pixel_frames_per_view, fps=fps)
            else:
                save_multiview_output(to_show, base_fp_wo_ext, n_views, pixel_frames_per_view, fps=fps)

        return video

    def cleanup(self) -> None:

        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Causal I2V inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specified in the config",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="gamma_world/_src/gamma_world/configs/causal_cosmos2/config.py",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )

    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance value")
    parser.add_argument("--shift", type=float, default=5.0, help="Shift parameter for diffusion")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of sampling steps")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    parser.add_argument(
        "--use_negative_prompt",
        action="store_true",
        default=False,
        help="Use negative prompt for classifier-free guidance (default: False)",
    )
    parser.add_argument(
        "--no_negative_prompt",
        action="store_false",
        dest="use_negative_prompt",
        help="Disable negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="This is ignored in multiview inference. The model always loades the default negative prompt.",
    )

    parser.add_argument(
        "--input_is_train_data",
        action="store_true",
        help="Inference on the training data, the input_root will be ignored if this is set",
    )
    parser.add_argument("--input_root", type=str, default="assets/i2v", help="Input root")
    parser.add_argument("--save_root", type=str, default="results/causal_i2v", help="Save root")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to generate")
    parser.add_argument(
        "--save_output_for_viz",
        action="store_true",
        help="Save output videos with ground truth for visualization",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=None,
        help="Chunk duration (num_frame_per_block). If specified, overwrites config.model.config.num_frame_per_block",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["NVTE_FUSED_ATTN"] = "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)

    args = parse_arguments()
    if args.negative_prompt != "":
        log.warning(
            "The negative prompt is ignored in multiview inference. The model always loades the default negative prompt if use_negative_prompt is True and zeros if use_negative_prompt is False."
        )


    i2v_cli = I2VInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        context_parallel_size=args.context_parallel_size,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_steps=args.num_steps,
        seed=args.seed,
        chunk_duration=args.chunk_duration,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")


    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    os.makedirs(args.save_root, exist_ok=True)

    if args.input_is_train_data:

        dataloader = instantiate(i2v_cli.config.dataloader_train, is_train=False, num_workers=0)
        for i, batch in enumerate(dataloader):
            if i >= args.max_samples:
                break
            save_name = f"infer_from_train_{i}_multiview.mp4"
            save_path = os.path.join(args.save_root, save_name)
            log.info(f"Saving to {save_path}")
            if os.path.exists(save_path):
                log.info(f"Skipping sample {i} because it already exists")
                continue
            log.info(batch["ai_caption"][0][0])


            i2v_cli.clear_cache()


            batch[NUM_CONDITIONAL_FRAMES_KEY] = args.num_conditional_frames

            import gc

            gc.collect()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[debug] PyTorch VRAM usage: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

            video = i2v_cli.generate_from_batch(
                batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
                shift=args.shift,
                use_negative_prompt=args.use_negative_prompt,
                save_output_for_viz=args.save_output_for_viz,
                output_path=args.save_root,
                output_name=f"sample_{i}",
                fps=args.fps,
            )

            if rank0:

                video_normalized = ((video + 1.0) / 2.0).clamp(0, 1)


                state_t = (
                    i2v_cli.model.config.state_t if hasattr(i2v_cli.model.config, "state_t") else i2v_cli.model.state_t
                )
                pixel_frames_per_view = int(i2v_cli.model.tokenizer.get_pixel_num_frames(state_t))
                n_views = video_normalized.shape[2] // pixel_frames_per_view


                save_name = f"infer_from_train_{i}_multiview"

                video_mv = rearrange(
                    video_normalized[0],
                    "c (v t) h w -> c t h (v w)",
                    v=n_views,
                    t=pixel_frames_per_view,
                )
                save_img_or_video(video_mv, f"{args.save_root}/{save_name}", fps=args.fps)
                log.info(f"Saved multiview sample {i} with {n_views} views to {args.save_root}/{save_name}")


                for v in range(n_views):
                    start_frame = v * pixel_frames_per_view
                    end_frame = start_frame + pixel_frames_per_view
                    view_video = video_normalized[0, :, start_frame:end_frame, :, :]
                    view_save_name = f"infer_from_train_{i}_view{v}"
                    save_img_or_video(view_video, f"{args.save_root}/{view_save_name}", fps=args.fps)
                log.info(f"Saved {n_views} individual view videos for sample {i}")
    else:
        raise NotImplementedError("Custom input inference not implemented yet")


    i2v_cli.cleanup()
