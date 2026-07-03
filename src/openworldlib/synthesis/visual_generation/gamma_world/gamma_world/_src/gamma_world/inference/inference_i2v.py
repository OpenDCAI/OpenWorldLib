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
import time
from typing import Any

import mediapy as media
import numpy as np
from PIL import Image
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
        if not torch.is_floating_point(data):
            data = data.to(
                device=device,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        elif key is not None and key in skip_tensor_name:
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
        experiment_opts: list[str] | None = None,
        vae_pth: str | None = None,
        text_encoder_pth: str | None = None,
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

        if experiment_opts:
            log.info(f"[InferenceI2V] experiment_opts={experiment_opts}")

        self.model, self.config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=self.config_file,
            load_ema_to_reg=False,
            instantiate_ema=False,
            cache_text_encoder=True,
            local_cache_dir=os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire")),
            experiment_opts=experiment_opts or [],
            vae_pth=vae_pth,
            text_encoder_pth=text_encoder_pth,
        )

        net_cfg = getattr(self.config.model.config, "net", None)
        if net_cfg is not None:
            log.info(
                "[InferenceI2V] resolved net config: "
                f"use_multi_agent_rope={getattr(net_cfg, 'use_multi_agent_rope', None)} "
                f"num_agents={getattr(net_cfg, 'multi_agent_rope_num_agents', None)} "
                f"simplex_pool_size={getattr(net_cfg, 'multi_agent_rope_simplex_pool_size', None)} "
                f"agent_encoding={getattr(net_cfg, 'multi_agent_rope_agent_encoding', None)} "
                f"agent_scale={getattr(net_cfg, 'multi_agent_rope_agent_scale', None)} "
                f"agent_id_offset={getattr(net_cfg, 'multi_agent_rope_agent_id_offset', None)}"
            )

        self.rank0 = True
        if self.context_parallel_size > 1:
            self.model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model.eval()
        self.model = self.model.to(dtype=torch.bfloat16)

        if hasattr(self.model, "net") and hasattr(self.model.net, "pos_embedder"):
            log.info("Resetting pos_embedder parameters to restore float32 precision after bf16 cast")
            self.model.net.pos_embedder.reset_parameters()
        else:
            log.warning("self.model.net.pos_embedder not available, skipping reset_parameters()")


        self.model.config.split_cp_in_model = False
        self.batch_size = 1
        self.generate_cnt = 0
        torch.cuda.empty_cache()

        if hasattr(self.model, "net") and getattr(self.model.net, "use_multi_agent_rope", False):
            log.info(
                "[multi-agent RoPE] "
                f"use_multi_agent_rope={getattr(self.model.net, 'use_multi_agent_rope', None)} "
                f"num_agents={getattr(self.model.net, 'num_agents', None)} "
                f"simplex_pool_size={getattr(self.model.net, 'simplex_pool_size', None)} "
                f"agent_encoding={getattr(self.model.net, 'agent_encoding', None)} "
                f"agent_scale={getattr(self.model.net, 'agent_scale', None)} "
                f"agent_id_offset={getattr(self.model.net, 'agent_id_offset', None)}"
            )
            if hasattr(self.model.net, "multi_agent_action_control") and self.model.net.multi_agent_action_control is not None:
                mac = self.model.net.multi_agent_action_control
                log.info(
                    f"[multi-agent action] type={type(mac).__name__} "
                    f"num_agents={getattr(mac, 'num_agents', None)} "
                    f"pool_size={getattr(mac, 'pool_size', None)}"
                )

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

    def build_inference_batch(
        self,
        init_images: list[np.ndarray],
        prompt: str,
        actions: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
        *,
        num_frames: int,
        num_conditional_frames: int = 1,
    ) -> dict[str, Any]:

        n_players = len(init_images)
        if actions is not None and len(actions) != n_players:
            raise ValueError(f"len(actions)={len(actions)} != len(init_images)={n_players}")
        height, width = init_images[0].shape[:2]
        per_view = []
        for image in init_images:
            array = np.ascontiguousarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            chw = torch.from_numpy(array).permute(2, 0, 1)
            video_view = torch.zeros((chw.shape[0], num_frames, height, width), dtype=torch.uint8)
            if num_conditional_frames > 0:
                video_view[:, :num_conditional_frames] = chw.unsqueeze(1)
            per_view.append(video_view)
        video = torch.cat(per_view, dim=1).unsqueeze(0)
        view_indices = torch.tensor(
            [view for view in range(n_players) for _ in range(num_frames)], dtype=torch.int64
        ).unsqueeze(0)
        batch = {
            "video": video,
            self.model.input_caption_key: [[prompt]],
            "view_indices": view_indices,
            "view_indices_selection": torch.arange(n_players, dtype=torch.int64).unsqueeze(0),
            "num_video_frames_per_view": torch.tensor([num_frames], dtype=torch.int64),
            "sample_n_views": torch.tensor([n_players], dtype=torch.int64),
            "fps": torch.tensor([float(getattr(self, "fps", 16.0))], dtype=torch.float64),
            "frame_indices": torch.arange(num_frames, dtype=torch.int64).unsqueeze(0),
            "padding_mask": torch.zeros(1, 1, height, width, dtype=torch.float32),
            "original_hw": torch.tensor([[[height, width]] * n_players], dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor([0], dtype=torch.int64),
            "ref_cam_view_idx_sample_position": torch.tensor([-1], dtype=torch.int64),
            NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
        }
        for index, (keyboard, camera) in enumerate(actions or []):
            batch[f"action_{index}_keyboard"] = keyboard
            if camera is not None:
                batch[f"action_{index}_camera"] = camera
        return batch

    def inplace_compute_text_embeddings_online(
        self,
        data_batch: dict[str, torch.Tensor],
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> None:

        if (
            self.model.config.text_encoder_config is not None
            and self.model.config.text_encoder_config.compute_online
            and self.model.text_encoder is not None
        ):
            text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                data_batch, self.model.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

            if use_negative_prompt:
                batch_size = text_embeddings.shape[0]
                neg_data_batch = {self.model.input_caption_key: [negative_prompt] * batch_size, "images": None}
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    neg_data_batch, self.model.input_caption_key
                )
                data_batch["neg_t5_text_embeddings"] = neg_text_embeddings


    def generate_from_batch(
        self,
        data_batch: dict,
        guidance: float | None = None,
        seed: int = 1,
        num_steps: int | None = None,
        shift: float | None = None,
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        save_output_for_viz: bool = False,
        output_path: str | None = None,
        output_name: str | None = None,
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
            data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"].float()
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"] / 127.5 - 1.0
            data_batch["control_input_hdmap_bbox"] = torch.clamp(data_batch["control_input_hdmap_bbox"], -1, 1)

        data_batch[IS_PREPROCESSED_KEY] = True
        data_batch = to_with_skip_tensor(data_batch, **self.model.tensor_kwargs)

        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=use_negative_prompt,
            negative_prompt=negative_prompt,
        )

        if hasattr(self.model, "net") and hasattr(self.model.net, "multi_agent_action_control"):
            mac = self.model.net.multi_agent_action_control
            if mac is not None and "action_inference_scale" in data_batch:
                override_val = data_batch.pop("action_inference_scale")
                log.info(f"Overriding action scale: {mac.scale.item():.6f} -> {override_val}")
                mac.scale.data.fill_(override_val)

        control_input_hdmap_bbox_viz = data_batch.get("control_input_hdmap_bbox")
        data_batch = self.model.get_data_batch_with_latent_view_indices(data_batch)
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        with torch.no_grad():
            log.info("Start inference", rank0_only=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _gen_t0 = time.perf_counter()
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
            torch.cuda.synchronize()
            _sample_elapsed = time.perf_counter() - _gen_t0
            _sample_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            _sample_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            torch.cuda.reset_peak_memory_stats()
            _dec_t0 = time.perf_counter()
            with sync_timer("decode"):
                video = self.model.decode(sample)
            torch.cuda.synchronize()
            _decode_elapsed = time.perf_counter() - _dec_t0
            _decode_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            _decode_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            log.info(
                f"[gen-stats] sample: time={_sample_elapsed:.2f}s "
                f"peak_alloc={_sample_peak_alloc:.2f}GB peak_reserved={_sample_peak_reserved:.2f}GB | "
                f"decode: time={_decode_elapsed:.2f}s "
                f"peak_alloc={_decode_peak_alloc:.2f}GB peak_reserved={_decode_peak_reserved:.2f}GB | "
                f"total_gen_time={_sample_elapsed + _decode_elapsed:.2f}s",
                rank0_only=True,
            )
            log.info("End inference", rank0_only=True)

        n_views = int(data_batch["sample_n_views"].cpu().item())
        if n_views > 1:
            video = rearrange(video, "B C (V T) H W -> B C T H (V W)", V=n_views)

        if save_output_for_viz and output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if output_name is not None:
                base_fp_wo_ext = os.path.join(output_path, output_name + "_with_hdmap.mp4")
            else:
                base_fp_wo_ext = os.path.join(output_path, f"_Sample_Iter{self.generate_cnt:03d}.mp4")
            self.generate_cnt += 1
            to_show = [
                video.float().cpu(),
            ]
            if control_input_hdmap_bbox_viz is not None:
                to_show.insert(0, control_input_hdmap_bbox_viz.float().cpu())
            if self.context_parallel_size > 1:
                if is_tp_cp_pp_rank0():
                    save_output(to_show, base_fp_wo_ext)
            else:
                save_output(to_show, base_fp_wo_ext)

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
        default=True,
        help="Use negative prompt for classifier-free guidance (default: True)",
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
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Custom negative prompt for classifier-free guidance. If not specified, uses default negative prompt.",
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
        "--num_frames",
        type=int,
        default=93,
        help="Number of frames to generate for demo examples",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default="A driving scene video.",
        help="Default prompt for demo examples",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        choices=["480p", "720p"],
        help="Output resolution preset: 480p (480x832) or 720p (704x1280). Overridden by --height/--width if provided.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels. Overrides --resolution preset when both --height and --width are provided.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels. Overrides --resolution preset when both --height and --width are provided.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single input image (PNG/JPG) for custom I2V inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for custom I2V inference",
    )
    parser.add_argument(
        "--split_width_for_multiview",
        action="store_true",
        help="Split input image width into ``--n_players`` equal segments to create N-view multiview input "
             "(e.g. for Solaris 2-player or 4-player data).",
    )
    parser.add_argument(
        "--spatial_concat",
        action="store_true",
        help="Spatial-concat ablation: feed both players as a single horizontally-concatenated "
             "wide frame (n_views=1). The input image is split into ``--n_players`` halves, each "
             "half is resized independently to ``(H, W // n_players)`` to preserve per-player "
             "aspect ratio, then concatenated back along width before being passed to the model. "
             "Actions are still loaded as ``--n_players`` streams; the spatial-concat action "
             "module routes each agent's bias to its corresponding W slab. Mutually exclusive "
             "with ``--split_width_for_multiview``.",
    )
    parser.add_argument(
        "--n_players",
        type=int,
        default=2,
        help="Number of players (views) when ``--split_width_for_multiview`` is set, or number of "
             "players whose actions are loaded under ``--spatial_concat``. "
             "When 2: uses legacy ``action_left/right_*`` keys and ``--action_left/right_json``. "
             "When >=2: uses ``action_<i>_*`` keys and ``--action_jsons`` (one path per player).",
    )
    parser.add_argument(
        "--experiment_opts",
        nargs="*",
        default=[],
        help="Hydra-style experiment overrides forwarded to load_model_from_checkpoint. "
             "E.g. 'model.config.net.multi_agent_rope_num_agents=4' to switch a 2-player simplex-pool "
             "checkpoint to 4-player inference.",
    )
    parser.add_argument(
        "--action_left_json",
        type=str,
        default=None,
        help="(2-player legacy) Path to JSON file with left player action sequence: "
             "{\"keyboard\": [[...]], \"camera\": [[...]]}. Used only when --n_players=2.",
    )
    parser.add_argument(
        "--action_right_json",
        type=str,
        default=None,
        help="(2-player legacy) Path to JSON file with right player action sequence. Used only when --n_players=2.",
    )
    parser.add_argument(
        "--action_jsons",
        nargs="*",
        default=None,
        help="(N-player) List of action JSON paths, one per player (length must equal --n_players). "
             "Each JSON has the same schema as --action_left_json.",
    )
    parser.add_argument(
        "--action_inference_scale",
        type=float,
        default=None,
        help="Override action control scale at inference time (e.g. 0.1 to reduce action strength)",
    )
    parser.add_argument(
        "--gt_video_path",
        type=str,
        default=None,
        help="Path to ground truth video (mp4) to concatenate below generated video in visualization",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Directory of prepared eval samples. Each subdirectory has first_frame.png or first_frame_path.txt, prompt.txt, "
             "an optional gt_combined.mp4, and per-player action JSONs: either action_left/right.json "
             "(--n_players=2) or action_<i>.json for i in [0, n_players) (--n_players>=2). "
             "Model is loaded once and all samples are processed sequentially.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Max number of eval samples to process per subset (default: all)",
    )
    parser.add_argument(
        "--gt_fps_downsample",
        type=int,
        default=1,
        help="Downsample factor for GT video frames (e.g. 2 = 20fps->10fps)",
    )
    args = parser.parse_args()

    if args.split_width_for_multiview and args.spatial_concat:
        parser.error(
            "--split_width_for_multiview and --spatial_concat are mutually exclusive: "
            "the former produces n_views=n_players (sequence-concat layout), the latter "
            "produces n_views=1 (spatial-concat layout)."
        )

    if args.height is not None and args.width is not None:
        args.resolved_resolution = (args.height, args.width)
    elif args.height is not None or args.width is not None:
        parser.error("--height and --width must be specified together")
    else:
        _RESOLUTION_PRESETS = {"480p": (480, 832), "720p": (704, 1280)}
        args.resolved_resolution = _RESOLUTION_PRESETS[args.resolution]

    return args


if __name__ == "__main__":
    os.environ["NVTE_FUSED_ATTN"] = "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)

    args = parse_arguments()

    i2v_cli = I2VInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        context_parallel_size=args.context_parallel_size,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_steps=args.num_steps,
        seed=args.seed,
        experiment_opts=args.experiment_opts,
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

            i2v_cli.clear_cache()

            batch[NUM_CONDITIONAL_FRAMES_KEY] = args.num_conditional_frames

            video = i2v_cli.generate_from_batch(
                batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
                shift=args.shift,
                use_negative_prompt=args.use_negative_prompt,
                negative_prompt=args.negative_prompt,
                save_output_for_viz=args.save_output_for_viz,
                output_path=args.save_root,
            )

            if rank0:
                video_normalized = ((video + 1.0) / 2.0).clamp(0, 1)
                save_name = f"infer_from_train_{i}"
                save_img_or_video(video_normalized[0], f"{args.save_root}/{save_name}", fps=args.fps)
                log.info(f"Saved sample {i} to {args.save_root}/{save_name}")
    else:

        def _load_action_json(path: str, num_frames: int) -> tuple[torch.Tensor, torch.Tensor | None]:
            import json as _json

            with open(path) as f:
                data = _json.load(f)
            keyboard_all = data["keyboard"]
            camera_all = data.get("camera")
            keyboard = [keyboard_all[i] if i < len(keyboard_all) else [0.0] * 23 for i in range(num_frames)]
            keyboard_tensor = torch.tensor(keyboard, dtype=torch.float32).unsqueeze(0)
            camera_tensor = None
            if camera_all is not None:
                camera = [camera_all[i] if i < len(camera_all) else [0.0, 0.0] for i in range(num_frames)]
                camera_tensor = torch.tensor(camera, dtype=torch.float32).unsqueeze(0)
            return keyboard_tensor, camera_tensor

        def _run_single_sample(
            image_path: str,
            prompt: str,
            save_root: str,
            action_jsons: list[str | None] | None = None,
            action_left_json: str | None = None,
            action_right_json: str | None = None,
            gt_video_path: str | None = None,
        ) -> None:

            resolution = args.resolved_resolution
            image = media.read_image(image_path)
            if args.spatial_concat:
                target_h, target_w = resolution
                assert target_w % args.n_players == 0, (
                    f"--spatial_concat: resolution width {target_w} not divisible by "
                    f"--n_players={args.n_players}"
                )
                in_h, in_w, _ = image.shape
                assert in_w % args.n_players == 0, (
                    f"--spatial_concat: input image width {in_w} not divisible by "
                    f"--n_players={args.n_players}"
                )
                per_view_w = in_w // args.n_players
                target_per_view = (target_h, target_w // args.n_players)
                halves = [
                    media.resize_image(
                        image[:, i * per_view_w : (i + 1) * per_view_w], target_per_view
                    )
                    for i in range(args.n_players)
                ]
                image = np.concatenate(halves, axis=1)
                log.info(
                    f"Loaded image from {image_path}, spatial-concat resized "
                    f"({args.n_players} halves -> {target_per_view} each, concat -> {image.shape})"
                )
            else:
                image = media.resize_image(image, resolution)
                log.info(f"Loaded image from {image_path}, resized to {resolution}, shape={image.shape}")

            batch = i2v_cli.create_data_batch_from_demo(
                image_or_video=image,
                prompt=prompt,
                num_frames=args.num_frames,
                num_conditional_frames=args.num_conditional_frames,
                split_width_for_multiview=args.split_width_for_multiview,
                n_players=args.n_players,
                spatial_concat=args.spatial_concat,
            )

            if action_jsons is not None:
                assert len(action_jsons) == args.n_players, (
                    f"--action_jsons length {len(action_jsons)} != --n_players {args.n_players}"
                )
                for i, jp in enumerate(action_jsons):
                    if jp is None:
                        continue
                    kb, cam = _load_action_json(jp, args.num_frames)
                    batch[f"action_{i}_keyboard"] = kb
                    if cam is not None:
                        batch[f"action_{i}_camera"] = cam
                loaded = sum(int(jp is not None) for jp in action_jsons)
                log.info(f"Loaded N-player action inputs: {loaded}/{args.n_players} players")
            elif action_left_json is not None and action_right_json is not None:
                assert args.n_players == 2, (
                    f"--action_left/right_json only supported with --n_players=2, got {args.n_players}"
                )
                left_kb, left_cam = _load_action_json(action_left_json, args.num_frames)
                right_kb, right_cam = _load_action_json(action_right_json, args.num_frames)
                batch["action_left_keyboard"] = left_kb
                batch["action_right_keyboard"] = right_kb
                if left_cam is not None:
                    batch["action_left_camera"] = left_cam
                if right_cam is not None:
                    batch["action_right_camera"] = right_cam
                log.info(f"Loaded 2P action inputs: left_keyboard={left_kb.shape}, right_keyboard={right_kb.shape}")

            if hasattr(args, "action_inference_scale") and args.action_inference_scale is not None:
                batch["action_inference_scale"] = args.action_inference_scale

            i2v_cli.clear_cache()

            video = i2v_cli.generate_from_batch(
                batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
                shift=args.shift,
                use_negative_prompt=args.use_negative_prompt,
                negative_prompt=args.negative_prompt,
                save_output_for_viz=args.save_output_for_viz,
                output_path=save_root,
                output_name="custom_i2v",
            )

            if rank0:
                video_normalized = ((video + 1.0) / 2.0).clamp(0, 1)
                save_img_or_video(video_normalized[0], f"{save_root}/custom_i2v", fps=args.fps)
                log.info(f"Saved custom I2V output to {save_root}/custom_i2v")

                if gt_video_path is not None and os.path.exists(gt_video_path):
                    gt_video_np = media.read_video(gt_video_path)
                    num_gen_frames = video_normalized.shape[2]
                    gen_h, gen_w = video_normalized.shape[3], video_normalized.shape[4]

                    ds = args.gt_fps_downsample
                    gt_frames = gt_video_np[: num_gen_frames * ds : ds]
                    if len(gt_frames) < num_gen_frames:
                        pad = np.zeros(
                            (num_gen_frames - len(gt_frames), *gt_frames.shape[1:]), dtype=gt_frames.dtype
                        )
                        gt_frames = np.concatenate([gt_frames, pad], axis=0)

                    gt_h, gt_w = gt_frames.shape[1], gt_frames.shape[2]
                    if gt_h > gt_w:
                        half_h = gt_h // 2
                        top = gt_frames[:, :half_h, :, :]
                        bottom = gt_frames[:, half_h:, :, :]
                        gt_frames = np.concatenate([top, bottom], axis=2)

                    gt_resized = np.stack([
                        np.array(Image.fromarray(f).resize((gen_w, gen_h), Image.LANCZOS))
                        for f in gt_frames
                    ])
                    gt_tensor = torch.from_numpy(gt_resized).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0

                    combined = torch.cat([video_normalized.cpu(), gt_tensor], dim=3)
                    save_img_or_video(combined[0], f"{save_root}/custom_i2v_with_gt", fps=args.fps)
                    log.info(f"Saved combined (gen+GT) to {save_root}/custom_i2v_with_gt")

        if args.eval_dir is not None:
            assert os.path.isdir(args.eval_dir), f"Eval dir not found: {args.eval_dir}"
            sample_dirs = []
            for root, _, files in os.walk(args.eval_dir):
                if "first_frame.png" in files or "first_frame_path.txt" in files:
                    sample_dirs.append(os.path.relpath(root, args.eval_dir))
            sample_dirs = sorted(sample_dirs)
            if args.max_eval_samples is not None:
                sample_dirs = sample_dirs[: args.max_eval_samples]

            total_samples = len(sample_dirs)
            import torch.distributed as _dist
            if (
                args.context_parallel_size == 1
                and _dist.is_available()
                and _dist.is_initialized()
                and _dist.get_world_size() > 1
            ):
                ws = _dist.get_world_size()
                rk = _dist.get_rank()
                sample_dirs = sample_dirs[rk::ws]
                log.info(
                    f"DP shard: rank {rk}/{ws} -> {len(sample_dirs)}/{total_samples} samples"
                )

            log.info(
                f"Eval mode: {len(sample_dirs)}/{total_samples} samples from "
                f"{args.eval_dir} (n_players={args.n_players})"
            )
            for sample_idx, sample_name in enumerate(sample_dirs):
                sample_path = os.path.join(args.eval_dir, sample_name)
                image_path = os.path.join(sample_path, "first_frame.png")
                pointer_path = os.path.join(sample_path, "first_frame_path.txt")
                if not os.path.exists(image_path) and os.path.exists(pointer_path):
                    with open(pointer_path, "r", encoding="utf-8") as f:
                        referenced_path = f.read().strip()
                    image_path = referenced_path
                    if not os.path.isabs(image_path):
                        image_path = os.path.normpath(os.path.join(sample_path, image_path))
                prompt_path = os.path.join(sample_path, "prompt.txt")
                gt_path = os.path.join(sample_path, "gt_combined.mp4")

                if not os.path.exists(image_path):
                    log.warning(f"[{sample_idx}] SKIP {sample_name}: first_frame image not found")
                    continue

                prompt = args.prompt or "Multiple Minecraft players in a flat world"
                if os.path.exists(prompt_path):
                    with open(prompt_path) as f:
                        prompt = f.read().strip()

                numbered_paths = [
                    os.path.join(sample_path, f"action_{i}.json") for i in range(args.n_players)
                ]
                numbered_present = [os.path.exists(p) for p in numbered_paths]
                action_left = os.path.join(sample_path, "action_left.json")
                action_right = os.path.join(sample_path, "action_right.json")

                action_jsons_arg: list[str | None] | None = None
                action_left_arg: str | None = None
                action_right_arg: str | None = None
                if any(numbered_present):
                    action_jsons_arg = [
                        p if exists else None for p, exists in zip(numbered_paths, numbered_present)
                    ]
                elif args.n_players == 2 and os.path.exists(action_left) and os.path.exists(action_right):
                    action_left_arg = action_left
                    action_right_arg = action_right
                else:
                    log.warning(
                        f"[{sample_idx}] {sample_name}: no action JSONs found "
                        f"(checked {numbered_paths} and 2P legacy left/right); proceeding without actions"
                    )

                gt = gt_path if os.path.exists(gt_path) else None

                save_root = os.path.join(args.save_root, sample_name)
                log.info(f"[{sample_idx + 1}/{len(sample_dirs)}] Processing {sample_name}")
                _run_single_sample(
                    image_path=image_path,
                    prompt=prompt,
                    save_root=save_root,
                    action_jsons=action_jsons_arg,
                    action_left_json=action_left_arg,
                    action_right_json=action_right_arg,
                    gt_video_path=gt,
                )

        else:
            assert args.image_path is not None, "--image_path is required for custom inference"
            assert args.prompt is not None, "--prompt is required for custom inference"
            assert os.path.exists(args.image_path), f"Image not found: {args.image_path}"

            _run_single_sample(
                image_path=args.image_path,
                prompt=args.prompt,
                save_root=args.save_root,
                action_jsons=args.action_jsons,
                action_left_json=args.action_left_json,
                action_right_json=args.action_right_json,
                gt_video_path=args.gt_video_path,
            )

    i2v_cli.cleanup()
