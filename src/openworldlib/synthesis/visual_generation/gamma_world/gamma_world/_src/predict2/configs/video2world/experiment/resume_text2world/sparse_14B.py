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

import math

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.models.video2world_model import HighSigmaStrategy

NATTEN_PARAMETERS_14B_COMB01 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

NATTEN_PARAMETERS_14B_COMB02 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=25,
    logging_iter=2,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=12,
        ),
        every_n_sample_ema=dict(
            every_n=12,
        ),
        reg_model_image2video_sora_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_sora_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        reg_model_image2video_vbench_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_vbench_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
    ),
)
_CKPT_DEBUG_CONFIG = dict(
    save_iter=10,
    load_path="",
    load_training_state=False,
    strict_resume=False,
)

def build_debug_runs(job):
    wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    mock_wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_MOCK_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    return [wo_resume, mock_wo_resume]

I2V_STAGE_C_PT_4_INDEX_200_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_9Dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_1_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-200-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_40_sparse-attn_9dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-38-Size-14B-Res-720-Fps-16-Note-T24_HQV3_from_35",
            load_training_state=True,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=9,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="gt720p",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_201_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb01_1dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-201-Size-14B-Res-720-Fps-16-Note-T24_HQV2_1_from_43_sparse-attn_comb01-1dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-40-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_38/checkpoints/iter_000015000",
            load_training_state=True,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB01,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[20_001],
        ),
        trainer=dict(
            max_iter=18_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="all",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_202_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb02_1dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-202-Size-14B-Res-720-Fps-16-Note-T24_HQV2_1_from_43_sparse-attn_comb02-1dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-40-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_38/checkpoints/iter_000015000",
            load_training_state=True,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB02,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[20_001],
        ),
        trainer=dict(
            max_iter=18_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="all",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_203_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_Comb02_1dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_1_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-203-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_40_sparse-attn_comb02-1dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-38-Size-14B-Res-720-Fps-16-Note-T24_HQV3_from_35",
            load_training_state=True,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB02,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="gt720p",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_204_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-204-Size-14B-Res-720-Fps-16-Note-T24_HQV2_1_from_203_sparse-attn_comb02-1dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-203-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_40_sparse-attn_comb02-1dense/checkpoints/iter_000031000",
            load_training_state=True,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB02,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[50_001],
        ),
        trainer=dict(
            max_iter=50_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="all",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_205_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense_altLR: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-205-Size-14B-Res-720-Fps-16-Note-T24_HQV2_1_from_203_sparse-attn_comb02-1dense-altLR",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-203-Size-14B-Res-720-Fps-16-Note-T24_HQV3pt1_from_40_sparse-attn_comb02-1dense/checkpoints/iter_000031000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=24,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB02,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[20_001],
        ),
        trainer=dict(
            max_iter=18_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="all",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_230_SIZE_14B_RES_720_FPS10_T15_HQV5_from_205_SparseAttn_Comb02_1dense: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-230-Size-14B-Res-720-Fps-10-Note-T15_HQV5_from_205_sparse-attn_comb02-1dense",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-205-Size-14B-Res-720-Fps-16-Note-T24_HQV2_1_from_203_sparse-attn_comb02-1dense-altLR/checkpoints/iter_000018000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                state_t=16,
                resize_online=True,
                tokenizer=dict(
                    temporal_window=16,
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=16.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720",
                    ),
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_14B_COMB02,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[20_001],
        ),
        trainer=dict(
            max_iter=18_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type="t5_xxl",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=61,
                            dataset_resolution_type="all",
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        I2V_STAGE_C_PT_4_INDEX_200_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_9Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_200_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_9Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_201_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb01_1dense,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_201_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb01_1dense
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_202_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb02_1dense,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_202_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_43_SparseAttn_Comb02_1dense
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_203_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_Comb02_1dense,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_203_SIZE_14B_RES_720_FPS16_T24_HQV3pt1_from_40_SparseAttn_Comb02_1dense
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_204_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_204_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_205_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense_altLR,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_205_SIZE_14B_RES_720_FPS16_T24_HQV2_1_from_203_SparseAttn_Comb02_1dense_altLR
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_230_SIZE_14B_RES_720_FPS10_T15_HQV5_from_205_SparseAttn_Comb02_1dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_230_SIZE_14B_RES_720_FPS10_T15_HQV5_from_205_SparseAttn_Comb02_1dense),
    ],
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
    if _item_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_wo_resume",
            node=_item_wo_resume,
        )
    if _item_mock_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_mock_wo_resume",
            node=_item_mock_wo_resume,
        )
