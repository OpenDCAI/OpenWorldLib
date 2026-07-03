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

import functools
import math

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.datasets.cached_replay_dataloader import duplicate_batches_random
from gamma_world._src.predict2.models.video2world_model import HighSigmaStrategy

NATTEN_PARAMETERS_2B_COMB01 = [
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

NATTEN_PARAMETERS_2B_COMB02 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

NATTEN_PARAMETERS_2B_COMB03 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

NATTEN_PARAMETERS_2B_COMB04 = [
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
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
]

NATTEN_PARAMETERS_2B_COMB05 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 4, 24), "stride": (1, 1, 8), "dilation": (1, 11, 1), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    None,
    None,
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
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

I2V_STAGE_C_PT_4_INDEX_200_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-200-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_7dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=7,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_201_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_6Dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-201-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_6dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=6,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_202_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_4Dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-202-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_4dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=4,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_203_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_9Dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-203-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_9dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=9,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_204_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_12Dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-204-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_12dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=12,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_205_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense_NA = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-205-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_7dense-na",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=7,
                    natten_parameters={"window_size": (-1, 12, 24), "stride": (1, 1, 1), "base_size": (-1, 44, 80)},
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_206_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb01_4dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-206-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb01-4dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_207_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb02_0dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-207-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb02-0dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB02,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_208_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb03_1dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-208-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb03-1dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB03,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_209_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb04_0dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-209-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb04-0dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB04,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_210_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb05_2dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v3_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-210-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb05-2dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB05,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-20-Size-2B-Res-720-Fps-16-Note-06_04_accumulated_hq_wan_from_19/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_211_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v6_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-211-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_207_sparse-attn_comb02-0dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB02,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-207-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb02-0dense/checkpoints/iter_000032000",
            load_training_state=True,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=48_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_212_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v6_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-212-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_207_sparse-attn_comb02-0dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[60_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB02,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-207-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb02-0dense/checkpoints/iter_000032000",
            load_training_state=True,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=60_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_213_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense_altLR = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v6_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-213-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_207_sparse-attn_comb02-0dense-altLR",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB02,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-207-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_22_sparse-attn_comb02-0dense/checkpoints/iter_000032000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=26_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=93,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_230_SIZE_2B_RES_720_FPS10_HQ_V5_from_212_SparseAttn_Comb02_0dense = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v5_20250607_s3"
            },
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-c_pt_4-Index-230-Size-2B-Res-720-Fps-10-Note-HQ_V5_from_212_sparse-attn_comb02-0dense",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=16,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=16.0 / 24,
                    n_dense_blocks=0,
                    natten_parameters=NATTEN_PARAMETERS_2B_COMB02,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_gna/Stage-c_pt_4-Index-212-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_207_sparse-attn_comb02-0dense/checkpoints/iter_000050000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=26_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
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
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type="t5_xxl",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=61,
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
        I2V_STAGE_C_PT_4_INDEX_200_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_200_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_201_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_6Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_201_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_6Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_202_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_4Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_202_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_4Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_203_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_9Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_203_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_9Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_204_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_12Dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_204_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_12Dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_205_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense_NA,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_205_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_7Dense_NA),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_206_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb01_4dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_206_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb01_4dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_207_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb02_0dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_207_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb02_0dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_208_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb03_1dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_208_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb03_1dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_209_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb04_0dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_209_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb04_0dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_210_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb05_2dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_210_SIZE_2B_RES_720_FPS16_HQ_V3_from_22_SparseAttn_Comb05_2dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_211_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_211_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_212_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_212_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_213_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense_altLR,
        *build_debug_runs(
            I2V_STAGE_C_PT_4_INDEX_213_SIZE_2B_RES_720_FPS16_HQ_V6_from_207_SparseAttn_Comb02_0dense_altLR
        ),
    ],
    [
        I2V_STAGE_C_PT_4_INDEX_230_SIZE_2B_RES_720_FPS10_HQ_V5_from_212_SparseAttn_Comb02_0dense,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_230_SIZE_2B_RES_720_FPS10_HQ_V5_from_212_SparseAttn_Comb02_0dense),
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
