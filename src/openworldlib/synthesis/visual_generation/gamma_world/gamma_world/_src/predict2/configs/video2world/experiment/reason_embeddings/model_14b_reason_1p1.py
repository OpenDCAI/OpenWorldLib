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

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.configs.video2world.experiment.reason_embeddings.stage3_14B_index_3 import (
    I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR,
)
from gamma_world._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from gamma_world._src.predict2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from gamma_world._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader
from gamma_world._src.predict2.models.video2world_model import HighSigmaStrategy
from gamma_world._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=25,
    logging_iter=2,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=1,
        ),
        every_n_sample_ema=dict(
            every_n=1,
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

I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_gcp",
            },
            {"override /checkpoint": "gcp"},
            {"override /tokenizer": "wan2pt1_tokenizer_gcp"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_gcp",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat_v2/checkpoints/iter_000038000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
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
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                scaling="rectified_flow",
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/gcp_checkpoint.secret",
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.3],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=False,
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
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
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
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="all",
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_gcp",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_gcp_resume2",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_gcp/checkpoints/iter_000011500",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": None,
            },
            "_self_",
        ],
        job=dict(
            group=I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_improved_pretraining_data_gcp",
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=3,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=3,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                    ),
                    ratio=1,
                ),
                "video_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            max_fps_thres=38,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="gt720p",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=1,
                        num_workers=8,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                        use_cache=False,
                    ),
                    ratio=2,
                ),
            }
        ),
        checkpoint=dict(
            save_iter=250,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_gcp_resume2/checkpoints/iter_000043750",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16),
    ],
    [
        I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME2,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME2),
    ],
    [
        I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA
        ),
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
