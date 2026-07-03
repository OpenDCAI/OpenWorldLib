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
from gamma_world._src.predict2.configs.video2world.experiment.reason_embeddings.stage3_2B import (
    I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY,
    I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2,
)

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=1000,
    logging_iter=50,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=10,
        ),
        every_n_sample_ema=dict(
            every_n=10,
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

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                scaling="rectified_flow",
                conditioner=dict(
                    text=dict(
                        use_empty_string=False,
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000065000/",
            load_training_state=False,
            strict_resume=True,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        use_cache=False,
                    ),
                    ratio=3,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

WEEKYLY_FINETUNING = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250602_dedup_20250825_s3",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],

            name="weekly_finetuning_pretrainvideo_20250602_dedup_20250825",
        ),
        trainer=dict(
            max_iter=50_000,
            logging_iter=20,
            callbacks=dict(
                iter_speed=dict(
                    hit_thres=200,
                ),
                every_n_sample_reg=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
            ),

            run_validation=False,
            validation_iter=999999999,
        ),
        checkpoint=dict(
            save_iter=5_000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_COOLDOWN_FROM_10K = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v6_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_cooldown_from_10K",
        ),
        trainer=dict(
            max_iter=30_000,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[30_000],
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            use_native_fps=True,
                            min_fps_thres=14,
                            max_fps_thres=30,
                        ),
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_COOLDOWN_FROM_10K = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v6_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_HQ_cooldown_from_10K",
        ),
        trainer=dict(
            max_iter=50_000,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16_hq_cooldown_from_57K/checkpoints/iter_000035000/",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_4K_COOLDOWN_FROM_10K = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_4K_cooldown_from_10K",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16_hq_cooldown_from_57K_4K_videos/checkpoints/iter_000035000/",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V8_COOLDOWN_FROM_10K = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v8_20250822_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_HQ_V8_cooldown_from_10K",
        ),
        trainer=dict(
            max_iter=50_000,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_noise_analysis_shift2",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=True,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                sde=dict(
                    p_mean=math.log(2.0),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT5_CONTCHUNK = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_noise_analysis_shift5_contchunk",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=True,
            strict_resume=True,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            use_native_fps=True,
                            min_fps_thres=14,
                            max_fps_thres=30,
                        ),
                    ),
                ),
            ),
        ),
        model=dict(
            config=dict(
                sde=dict(
                    p_mean=math.log(5.0),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT7 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_noise_analysis_shift7",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=True,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                sde=dict(
                    p_mean=math.log(7.0),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

T2V_QWEN05B_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-qwen05b-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
        ),
        model=dict(
            config=dict(
                net=dict(
                    crossattn_proj_in_channels=21504,
                ),
                text_encoder_class="qwen0.5B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/pretrained/qwen2p5_0p5b/checkpoints/iter_000000001/model/",
                    model_config=dict(
                        model_config=dict(
                            name_or_path="Qwen/Qwen2.5-0.5B",
                            model_type="qwen2_5",
                            tokenizer_type="Qwen/Qwen2.5-0.5B",
                        ),
                        tokenizer=dict(
                            tokenizer_type="Qwen/Qwen2.5-0.5B",
                        ),
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000065000/",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_COOLDOWN_FROM_10K,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_COOLDOWN_FROM_10K),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_COOLDOWN_FROM_10K,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_COOLDOWN_FROM_10K),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_4K_COOLDOWN_FROM_10K,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_4K_COOLDOWN_FROM_10K),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT2,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT2),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT7,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT7),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT5_CONTCHUNK,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_NOISE_ANALYSIS_SHIFT5_CONTCHUNK
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V8_COOLDOWN_FROM_10K,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V8_COOLDOWN_FROM_10K
        ),
    ],
    [
        T2V_QWEN05B_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16,
        *build_debug_runs(T2V_QWEN05B_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16),
    ],
    [
        WEEKYLY_FINETUNING,
        *build_debug_runs(WEEKYLY_FINETUNING),
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
