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
from gamma_world._src.predict2.datasets.cached_replay_dataloader import duplicate_batches, duplicate_batches_random
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

I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "image_cosmos_pretrain_qwen_20250415_video_cosmos_pretrain_v1_3_20250426_s3"},
            {"override /model": "fsdp"},
            {"override /net": "cosmos_v1_14B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.2,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[300_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=2,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=32,
                resolution="480",
                state_t=20,
                resize_online=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=20.0 / 24,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-a_pt_3-Vid2Vid-Index-5-Size-14B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise_joint_2framecond/checkpoints/iter_000052500",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500000000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=500000000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=20 // 4,
                        num_workers=6,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                        dataset=dict(
                            resolution="480",
                        ),
                    ),
                    ratio="${trainer.grad_accum_iter}",
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                        dataset=dict(
                            resolution="480",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            use_native_fps=True,
                        ),
                    ),
                    ratio="${trainer.grad_accum_iter}",
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {

                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40/checkpoints/iter_000018000",
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
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
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
            max_iter=200_000,
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
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_native_fps_qwen_concat",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat/checkpoints/iter_000020000",
            load_training_state=False,
            strict_resume=False,
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        trainer=dict(
            max_iter=200_000,
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

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_14B_RES_720_FPS10_T15_HQV5_from_43: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-100-Size-14B-Res-720-Fps-10-Note-T15_HQV5_from_43_qwen_concat",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-100-Size-14B-Res-720-Fps-10-Note-T15_HQV5_from_43/checkpoints/iter_000010000",
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
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
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
                            num_video_frames=61,
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

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_14B_RES_480_FPS10_T15_HQV5_from_43: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-101-Size-14B-Res-480-Fps-10-Note-T15_HQV5_from_43_qwen_concat",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-101-Size-14B-Res-480-Fps-10-Note-T15_HQV5_from_43/checkpoints/iter_000010000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="480",
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
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
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
                            num_video_frames=61,
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

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_14B_RES_480_FPS16_T24_HQV5_from_43: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-102-Size-14B-Res-480-Fps-16-Note-T24_HQV5_from_43_qwen_concat",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-102-Size-14B-Res-480-Fps-16-Note-T24_HQV5_from_43/checkpoints/iter_000010000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                resolution="480",
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
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        trainer=dict(
            max_iter=200_000,
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

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS_LOSS_REWEIGHTED: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS["job"][
                "group"
            ],
            name="Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_native_fps_qwen_concat_loss_reweighted",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_native_fps_qwen_concat/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=False,
        ),
        scheduler=dict(
            f_max=[0.3],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[100000],
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        model=dict(
            config=dict(
                adjust_video_noise=False,
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                rectified_flow_loss_weight_uniform=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_V2: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR['job']['name']}",
            {

                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_14B_RES_480_NEW_VIDEO_DATA1PT3_AUGMENTOR["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat_v2",
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_native_fps_qwen_concat/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
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
                        mode="predict2_14b_720",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        trainer=dict(
            max_iter=200_000,
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
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_14B_RES_720_FPS10_T15_HQV5_from_43,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_14B_RES_720_FPS10_T15_HQV5_from_43),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_14B_RES_480_FPS10_T15_HQV5_from_43,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_14B_RES_480_FPS10_T15_HQV5_from_43),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_14B_RES_480_FPS16_T24_HQV5_from_43,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_14B_RES_480_FPS16_T24_HQV5_from_43),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS_LOSS_REWEIGHTED,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_NATIVE_FPS_LOSS_REWEIGHTED
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_V2,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_T24_HQV5_from_40_V2),
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
