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

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.datasets.cached_replay_dataloader import duplicate_batches, duplicate_batches_random

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

Text2World_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS16_QWEN_IMAGE: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "image_cosmos_pretrain_qwen_20250415_video_cosmos_pretrain_20241219_s3"},
            {"override /model": "fsdp"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "add_fps_padding_mask"},
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
            group="official_runs",
            name="Stage-a_pt_3-Index-1-Size-2B-Res-480-Fps-16-Note-qwen_imagecaption",
        ),
        optimizer=dict(
            lr=2 ** (-14),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="480",
                state_t=24,
                resize_online=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
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
            load_path="cosmos_diffusion_v2/ablation_2B_0321_tokenizer_data/ablation_2B_0321_tokenizer_data_204pt1_wan2pt1tokenizer_resizeT24_Res480V250322-241219/checkpoints/iter_000150000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5_000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5_000,
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
                        batch_size=24,
                        num_workers=6,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                        dataset=dict(
                            resolution="480",
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
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.1),
                        dataset=dict(
                            resolution="480",
                            num_video_frames=93,
                            video_decoder_name="chunked_video_decoder_with_fixed_fps",
                            min_fps_thres=16,
                            max_fps_thres=45,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

Text2World_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{Text2World_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS16_QWEN_IMAGE['job']['name']}",
            {"override /net": "cosmos_v1_14B"},
            "_self_",
        ],
        checkpoint=dict(
            save_iter=2_500,
            load_path="cosmos_diffusion_v2/ablation_2B_0321_tokenizer_data/ablation_2B_0321_tokenizer_data_218pt1_wan2pt1tokenizer_resizeT20_Res480V250322-241219_14B_perframe_noise/checkpoints/iter_000110000",
            load_training_state=False,
            strict_resume=False,
        ),
        job=dict(
            group=Text2World_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS16_QWEN_IMAGE["job"]["group"],
            name="Stage-a_pt_3-Index-4-Size-14B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise",
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        model=dict(
            config=dict(
                state_t=20,
                net=dict(
                    rope_t_extrapolation_ratio=20.0 / 24,
                ),
                resize_online=True,
                fsdp_shard_size=32,
            ),
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
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=20 // 4,
                        num_workers=4,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                        dataset=dict(resolution="480"),
                    ),
                    ratio="${trainer.grad_accum_iter}",
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                        dataset=dict(
                            resolution="480",
                            num_video_frames=100,
                            video_decoder_name="chunked_video_decoder_with_fixed_fps",
                            min_fps_thres=16,
                            max_fps_thres=45,
                        ),
                    ),
                    ratio="${trainer.grad_accum_iter}",
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{Text2World_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE['job']['name']}",
            {"override /conditioner": "video_prediction_conditioner"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "video2world_val_sampling_image2video_vbench",
                    "video2world_val_sampling_image2video_sora",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_video2world",
            name="Stage-a_pt_3-Video2World-Index-4-Size-14B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise_joint_2framecond",
        ),
        trainer=dict(
            max_iter=200_000,
            grad_accum_iter=2,
            callbacks=dict(
                reg_model_image2video_sora_val_sampling=dict(
                    every_n=5_000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_sora_val_sampling=dict(
                    every_n=5_000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                reg_model_image2video_vbench_val_sampling=dict(
                    every_n=5_000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_vbench_val_sampling=dict(
                    every_n=5_000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
            ),
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=2,
            )
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs/Stage-a_pt_3-Index-4-Size-14B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise/checkpoints/iter_000035000",
        ),
    ),
    flags={"allow_objects": True},
)

V2V_STAGE_A_PT_3_INDEX_5_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND['job']['name']}",
            {"override /conditioner": "video_prediction_conditioner_v2"},
            "_self_",
        ],
        job=dict(
            group=V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND["job"]["group"],
            name="Stage-a_pt_3-Video2World-Index-5-Size-14B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise_joint_2framecond",
        ),
        trainer=dict(
            max_iter=200_000,
            grad_accum_iter=1,
        ),
    )
)

V2V_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND['job']['name']}",
            {"override /net": "cosmos_v1_2B"},
            "_self_",
        ],
        job=dict(
            group=V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND["job"]["group"],
            name="Stage-a_pt_3-Video2World-Index-1-Size-2B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise_joint_2framecond",
        ),
        optimizer=dict(
            lr=2 ** (-14),
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_500,
            load_path="cosmos_diffusion_v2/official_runs/Stage-a_pt_3-Index-1-Size-2B-Res-480-Fps-16-Note-qwen_imagecaption/checkpoints/iter_000057500",
            load_training_state=False,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="480",
                state_t=24,
                resize_online=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
                ),
            )
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=24,
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
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.1),
                        dataset=dict(
                            resolution="480",
                            num_video_frames=93,
                            video_decoder_name="chunked_video_decoder_with_fixed_fps",
                            min_fps_thres=16,
                            max_fps_thres=45,
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

V2V_STAGE_A_PT_3_INDEX_2_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND_GRADACCM_1: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{V2V_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=V2V_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND["job"]["group"],
            name="Stage-a_pt_3-Video2World-Index-2-Size-2B-Res-480-Fps-16-Note-qwen_imagecaption_sync_noise_joint_2framecond_gradaccm_1",
        ),
        trainer=dict(
            grad_accum_iter=1,
        ),
    )
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        Text2World_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS16_QWEN_IMAGE,
        *build_debug_runs(Text2World_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS16_QWEN_IMAGE),
    ],
    [
        Text2World_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE,
        *build_debug_runs(Text2World_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE),
    ],
    [
        V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND,
        *build_debug_runs(V2V_STAGE_A_PT_3_INDEX_4_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND),
    ],
    [
        V2V_STAGE_A_PT_3_INDEX_5_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND,
        *build_debug_runs(V2V_STAGE_A_PT_3_INDEX_5_SIZE_14B_RES_480_FPS_16_QWEN_IMAGE_JOINT_2FRAMECOND),
    ],
    [
        V2V_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND,
        *build_debug_runs(V2V_STAGE_A_PT_3_INDEX_1_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND),
    ],
    [
        V2V_STAGE_A_PT_3_INDEX_2_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND_GRADACCM_1,
        *build_debug_runs(V2V_STAGE_A_PT_3_INDEX_2_SIZE_2B_RES_480_FPS_16_IMAGE_JOINT_2FRAMECOND_GRADACCM_1),
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
