# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.datasets.cached_replay_dataloader import (
    duplicate_batches_random,
    get_cached_replay_dataloader,
)
from gamma_world._src.predict2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from gamma_world._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader
from gamma_world._src.predict2.models.video2world_model import HighSigmaStrategy
from gamma_world._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=25,
    logging_iter=2,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=20,
        ),
        every_n_sample_ema=dict(
            every_n=20,
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

def build_gcp_config(job):

    gcp_config = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /checkpoint": "gcp"},
            {"override /tokenizer": "wan2pt1_tokenizer_gcp"},
            "_self_",
        ],
        model=dict(
            config=dict(
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/gcp_checkpoint.secret",
                ),
                conditioner=dict(
                    text=dict(
                        empty_string_embeddings_path="s3://bucket/predict2_assets/reason1_empty_string_embeddings.pt",
                        credential_path="credentials/gcp_training.secret",
                    ),
                ),
            )
        ),
        job=dict(
            name=(
                job["job"]["name"].replace("_s3", "_gcp")
                if "_s3" in job["job"]["name"]
                else job["job"]["name"] + "_gcp"
            ),
        ),
    )

    gcp_config["dataloader_train"] = dict(dataloaders=dict())
    for dataset_name in job["dataloader_train"]["dataloaders"].keys():
        gcp_config["dataloader_train"]["dataloaders"][dataset_name] = dict(
            dataloader=dict(
                dataset=dict(
                    object_store="gcp",
                ),
            ),
        )
    return [gcp_config]

T2W_7B_MODEL_401_256RES_S3: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /model": "fsdp"},
            {"override /net": "cosmos_v1_7B"},
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
            group="official_runs_text2world",
            name="official_runs_text2world_401_7b_text_to_world_256res_s3",
        ),
        optimizer=dict(
            lr=2 ** (-14),
            weight_decay=0.05,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[300_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="256",
                state_t=16,
                resize_online=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.1,
                        use_empty_string=True,
                    ),
                ),
                sde=dict(
                    p_mean=0.0,
                    p_std=1.2,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                denoise_replace_gt_frames=True,
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
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
        ),
        trainer=dict(
            max_iter=500_000,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data_qwen_2p5_7b_v4_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=64,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=64,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "video_data_cosmos_pretrain": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250707_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="s3",
                            resolution="256",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=61,
                            dataset_resolution_type="all",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=6,
                        num_workers=8,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=2),
                    ),
                    ratio=2,
                ),
            }
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

T2W_7B_MODEL_402_256RES_S3_REASON_1P1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2W_7B_MODEL_401_256RES_S3['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=T2W_7B_MODEL_401_256RES_S3["job"]["group"],
            name="official_runs_text2world_402_7b_reason_1p1_text_to_world_256res_s3",
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.1,
                        use_empty_string=False,
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/s3_checkpoint.secret",
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
            load_path="cosmos_diffusion_v2/official_runs_text2world/official_runs_text2world_401_7b_text_to_world_256res_gcp/checkpoints/iter_000050000/",
            load_training_state=True,
            strict_resume=True,
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data_qwen_2p5_7b_v4_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=64,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=64,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "video_data_cosmos_pretrain": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="s3",
                            resolution="256",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=61,
                            dataset_resolution_type="all",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=6,
                        num_workers=8,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=2),
                    ),
                    ratio=2,
                ),
            }
        ),
    ),
    flags={"allow_objects": True},
)

T2W_7B_MODEL_403_256RES_S3_REASON_1P1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2W_7B_MODEL_402_256RES_S3_REASON_1P1['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=T2W_7B_MODEL_402_256RES_S3_REASON_1P1["job"]["group"],
            name="official_runs_text2world_403_7b_reason_1p1_text_to_world_256res_s3",
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/official_runs_text2world_402_7b_reason_1p1_text_to_world_256res_s3/checkpoints/iter_000055000/",
            load_training_state=True,
            strict_resume=True,
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data_qwen_2p5_7b_v4_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=24,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="s3",
                            resolution="256",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="all",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=24,
                        num_workers=16,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        cache_size=32,
                        use_cache=False,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=1,
                ),
                "video_data_cosmos_pretrain": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="s3",
                            resolution="256",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=61,
                            dataset_resolution_type="all",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=2,
                        num_workers=8,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1),
                    ),
                    ratio=2,
                ),
            }
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
for _item, _item_wo_resume, _item_mock_wo_resume, _item_gcp in [
    [
        T2W_7B_MODEL_401_256RES_S3,
        *build_debug_runs(T2W_7B_MODEL_401_256RES_S3),
        *build_gcp_config(T2W_7B_MODEL_401_256RES_S3),
    ],
    [
        T2W_7B_MODEL_402_256RES_S3_REASON_1P1,
        *build_debug_runs(T2W_7B_MODEL_402_256RES_S3_REASON_1P1),
        *build_gcp_config(T2W_7B_MODEL_402_256RES_S3_REASON_1P1),
    ],
    [
        T2W_7B_MODEL_403_256RES_S3_REASON_1P1,
        *build_debug_runs(T2W_7B_MODEL_403_256RES_S3_REASON_1P1),
        *build_gcp_config(T2W_7B_MODEL_403_256RES_S3_REASON_1P1),
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
    if _item_gcp is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=_item_gcp["job"]["name"],
            node=_item_gcp,
        )