# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
BIDIRECTIONAL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_solaris_action"},
            {"override /model": "fsdp_mv"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "viz_online_sampling", "wandb", "cluster_speed"]},
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="causal_cosmos2",
            name="bidirectional",
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                noise_scheme="consistent_noise",
                num_frame_per_block=48,
                max_latent_frames_per_gpu=48,
                ema=dict(
                    enabled=False,
                ),
                denoise_replace_gt_frames=True,
                state_t=48,
                fsdp_shard_size=8,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                split_cp_in_model=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
                    timestep_scale=0.001,
                    enable_action_control=True,
                    action_keyboard_dim=23,
                    action_camera_dim=2,
                    action_use_camera=True,
                    action_embed_dim=256,
                    action_temporal_downsample=4,
                    use_multi_agent_rope=True,
                    multi_agent_rope_num_agents=2,
                    multi_agent_rope_agent_id_offset=0,
                    multi_agent_rope_simplex_pool_size=4,
                    multi_agent_rope_agent_encoding="simplex",
                    multi_agent_rope_agent_scale=1.0,
                    multi_agent_rope_share_action_encoder=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy="full_concat",
                    compute_online=True,
                    ckpt_path="hf://nvidia/Cosmos-Reason1-7B",
                ),
                use_action_control=True,
                train_sample_views_range=[2, 2],
                condition_locations=["first_random_n"],
                min_num_conditional_frames_per_view=1,
                max_num_conditional_frames_per_view=1,
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                shuffle_agents=True,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=10,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100_000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=10,
                ),
                grad_clip=dict(clip_norm=0.1),
                iter_speed=dict(hit_thres=100),
            ),
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                resolution_hw=(320, 480),
                take_half_width=False,
                num_video_frames=189,
                load_action=True,
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": False},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=BIDIRECTIONAL["job"]["name"],
    node=BIDIRECTIONAL,
)
