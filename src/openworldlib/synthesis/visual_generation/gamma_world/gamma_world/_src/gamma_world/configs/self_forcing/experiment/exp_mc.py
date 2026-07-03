# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict

CAUSAL_FEW_STEP: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
            {"override /net": "causal_cosmosv2_2b"},
            {"override /net_real_score": "cosmos_v1_2B_mc"},
            {"override /net_fake_score": "cosmos_v1_2B_mc"},
            {"override /model": "fsdp_mv"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"override /data_train": "video_solaris_action"},
            "_self_",
        ],
        job=dict(
            group="causal_cosmos2",
            name="causal_few_step",
        ),
        model=dict(
            config=dict(
                ema=dict(
                    enabled=False,
                ),
                state_t=48,
                train_sample_views_range=[2, 2],
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                denoise_replace_gt_frames=True,
                use_action_control=True,
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
                    use_sparse_hub=True,
                    z_num=8,
                    multi_agent_rope_simplex_pool_size=4,
                    multi_agent_rope_agent_encoding="simplex",
                    multi_agent_rope_agent_scale=1.0,
                    multi_agent_rope_share_action_encoder=True,
                    local_attn_size=24,
                    sink_size=0,
                ),
                net_real_score=dict(
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
                net_fake_score=dict(
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
                net_ckpt="",
                net_real_score_ckpt="",
                net_fake_score_ckpt="",
                context_noise=128,
                real_guidance_scale=0.0,
                optimizer_fake_score_config=dict(
                    lr=4e-7,
                    weight_decay=1e-2,
                    betas=(0.0, 0.999),
                ),
                model_type="i2v",
                i2v_zero_latent_condition=True,
                use_vidprom_dataset=False,
                fsdp_shard_size=64,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                shuffle_agents=True,
            ),
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=1e-2,
            betas=(0.0, 0.999),
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[10_000],
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            save_iter=50,
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
            max_iter=10_000,
            logging_iter=10,
            callbacks=dict(
                grad_clip=dict(clip_norm=10.0),
                iter_speed=dict(hit_thres=50),
                compile_tokenizer=dict(enabled=False),
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
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=CAUSAL_FEW_STEP["job"]["name"],
    node=CAUSAL_FEW_STEP,
)
