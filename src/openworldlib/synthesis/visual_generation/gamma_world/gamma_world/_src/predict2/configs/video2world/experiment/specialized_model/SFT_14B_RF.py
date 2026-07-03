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

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.configs.video2world.experiment.reason_embeddings.model_14b_reason_1p1_rectified_flow import (
    T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA,
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_FACE_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_face_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_CROWDED_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_crowded_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_HIGH_MOTION_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_high_motion_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_high_motion_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_robotics_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_AV_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_av_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_av_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_PHYSICAL_AI_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_physical_ai_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_physical_ai_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),

    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_4K_COOLDOWN_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_4k_cooldown_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),

    ),
    flags={"allow_objects": True},
)

variants = [
    (
        "variant1",
        {
            "scheduler.f_max": [0.4],
            "scheduler.f_min": [0.1],
        },
    ),
]

def apply_kv_to_config(config, key, value):

    new_config = deepcopy(config)

    parts = key.split(".")
    current = new_config

    for i, part in enumerate(parts[:-1]):
        if isinstance(current, (dict, LazyDict)):
            if part not in current:

                current[part] = dict()
            current = current[part]
        else:

            if not hasattr(current, part):

                setattr(current, part, dict())
            current = getattr(current, part)

    final_key = parts[-1]
    if isinstance(current, (dict, LazyDict)):
        current[final_key] = value
    else:
        setattr(current, final_key, value)

    return new_config

cs = ConfigStore.instance()

for _item in [
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_FACE_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_CROWDED_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_HIGH_MOTION_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_AV_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_PHYSICAL_AI_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_4K_COOLDOWN_RF_HIGH_SIGMA,
]:
    if isinstance(_item, LazyDict):
        cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
        for variant_name, variant_config in variants:
            _item_variant = deepcopy(_item)

            for key, value in variant_config.items():
                _item_variant = apply_kv_to_config(_item_variant, key, value)

            _item_variant["job"]["name"] = f"{_item['job']['name']}_{variant_name}"

            cs.store(group="experiment", package="_global_", name=f"{_item_variant['job']['name']}", node=_item_variant)
