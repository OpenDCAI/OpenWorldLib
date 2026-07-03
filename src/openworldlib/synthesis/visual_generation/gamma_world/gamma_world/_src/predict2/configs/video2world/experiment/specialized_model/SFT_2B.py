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
from copy import deepcopy

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.configs.video2world.experiment.reason_embeddings.model_2B_reason_1p1 import (
    T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16,
)
from gamma_world._src.predict2.configs.video2world.experiment.reason_embeddings.stage3_2B import (
    T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED,
)

I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-3-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_face_focused",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-3-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_face_focused_formal",
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
                scaling="rectified_flow",
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_crowded",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_crowded_formal",
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
                scaling="rectified_flow",
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_4_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_high_motion_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-4-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_high_motion",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_4_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_high_motion_20250607_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-4-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_high_motion_formal",
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
                scaling="rectified_flow",
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_720_FPS16_ROBOTICS = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-101-Size-2B-Res-720-Fps-16-Note-HQ_V7_robotics",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_720_FPS16_ROBOTICS_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-101-Size-2B-Res-720-Fps-16-Note-HQ_V7_robotics_formal",
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
                scaling="rectified_flow",
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_201_SIZE_2B_RES_720_FPS16_FISHEYE = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {"override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_fisheye_20250806_s3"},
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-201-Size-2B-Res-720-Fps-16-Note-HQ_V7_fisheye",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

I2V_STAGE_C_PT_4_INDEX_301_SIZE_2B_RES_720_FPS16_AV = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_av_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-301-Size-2B-Res-720-Fps-16-Note-HQ_V7_av",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_301_SIZE_2B_RES_720_FPS16_AV_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_av_only_20250717_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-301-Size-2B-Res-720-Fps-16-Note-HQ_V7_av_formal",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_401_SIZE_2B_RES_720_FPS16_PHYSICAL_AI = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_physical_ai_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-401-Size-2B-Res-720-Fps-16-Note-HQ_V7_physical_ai",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
    ),
    flags={"allow_objects": True},
)
I2V_STAGE_C_PT_4_INDEX_401_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_FORMAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_physical_ai_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-401-Size-2B-Res-720-Fps-16-Note-HQ_V7_physical_ai_formal",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(

            config=dict(
                resolution="720",
                scaling="rectified_flow",
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,

            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
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
    I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED,
    I2V_STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED,
    I2V_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_720_FPS16_ROBOTICS,
    I2V_STAGE_C_PT_4_INDEX_201_SIZE_2B_RES_720_FPS16_FISHEYE,
    I2V_STAGE_C_PT_4_INDEX_301_SIZE_2B_RES_720_FPS16_AV,
    I2V_STAGE_C_PT_4_INDEX_4_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION,
    I2V_STAGE_C_PT_4_INDEX_401_SIZE_2B_RES_720_FPS16_PHYSICAL_AI,
    I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_FORMAL,
    I2V_STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_FORMAL,
    I2V_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_720_FPS16_ROBOTICS_FORMAL,
    I2V_STAGE_C_PT_4_INDEX_301_SIZE_2B_RES_720_FPS16_AV_FORMAL,
    I2V_STAGE_C_PT_4_INDEX_4_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_FORMAL,
    I2V_STAGE_C_PT_4_INDEX_401_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_FORMAL,
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
    for variant_name, variant_config in variants:
        _item_variant = deepcopy(_item)

        for key, value in variant_config.items():
            _item_variant = apply_kv_to_config(_item_variant, key, value)

        _item_variant["job"]["name"] = f"{_item['job']['name']}_{variant_name}"

        cs.store(group="experiment", package="_global_", name=f"{_item_variant['job']['name']}", node=_item_variant)
