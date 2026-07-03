# Copyright Alibaba Inc. All Rights Reserved.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import torch.cuda.amp as amp

from ..models.aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead_3D_Causal
from ..heads.track_head import TrackHead
from openworldlib.base_models.diffusion_model.video.wan_2p1.modules.model import sinusoidal_embedding_1d
class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=518,
                 patch_size=16,
                 embed_dim=1024,
                 number_frame=81,
                 freq_dim = 256,
                 enable_camera = True,
                 enable_depth = True,
                 enable_point = True,
                 enable_track = True,
                 load_path = None,
                 DPT_patch_size = 16,
                 ):
        super().__init__()

        self.spatial_frame = (number_frame-1)//4 + 1
        self.freq_dim = freq_dim
        self.embed_dim = embed_dim
        self.projection_head = nn.Conv3d(5120, 1024, kernel_size=(1,1,1), stride=(1,1,1))
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, spatial_time=self.spatial_frame)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head = DPTHead_3D_Causal(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", patch_size=DPT_patch_size) if enable_depth else None
        self.point_head = DPTHead_3D_Causal(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", patch_size=DPT_patch_size) if enable_point else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 6))

        if load_path is not None:
            self.load_state_dict(torch.load(load_path)['model'],strict=True)

    def forward(self, patch_token: torch.Tensor, query_points: torch.Tensor = None, camera_token: torch.Tensor = None, t = None):

        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            camera_token (torch.Tensro, optional): [B, S, 9]

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        patch_token = self.projection_head(patch_token)
        patch_token = patch_token.permute(0, 2, 3, 4, 1)
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.embed_dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32


        aggregated_tokens_list, patch_start_idx = self.aggregator(patch_token,camera_token, e0)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=True, dtype= torch.bfloat16):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=patch_token, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=patch_token, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=patch_token, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf


        return predictions
    def _process_wan_input(self,patch_token: torch.Tensor, query_points: torch.Tensor = None, camera_token: torch.Tensor = None, t = None):

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        patch_token = self.projection_head(patch_token)
        patch_token = patch_token.permute(0, 2, 3, 4, 1)

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.embed_dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        return patch_token, camera_token, e0


    def _head_predction(self, patch_token,patch_start_idx,aggregated_tokens_list):
        predictions = {}
        with torch.cuda.amp.autocast(enabled=True, dtype= torch.bfloat16):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=patch_token, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=patch_token, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
        return predictions
