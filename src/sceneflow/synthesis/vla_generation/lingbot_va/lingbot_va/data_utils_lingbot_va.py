# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import numpy as np
import torch

__all__ = ['get_mesh_id', 'data_seq_to_patch']


def data_seq_to_patch(
    patch_size,
    data_seq,
    latent_num_frames,
    latent_height,
    latent_width,
    batch_size=1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(batch_size, post_patch_num_frames,
                                  post_patch_height, post_patch_width, p_t,
                                  p_h, p_w, -1)
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


