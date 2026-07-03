# Copyright Alibaba Inc. All Rights Reserved.
from .wan_video_dit import flash_attention,WanModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from safetensors import safe_open
import torch
import torch.nn as nn

class PoseProjModel(nn.Module):
    def __init__(self, pose_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(pose_in_dim, cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, pose_embeds):
        context_tokens = self.proj(pose_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens #[B,L,C]


class GroupLinearDualK(nn.Module):
    def __init__(self, context_dim, hidden_dim, groups=2):
        super().__init__()
        self.group1 = nn.Linear(context_dim, context_dim)
        # self.group2 = nn.Linear(hidden_dim, context_dim)
        intermediate_dim = min(hidden_dim, context_dim) // 2  # 1024
        self.group2 = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),    # 5120 -> 1024
            nn.ReLU(),
            nn.Linear(intermediate_dim, context_dim)    # 1024 -> 2048
        )

    def forward(self, x1, x2):
        out1 = self.group1(x1)
        out2 = self.group2(x2)
        return out1, out2


class GroupLinearDualV(nn.Module):
    def __init__(self, context_dim, hidden_dim, groups=2):
        super().__init__()
        reduction_factor = 5
        reduced_dim = context_dim // reduction_factor
        self.group2 = nn.Sequential(
            nn.Linear(context_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim)
        )

        last_layer = self.group2[-1]
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)


    def forward(self, x):
#        out1 = self.group1(x)
        out1 = 0.
        out2 = self.group2(x)
        return out1, out2


def get_processor(method, context_dim, hidden_dim):
    if method == 'latent_split' or method == 'latent_overall':
        k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        v_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        nn.init.zeros_(k_proj.weight)
        nn.init.zeros_(v_proj.weight)

        return k_proj, v_proj

    elif method == 'adaln':
        k_proj = GroupLinearDualK(context_dim, hidden_dim)
        v_proj = GroupLinearDualV(context_dim, hidden_dim)

        return k_proj, v_proj

class CrossAttentionAdapterProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim, pose_inject_method='latent_split'):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.pose_inject_method = pose_inject_method

        self.k_proj, self.v_proj = get_processor(pose_inject_method, context_dim, hidden_dim)


    def __call__(self, attn: nn.Module, x: torch.Tensor, y: torch.Tensor,
                plucker_fea: torch.Tensor, plucker_context_lens: torch.Tensor, pose_scale: float = 1.0):
        if attn.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = attn.norm_q(attn.q(x))
        k = attn.norm_k(attn.k(ctx))
        v = attn.v(ctx)
        x = attn.attn(q, k, v)
        if attn.has_image_input:
            k_img = attn.norm_k_img(attn.k_img(img))
            v_img = attn.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=attn.num_heads)
            x = x + y

        b, _, d = x.size()
        latents_num_frames = len(plucker_context_lens)
        is_all_zeros = torch.all(plucker_fea == 0).item()

        if self.pose_inject_method=='adaln':
            plucker_fea, combined = self.k_proj(plucker_fea, x)

            combined = combined+plucker_fea

            scale, shift = self.v_proj(combined)

            scale = scale*pose_scale
            shift = shift*pose_scale


            if is_all_zeros:
                x = x
            else:
                x = x * (scale + 1.) + shift

        elif self.pose_inject_method=='latent_split':
            pose_q = q.view(b*latents_num_frames, -1, d)  # [b, 21, l1, d]
            ip_key = self.k_proj(plucker_fea).view(b*latents_num_frames, -1, d)
            ip_value = self.v_proj(plucker_fea).view(b*latents_num_frames, -1, d)
            pose_x = flash_attention(pose_q, ip_key, ip_value, num_heads=attn.num_heads)
            pose_x = pose_x.view(b, q.size(1), d)
            pose_x = pose_x.flatten(2)

            x = x + pose_x * pose_scale
        elif self.pose_inject_method=='latent_overall':
            ip_key = self.k_proj(plucker_fea).view(b, -1, d)
            ip_value = self.v_proj(plucker_fea).view(b, -1, d)
            pose_x = flash_attention(q, ip_key, ip_value, num_heads=attn.num_heads)
            pose_x = pose_x.flatten(2)

            x = x + pose_x * pose_scale
        else:
            raise NotImplementedError

        return attn.o(x)



class CameraConditionModel(nn.Module):
    def __init__(self, wan_dit: WanModel, pose_in_dim: int, plucker_fea_dim: int, pose_inject_method: str, use_info: str ):
        super().__init__()

        self.pose_in_dim = pose_in_dim
        self.plucker_fea_dim = plucker_fea_dim
        self.pose_inject_method = pose_inject_method


        self.proj_model = nn.Identity()
        self.set_pose_processor(wan_dit)

        pose_encoder_kwargs = {
            "downscale_factor": 8,
            "channels": [320, 640, 1280, 1280, 2048],
            "nums_rb": 2,
            "cin": 384,
            "ksize": 1,
            "sk": True,
            "use_conv": False,
            "compression_factor": 1,
            "temporal_attention_nhead": 8,
            "attention_block_types": ["Temporal_Self"],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 16,
            "pose_inject_method": pose_inject_method,
            "context_dim": plucker_fea_dim,
        }

        from .pose_adaptor_ac3d import CameraPoseEncoder
        if use_info=="all":
            pose_encoder_kwargs['in_channels'] = 12 # plucker+rgb+depth+conf+mask
        elif use_info=="rgb_conf":
            pose_encoder_kwargs['in_channels'] = 4 # plucker
        elif use_info=="plucker":
            pose_encoder_kwargs['in_channels'] = 6 # plucker
        else:
            raise NotImplementedError

        self.pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)

    def init_proj(self,cross_attention_dim=5120):
        proj_model = PoseProjModel(
            pose_in_dim=self.pose_in_dim,
            cross_attention_dim=cross_attention_dim
        )
        return proj_model
    def set_pose_processor(self,wan_dit):
        attn_procs = {}

        for name in wan_dit.attn_processors.keys():
            attn_procs[name] = CrossAttentionAdapterProcessor(
            # attn_procs[name] = CrossAttentionAdaLNProcessor(
                context_dim=self.plucker_fea_dim,
                hidden_dim=wan_dit.dim,
                pose_inject_method=self.pose_inject_method,
            )
        wan_dit.set_attn_processor(attn_procs)

    def load_pose_processor(self, ip_ckpt: str, wan_dit):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"proj_model": {}, "pose_processor": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model."):
                        state_dict["proj_model"][key.replace("proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("pose_processor."):
                        state_dict["pose_processor"][key.replace("pose_processor.", "")] = f.get_tensor(key)
                    elif key.startswith("pose_encoder."):
                        state_dict["pose_encoder"][key.replace("pose_encoder.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")
        self.proj_model.load_state_dict(state_dict["proj_model"])
        self.pose_encoder.load_state_dict(state_dict["pose_encoder"])
        wan_dit.load_state_dict(state_dict["pose_processor"],strict=False)


    def get_proj_fea(self, pose_fea=None):

        return self.proj_model(pose_fea) if pose_fea is not None else None

    def get_pose_fea(self, plucker=None):
        return self.pose_encoder(plucker) if plucker is not None else None

    def split_pose_sequence(self, plucker_fea_length, num_frames=81):
        """
        Map the pose feature sequence to corresponding latent frame slices.

        Args:
            plucker_fea_length (int): The total length of the pose feature sequence
                                    (e.g., 173 in plucker_fea[1, 173, 768]).
            num_frames (int): The number of video frames in the training data (default: 81).

        Returns:
            list: A list of [start_idx, end_idx] pairs. Each pair represents the index range
                (within the pose feature sequence) corresponding to a latent frame.
        """
        # Average number of tokens per original video frame
        tokens_per_frame = plucker_fea_length / num_frames

        # Each latent frame covers 4 video frames, and we want the center
        tokens_per_latent_frame = tokens_per_frame * 4
        half_tokens = int(tokens_per_latent_frame / 2)

        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1
                pos_indices.append(center_token)

        # Build index ranges centered around each position
        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

        # Adjust the first range to avoid negative start index
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0]
        ]

        return pos_idx_ranges


    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """
        Split the input tensor into subsequences based on index ranges, and apply right-side zero-padding
        if the range exceeds the input boundaries.

        Args:
            input_tensor (Tensor): Input pose tensor of shape [1, L, 768].
            pos_idx_ranges (list): A list of index ranges, e.g. [[-7, 1], [1, 9], ..., [165, 173]].
            expand_length (int): Number of tokens to expand on both sides of each subsequence.

        Returns:
            sub_sequences (Tensor): A tensor of shape [1, F, L, 768], where L is the length after padding.
                                    Each element is a padded subsequence.
            k_lens (Tensor): A tensor of shape [F], representing the actual (unpadded) length of each subsequence.
                            Useful for ignoring padding tokens in attention masks.
        """
        pos_idx_ranges = [[idx[0]-expand_length,idx[1]+expand_length] for idx in pos_idx_ranges]
        sub_sequences = []
        seq_len = input_tensor.size(1)  # 173
        max_valid_idx = seq_len - 1    # 172
        k_lens_list = []
        for start, end in pos_idx_ranges:
            # Calculate the fill amount
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)

            # Calculate the start and end indices of the valid part
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            # Extract the valid part
            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start:valid_end+1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))  # 空张量

            # In the sequence dimension (the 1st dimension) perform padding
            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back+pad_front, 0, 0),
                mode='constant',
                value=0
            )
            k_lens_list.append(padded_subseq.size(-2)-pad_back-pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences,dim=1), torch.tensor(k_lens_list, dtype=torch.long)
