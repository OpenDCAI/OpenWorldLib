#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ezcolorlog import root_logger as logger

from .multimodal_encoder.builder import build_vision_tower_aux_list
from .multimodal_projector.builder import build_vision_projector

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from cambrian.utils import IS_XLA_AVAILABLE, inspect_tensor_sharding

try:
    import torch_xla.distributed.spmd as xs
except ImportError:
    xs = None


class CustomKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_embeds, newline_tokens, img_embeds, token_indices):
        ctx.full_input_shape = input_embeds.shape
        ctx.full_img_shape = img_embeds.shape
        ctx.dtype = input_embeds.dtype
        ctx.device = input_embeds.device
        
        sharded_input_embeds = xs.enable_manual_sharding(input_embeds, ("fsdp", None, None)).global_tensor
        sharded_newline_tokens = xs.enable_manual_sharding(newline_tokens, ("fsdp", None, None)).global_tensor
        sharded_img_embeds = xs.enable_manual_sharding(img_embeds, ("fsdp", None, None)).global_tensor
        sharded_token_indices = xs.enable_manual_sharding(token_indices, ("fsdp", None)).global_tensor

        sharded_embeds = torch.cat([sharded_input_embeds, sharded_newline_tokens, sharded_img_embeds], dim=1)
        sharded_embeds = torch.gather(sharded_embeds, 1, sharded_token_indices.unsqueeze(-1).expand(-1, -1, sharded_embeds.size(-1)))

        full_output_shape = (input_embeds.shape[0], token_indices.shape[1], input_embeds.shape[2])
        output_embeds = xs.disable_manual_sharding(sharded_embeds, ("fsdp", None, None), full_output_shape, mesh=xs.get_global_mesh()).global_tensor

        ctx.save_for_backward(token_indices)
        return output_embeds

    @staticmethod
    def backward(ctx, grad_output):

        bs = ctx.full_input_shape[0]
        input_seqlen = ctx.full_input_shape[1]
        img_seqlen = ctx.full_img_shape[1]
        dim = ctx.full_input_shape[2]
        token_indices, = ctx.saved_tensors
        
        sharded_token_indices = xs.enable_manual_sharding(token_indices, ("fsdp", None)).global_tensor
        sharded_grad_output = xs.enable_manual_sharding(grad_output, ("fsdp", None, None)).global_tensor
        lbs = sharded_grad_output.shape[0]

        sharded_embeds_grad = torch.zeros(
            lbs, input_seqlen + 1 + img_seqlen, dim,
            dtype=ctx.dtype, device=sharded_token_indices.device)
        sharded_embeds_grad = sharded_embeds_grad.scatter_add(1, sharded_token_indices.unsqueeze(-1).expand(-1, -1, dim), sharded_grad_output)

        full_grad_shape = (bs, input_seqlen + 1 + img_seqlen, dim)
        full_grad = xs.disable_manual_sharding(sharded_embeds_grad, ("fsdp", None, None), full_grad_shape, mesh=xs.get_global_mesh()).global_tensor

        return full_grad[:, :input_seqlen].clone(), full_grad[:, input_seqlen:input_seqlen+1].clone(), full_grad[:, input_seqlen+1:].clone(), None

def apply_custom_kernel(input_embeds, newline_tokens, img_embeds, token_indices):
    return CustomKernel.apply(input_embeds, newline_tokens, img_embeds, token_indices)

class CustomScatterKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tgt_embeds, src_embeds, indices):
        ctx.save_for_backward(indices)
        ctx.tgt_embeds_shape = tgt_embeds.shape
        ctx.src_embeds_shape = src_embeds.shape

        sharded_tgt_embeds = xs.enable_manual_sharding(tgt_embeds, ("fsdp", None, None)).global_tensor
        sharded_src_embeds = xs.enable_manual_sharding(src_embeds, ("fsdp", None, None)).global_tensor
        sharded_indices = xs.enable_manual_sharding(indices, ("fsdp", None)).global_tensor
        
        sharded_tgt_embeds.scatter_(1, sharded_indices.unsqueeze(-1).expand(-1, -1, sharded_tgt_embeds.size(-1)), sharded_src_embeds)
        
        tgt_embeds = xs.disable_manual_sharding(sharded_tgt_embeds, ("fsdp", None, None), tgt_embeds.shape, mesh=xs.get_global_mesh()).global_tensor
        return tgt_embeds

    @staticmethod
    def backward(ctx, grad):

        indices, = ctx.saved_tensors
        sharded_grad = xs.enable_manual_sharding(grad, ("fsdp", None, None)).global_tensor
        sharded_indices = xs.enable_manual_sharding(indices, ("fsdp", None)).global_tensor

        expanded_sharded_indices = sharded_indices.unsqueeze(-1).expand(-1, -1, sharded_grad.size(-1))
        sharded_tgt_embeds_grad = sharded_grad.clone().scatter_(1, expanded_sharded_indices, 0.)
        sharded_src_embeds_grad = sharded_grad.gather(1, expanded_sharded_indices)
        
        tgt_embeds_grad = xs.disable_manual_sharding(sharded_tgt_embeds_grad, ("fsdp", None, None), ctx.tgt_embeds_shape, mesh=xs.get_global_mesh()).global_tensor
        src_embeds_grad = xs.disable_manual_sharding(sharded_src_embeds_grad, ("fsdp", None, None), ctx.src_embeds_shape, mesh=xs.get_global_mesh()).global_tensor

        return tgt_embeds_grad, src_embeds_grad, None

def apply_custom_scatter_kernel(tgt_embeds, src_embeds, indices):
    return CustomScatterKernel.apply(tgt_embeds, src_embeds, indices)

class ShardedBLD2BCHW(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, hw):
        ctx.input_tensor_shape = input_tensor.shape

        sharded_input_tensor = xs.enable_manual_sharding(input_tensor, ("fsdp", None, None)).global_tensor
        sharded_input_tensor = sharded_input_tensor.unflatten(1, hw).permute(0, 3, 1, 2)
        
        output_shape = (input_tensor.size(0), input_tensor.size(2), hw[0], hw[1])
        output_tensor = xs.disable_manual_sharding(sharded_input_tensor, ("fsdp", None, None, None), output_shape, mesh=xs.get_global_mesh()).global_tensor

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        sharded_grad_output = xs.enable_manual_sharding(grad_output, ("fsdp", None, None, None)).global_tensor
        sharded_grad_output = sharded_grad_output.permute(0, 2, 3, 1).flatten(1, 2)

        full_grad_shape = ctx.input_tensor_shape
        full_grad = xs.disable_manual_sharding(sharded_grad_output, ("fsdp", None, None), full_grad_shape, mesh=xs.get_global_mesh()).global_tensor

        return full_grad, None

def apply_sharded_bld2bchw(input_tensor, hw):
    return ShardedBLD2BCHW.apply(input_tensor, hw)


class ShardedBCHW2BLD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, bs, nimgs_per_sample):
        ctx.shape = input_tensor.shape
        ctx.bs = bs
        ctx.nimgs_per_sample = nimgs_per_sample
        ctx.hw = (input_tensor.shape[2], input_tensor.shape[3])

        sharded_input_tensor = xs.enable_manual_sharding(input_tensor, ("fsdp", None, None, None)).global_tensor
        sharded_input_tensor = sharded_input_tensor.flatten(2, 3).permute(0, 2, 1) # BT C H W -> BT C HW -> BT L C
        sharded_input_tensor = sharded_input_tensor.unflatten(0, (sharded_input_tensor.size(0) // nimgs_per_sample, nimgs_per_sample)).flatten(1, 2) # BT L C -> B T L C -> B TL C
        
        output_shape = (bs, *sharded_input_tensor.shape[1:])
        output_tensor = xs.disable_manual_sharding(sharded_input_tensor, ("fsdp", None, None), output_shape, mesh=xs.get_global_mesh()).global_tensor

        return output_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        sharded_grad_output = xs.enable_manual_sharding(grad_output, ("fsdp", None, None)).global_tensor
        sharded_grad_output = sharded_grad_output.unflatten(1, (ctx.nimgs_per_sample, *ctx.hw)).permute(0, 1, 4, 2, 3) # B TL C -> B T H W C -> B T C H W
        sharded_grad_output = sharded_grad_output.flatten(0, 1) # B T C H W -> BT C H W

        full_grad_shape = ctx.shape
        full_grad = xs.disable_manual_sharding(sharded_grad_output, ("fsdp", None, None, None), full_grad_shape, mesh=xs.get_global_mesh()).global_tensor

        return full_grad, None, None

def apply_sharded_bchw2bld(input_tensor, bs, nimgs_per_sample):
    return ShardedBCHW2BLD.apply(input_tensor, bs, nimgs_per_sample)

class CambrianMetaModel:

    def __init__(self, config):
        super(CambrianMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):

            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if projector_type == 'sva':
                raise NotImplementedError
            else:
                self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True) # NOTE: why delay_load=True?
                config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in self.vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(config)
                self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )

        if hasattr(config, "nfp_head"):
            self.nfp_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
            )

    def get_vision_tower_aux_list(self):
        vision_tower_aux_list = getattr(self, 'vision_tower_aux_list', None)
        return vision_tower_aux_list

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        connector_only = model_args.connector_only

        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only = connector_only
        
        self.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower

        if self.get_vision_tower_aux_list() is None:
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            if model_args.unfreeze_mm_vision_tower:
                self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
            else:
                self.vision_tower_aux_list = vision_tower_aux_list
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.vision_hidden_size = vision_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.nfp_head = model_args.nfp_head
        self.config.nfp_mse_loss_weight = model_args.nfp_mse_loss_weight
        self.config.nfp_cosine_loss_weight = model_args.nfp_cosine_loss_weight

        if getattr(self, 'mm_projector', None) is None:

            if self.config.mm_projector_type == 'sva':
                raise NotImplementedError
            else:
                self.config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(self.config)
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if self.config.nfp_head:
            self.nfp_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
            )
            trunc_normal_(self.nfp_head[0].weight, std=0.02)
            nn.init.constant_(self.nfp_head[0].bias, 0.)
            trunc_normal_(self.nfp_head[2].weight, std=0.02)
            nn.init.constant_(self.nfp_head[2].bias, 0.)

        if pretrain_mm_mlp_adapter is not None:
            logger.info("Loading pretrained mm_projector weights from %s" % pretrain_mm_mlp_adapter)
            import gcsfs
            fs = gcsfs.GCSFileSystem(project='nyu-vision-lab')
            with fs.open(pretrain_mm_mlp_adapter, "rb") as f:
                mm_projector_weights = torch.load(f, map_location='cpu')

                # NOTE: if the weight is saved by spmd
                if "model"  in mm_projector_weights:
                    mm_projector_weights = mm_projector_weights["model"]

                for key in list(mm_projector_weights.keys()):
                    if key.startswith("_orig_module."):
                        mm_projector_weights[key.replace("_orig_module.", "")] = mm_projector_weights[key]
                        del mm_projector_weights[key]

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword+'.' in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict=True)

            if self.config.mm_projector_type == 'sva':
                raise NotImplementedError
            self.image_newline.data = mm_projector_weights['model.image_newline']

            logger.info(f"Load pretrained mm_projector weights from {pretrain_mm_mlp_adapter}.")


def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :]=0
            mask[:, -padding:, :]=0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding]=0
            mask[:, :, -padding:]=0
        return mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class CambrianMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def encode_images(self, image_aux_list):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        image_aux_features_list = []
        for image_aux, vision_tower_aux in zip(image_aux_list, vision_tower_aux_list):
            image_aux_features = vision_tower_aux(image_aux)
            image_aux_features_list.append(image_aux_features)
        return image_aux_features_list

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, newline_token_indices=None, si_token_indices=None, miv_token_indices=None,
        nfp_token_indices=None, nfp_loss_masks=None,
    ):
        if os.getenv("CAMBRIAN_LAUNCHER", "") == "TORCHXLA_SPMD":
            import torch_xla.distributed.spmd as xs

        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None, None

        image_aux_list = [images]
        bs, nimgs_per_sample = input_ids.size(0), images.size(0) // input_ids.size(0)

        si_token_len = self.get_model().config.si_token_len
        miv_token_len = self.get_model().config.miv_token_len

        if si_token_len > 0:
            si_side_len = int(si_token_len**0.5)
        else:
            si_side_len = None

        if miv_token_len > 0:
            miv_side_len = int(miv_token_len**0.5)
        else:
            miv_side_len = None

        assert si_side_len is not None or miv_side_len is not None

        if os.getenv("CAMBRIAN_LAUNCHER", "") == "TORCHXLA_SPMD":
            if self.get_model().config.unfreeze_mm_vision_tower:
                image_features = self.encode_images([images])[0]
            else:
                sharded_images = xs.enable_manual_sharding(images, ("fsdp", None, None, None)).global_tensor
                sharded_image_features = self.encode_images([sharded_images])[0]
                image_features = xs.disable_manual_sharding(sharded_image_features, ("fsdp", None, None), (images.size(0), sharded_image_features.size(1), sharded_image_features.size(2)), mesh=xs.get_global_mesh()).global_tensor
            image_aux_features_list = [image_features]
        elif os.getenv("CAMBRIAN_LAUNCHER", "") == "TORCHXLA_MP":
            image_aux_features_list = self.encode_images(image_aux_list)
        else:
            raise NotImplementedError

        assert len(image_aux_features_list) == 1
        image_features = image_aux_features_list[0]
        image_features = self.get_model().mm_projector(image_features).bfloat16()
        if os.getenv("CAMBRIAN_LAUNCHER", "") == "TORCHXLA_SPMD":
            xs.mark_sharding(image_features, xs.get_global_mesh(), ("fsdp", None, None))

        new_input_ids_padded_for_emb = torch.where(input_ids==IMAGE_TOKEN_INDEX, 0, input_ids)
        input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb).bfloat16()
        if not self.get_model().embed_tokens.weight.requires_grad:
            input_embeds = input_embeds.clone()

        feature_side_len = int(image_features.size(1) ** .5)

        newline_tokens = self.model.image_newline[None, None, :].expand(input_ids.size(0), -1, -1).clone()

        image_features = apply_sharded_bld2bchw(image_features, (feature_side_len, feature_side_len))

        nfp_tgt_embeds = None

        if si_side_len is not None:
            if self.get_model().config.image_aspect_ratio == "pad":
                nimgs_per_sample_si = 1
            elif self.get_model().config.image_aspect_ratio == "anyres":
                nimgs_per_sample_si = self.get_model().config.anyres_max_subimages + 1
            else:
                raise NotImplementedError
            si_features = image_features.unflatten(0, (bs, nimgs_per_sample))[:, :nimgs_per_sample_si].clone().flatten(0, 1)
            if si_side_len != feature_side_len:
                si_features = F.interpolate(si_features.clone(), size=(si_side_len, si_side_len), mode="bilinear", align_corners=False).type_as(si_features)
            else:
                si_features = si_features.clone()
            si_features = apply_sharded_bchw2bld(si_features, bs, nimgs_per_sample_si).type_as(input_embeds)
            if si_features.size(1) > input_ids.size(1):
                si_features = si_features[:, :input_ids.size(1)].clone()
            input_embeds = apply_custom_kernel(input_embeds, newline_tokens.type_as(input_embeds), si_features.type_as(input_embeds), si_token_indices)

        if miv_side_len is not None:
            if miv_side_len != feature_side_len:
                miv_features = F.interpolate(image_features.clone(), size=(miv_side_len, miv_side_len), mode="bilinear", align_corners=False).type_as(image_features)
            else:
                miv_features = image_features.clone()
            miv_features = apply_sharded_bchw2bld(miv_features, bs, nimgs_per_sample).type_as(input_embeds)
            if miv_features.size(1) > input_ids.size(1):
                miv_features = miv_features[:, :input_ids.size(1)].clone()
            input_embeds = apply_custom_kernel(input_embeds, newline_tokens.type_as(input_embeds), miv_features.type_as(input_embeds), miv_token_indices)

        if self.get_model().config.nfp_head:
            raw_miv_features = image_aux_features_list[0]
            raw_miv_features = apply_sharded_bld2bchw(raw_miv_features, (feature_side_len, feature_side_len))
            if miv_side_len != feature_side_len:
                raw_miv_features = F.interpolate(raw_miv_features.clone(), size=(miv_side_len, miv_side_len), mode="bilinear", align_corners=False).type_as(raw_miv_features)
            else:
                raw_miv_features = raw_miv_features.clone()
            raw_miv_features = apply_sharded_bchw2bld(raw_miv_features, bs, nimgs_per_sample).type_as(input_embeds)

            nfp_tgt_embeds = torch.zeros(input_embeds.size(0), input_embeds.size(1), raw_miv_features.size(2), device=raw_miv_features.device, dtype=raw_miv_features.dtype)
            pseudo_newline_tokens = torch.zeros(input_embeds.size(0), 1, raw_miv_features.size(2), device=raw_miv_features.device, dtype=raw_miv_features.dtype)
            nfp_tgt_embeds = apply_custom_kernel(nfp_tgt_embeds, pseudo_newline_tokens, raw_miv_features.type_as(nfp_tgt_embeds), nfp_token_indices)

        if IS_XLA_AVAILABLE:
            return None, position_ids, attention_mask, past_key_values, input_embeds, labels, nfp_tgt_embeds
        else:
            raise NotImplementedError

    def prepare_inputs_labels_for_multimodal_for_generation(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None,
    ):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if len(images) == 1: # single image, multiple images, single video
            images = images[0].flatten(0, 1).unsqueeze(0)
            image_aux_list = [images]
        else: # mixed videos and images
            images = torch.cat(images, dim=1)
            image_aux_list = [images]

        bs, nimgs_per_sample = image_aux_list[0].shape[:2]
        dtype = image_aux_list[0].dtype

        assert bs == 1 # ! NOTE: for inference, we only support batch size 1

        image_aux_list = [_.flatten(0, 1) for _ in image_aux_list]

        si_token_len = self.get_model().config.si_token_len
        miv_token_len = self.get_model().config.miv_token_len
        from transformers.trainer_pt_utils import logger
        logger.warning_once(f"si_token_len: {si_token_len}, miv_token_len: {miv_token_len}")

        if si_token_len > 0:
            si_side_len = int(si_token_len**0.5)
        else:
            si_side_len = None
        
        if miv_token_len <= 0:
            miv_side_len = None
        else:
            miv_side_len = int(miv_token_len**0.5)

        assert si_side_len is not None or miv_side_len is not None

        image_aux_features_list = self.encode_images(image_aux_list)
        assert len(image_aux_features_list) == 1 # NOTE: hard code to only use one encoder
        image_features = image_aux_features_list[0]

        image_features = image_features.to(self.get_model().mm_projector[0].weight.dtype)
        image_features = self.get_model().mm_projector(image_features).to(dtype)

        if len(set(len(_) for _ in image_sizes)) != 1:
            num_splits = []
            for size in image_sizes:
                if len(size) == 3:
                    num_splits.append(size[2])
                elif len(size) == 2:
                    num_splits.append(1)
                else:
                    raise NotImplementedError
            image_features = torch.split(image_features, num_splits, dim=0)

        # NOTE: gpu inference with bs 1
        if isinstance(image_features, tuple): # mixed video and images
            all_image_features = []
            for image_feature, image_size in zip(image_features, image_sizes):

                modality = ""
                if len(image_size) == 3: # video:
                    modality = "video"
                elif len(image_size) == 2: # image:
                    modality = "image"
                else:
                    raise NotImplementedError
                
                if modality == "video":
                    feature_side_len = int(image_feature.size(1) ** .5)
                    image_feature = image_feature.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2) # B, C, H, W
                    if miv_side_len != feature_side_len:
                        image_feature = F.interpolate(image_feature.float(), size=(miv_side_len, miv_side_len), mode="bilinear", align_corners=False).type_as(image_feature)
                    image_feature = image_feature.permute(0, 2, 3, 1) # bchw -> bhwc
                    original_size = image_size[:2]
                    image_feature = unpad_image(image_feature, original_size)
                    if not hasattr(self.get_model().config, "mm_use_im_newline_token") or self.get_model().config.mm_use_im_newline_token: # if not set, default to True
                        image_feature = torch.cat([image_feature, self.model.image_newline[None, None, None, :].expand(*image_feature.size()[:2], 1, -1)], dim=2)
                    image_feature = image_feature.flatten(1, 2) # T, HW, C
                    image_feature = image_feature.flatten(0, 1).unsqueeze(0) # 1, T*HW, C
                    all_image_features.append(image_feature)
                elif modality == "image":
                    feature_side_len = int(image_feature.size(1) ** .5)
                    image_feature = image_feature.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2) # B, C, H, W
                    if si_side_len != feature_side_len:
                        image_feature = F.interpolate(image_feature.float(), size=(si_side_len, si_side_len), mode="bilinear", align_corners=False).type_as(image_feature)
                    image_feature = image_feature.permute(0, 2, 3, 1) # bchw -> bhwc
                    original_size = image_size[:2]
                    image_feature = unpad_image(image_feature, original_size)
                    if not hasattr(self.get_model().config, "mm_use_im_newline_token") or self.get_model().config.mm_use_im_newline_token: # if not set, default to True
                        image_feature = torch.cat([image_feature, self.model.image_newline[None, None, None, :].expand(*image_feature.size()[:2], 1, -1)], dim=2)
                    image_feature = image_feature.flatten(1, 2)
                    all_image_features.append(image_feature)
                else:
                    raise NotImplementedError
        elif len(image_sizes[0]) == 2:
            # image
            feature_side_len = int(image_features.size(1) ** .5)
            image_features = image_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2) # B, C, H, W
            if si_side_len != feature_side_len:
                image_features = F.interpolate(image_features.float(), size=(si_side_len, si_side_len), mode="bilinear", align_corners=False).type_as(image_features)
            
            image_features = image_features.permute(0, 2, 3, 1) # bchw -> bhwc
            # ! NOTE: remove padding features!!!
            if len(image_sizes) == 1: # single images:
                original_size = image_sizes[0]
                image_features = unpad_image(image_features, original_size)
                if not hasattr(self.get_model().config, "mm_use_im_newline_token") or self.get_model().config.mm_use_im_newline_token: # if not set, default to True
                    image_features = torch.cat([image_features, self.model.image_newline[None, None, None, :].expand(*image_features.size()[:2], 1, -1)], dim=2)
                image_features = image_features.flatten(1, 2)
            else:
                # multiple images
                all_image_features = []
                for _ in range(len(image_sizes)):
                # for org_size, per_img_feat in zip(image_sizes, image_features):
                    org_size = image_sizes[_]
                    per_img_feat = image_features[_].unsqueeze(0)
                    per_img_feat = unpad_image(per_img_feat, org_size)
                    if not hasattr(self.get_model().config, "mm_use_im_newline_token") or self.get_model().config.mm_use_im_newline_token: # if not set, default to True
                        per_img_feat = torch.cat([per_img_feat, self.model.image_newline[None, None, None, :].expand(*per_img_feat.size()[:2], 1, -1)], dim=2)
                    all_image_features.append(per_img_feat.flatten(1, 2))
        elif len(image_sizes[0]) == 3:
            # video
            feature_side_len = int(image_features.size(1) ** .5)
            image_features = image_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2) # B, C, H, W
            if miv_side_len != feature_side_len:
                image_features = F.interpolate(image_features.float(), size=(miv_side_len, miv_side_len), mode="bilinear", align_corners=False).type_as(image_features)
            
            image_features = image_features.permute(0, 2, 3, 1) # bchw -> bhwc
            # ! NOTE: remove padding features!!!
            original_size = image_sizes[0][:2]
            image_features = unpad_image(image_features, original_size) # ! NOTE: comment this out if you are evaluating llava-onevision trained model cuz they do not pad the image
            image_features = torch.cat([image_features, self.model.image_newline[None, None, None, :].expand(*image_features.size()[:2], 1, -1)], dim=2)
            image_features = image_features.flatten(1, 2)
            image_features = image_features.unflatten(0, (bs, nimgs_per_sample)).flatten(1, 2)
        elif len(image_sizes[0]) == 4:
            # image and anyres
            feature_side_len = int(image_features.size(1) ** .5)
            image_features = image_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2) # B, C, H, W
            if si_side_len != feature_side_len:
                image_features = F.interpolate(image_features.float(), size=(si_side_len, si_side_len), mode="bilinear", align_corners=False).type_as(image_features)

            image_features = image_features.permute(0, 2, 3, 1) # bchw -> bhwc
            snapshot_features = image_features[0].unsqueeze(0)

            anyres_features = image_features[1:].unflatten(0, image_sizes[0][2:])
            anyres_features = anyres_features.permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1).unsqueeze(0)
            
            # ! NOTE: remove padding features!!!
            original_size = image_sizes[0][:2]
            snapshot_features = unpad_image(snapshot_features, original_size)
            anyres_features = unpad_image(anyres_features, original_size)

            snapshot_features = torch.cat([snapshot_features, self.model.image_newline[None, None, None, :].expand(*snapshot_features.size()[:2], 1, -1)], dim=2)
            snapshot_features = snapshot_features.flatten(1, 2)
            anyres_features = torch.cat([anyres_features, self.model.image_newline[None, None, None, :].expand(*anyres_features.size()[:2], 1, -1)], dim=2)
            anyres_features = anyres_features.flatten(1, 2)
            image_features = torch.cat([snapshot_features, anyres_features], dim=1)
            
        else:
            raise NotImplementedError

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        new_input_ids_padded_for_emb = torch.where(input_ids==IMAGE_TOKEN_INDEX, 0, input_ids)
        input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb)
        assert input_ids.size(0)
        image_pos = input_ids[0].eq(IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
        if len(image_pos) == 1: # single image or video
            image_pos = image_pos[0]
            new_input_embeds = torch.cat([
                input_embeds[:, :image_pos],
                image_features,
                input_embeds[:, image_pos+1:],
            ], dim=1).type_as(input_embeds)
        else: # multiple images or mixed video and images
            all_input_embeds = []
            assert len(all_image_features) == len(image_pos)
            pre_pos = 0
            for _ in range(len(image_pos)):
                _pos = image_pos[_]
                all_input_embeds.append(input_embeds[:, pre_pos:_pos])
                all_input_embeds.append(all_image_features[_])
                pre_pos = _pos + 1 # ! NOTE: replace -200 token with image token
            all_input_embeds.append(input_embeds[:, pre_pos:])
            new_input_embeds = torch.cat(all_input_embeds, dim=1).type_as(input_embeds)
        attention_mask = torch.ones(new_input_embeds.size(0), new_input_embeds.size(1)).to(new_input_embeds.device).bool()
        return None, None, attention_mask, past_key_values, new_input_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                import gcsfs
                fs = gcsfs.GCSFileSystem(project='nyu-vision-lab')
                with fs.open(model_args.pretrain_mm_mlp_adapter, "rb") as f:
                    mm_projector_weights = torch.load(f, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
