#    Copyright 2024 Hao Zhang
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
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

from ..cambrian_arch import CambrianMetaModel, CambrianMetaForCausalLM

from cambrian.utils import IS_XLA_AVAILABLE


class CambrianQwenConfig(Qwen2Config):
    model_type = "cambrian_qwen"


class CambrianQwenModel(CambrianMetaModel, Qwen2Model):
    config_class = CambrianQwenConfig

    def __init__(self, config: Qwen2Config):
        # config.num_hidden_layers = 1 # NOTE: for debug only!!!
        if IS_XLA_AVAILABLE:
            config._attn_implementation = "eager"
        super(CambrianQwenModel, self).__init__(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            if os.getenv("CAMBRIAN_LAUNCHER", "") == "TORCHXLA_SPMD":
                # ! NOTE: this is a hack to speed up the training
                # ! NOTE: we use torch_xla's flash attention which does not require mask
                attention_mask = None
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.config.sliding_window,
                )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            ############################################################################################
            # Cambrian: For SVA
            ############################################################################################

            if not self.config.connector_only:

                cross_layers_start_idx = self.config.start_of_vision_sampler_layers
                cross_index_step = self.config.stride_of_vision_sampler_layers
                cross_layers_start_idx_list = [cross_layers_start_idx+cross_index*cross_index_step for cross_index in range(len(self.vision_sampler_layers))]

                if vision_tower_aux_feature_list is not None and i in cross_layers_start_idx_list:
                    latent_query_start_idx = self.config.image_position

                    if IS_XLA_AVAILABLE:
                        image_token_len_per_side = int(self.config.image_token_len**0.5)
                        latent_query_newline_num = self.config.image_token_len + image_token_len_per_side
                        latent_query_num = self.config.image_token_len
                        if self.config.video_max_frames > 0:
                            latent_query_newline_num *= self.config.video_max_frames
                        latent_query_with_newline = hidden_states[:, latent_query_start_idx:latent_query_start_idx+latent_query_newline_num, :].clone()
                        if self.config.video_max_frames > 0:
                            latent_query_with_newline = latent_query_with_newline.reshape(latent_query_with_newline.size(0), self.config.video_max_frames, -1, latent_query_with_newline.size(-1)).flatten(0, 1)
                        bs = latent_query_with_newline.shape[0]
                        latent_query_with_newline = latent_query_with_newline.view(bs, image_token_len_per_side, image_token_len_per_side+1, -1)
                        latent_query = latent_query_with_newline[:, :, :-1, :]
                        newline_embd = latent_query_with_newline[:, :, -1:, :]
                        vision_tower_aux_feature_list = [vision_tower_aux_feature.to(latent_query.dtype) for vision_tower_aux_feature in vision_tower_aux_feature_list]
                        bs = latent_query.shape[0]
                        latent_query = latent_query.view(bs*latent_query_num, 1, -1)
                        if self.gradient_checkpointing and self.training:
                            latent_query = self._gradient_checkpointing_func(
                            self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step].__call__,
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                            )
                        else:
                            latent_query = self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                            )
                        # latent_query = latent_query.view(bs, self.latent_query_num, -1)
                        latent_query = latent_query.view(bs, image_token_len_per_side, image_token_len_per_side, -1)
                        latent_query_with_newline = torch.cat([latent_query, newline_embd], 2).flatten(1,2)
                        if self.config.video_max_frames > 0: # video is enabled
                            latent_query_with_newline = latent_query_with_newline.reshape(hidden_states.size(0), self.config.video_max_frames, *latent_query_with_newline.size()[1:]).flatten(1, 2)
                        hidden_states[:, latent_query_start_idx:latent_query_start_idx+latent_query_newline_num] = latent_query_with_newline[:, :, :]
                    else:
                        bs = len(final_vision_feature_size)
                        latent_query_num_list = []
                        newline_embd_list = []
                        latent_query_list = []
                        for batch_i in range(bs):
                            cur_h, cur_w = final_vision_feature_size[batch_i]
                    
                            cur_latent_query_num = cur_h*cur_w
                            cur_latent_query_newline_num = cur_h * (cur_w+1)
                            cur_latent_query_with_newline = hidden_states[batch_i:batch_i+1, latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num, :].clone()

                            cur_latent_query_with_newline = cur_latent_query_with_newline.view(1, cur_h, cur_w+1, -1)
                            cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
                            cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]

                            latent_query_num_list.append(cur_latent_query_num)
                            latent_query_list.append(cur_latent_query.contiguous().view(cur_latent_query_num, 1, -1))
                            newline_embd_list.append(cur_newline_embd)

                        latent_query = torch.cat(latent_query_list, 0)
                        if self.gradient_checkpointing and self.training:
                            latent_query = self._gradient_checkpointing_func(
                            self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step].__call__,
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                            )
                        else:
                            latent_query = self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                            )

                        latent_query = torch.split(latent_query, latent_query_num_list, 0)
                        for batch_i in range(bs):
                            cur_h, cur_w = final_vision_feature_size[batch_i]
                            cur_latent_query = latent_query[batch_i]
                            cur_newline_embd = newline_embd_list[batch_i]
                            cur_latent_query_newline_num = cur_h * (cur_w+1)
                            cur_latent_query = cur_latent_query.view(1, cur_h, cur_w, -1)
                            cur_latent_query_with_newline = torch.cat([cur_latent_query, cur_newline_embd], 2).flatten(1,2)
                            hidden_states[batch_i:batch_i+1, latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num] = cur_latent_query_with_newline[:, :, :]
            ############################################################################################

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CambrianQwenForCausalLM(Qwen2ForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        config.rope_scaling = None

        self.model = CambrianQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def nfp_loss(self, nfp_outputs, nfp_targets, nfp_loss_masks):

        nfp_outputs = nfp_outputs.view(-1, nfp_outputs.size(-1))
        nfp_targets = nfp_targets.view(-1, nfp_targets.size(-1))
        nfp_loss_masks = nfp_loss_masks.view(-1)

        mse_loss = torch.nn.functional.mse_loss(nfp_outputs, nfp_targets, reduction='none').mean(-1)
        mse_loss = mse_loss * nfp_loss_masks
        mse_loss = mse_loss.sum() / (nfp_loss_masks.sum() + 1e-12)

        cosine_loss = torch.nn.functional.cosine_embedding_loss(nfp_outputs, nfp_targets, torch.ones(nfp_outputs.size(0)).to(nfp_outputs.device), reduction='none')
        cosine_loss = cosine_loss * nfp_loss_masks
        cosine_loss = cosine_loss.sum() / (nfp_loss_masks.sum() + 1e-12)

        return mse_loss, cosine_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        newline_token_indices: Optional[torch.Tensor] = None,
        si_token_indices: Optional[torch.Tensor] = None,
        miv_token_indices: Optional[torch.Tensor] = None,
        nfp_token_indices: Optional[torch.Tensor] = None,
        nfp_loss_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                nfp_tgt_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                newline_token_indices=newline_token_indices,
                si_token_indices=si_token_indices,
                miv_token_indices=miv_token_indices,
                nfp_token_indices=nfp_token_indices,
                nfp_loss_masks=nfp_loss_masks,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            #self.model.gradient_checkpointing = False
                
            from torch_xla.utils.checkpoint import checkpoint
            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # training
        if IS_XLA_AVAILABLE:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else: # inference
            if hasattr(self, "vision_tower_aux_feature_list"):
                raise NotImplementedError("vision_tower_aux_feature_list should not be set in inference mode")
            else:
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.config.nfp_head:
            nfp_outputs = self.model.nfp_head(hidden_states)
            nfp_mse_loss, nfp_cosine_loss = self.nfp_loss(nfp_outputs, nfp_tgt_embeds, nfp_loss_masks)
            nfp_mse_loss = nfp_mse_loss * self.config.nfp_mse_loss_weight
            nfp_cosine_loss = nfp_cosine_loss * self.config.nfp_cosine_loss_weight

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.config.nfp_head:
            total_loss = (
                loss + nfp_mse_loss + nfp_cosine_loss,
                loss,
                nfp_mse_loss,
                nfp_cosine_loss,
            )
        else:
            total_loss = loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal_for_generation(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("cambrian_qwen", CambrianQwenConfig)
AutoModelForCausalLM.register(CambrianQwenConfig, CambrianQwenForCausalLM)
