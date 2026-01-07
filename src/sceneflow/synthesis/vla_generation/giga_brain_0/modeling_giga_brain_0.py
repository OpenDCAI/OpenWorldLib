import math

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn

from .paligemma2_with_expert import PaliGemma2WithExpertModel
from .paligemma_with_expert import PaliGemmaWithExpertModel


class GigaBrain0Policy(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        vlm_type: str = 'paligemma2',
        vlm_hidden_size: int = 2304,
        n_action_steps: int = 50,
        num_steps: int = 10,
        use_cache: bool = True,
        vision_in_channels: int = 3,
        enable_knowledge_insulation: bool = False,
        enable_next_token_prediction: bool = True,
        enable_learnable_traj_token: bool = False,
        num_traj_tokens: int = 10,
        max_traj_dim: int = 4,
        traj_hidden_dim: int = 256,
        num_embodiments: int = 1,
        **kwargs,
    ):
        super().__init__()

        # Store the parameters
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.proj_width = proj_width
        self.n_action_steps = n_action_steps
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.vision_in_channels = vision_in_channels

        self.enable_knowledge_insulation = enable_knowledge_insulation
        self.vlm_type = vlm_type
        self.vlm_hidden_size = vlm_hidden_size

        if self.vlm_type == 'paligemma2':
            self.paligemma_with_expert = PaliGemma2WithExpertModel(
                vision_in_channels=vision_in_channels,
                enable_next_token_prediction=enable_next_token_prediction,
            )
        elif self.vlm_type == 'paligemma':
            self.paligemma_with_expert = PaliGemmaWithExpertModel(
                vision_in_channels=vision_in_channels,
                enable_next_token_prediction=enable_next_token_prediction,
                pi05_enabled=True,
            )

        # Projections are float32
        self.action_in_proj = EmbodimentSpecificLinear(self.max_action_dim, self.proj_width, num_categories=num_embodiments, dtype=torch.float32)
        self.action_out_proj = EmbodimentSpecificLinear(self.proj_width, self.max_action_dim, num_categories=num_embodiments, dtype=torch.float32)
        self.time_mlp_in = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
        self.time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)

        self.enable_next_token_prediction = enable_next_token_prediction
        self.enable_learnable_traj_token = enable_learnable_traj_token
        if self.enable_learnable_traj_token:
            self.num_traj_tokens = num_traj_tokens
            self.max_traj_dim = max_traj_dim
            self.traj_hidden_dim = traj_hidden_dim
            self.traj_token = nn.Parameter(torch.randn(num_traj_tokens, self.vlm_hidden_size), requires_grad=True)
            self.traj_decoder = nn.GRU(input_size=self.vlm_hidden_size, hidden_size=self.traj_hidden_dim, batch_first=True)
            self.traj_out_proj = nn.Linear(self.traj_hidden_dim, self.max_traj_dim, dtype=torch.float32)

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Runs the full model forward pass.

        This method processes image and language inputs, generates key-value caches,
        and then uses them to denoise an action sequence `x_t` at a given `timestep`.
        It also computes language logits and, if enabled, trajectory predictions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            x_t: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action steps.

        Returns:
            A dictionary containing:
            - 'v_t': The predicted velocity (denoised action).
            - 'lang_logits': Logits for the language model output.
            - 'traj_pred' (optional): The predicted trajectory if enabled.
        """
        lang_length = lang_masks.shape[-1]

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, traj_token_start_idx = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_att_masks, fast_action_indicator
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        # Construct 2d attention mask and position ids
        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        if fast_action_indicator is not None:
            suffix_fast_indicator = fast_action_indicator.new_zeros(batch_size, suffix_len)
            full_fast_indicator = torch.cat([fast_action_indicator, suffix_fast_indicator], dim=1)
            start = full_att_2d.shape[1] - suffix_len
            full_att_2d[:, start:, :] &= ~full_fast_indicator[:, None, :]

        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if fast_action_indicator is not None:
            suffix_position_ids = suffix_position_ids - fast_action_indicator.sum(dim=-1, keepdim=True)

        # VLM forward pass
        if self.enable_knowledge_insulation:
            # Two-stage forward: prefix first, then suffix with cached KV
            prefix_att_2d_masks = full_att_2d[:, :prefix_len, :prefix_len]

            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )

            # Detach KV cache for knowledge insulation
            for key in past_key_values.keys():
                past_key_values[key]['key_states'] = past_key_values[key]['key_states'].detach()
                past_key_values[key]['value_states'] = past_key_values[key]['value_states'].detach()

            suffix_att_2d_masks = full_att_2d[:, prefix_len:, :]

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=suffix_att_2d_masks,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        else:
            full_position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        output = {}

        # Compute denoised output
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out, emb_ids=emb_ids)
        output['v_t'] = v_t

        # Compute language logits
        if self.enable_next_token_prediction:
            lang_out = prefix_out[:, -lang_length:, :]
            lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
            lang_logits = self.paligemma_with_expert.language_out_proj(lang_out)
            output['lang_logits'] = lang_logits

        # Compute trajectory predictions
        if self.enable_learnable_traj_token:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)
            output['traj_pred'] = traj_pred

        return output

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, int | None]:
        """Embeds prefix inputs (images, optional trajectory tokens, language)
        into a single sequence.

        This method uses a vision model (SigLIP) for images and an embedding layer
        for language tokens, combining them into a unified embedding sequence for
        the transformer.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action steps.

        Returns:
            A tuple containing:
            - embs: The combined prefix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - fast_action_indicator: Updated fast action indicator with prefix padding.
            - traj_token_start_idx: The starting index of trajectory tokens, if they exist.
        """
        num_images = len(images)

        # Stack images and masks for batch processing
        images_stacked = torch.stack(images, dim=0)  # (num_images, bsize, ...)
        img_masks_stacked = torch.stack(img_masks, dim=0)  # (num_images, bsize)

        # Batch embed all images at once
        # Reshape to (num_images * bsize, ...)
        orig_shape = images_stacked.shape
        images_flat = images_stacked.reshape(-1, *orig_shape[2:])
        img_embs_flat = self.paligemma_with_expert.embed_image(images_flat)

        # Reshape back to (num_images, bsize, num_img_embs, emb_dim)
        bsize = orig_shape[1]
        img_embs = img_embs_flat.reshape(num_images, bsize, *img_embs_flat.shape[1:])

        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        num_img_embs = img_embs.shape[2]

        # Expand masks: (num_images, bsize) -> (num_images, bsize, num_img_embs)
        img_masks_expanded = img_masks_stacked[:, :, None].expand(num_images, bsize, num_img_embs)

        # Reshape to (bsize, num_images * num_img_embs, emb_dim)
        img_embs_concat = img_embs.transpose(0, 1).reshape(bsize, num_images * num_img_embs, img_emb_dim)
        img_masks_concat = img_masks_expanded.transpose(0, 1).reshape(bsize, num_images * num_img_embs)

        # Process language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_emb = lang_emb.to(dtype=img_embs_concat.dtype)

        num_lang_embs = lang_emb.shape[1]
        num_img_embs_total = num_images * num_img_embs
        num_traj_embs = self.num_traj_tokens if self.enable_learnable_traj_token else 0
        total_seq_len = num_img_embs_total + num_traj_embs + num_lang_embs

        # Pre-allocate final tensors
        embs = torch.empty(bsize, total_seq_len, img_emb_dim, dtype=img_embs_concat.dtype, device=img_embs_concat.device)
        pad_masks = torch.empty(bsize, total_seq_len, dtype=torch.bool, device=img_embs_concat.device)

        # Fill pre-allocated tensors
        embs[:, :num_img_embs_total] = img_embs_concat
        pad_masks[:, :num_img_embs_total] = img_masks_concat

        num_prefix_embs = num_img_embs_total
        traj_token_start_idx = None
        if self.enable_learnable_traj_token:
            traj_emb = self.traj_token[None].repeat(bsize, 1, 1)
            embs[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = traj_emb
            pad_masks[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = torch.ones(
                bsize, self.num_traj_tokens, dtype=torch.bool, device=traj_emb.device
            )
            traj_token_start_idx = num_prefix_embs
            num_prefix_embs += num_traj_embs

        embs[:, num_img_embs_total + num_traj_embs :] = lang_emb
        pad_masks[:, num_img_embs_total + num_traj_embs :] = lang_masks

        # Create attention masks (all zeros for full attention between image and language)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, total_seq_len).clone()
        if lang_att_masks is not None:
            att_masks[:, -num_lang_embs:] = lang_att_masks
        if fast_action_indicator is not None:
            fast_action_indicator = F.pad(fast_action_indicator, (num_prefix_embs, 0))

        return embs, pad_masks, att_masks, fast_action_indicator, traj_token_start_idx

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, emb_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embeds suffix inputs (noisy actions and timestep) for the
        transformer.

        Args:
            noisy_actions: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.

        Returns:
            A tuple containing:
            - embs: The combined suffix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - adarms_cond: The conditioning vector for AdaRMSNorm.
        """
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions, emb_ids=emb_ids)
        bsize, action_dim = action_emb.shape[:2]
        dtype = action_emb.dtype
        device = action_emb.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_emb)

        action_time_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image and language do not attend to action tokens
        att_masks += [1] + ([0] * (self.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def sample_noise(self, shape: tuple, device: torch.device | str) -> Tensor:
        """Samples noise from a standard normal distribution.

        Args:
            shape: The desired shape of the noise tensor.
            device: The device to create the tensor on.

        Returns:
            A tensor of the given shape filled with noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        enable_2d_traj_output: bool = False,
        noise: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Performs a full inference pass to sample actions.

        This involves creating a key-value cache from image and language inputs,
        then iteratively denoising an initial noise tensor to produce the final actions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            enable_2d_traj_output: If True, also returns trajectory predictions.
            noise: Optional initial noise tensor. If None, it will be sampled.

        Returns:
            - The sampled action tensor.
            - If `enable_2d_traj_output` is True, a tuple of the action tensor
              and the trajectory prediction tensor.
        """
        bsize = images[0].shape[0]
        device = images[0].device

        if noise is None:
            actions_shape = (bsize, self.n_action_steps, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _, traj_token_start_idx = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        if enable_2d_traj_output:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)

        x_t = noise
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(bsize),
                emb_ids,
            )
            x_t += dt * v_t

        if enable_2d_traj_output:
            return x_t, traj_pred

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: dict,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
    ) -> Tensor:
        """Applies one denoising step to `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out, emb_ids=emb_ids)
        return v_t

    @torch.no_grad()
    def init_lang_generation(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Prefill stage: build KV cache from images + existing language prefix and
        return logits for the next token.

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: A dictionary containing the state for autoregressive generation:
                - 'past_key_values': The key-value cache from the transformer.
                - 'position_cursor': The current position in the sequence for position embeddings.
                - 'batch_size': The batch size.
                - 'device': The device of the tensors.
                - 'prefix_pad_masks': The padding mask for the prefix.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # embed prefix (images + lang)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # forward to build cache and get language hidden states
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        # Take the hidden state of the last valid language token per sample as
        # the condition for the next-token distribution
        lang_length = lang_masks.shape[-1]
        lang_region = prefix_out[:, -lang_length:, :]
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = lang_region[batch_indices, -1, :].unsqueeze(1)
        last_hidden = last_hidden.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(last_hidden)[:, 0, :]

        # Next-token position (0-indexed) equals the number of consumed tokens so far
        position_cursor = torch.sum(prefix_pad_masks, dim=-1).to(dtype=torch.long)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': position_cursor,
            'batch_size': bsize,
            'device': device,
            'prefix_pad_masks': prefix_pad_masks[:, None, :],
        }
        return next_logits, state

    @torch.no_grad()
    def next_lang_logits(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Decode step: given the previously generated token, return logits for
        the next token and update the KV cache/state.

        Args:
            state: A dictionary containing the state from a previous generation step.
                   See `init_lang_generation` for details on its contents.
            input_token: The token generated in the previous step, shape (batch_size, 1).

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: The updated state dictionary for the next generation step.
        """
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']
        bsize = state['batch_size']
        device = state['device']
        prefix_pad_masks = state['prefix_pad_masks']

        # Prepare single-step language token embedding
        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        token_emb = self.paligemma_with_expert.embed_language_tokens(input_token)
        # Match scaling used in embed_prefix to keep prefill/decoding consistent
        lang_emb_dim = token_emb.shape[-1]
        token_emb = token_emb * math.sqrt(lang_emb_dim)

        # Build attention mask: one-step query attends to all history and itself
        att_2d_masks = torch.cat([prefix_pad_masks, torch.ones((bsize, 1, 1), dtype=torch.bool, device=device)], dim=2)

        # Position id is the current cursor (0-indexed)
        position_ids = position_cursor.view(bsize, 1).to(device)

        (lang_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[token_emb, None],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, None],
            auto_regression_inference_mode=True,
        )

        # Single step only, shape: (b, 1, d)
        lang_out = lang_out[:, -1:, :]
        lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(lang_out)[:, 0, :]

        # Advance the cursor; KV cache is persistently appended inside the model
        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        state['prefix_pad_masks'] = att_2d_masks
        return next_logits, state


class EmbodimentSpecificLinear(nn.Module):
    """A linear layer with weights and biases specific to an embodiment
    category."""

    def __init__(self, input_dim: int, output_dim: int, num_categories: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_categories = num_categories
        self.weight = nn.Parameter(torch.empty(num_categories, input_dim, output_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(num_categories, output_dim, dtype=dtype))

        # Initialize using Linear's default initialization: U(-k, k) where k = 1/sqrt(in_features)
        k = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight, -k, k)
        nn.init.uniform_(self.bias, -k, k)

    def forward(self, x: Tensor, emb_ids: Tensor) -> Tensor:
        """Applies the embodiment-specific linear transformation.

        Args:
            x: The input tensor.
            emb_ids: A tensor of embodiment IDs, used to select the appropriate
                     weights and biases.

        Returns:
            The transformed tensor.
        """
        # Use one-hot to aggregate per-category weights/biases to avoid dynamic indexing (compile-friendly)
        if emb_ids.dtype != torch.long:
            emb_ids = emb_ids.to(torch.long)
        emb_ids = emb_ids.reshape(-1).to(device=self.weight.device)

        one_hot = F.one_hot(emb_ids, num_classes=self.num_categories).to(dtype=self.weight.dtype)

        selected_weight = torch.einsum('bc,cij->bij', one_hot, self.weight)
        selected_bias = torch.einsum('bc,cj->bj', one_hot, self.bias)

        return torch.bmm(x, selected_weight) + selected_bias.unsqueeze(1)

    def extra_repr(self):
        return f'EmbodimentSpecificLinear(num_categories={self.num_categories}, input_dim={self.weight.shape[1]}, output_dim={self.weight.shape[2]})'


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """MPS is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == 'mps' and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float, device='cpu') -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar
    positions."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Creates a 2D attention mask from 1D padding and attention-type masks.

    This function is copied from big_vision.

    The `att_masks` allow for different attention patterns:
    - `[[1, 1, 1, 1, 1, 1]]`: Purely causal attention.
    - `[[0, 0, 0, 1, 1, 1]]`: Prefix-LM attention. The first 3 tokens attend
      to each other, and the last 3 tokens are causal.
    - `[[1, 0, 1, 0, 1, 0]]`: Causal attention between blocks. Tokens can
      attend to previous blocks and tokens within the same block.

    Args:
        pad_masks: bool[B, N] indicating valid (true) vs. padding (false) tokens.
        att_masks: int[B, N] defining attention type. A `1` at a position
                   indicates the start of a new causal block.

    Returns:
        A 2D boolean attention mask of shape (B, N, N).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks