from typing import Any

import torch

from ...operators.giga_brain_0_operator import GigaBrain0Operator
from ...synthesis.vla_generation.giga_brain_0.giga_brain_0_synthesis import GigaBrain0Synthesis
from ...synthesis.vla_generation.giga_brain_0.modeling_giga_brain_0 import GigaBrain0Policy


class GigaBrain0Pipeline:
    """Pipeline wrapper for GigaBrain0 policy inference using a dedicated operator."""

    def __init__(
        self,
        policy: GigaBrain0Policy,
        operator: GigaBrain0Operator,
        embodiment_id: int,
        original_action_dim: int,
        synthesis: GigaBrain0Synthesis | None = None,
        device: str | torch.device | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesis = synthesis or GigaBrain0Synthesis(policy=policy)
        self.synthesis.to(self.device)
        self.policy = self.synthesis.policy  # keep reference for AR utilities
        self.policy.eval()
        self.operator = operator.to(self.device)
        self.operator.set_action_dim(self.policy.max_action_dim)
        self.embodiment_id = embodiment_id
        self.original_action_dim = original_action_dim
        self.resize_imgs_with_padding = (224, 224)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        embodiment_id: int,
        state_norm_stats: dict,
        action_norm_stats: dict,
        delta_mask: list[bool],
        original_action_dim: int,
        discrete_state_input: bool = True,
        autoregressive_inference_mode: bool = False,
        depth_img_prefix_name: str | None = None,
        device: str | torch.device | None = None,
        present_img_keys: list[str] | None = None,
        **policy_kwargs: Any,
    ) -> 'GigaBrain0Pipeline':
        policy = GigaBrain0Policy.from_pretrained(model_path, **policy_kwargs)
        operator = GigaBrain0Operator(
            embodiment_id=embodiment_id,
            state_norm_stats=state_norm_stats,
            action_norm_stats=action_norm_stats,
            delta_mask=delta_mask,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            resize_imgs_with_padding=(224, 224),
            enable_depth_img=policy.vision_in_channels == 4,
            depth_img_prefix_name=depth_img_prefix_name,
            discrete_state_input=discrete_state_input,
            autoregressive_inference_mode=autoregressive_inference_mode,
            text_max_length=200,
            present_img_keys=present_img_keys,
        )
        synthesis = GigaBrain0Synthesis(policy=policy, device=device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        return cls(policy=policy, operator=operator, embodiment_id=embodiment_id, original_action_dim=original_action_dim, synthesis=synthesis, device=device)

    def to(self, device: str | torch.device):
        self.device = device
        self.synthesis.to(device)
        self.policy = self.synthesis.policy
        self.operator.to(device)
        return self

    def quantize(self) -> None:
        """Apply dynamic float8 quantization to the Paligemma blocks only."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        layers = self.policy.paligemma_with_expert.layers
        for i in range(len(layers)):
            quantize_(layers[i].mlps[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.q_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.k_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.v_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.o_proj[0], Float8DynamicActivationFloat8WeightConfig())

    def compile(self, **kwargs: Any) -> None:
        """Compile the `sample_actions` method using `torch.compile` for improved runtime speed."""
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)

    def _prepare_action_inputs(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        task: str,
        pad_state: bool = True,
        add_batch_dim: bool = True,
    ):
        ori_device = state.device
        images = {k: v.to(self.device) for k, v in images.items()}
        state = state.to(self.device)

        images, img_masks, image_transform_params, state = self.operator.process_perception(images, state, pad_state=pad_state)
        lang_tokens, lang_masks, _, _, _, _ = self.operator.process_interaction(task=task, state=state)

        if add_batch_dim:
            images = [img.unsqueeze(0) for img in images]
            img_masks = [mask.unsqueeze(0) for mask in img_masks]
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
            emb_ids = torch.tensor([self.embodiment_id], dtype=torch.long, device=self.device)
        else:
            emb_ids = torch.tensor(self.embodiment_id, dtype=torch.long, device=self.device)

        return images, img_masks, lang_tokens, lang_masks, state, image_transform_params, emb_ids, ori_device

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
        enable_2d_traj_output: bool = False,
        autoregressive_mode_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if autoregressive_mode_only:
            return self.predict_autoregressive_actions(images, task, state)

        images, img_masks, lang_tokens, lang_masks, state, image_transform_params, emb_ids, ori_device = self._prepare_action_inputs(
            images, state, task, pad_state=True, add_batch_dim=True
        )

        outputs = self.synthesis.predict(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            emb_ids=emb_ids,
            enable_2d_traj_output=enable_2d_traj_output,
        )
        if enable_2d_traj_output:
            pred_action, traj_pred = outputs
        else:
            pred_action = outputs

        pred_action = self.operator.process_output(
            pred_action[0],
            state,
            self.original_action_dim,
            image_transform_params=image_transform_params,
            traj_pred=None,
        )
        if isinstance(pred_action, tuple):
            pred_action = pred_action[0]
        pred_action = pred_action.to(ori_device)

        if enable_2d_traj_output:
            traj_pred = traj_pred[0]
            if 'resize_with_pad' in image_transform_params:
                ratio = image_transform_params['resize_with_pad']['ratio']
                pad_x, pad_y = image_transform_params['resize_with_pad']['padding']
                traj_pred[:, ::2] = (traj_pred[:, ::2] * self.resize_imgs_with_padding[0] - pad_x) * ratio
                traj_pred[:, 1::2] = (traj_pred[:, 1::2] * self.resize_imgs_with_padding[1] - pad_y) * ratio
            traj_pred = traj_pred.to(ori_device)
            return pred_action, traj_pred

        return pred_action

    @torch.no_grad()
    def predict_current_subtask(self, images: dict[str, torch.Tensor], task: str) -> list[str]:
        tokenizer = self.operator.tokenizer

        images = {k: v.to(self.device) for k, v in images.items()}
        images, img_masks, _, _ = self.operator.process_perception(images, state=torch.empty(0), pad_state=False)
        lang_tokens, lang_masks, _, _, _, _ = self.operator.process_interaction(task=task)

        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]

        generated = self.generate_autoregressive_tokens(images, img_masks, lang_tokens, lang_masks)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def predict_autoregressive_actions(
        self, images: dict[str, torch.Tensor], task: str, state: torch.Tensor, max_new_tokens: int = 200
    ) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, state, _, _, ori_device = self._prepare_action_inputs(
            images, state, task, pad_state=False, add_batch_dim=False
        )

        generated = self.generate_autoregressive_tokens(images, img_masks, lang_tokens, lang_masks, max_new_tokens=max_new_tokens)

        pred_action = self.operator.extract_actions(generated, self.policy.n_action_steps, self.original_action_dim)
        pred_action = pred_action.to(self.device)
        pred_action = self.operator.process_output(pred_action, state, self.original_action_dim)
        if isinstance(pred_action, tuple):
            pred_action = pred_action[0]
        pred_action = pred_action.to(ori_device)
        return pred_action

    @torch.no_grad()
    def generate_autoregressive_tokens(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> list[list[int]]:
        for i in range(len(images)):
            if images[i].ndim == 3:
                images[i] = images[i][None, ...]
            if img_masks[i].ndim == 1:
                img_masks[i] = img_masks[i][None, ...]
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens[None, ...]
        if lang_masks.ndim == 1:
            lang_masks = lang_masks[None, ...]

        next_logits, gen_state = self.policy.init_lang_generation(images, img_masks, lang_tokens, lang_masks)

        tokenizer = self.operator.tokenizer
        eos_id = tokenizer.eos_token_id
        generated: list[list[int]] = [[] for _ in range(lang_tokens.shape[0])]
        finished = torch.zeros(lang_tokens.shape[0], dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            step_token = torch.argmax(next_logits, dim=-1).to(torch.long)
            step_token = torch.where(finished, torch.tensor(eos_id, device=step_token.device), step_token)
            for i in range(len(generated)):
                if not finished[i].item():
                    generated[i].append(step_token[i].item())
            finished = finished | (step_token == eos_id)
            if torch.all(finished):
                break
            input_token = step_token.view(lang_tokens.shape[0], 1)
            next_logits, gen_state = self.policy.next_lang_logits(gen_state, input_token)

        return generated
