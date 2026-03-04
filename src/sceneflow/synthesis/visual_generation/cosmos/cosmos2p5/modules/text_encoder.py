import torch
import torch.nn as nn
from diffusers.utils.accelerate_utils import apply_forward_hook
from transformers.models import Qwen2_5_VLProcessor, Qwen2_5_VLTextModel


class Reason1TextEncoder(nn.Module):
    """A text encoder that uses the Qwen2.5-VL model to generate embeddings
    from text prompts.

    This class wraps the `Qwen2_5_VLTextModel` and its processor to provide a simple interface
    for encoding prompts into a format suitable for the Cosmos2.5 diffusion model. It handles
    tokenization, padding, and extracting and normalizing hidden states from multiple layers.

    Args:
        pretrained (`str`): The path or Hugging Face identifier for the pretrained Qwen2.5-VL model.
    """

    def __init__(self, pretrained):
        super().__init__()
        # Load the processor and the text model from the pretrained path
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained)
        self.model = Qwen2_5_VLTextModel.from_pretrained(pretrained)
        self.model.eval()  # Set the model to evaluation mode
        # Define the target sequence length for padding/truncation
        self.num_embedding_padding_tokens = 512

    @property
    def device(self):
        """The device the model is currently on."""
        return self.model.device

    @property
    def dtype(self):
        """The data type of the model's parameters."""
        return self.model.dtype

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Mean-normalize a tensor along its last dimension.

        This involves subtracting the mean and dividing by the standard deviation.

        Args:
            tensor (`torch.Tensor`): The tensor to normalize.

        Returns:
            `torch.Tensor`: The normalized tensor.
        """
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    @apply_forward_hook
    @torch.inference_mode()
    def encode_prompts(self, prompts):
        """Encodes a batch of text prompts into embeddings.

        Args:
            prompts (`str` or `List[str]`): A single prompt or a list of prompts.

        Returns:
            `torch.Tensor`: The concatenated and normalized text embeddings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        input_ids_batch = []
        for prompt in prompts:
            # Format the prompt using the model's chat template
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a helpful assistant who will provide prompts to an image generator.'}],
                },
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': prompt}],
                },
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            # Tokenize the formatted text
            inputs = self.processor(
                text=[text],
                padding=False,
                return_tensors='pt',
            )
            input_ids = inputs['input_ids'][0]

            # Pad or truncate the token IDs to a fixed length
            pad_id = self.processor.tokenizer.pad_token_id
            if self.num_embedding_padding_tokens > len(input_ids):
                pad_len = self.num_embedding_padding_tokens - len(input_ids)
                input_ids = input_ids.tolist() + [pad_id] * pad_len
            else:
                input_ids = input_ids.tolist()[: self.num_embedding_padding_tokens]

            input_ids = torch.LongTensor(input_ids).to(device=self.device)
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        # Get hidden states from the text model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_batch, output_hidden_states=True, use_cache=False)
        hidden_states = outputs['hidden_states']

        # Normalize the hidden states from each layer (except the first)
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)

        # Concatenate the normalized states to form the final text embeddings
        text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        return text_embeddings
