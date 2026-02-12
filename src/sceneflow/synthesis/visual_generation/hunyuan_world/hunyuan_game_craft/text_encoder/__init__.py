from typing import Optional, Tuple
from copy import deepcopy
import torch, os
import torch.nn as nn
from transformers import (
    CLIPTextModel, CLIPTokenizer, LlavaForConditionalGeneration, LlamaModel,
    LlamaTokenizerFast
)


from ......base_models.diffusion_model.video.hunyuan_video.text_encoder import (
        TextEncoder as BaseTextEncoder,
        TextEncoderModelOutput, 
        use_default
    )


from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH, PRECISION_TO_TYPE

CPU_OFFLOAD = int(os.environ.get("CPU_OFFLOAD", 0))
print(f'text_encoder: cpu_offload={CPU_OFFLOAD}')

def load_text_encoder(text_encoder_type,
                      text_encoder_precision=None,
                      text_encoder_path=None,
                      logger=None,
                      device=None,
                      model_base="tencent/Hunyuan-GameCraft-1.0"
                      ):
    if text_encoder_path is None:
        text_encoder_path = f"{model_base}/{TEXT_ENCODER_PATH[text_encoder_type]}"
    if logger is not None:
        logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llava-llama-3-8b":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
        import transformers
        transformers_version = transformers.__version__
        if transformers_version >= "4.53.0":
            text_encoder.final_layer_norm = text_encoder.language_model.norm
        else:
            text_encoder.final_layer_norm = text_encoder.language_model.model.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path

def load_tokenizer(tokenizer_type,
                   tokenizer_path=None,
                   padding_side="right",
                   logger=None,
                   model_base="tencent/Hunyuan-GameCraft-1.0"
                   ):
    if tokenizer_path is None:
        tokenizer_path = f"{model_base}/{TOKENIZER_PATH[tokenizer_type]}"
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llava-llama-3-8b":
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, padding_side=padding_side)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path


class TextEncoder(BaseTextEncoder):
    """
    Inherits from the base Hunyuan Video TextEncoder but overrides methods specific
    to Hunyuan GameCraft (LLaVA support, specific templating, CPU offloading).
    """
    def __init__(self,
                 text_encoder_type: str,
                 max_length: int,
                 text_encoder_precision: Optional[str] = None,
                 text_encoder_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 output_key: Optional[str] = None,
                 use_attention_mask: bool = True,
                 input_max_length: Optional[int] = None,
                 prompt_template_video: Optional[dict] = None,
                 hidden_state_skip_layer: Optional[int] = None,
                 apply_final_norm: bool = False,
                 reproduce: bool = False,
                 logger=None,
                 device=None,
                 model_base="tencent/Hunyuan-GameCraft-1.0"
                 ):
        nn.Module.__init__(self)
        
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask
        
        if prompt_template_video is not None: 
            assert use_attention_mask is True, "Attention mask is True required when training videos."
        
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert isinstance(self.prompt_template_video, dict) and "template" in self.prompt_template_video, (
                    f"`prompt_template_video` must be a dictionary with a key 'template', \
                    got {self.prompt_template_video}"
                )
            assert '{}' in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llama" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
            model_base=model_base
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
            model_base=model_base
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    def text2tokens(self, text, data_type='video', name='person'):
        """
        Tokenize the input text.
        Override to support LLaVA specific logic (<image> token).
        """
        tokenize_input_type = 'str'
        if self.use_video_template:
            if data_type == 'video': 
                prompt_template = self.prompt_template_video["template"]
            else: 
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = 'list'
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = 'list'
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        kwargs = dict(truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        if self.text_encoder_type == "llava-llama-3-8b":
            if isinstance(text, list):
                for i in range(len(text)):
                    text[i] = text[i] + '\nThe %s looks like<image>' % name
            elif isinstance(text, str):
                text = text + '\nThe %s looks like<image>' % name
            else:
                raise NotImplementedError

        if tokenize_input_type == 'str':
            return self.tokenizer(text, 
                                  return_length=False, 
                                  return_overflowing_tokens=False, 
                                  return_attention_mask=True, 
                                  **kwargs, )
        elif tokenize_input_type == 'list':
            return self.tokenizer.apply_chat_template(text, 
                                                      add_generation_prompt=True, 
                                                      tokenize=True, 
                                                      return_dict=True, 
                                                      **kwargs, )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(self, batch_encoding, use_attention_mask=None, output_hidden_states=False, do_sample=None,
                hidden_state_skip_layer=None, return_texts=False, data_type='image'):
        """
        Override to support CPU_OFFLOAD and pixel_value_llava
        """
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        do_sample = use_default(do_sample, not self.reproduce)
        
        if CPU_OFFLOAD:
            self.model.to('cuda')
            print(f'encode prompt: move text_encoder to cuda')

        attention_mask = batch_encoding["attention_mask"].to(self.model.device) if use_attention_mask else None
        
        if 'pixel_value_llava' in batch_encoding:
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(self.model.device),
                attention_mask=attention_mask,
                pixel_values=batch_encoding["pixel_value_llava"].to(self.model.device),
                output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None)
        else:
            outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(self.model.device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,)
        
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if self.use_video_template:
            if data_type == 'video': 
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else: 
                raise ValueError(f"Unsupported data type: {data_type}")
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None
        
        if CPU_OFFLOAD:
            self.model.to('cpu')
            torch.cuda.empty_cache()
            print(f'encode prompt successful: move text_encoder to cpu')
            
        if output_hidden_states:
            return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)
        return TextEncoderModelOutput(last_hidden_state, attention_mask)