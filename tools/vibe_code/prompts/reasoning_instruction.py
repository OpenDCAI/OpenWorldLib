reasoning_prompt = """
The world model requires the implementation of multimodal reasoning capabilities, such as understanding and processing video, audio, and images. Our framework needs to possess multimodal reasoning capabilities; therefore, a Reasoning class must be defined.

The Reasoning class is invoked within the Pipeline class. It accepts processed inputs from the Operator and performs inference using the underlying model to generate reasoning outputs.
It should follow the structure below:
```python
class BaseReasoning(object):
    def __init__(self):
        ## Initialize the model used by the Reasoning class

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        ## Load the model weights required by the Reasoning class
    
    def api_init(self, api_key, endpoint):
        ## If calling an online model, initialize the API key or API URL

    @torch.no_grad()
    def inference(self):
        ## Accept external inputs and output the corresponding reasoning results
```
"""

example_reasoning_code = """
Here are the organized code results for qwen2.5-omni: https://github.com/QwenLM/Qwen2.5-Omni".
The Operator implementation is as follows:
```python

import numpy as np
from PIL import Image
import torch
from typing import Union, Optional, Dict, Any, List, Sequence
from pathlib import Path

from .base_operator import BaseOperator


class Qwen2p5OmniOperator(BaseOperator):
    
    def __init__(
        self,
        processor=None,
        use_audio_in_video: bool = True,
        system_prompt: Optional[str] = None,
        operation_types: List[str] = None,
    ):
        if operation_types is None:
            operation_types = [
                "text_processing",
                "image_processing",
                "audio_processing",
                "video_processing",
                "multimodal_processing"
            ]
        
        super().__init__(operation_types)
        
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        
        # Default system prompt for Qwen2.5-Omni
        if system_prompt is None:
            self.system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, "
                "Alibaba Group, capable of perceiving auditory and visual inputs, "
                "as well as generating text and speech."
            )
        else:
            self.system_prompt = system_prompt
        
        # Initialize interaction template
        self.interaction_template = [
            "text_prompt",
            "image_prompt",
            "audio_prompt",
            "video_prompt",
            "multimodal_prompt"
        ]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        if not isinstance(interaction, (str, dict, list)):
            raise TypeError(f"Invalid interaction type: {type(interaction)}")
        return True
    
        if self.check_interaction(interaction):
            self.current_interaction = interaction
    
    def load_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        else:
            pil_img = image_input
        
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        return pil_img
    
    def load_audio(self, audio_input: Union[str, Path, bytes]) -> Union[str, bytes]:
        if isinstance(audio_input, (str, Path)):
            return str(audio_input)
        return audio_input
    
    def load_video(self, video_input: Union[str, Path]) -> str:
        return str(video_input)
    
    def process_interaction(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        # Store current interaction
        self.get_interaction(text or messages)
        
        result = {}
        
        # Build or use provided messages
        if messages is not None:
            # Use existing messages and append text if provided
            result["messages"] = messages.copy() if isinstance(messages, list) else messages
            
            # If text is provided, append it to the last user message or create new one
            if text:
                # Find last user message
                last_user_idx = None
                for i in range(len(result["messages"]) - 1, -1, -1):
                    if result["messages"][i].get("role") == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx is not None:
                    # Append to existing user message
                    if isinstance(result["messages"][last_user_idx]["content"], list):
                        result["messages"][last_user_idx]["content"].append(
                            {"type": "text", "text": text}
                        )
                    else:
                        # Convert to list format if needed
                        result["messages"][last_user_idx]["content"] = [
                            {"type": "text", "text": result["messages"][last_user_idx]["content"]},
                            {"type": "text", "text": text}
                        ]
                else:
                    # No user message found, create new one
                    result["messages"].append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    })
            
            result["text"] = text
        else:
            built_messages = []
            
            # Add system prompt if requested
            if include_system_prompt and self.system_prompt:
                built_messages.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                })
            
            # Build user message content with text only
            content = []
            if text:
                content.append({"type": "text", "text": text})
            
            # Add user message
            if content:
                built_messages.append({
                    "role": "user",
                    "content": content
                })
            
            result["messages"] = built_messages
            result["text"] = text
        
        return result
    
    def process_perception(
        self,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        messages = []
        
        # Add system prompt if requested
        if include_system_prompt and self.system_prompt:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ]
            })
        
        # Build user message content
        content = []
        
        # Add images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                processed_img = self.load_image(img)
                content.append({"type": "image", "image": processed_img})
        
        # Add audios
        if audios is not None:
            if not isinstance(audios, list):
                audios = [audios]
            for audio in audios:
                processed_audio = self.load_audio(audio)
                content.append({"type": "audio", "audio": processed_audio})
        
        # Add videos
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            for video in videos:
                processed_video = self.load_video(video)
                content.append({"type": "video", "video": processed_video})
        
        # Add user message
        if content:
            messages.append({
                "role": "user",
                "content": content
            })
        
        result = {
            "messages": messages,
            "use_audio_in_video": self.use_audio_in_video,
        }
        
        return result
    
    def update_config(self, **kwargs):
        if "use_audio_in_video" in kwargs:
            self.use_audio_in_video = kwargs["use_audio_in_video"]
        
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]

```

The Pipeline implementation is as follows:
```python
import torch
import os
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
from PIL import Image
import soundfile as sf
from ...operators.qwen2p5_omni_operator import Qwen2p5OmniOperator
from ...reasoning.general_reasoning.qwen.qwen2p5_omni_reasoning import Qwen2p5OmniReasoning
from ...memories.reasoning.qwen.qwen_memory import QwenMemory

class Qwen2p5OmniPipeline:
    def __init__(
        self,
        operator: Optional[Qwen2p5OmniOperator] = None,
        reasoning_model: Optional[Qwen2p5OmniReasoning] = None,
        memory_module: Optional[QwenMemory] = None,
        device: str = 'cuda',
        use_audio_in_video: bool = True,
    ):
        self.operator = operator
        self.reasoning_model = reasoning_model
        self.memory_module = memory_module if memory_module else QwenMemory()
        self.device = device
        self.use_audio_in_video = use_audio_in_video
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path] = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        device_map: Union[str, dict] = "auto",
        use_audio_in_video: bool = False,
        system_prompt: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'Qwen2p5OmniPipeline':
        if logger:
            logger.info(f"Loading Qwen2.5-Omni pipeline from {pretrained_model_path}")
        
        # Load reasoning model
        if logger:
            logger.info("Loading Qwen2.5-Omni reasoning model...")
        
        reasoning_model = Qwen2p5OmniReasoning.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs
        )
        
        # Initialize operator
        if logger:
            logger.info("Initializing Qwen2.5-Omni operator...")
        
        operator = Qwen2p5OmniOperator(
            processor=reasoning_model.processor,
            use_audio_in_video=use_audio_in_video,
            system_prompt=system_prompt,
        )
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize memory module
        memory_module = QwenMemory()
        
        # Create pipeline instance
        pipeline = cls(
            operator=operator,
            reasoning_model=reasoning_model,
            memory_module=memory_module,
            device=device,
            use_audio_in_video=use_audio_in_video,
        )
        
        if logger:
            logger.info("Qwen2.5-Omni pipeline loaded successfully")
        
        return pipeline

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for Qwen2.5-Omni yet.
        raise NotImplementedError("API init is not supported for Qwen2.5-Omni.")


    
    def process(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # Process text interaction
        interaction_data = self.operator.process_interaction(
            text=text,
            messages=messages,
            **kwargs
        )
        
        # Process perception inputs
        perception_data = self.operator.process_perception(
            images=images,
            audios=audios,
            videos=videos,
            **kwargs
        )
        
        # Merge messages
        final_messages = interaction_data.get("messages", [])
        perception_messages = perception_data.get("messages", [])
        
        for msg in final_messages:
            if msg.get("role") == "user":
                for p_msg in perception_messages:
                    if p_msg.get("role") == "user":
                        msg["content"].extend(p_msg["content"])
        
        return {
            "messages": final_messages,
            "use_audio_in_video": perception_data.get("use_audio_in_video", self.use_audio_in_video)
        }
    
    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        messages: Optional[List[Dict]] = None,
        max_new_tokens: int = 128,
        generation_kwargs: Optional[dict] = None,
        return_audio: bool = False,
        use_operator: bool = True,
        **kwargs
    ) -> Union[List[str], tuple]:
        if self.reasoning_model is None:
            raise ValueError("Reasoning model is not initialized")
        
        # Process inputs through operator if enabled
        if use_operator:
            processed_data = self.process(
                text=text,
                images=images,
                audios=audios,
                videos=videos,
                messages=messages,
                **kwargs
            )
            
            # Extract messages and use_audio_in_video from processed data
            messages = processed_data.get("messages")
            use_audio_in_video = processed_data.get("use_audio_in_video", self.use_audio_in_video)
        else:
            # Use raw inputs
            use_audio_in_video = kwargs.get("use_audio_in_video", self.use_audio_in_video)
        
        # Run inference
        if not return_audio:
            result = self.reasoning_model.inference(
                messages=messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
        else:
            result, audio = self.reasoning_model.inference(
                messages=messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            return result, audio
        return result
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save operator config
        if self.operator:
            operator_config = {
                'use_audio_in_video': self.operator.use_audio_in_video,
                'system_prompt': self.operator.system_prompt,
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }
            torch.save(operator_config, os.path.join(save_directory, "operator_config.pt"))
        
        # Save pipeline config
        pipeline_config = {
            'device': self.device,
            'use_audio_in_video': self.use_audio_in_video,
        }
        torch.save(pipeline_config, os.path.join(save_directory, "pipeline_config.pt"))
        
        print(f"Qwen2.5-Omni Pipeline saved to {save_directory}")
    
    def update_operator_config(self, **kwargs):
        if self.operator:
            self.operator.update_config(**kwargs)
    
    def get_operator(self) -> Optional[Qwen2p5OmniOperator]:
        return self.operator
    
    def get_reasoning_model(self) -> Optional[Qwen2p5OmniReasoning]:
        return self.reasoning_model
    
    def stream(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        use_history: bool = True,
        max_new_tokens: int = 128,
        generation_kwargs: Optional[dict] = None,
        return_audio: bool = False,
        reset_memory: bool = False,
        **kwargs
    ) -> Union[List[str], tuple]:
        if reset_memory:
            self.memory_module.manage(action="reset")
            print("--- Stream Started (Memory Reset) ---")
        
        # Build current turn messages
        messages = None
        if use_history:
            messages = self.memory_module.select()
        
        # Process inputs through operator
        processed_data = self.process(
            text=text,
            images=images,
            audios=audios,
            videos=videos,
            messages=messages,
            **kwargs
        )
        
        current_messages = processed_data.get("messages")
        use_audio_in_video = processed_data.get("use_audio_in_video", self.use_audio_in_video)
        breakpoint()
        # Run inference
        if not return_audio:
            result = self.reasoning_model.inference(
                messages=current_messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            response_text = result[0] if isinstance(result, list) else result
        else:
            result, audio = self.reasoning_model.inference(
                messages=current_messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            response_text = result[0] if isinstance(result, list) else result
        
        # Record to memory
        self.memory_module.record({
            'messages': current_messages,
            'response': response_text,
            'metadata': {
                'max_new_tokens': max_new_tokens,
                'return_audio': return_audio
            }
        })
        
        if return_audio:
            return result, audio
        return result

```

The Reasoning class implementation is as follows:
```python

from typing import List, Optional, Sequence, Union

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from ...base_reasoning import BaseReasoning


ImageLike = Union[str, bytes]
AudioLike = Union[str, bytes]


class Qwen2p5OmniReasoning(BaseReasoning):
    def __init__(
        self,
        model: Qwen2_5OmniForConditionalGeneration,
        processor: Qwen2_5OmniProcessor,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = torch.device(device) if device is not None else self._get_default_device()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        device_map: Union[str, dict] = "auto",
        
        **kwargs,
    ) -> "Qwen2p5Omni":
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(pretrained_model_path)
        return cls(model=model, processor=processor, device=device)

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for Qwen2.5-Omni yet.
        raise NotImplementedError("API init is not supported for Qwen2.5-Omni.")

    def _get_default_device(self) -> torch.device:
        # Prefer model's device when device_map is set, otherwise fall back to CUDA/CPU.
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_messages(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, Sequence[AudioLike]]] = None,
        instruction: str = "",
    ):
        content = []
        
        # Process image inputs
        if image_paths is not None:
            if isinstance(image_paths, (str, bytes)):
                image_paths = [image_paths]
            content.extend([{"type": "image", "image": path} for path in image_paths])
        
        # Process audio inputs
        if audio_paths is not None:
            if isinstance(audio_paths, (str, bytes)):
                audio_paths = [audio_paths]
            content.extend([{"type": "audio", "audio": path} for path in audio_paths])
        
        # Add text instruction
        if instruction:
            content.append({"type": "text", "text": instruction})
        
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def inference(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, Sequence[AudioLike]]] = None,
        instruction: str = "",
        max_new_tokens: int = 128,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
        use_audio_in_video: bool = True,
        return_audio: bool = False,
    ) -> Union[List[str], tuple]:
        if messages is None:
            batched_messages = [
                self._build_messages(
                    image_paths=image_paths, 
                    audio_paths=audio_paths, 
                    instruction=instruction
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batched_messages
        ]

        # Process multimodal inputs (images, videos, audio)
        audios, images, videos = process_mm_info(batched_messages, use_audio_in_video=use_audio_in_video)

        # Prepare inputs for the model
        inputs = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.device).to(self.model.dtype)

        # Prepare generation kwargs
        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_audio_in_video": use_audio_in_video}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        # Generate
        if return_audio:
            text_ids, audio = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text, audio
        else:
            gen_kwargs["return_audio"] = False
            text_ids = self.model.generate(**inputs, **gen_kwargs)
            # Trim the input tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text
```
"""
