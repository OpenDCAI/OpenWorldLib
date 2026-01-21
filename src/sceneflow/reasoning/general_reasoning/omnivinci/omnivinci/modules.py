from typing import Optional, Union, List, Dict, Any
import os
import torch.nn as nn
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig, Qwen2AudioEncoder
from transformers.generation import GenerationMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.siglip import SiglipVisionModel
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, SiglipImageProcessor
from transformers.image_processing_utils import BaseImageProcessor

def resolve_pretrained_subfolder(
    pretrained_model_name_or_path: str,
    subfolder: str,
) -> str:
    """
    Resolve a HF model id or a local path, and return the concrete local path
    to the given subfolder.

    Args:
        pretrained_model_name_or_path:
            - HF repo id, e.g. "nvidia/omnivinci"
            - or local directory path
        subfolder:
            e.g. "sound_tower"

    Returns:
        str: local filesystem path to the subfolder
    """
    # Case 1: local path
    if os.path.isdir(pretrained_model_name_or_path):
        base_path = Path(pretrained_model_name_or_path)

    # Case 2: Hugging Face repo id
    else:
        base_path = Path(
            snapshot_download(
                repo_id=pretrained_model_name_or_path,
                local_files_only=True,  # do not re-download
            )
        )

    resolved_path = base_path / subfolder

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Subfolder '{subfolder}' not found under {base_path}"
        )

    return str(resolved_path)



def _dummy_check_and_enable_flash_attn_2(
    cls,
    config,
    torch_dtype=None,
    device_map=None,
    check_device_map=True,
    hard_check_only=False,
):
    # 直接什么都不做，只返回 config
    return config

# monkey patch
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
    _dummy_check_and_enable_flash_attn_2
)


class VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.cfg_only = None

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def _maybe_resize_pos_embeds(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        resolution: int = -1,
        interpolate_mode: str = "linear",
    ):
        if resolution in [model.config.image_size, -1]:
            return
        print(
            f"Resizing vision model's position embeddings to support higher vision resolution: from {model.config.image_size} to {resolution} ..."
        )
        embeddings = model.vision_model.embeddings
        patch_size = embeddings.patch_size
        num_new_tokens = int((resolution // patch_size) ** 2)

        old_embeddings = embeddings.position_embedding
        match interpolate_mode:
            case "linear":
                # Step 1: Calculate the corresponding patch ID (pid) in the current resolution (M patches) based on the target resolution (N patches). Formula: pid = pid / N * M
                import torch
                import torch.nn as nn

                if is_deepspeed_zero3_enabled():
                    try:
                        import deepspeed
                    except ImportError:
                        raise ImportError("DeepSpeed is not installed. Please install it with `pip install deepspeed`.")
                    with deepspeed.zero.GatheredParameters([old_embeddings.weight], modifier_rank=None):
                        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
                else:
                    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
                new_embeddings = nn.Embedding(
                    num_new_tokens,
                    old_embedding_dim,
                    dtype=old_embeddings.weight.dtype,
                    device=old_embeddings.weight.device,
                )
                mapped_indices = (
                    torch.arange(num_new_tokens).to(old_embeddings.weight.device)
                    / (num_new_tokens - 1)
                    * (old_num_tokens - 1)
                )
                floor_indices = torch.clamp(mapped_indices.floor().long(), min=0, max=old_num_tokens - 1)
                ceil_indices = torch.clamp(mapped_indices.ceil().long(), min=0, max=old_num_tokens - 1)
                if is_deepspeed_zero3_enabled():
                    params = [old_embeddings.weight, new_embeddings.weight]
                    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                        interpolated_embeds = (mapped_indices - floor_indices)[:, None] * old_embeddings.weight.data[
                            ceil_indices, :
                        ] + (ceil_indices - mapped_indices)[:, None] * old_embeddings.weight.data[floor_indices, :]
                else:
                    interpolated_embeds = (mapped_indices - floor_indices)[:, None] * old_embeddings.weight.data[
                        ceil_indices, :
                    ] + (ceil_indices - mapped_indices)[:, None] * old_embeddings.weight.data[floor_indices, :]
                new_embeddings.weight.data = interpolated_embeds
            case _:
                raise NotImplementedError

        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
        model.config.image_size = resolution
        if hasattr(image_processor, "crop_size"):
            # CLIP vision tower
            image_processor.crop_size = resolution
        else:
            # SIGLIP vision tower
            assert hasattr(image_processor, "size")
            image_processor.size = {"height": resolution, "width": resolution}

        embeddings.position_embedding = new_embeddings
        embeddings.image_size = resolution
        embeddings.num_patches = embeddings.num_positions = num_new_tokens
        embeddings.position_ids = (
            torch.arange(embeddings.num_positions).expand((1, -1)).to(old_embeddings.weight.device)
        )

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device="cuda", dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device="cuda", dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device="cuda", dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype            

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class VisionTowerDynamicS2(VisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.scales = list(map(int, args.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = args.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(args, "s2_resize_output_to_scale_idx", 0)

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device="cuda", dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        assert type(images) is not list
        image_features = self.forward_feature(images)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class SiglipVisionTowerDynamicS2(VisionTowerDynamicS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        if type(config.model_dtype) == str:
            model_dtype = eval(config.model_dtype)
        else:
            model_dtype = config.model_dtype

        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation=config._attn_implementation,
            torch_dtype=model_dtype,
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[0]
        self.is_loaded = True


class AudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.cfg_only = None

    def forward(self, sounds):
        if type(sounds) is list:
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features"):
                    sound = sound["input_features"]
                sound_feature = self.audio_tower(sound)
                sound_feature = sound_feature.last_hidden_state
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

class Qwen2AudioTower(AudioTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.audio_tower = Qwen2AudioEncoder.from_pretrained(model_name_or_path, attn_implementation="eager")
        self.is_loaded = True
        self.audio_chunk_unit_duration = 30
        self.audio_chunk_unit_length = 3000

    def forward(self, sounds):
        if type(sounds) is list:
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features") or (type(sound) is dict and "input_features" in sound):
                    sound = sound["input_features"]

                sound_feature = self.forward_audio_tower_batch(sound)
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            if len(sound_features) > 0:
                sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths

    def forward_audio_tower_batch(self, inp):
        """
        Process long audio input by splitting into fixed-size chunks (30 seconds),
        padding if needed, batching them together, and processing through the audio tower.

        Args:
            inp: Tensor of shape (batch_size, n_mels, seq_len)

        Returns:
            Tensor of shape (batch_size, num_chunks * chunk_seq_len, hidden_size)
        """
        batch_size, n_mels, seq_len = inp.shape
        chunk_length = self.audio_chunk_unit_length
        num_chunks = (seq_len + chunk_length - 1) // chunk_length  # Ceiling division

        padded_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)

            # Extract and pad chunk if necessary
            chunk = inp[:, :, start_idx:end_idx]
            if chunk.shape[2] < chunk_length:
                pad_len = chunk_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len), mode='constant', value=0)

            padded_chunks.append(chunk)

        # Stack chunks along batch dimension
        all_chunks = torch.cat(padded_chunks, dim=0).reshape(batch_size * num_chunks, n_mels, chunk_length)

        # Forward pass through the audio tower
        chunk_outputs = self.audio_tower(all_chunks)
        hidden_states = chunk_outputs.last_hidden_state

        # Reshape back to (batch_size, num_chunks * seq_len', hidden_size)
        _, chunk_seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_chunks * chunk_seq_len, hidden_size)

        return hidden_states



def add_generation_mixin_to_remote_model(model_class):
    """
    Adds `GenerationMixin` to the inheritance of `model_class`, if `model_class` is a PyTorch model.

    This function is used for backwards compatibility purposes: in v4.45, we've started a deprecation cycle to make
    `PreTrainedModel` stop inheriting from `GenerationMixin`. Without this function, older models dynamically loaded
    from the Hub may not have the `generate` method after we remove the inheritance.
    """
    # 1. If it is not a PT model (i.e. doesn't inherit Module), do nothing
    if "torch.nn.modules.module.Module" not in str(model_class.__mro__):
        return model_class

    # 2. If it already **directly** inherits from GenerationMixin, do nothing
    if "GenerationMixin" in str(model_class.__bases__):
        return model_class

    # 3. Prior to v4.45, we could detect whether a model was `generate`-compatible if it had its own `generate` and/or
    # `prepare_inputs_for_generation` method.
    has_custom_generate = "GenerationMixin" not in str(getattr(model_class, "generate"))
    has_custom_prepare_inputs = "GenerationMixin" not in str(getattr(model_class, "prepare_inputs_for_generation"))
    if has_custom_generate or has_custom_prepare_inputs:
        model_class_with_generation_mixin = type(
            model_class.__name__, (model_class, GenerationMixin), {**model_class.__dict__}
        )
        return model_class_with_generation_mixin
    return model_class

def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]

@classmethod
def vinci_from_pretrained(
    cls,
    pretrained_model_name_or_path: Optional[str] = None,
    *model_args,
    config: Optional[Union[str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: Optional[bool] = None,
    weights_only: bool = True,
    **kwargs,
):
    # print("DEBUG2", kwargs); input()
    if not config:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    if kwargs.get("torch_dtype", None) is not None:
        config.torch_dtype = kwargs.get("torch_dtype", None)
        config.model_dtype = kwargs.get("torch_dtype", None)
        if type(kwargs.get("torch_dtype", None)) == str:
            kwargs["torch_dtype"] = eval(kwargs.get("torch_dtype", None))
        else:
            kwargs["torch_dtype"] = kwargs.get("torch_dtype", None)
    return cls._from_config(config, **kwargs)



@classmethod
def from_pretrained_new(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    config = kwargs.pop("config", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    kwargs["_from_auto"] = True
    hub_kwargs_names = [
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "use_auth_token",
        "token",
    ]
    hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", None)

    token = hub_kwargs.pop("token", None)
    use_auth_token = hub_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        token = use_auth_token

    if token is not None:
        hub_kwargs["token"] = token

    if commit_hash is None:
        commit_hash = getattr(config, "_commit_hash", None)

    if not config:
        kwargs_orig = copy.deepcopy(kwargs)
        # ensure not to pollute the config object with torch_dtype="auto" - since it's
        # meaningless in the context of the config object - torch.dtype values are acceptable
        if kwargs.get("torch_dtype", None) == "auto":
            _ = kwargs.pop("torch_dtype")
        # to not overwrite the quantization_config if config has a quantization_config
        if kwargs.get("quantization_config", None) is not None:
            _ = kwargs.pop("quantization_config")

        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
            _commit_hash=commit_hash,
            **hub_kwargs,
            **kwargs,
        )

        # if torch_dtype=auto was passed here, ensure to pass it on
        if kwargs_orig.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"
        if kwargs_orig.get("quantization_config", None) is not None:
            kwargs["quantization_config"] = kwargs_orig["quantization_config"]

    has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
    has_local_code = type(config) in cls._model_mapping.keys()

    # Set the adapter kwargs
    kwargs["adapter_kwargs"] = adapter_kwargs

    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[cls.__name__]
        model_class = get_class_from_dynamic_module(
            class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
        )
        _ = hub_kwargs.pop("code_revision", None)
        cls.register(config.__class__, model_class, exist_ok=True)
        model_class = add_generation_mixin_to_remote_model(model_class)
        model_class.from_pretrained = vinci_from_pretrained
        # breakpoint()
        Model = model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
        ) 
        sound_tower_path = resolve_pretrained_subfolder(
            pretrained_model_name_or_path,
            "sound_tower",
        )
        vision_tower_path = resolve_pretrained_subfolder(
            pretrained_model_name_or_path,
            "vision_tower",
        )
        Model.sound_tower = Qwen2AudioTower(sound_tower_path,config).to(Model.device)
        Model.vision_tower = SiglipVisionTowerDynamicS2(vision_tower_path,config).to(Model.device)
        config.mm_hidden_size = Model.vision_tower.hidden_size
        return Model
    elif type(config) in cls._model_mapping.keys():
        model_class = _get_model_class(config, cls._model_mapping)
        return model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
        )
    raise ValueError(
        f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
        f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
    )

