from typing_extensions import Literal, TypeAlias

from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel


# Model loader configurations for WAN video models
# Each entry: (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
model_loader_configs = [
    # WAN Video DiT models
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),

    # WAN Video Text Encoder
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),

    # WAN Video Image Encoder
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),

    # WAN Video VAE
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),

    # WAN Video Motion Controller
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
]

# Huggingface model loader configurations
# Each entry: (architecture, huggingface_lib, model_name, redirected_architecture)
huggingface_model_loader_configs = [
    # WAN models can be loaded from HuggingFace if needed
    # Add WAN-specific HuggingFace configs here if applicable
]

# Patch model loader configurations
# Each entry: (keys_hash_with_shape, model_name, model_class, extra_kwargs)
patch_model_loader_configs = [
    # Add WAN-specific patch model configs here if applicable
]

# Preset model IDs type
Preset_model_id: TypeAlias = Literal[
    # WAN video models can be added here when they have preset IDs
    "Wan2.1-T2V-1.3B",
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B",
]

# Preset model website
Preset_model_website: TypeAlias = Literal["ModelScope", "HuggingFace"]

# Preset models on HuggingFace
preset_models_on_huggingface = {}

# Preset models on ModelScope
preset_models_on_modelscope = {}
