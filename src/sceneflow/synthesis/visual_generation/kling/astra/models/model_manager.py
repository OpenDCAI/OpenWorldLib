# src/sceneflow/synthesis/visual_generation/kling/astra/models/model_manager.py

import os
import torch
import json
from typing import List, TypeAlias, Literal, Optional

# =========================================================================
# 1. 手动定义类型别名，解决 NameError
# =========================================================================
Preset_model_id: TypeAlias = str
Preset_model_website: TypeAlias = str

from ..configs.model_config import model_loader_configs, huggingface_model_loader_configs, patch_model_loader_configs

from ......base_models.diffusion_model.diffsynth.models.utils import (
    load_state_dict, 
    init_weights_on_device, 
    hash_state_dict_keys, 
    split_state_dict_with_prefix
)

# =========================================================================
# 3. 核心加载函数 (保持原样，删除了不必要的引用)
# =========================================================================

def load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        
        # 尝试使用 state_dict_converter，如果没有则直接加载
        if hasattr(model_class, "state_dict_converter"):
            state_dict_converter = model_class.state_dict_converter()
            if model_resource == "civitai":
                state_dict_results = state_dict_converter.from_civitai(state_dict)
            elif model_resource == "diffusers":
                state_dict_results = state_dict_converter.from_diffusers(state_dict)
            else:
                # 默认为 huggingface 或直接加载
                state_dict_results = state_dict_converter.from_diffusers(state_dict) 
        else:
            state_dict_results = state_dict, {}

        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        
        # 初始化模型
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        
        if hasattr(model, "eval"):
            model = model.eval()
        
        # 加载权重
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=torch_dtype, device=device)
        
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models

# =========================================================================
# 4. 检测器类 (Detector)
# =========================================================================

class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        
        # 先尝试带 Shape 的 Hash
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        
        # 再尝试普通 Hash
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False

    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # 匹配并加载
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            return load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)

        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            return load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)

        return [], []

# =========================================================================
# 5. Model Manager 类 (精简版)
# =========================================================================

class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: List[Preset_model_id] = [], # 这里的 Preset_model_id 现在是 str，不会报错了
        downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
        file_path_list: List[str] = [],
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        
        # 我们不使用 downloader，所以忽略 model_id_list 和 downloading_priority
        if len(model_id_list) > 0:
            print("Warning: Auto-downloading via model_id_list is disabled in this Astra version.")

        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            # ModelDetectorFromSplitedSingleFile, ModelDetectorFromHuggingfaceFolder 等暂不需要
        ]
        self.load_models(file_path_list)

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        if device is None: device = self.device
        if torch_dtype is None: torch_dtype = self.torch_dtype
        
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            print(f"Error: File not found: {file_path}")
            return

        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path, state_dict,
                    device=device, torch_dtype=torch_dtype,
                    allowed_model_names=model_names, model_manager=self
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        else:
            print(f"    We cannot detect the model type. No models are loaded.")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names, device=device, torch_dtype=torch_dtype)
    
    def fetch_model(self, model_name, file_path=None, require_model_path=False):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            # print(f"Using {model_name} from {fetched_model_paths[0]}.")
            pass
        else:
            print(f"More than one {model_name} models are loaded. Using from {fetched_model_paths[0]}.")
        
        if require_model_path:
            return fetched_models[0], fetched_model_paths[0]
        else:
            return fetched_models[0]

    def to(self, device):
        for model in self.model:
            model.to(device)