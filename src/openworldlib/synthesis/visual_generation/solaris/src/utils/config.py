import importlib
import os

from hydra.utils import get_original_cwd
from omegaconf import open_dict


def instantiate_from_config(config, **kwargs):
    params = dict(config.get("params", {}))
    additional_params = params.pop("additional_params", {})
    params.update(additional_params)
    return get_obj_from_str(config["target"])(**params, **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def resolve_device_paths(cfg):
    """Resolve relative device paths to absolute using Hydra's original cwd."""
    orig_cwd = get_original_cwd()
    path_keys = [
        "data_dir", "eval_data_dir", "pretrained_model_dir",
        "output_dir", "checkpoint_dir", "jax_cache_dir",
    ]
    with open_dict(cfg):
        for key in path_keys:
            if key in cfg.device:
                val = cfg.device[key]
                if val and not os.path.isabs(val):
                    cfg.device[key] = os.path.join(orig_cwd, val)
