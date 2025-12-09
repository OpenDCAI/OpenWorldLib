import torch
from torch.nn import Parameter
from ..models.factory import create_model_from_config

def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'
    if model_type == 'mm_diffusion_cond':
        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            diffusion_objective=training_config.get("diffusion_objective","v"),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            max_mask_segments = training_config.get("max_mask_segments", 0)
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_demo_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    demo_config = training_config.get("demo", {})

    if model_type == 'mm_diffusion_cond':
        from .diffusion import DiffusionCondDemoCallback

        return DiffusionCondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250), 
            num_demos=demo_config["num_demos"],
            demo_cfg_scales=demo_config["demo_cfg_scales"],
            demo_conditioning=demo_config.get("demo_cond", {}),
            demo_cond_from_batch=demo_config.get("demo_cond_from_batch", False),
            display_audio_cond=demo_config.get("display_audio_cond", False),
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')