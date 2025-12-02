from typing import Any, Dict, Optional, Tuple
import os
os.environ["LITMODELS_HIDE_TIP"] = "1"

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf, ListConfig

import rootutils
rootutils.setup_root(__file__, indicator="License.txt", pythonpath=True)

from mirror_training.utils.logger import RankedLogger, setup_logging

log = RankedLogger(__name__, rank_zero_only=True)

def create_directories(cfg: DictConfig):
    """Create necessary directories.

    :param cfg: Configuration object managed by Hydra framework.
    """
    if "paths" in cfg:
        log_dir = cfg.paths.get("log_dir", None)
        ckpt_dir = cfg.paths.get("ckpt_dir", None)
        vis_log_dir = cfg.paths.get("vis_log_dir", None)
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log.info(f"Created/verified log directory: {log_dir}")
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Created/verified checkpoint directory: {ckpt_dir}")
        if vis_log_dir:
            os.makedirs(vis_log_dir, exist_ok=True)
            log.info(f"Created/verified NVS output directory: {vis_log_dir}")

def train(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Training workflow implementation.

    :param cfg: Configuration object managed by Hydra framework.
    :return: Tuple containing metric dictionary and additional information.
    """
    log.info(f"\n{'-' * 60}\n{'Training Initialization'.center(60)}\n{'-' * 60}")
    
    # Create necessary directories
    create_directories(cfg)
    
    # Set random seed
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, verbose=False)
    
    # Instantiate data loader
    log.info(f"Instantiating data loader <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Instantiate model wrapper
    model_wrapper = hydra.utils.instantiate(cfg.wrapper)
    
    # Instantiate logger(s)
    if isinstance(cfg.logger, DictConfig):
        logger = hydra.utils.instantiate(cfg.logger)
    elif isinstance(cfg.logger, ListConfig):
        logger = [hydra.utils.instantiate(log_cfg) for log_cfg in cfg.logger]
    else:
        logger = None
    # Instantiate callbacks
    if isinstance(cfg.callbacks, DictConfig):
        callbacks = hydra.utils.instantiate(cfg.callbacks)
    elif isinstance(cfg.callbacks, ListConfig):
        callbacks = [hydra.utils.instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks = None
        
    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    # Train model
    trainer.fit(model_wrapper, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path", None))


def eval(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluation workflow implementation.

    :param cfg: Configuration object managed by Hydra framework.
    :return: Tuple containing metric dictionary and additional information.
    """
    log.info(f"\n{'-' * 60}\n{'Evaluation Initialization'.center(60)}\n{'-' * 60}")
    
    # Create necessary directories
    create_directories(cfg)
    
    # Set random seed
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, verbose=False)
    
    # Instantiate data loader
    log.info(f"Instantiating data loader <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Instantiate model wrapper
    model_wrapper = hydra.utils.instantiate(cfg.wrapper)
    
    # Instantiate logger(s)
    if isinstance(cfg.logger, DictConfig):
        logger = hydra.utils.instantiate(cfg.logger)
    elif isinstance(cfg.logger, list):
        logger = [hydra.utils.instantiate(log_cfg) for log_cfg in cfg.logger]
    else:
        logger = None
    
    # Instantiate callbacks
    if isinstance(cfg.callbacks, DictConfig):
        callbacks = hydra.utils.instantiate(cfg.callbacks)
    elif isinstance(cfg.callbacks, list):
        callbacks = [hydra.utils.instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks = None

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    
    # Evaluate model
    trainer.validate(model_wrapper, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Primary function for model training or evaluation workflows.

    :param cfg: Configuration object managed by Hydra framework.
    :return: Metric value for optimization, or None if not applicable.
    """
    # Suppress warning messages
    setup_logging()
    
    # Print configuration only on the main process
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    log.info(f"\n{'-' * 60}\n{'Configuration'.center(60)}\n{'-' * 60}\n{config_yaml}")
        
    # Extract configuration name to decide execution mode
    config_name = hydra.core.hydra_config.HydraConfig.get().job.config_name
    if config_name == "train.yaml":
        train(cfg)
    elif config_name == "eval.yaml":
        eval(cfg)
    else:
        raise ValueError(f"Invalid config name: {config_name}")

if __name__ == "__main__":
    main()