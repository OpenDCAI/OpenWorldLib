import logging
import os

from src.utils.config import instantiate_from_config


def setup_jax_cache(cache_dir):
    """Setup JAX compilation cache for faster recompilation."""
    # Set up compilation cache directory
    os.environ["JAX_COMPILATION_CACHE"] = "1"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    import jax

    # Configure JAX compilation cache
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_compilation_cache_max_size", -1)  # No size limit

    logging.info(f"JAX compilation cache enabled at: {cache_dir}")


def init_jax_distributed(cfg):
    import jax

    logging.info("Setting up jax distributed")

    if (cfg.device.name == "tpu") and not jax.distributed.is_initialized():
        instantiate_from_config(cfg.device.jax_distributed_config)
