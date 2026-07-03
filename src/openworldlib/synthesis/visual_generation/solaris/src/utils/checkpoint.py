"""
File containing the functions for model checkpointing.
Source: https://github.com/willisma/jax_measure_transport/blob/main/utils/checkpoint.py
"""

import orbax.checkpoint as ocp

def build_checkpoint_manager(
    ckpt_dir,
    *,
    save_interval_steps,
    max_to_keep,
    keep_period,
    step_prefix="checkpoint",
    enable_async_checkpointing=True,
):
    """Create a checkpoint manager for saving and restoring checkpoints during training."""

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,  # this handles the control flow of how many steps to save
        max_to_keep=max_to_keep,  # this handles the control flow of how many checkpoints to keep
        step_prefix=step_prefix,
        keep_period=keep_period,  # this keeps step % keep_period == 0; can be used as backup
        enable_async_checkpointing=enable_async_checkpointing,
    )
    return ocp.CheckpointManager(ckpt_dir, options=options)
