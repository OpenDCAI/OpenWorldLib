import abc
import functools
import os
import time
from collections import defaultdict

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import orbax.checkpoint as ocp
import torch.multiprocessing as mp
from absl import logging
from flax import nnx
from tqdm import tqdm

import src.utils.sharding as sharding_utils
import src.utils.wandb as wandb_utils
from src.data.dataset import VideoReadError
from src.models.model_loaders import get_jax_clip_model, get_vae_model
from src.utils.config import get_obj_from_str, instantiate_from_config




def ensure_torch_spawn_start_method():
    """Make dataloader worker behavior consistent across platforms."""
    mp.set_start_method("spawn", force=True)


def init_rngs(base_seed):
    rngs = jax.random.PRNGKey(base_seed + 1)
    rngs, rngs_eval = jax.random.split(rngs)
    rngs, rngs_test = jax.random.split(rngs)
    return rngs, rngs_eval, rngs_test


def load_clip_and_vae_models(
    *,
    clip_checkpoint_path,
    vae_checkpoint_path,
    repl_sharding,
    log_prefix="",
):
    """Load CLIP + VAE weights once and replicate them."""
    start_time = time.time()

    def load_models():
        clip_model = get_jax_clip_model()
        vae_model = get_vae_model()
        clip_graph, clip_state = nnx.split(clip_model)
        vae_graph, vae_state = nnx.split(vae_model)
        return clip_graph, clip_state, vae_graph, vae_state

    clip_graph, clip_state, vae_graph, vae_state = jax.jit(
        load_models,
        out_shardings=(
            repl_sharding,
            repl_sharding,
            repl_sharding,
            repl_sharding,
        ),
    )()
    if log_prefix:
        logging.info(
            "%sFinished initializing clip and vae (%.2fs)",
            log_prefix,
            time.time() - start_time,
        )

    checkpointer = ocp.StandardCheckpointer()
    clip_state = checkpointer.restore(clip_checkpoint_path, clip_state)
    vae_state = checkpointer.restore(vae_checkpoint_path, vae_state)

    clip_model = nnx.merge(clip_graph, clip_state)
    vae_model = nnx.merge(vae_graph, vae_state)
    return clip_model, vae_model


def build_eval_dataloaders(
    *,
    eval_datasets,
    eval_data_dir,
    obs_resolution,
    converters,
    eval_dataloader_config,
    allow_additional_params=True,
):
    eval_dataloaders = {}
    for eval_dataset_config in eval_datasets.values():
        eval_dataset_cls = get_obj_from_str(eval_dataset_config["class"])
        eval_dataset_name = eval_dataset_config["name"]
        dataset_kwargs = {
            "data_dir": os.path.join(
                eval_data_dir, eval_dataset_config["test_dataset_name"]
            ),
            "dataset_name": eval_dataset_name,
            "obs_resize": obs_resolution,
            "converters": converters,
        }
        if allow_additional_params:
            dataset_kwargs.update(eval_dataset_config.get("additional_params", {}))

        eval_dataset = eval_dataset_cls(**dataset_kwargs)
        eval_dataloader, eval_local_num_batches = instantiate_from_config(
            eval_dataloader_config, dataset=eval_dataset
        )
        eval_dataloaders[eval_dataset_name] = {
            "dataloader": eval_dataloader,
            "local_num_batches": eval_local_num_batches,
        }
    return eval_dataloaders


class BaseRunner(abc.ABC):

    def __init__(
        self,
        base_seed,
        eval_num_samples,
        eval_save_dir,
        network_config,
        eval_dataloader_config,
        sharding_config,
        eval_datasets,
        eval_data_dir,
        obs_resolution,
        converters,
        clip_checkpoint_path,
        vae_checkpoint_path,
        pretrained_model_dir,
        experiment_name,
    ):
        logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
        logging.info("JAX local devices: %r", jax.local_devices())
        logging.info("JAX local devices count: %d", jax.local_device_count())
        self.eval_save_dir = eval_save_dir
        self.network_config = network_config
        self.pretrained_model_dir = pretrained_model_dir
        self.experiment_name = experiment_name

        self.left_action_padding = (
            network_config.params.action_config.left_action_padding
        )

        (self.mesh, self.ddp_sharding, self.repl_sharding) = instantiate_from_config(
            sharding_config
        )

        self.pretrained_checkpointer = ocp.StandardCheckpointer()
        self.rngs, self.rngs_eval, self.rngs_test = init_rngs(base_seed)
        self.clip_model, self.vae_model = load_clip_and_vae_models(
            clip_checkpoint_path=clip_checkpoint_path,
            vae_checkpoint_path=vae_checkpoint_path,
            repl_sharding=self.repl_sharding,
        )
        mp.set_start_method("spawn", force=True)

        self.eval_dataloaders = build_eval_dataloaders(
            eval_datasets=eval_datasets,
            eval_data_dir=eval_data_dir,
            obs_resolution=obs_resolution,
            converters=converters,
            eval_dataloader_config=eval_dataloader_config,
            allow_additional_params=True,
        )

    def robust_batch_sample(self, loader_iter, num_retries=5):

        for _ in range(num_retries):
            try:
                return next(loader_iter)
            except VideoReadError:
                logging.info("Process %s retrying to load batch", jax.process_index())
        raise RuntimeError("Failed to load batch")

    def globalize_batch(self, batch):
        return jax.tree.map(
            lambda x: sharding_utils.make_fsarray_from_local_slice(
                x, self.mesh.devices.flatten()
            ),
            batch,
        )

    @abc.abstractmethod
    def _get_curr_batch(self, train_loader_iter):
        pass

    @abc.abstractmethod
    def _evaluate(
        self,
        model_state,
        model_graph,
        vae_state,
        vae_graph,
        clip_state,
        clip_graph,
        video,
        mouse_actions,
        keyboard_actions,
        real_lengths,
        eval_dir,
        mesh,
        left_action_padding,
        num_denoising_steps=None,
    ):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def run_eval(
        self,
        model,
        num_denoising_steps,
        eval_dataloader_info,
        eval_dir_name,
    ):
        _, video_unprocessed, actions_mouse, actions_keyboard, real_lengths = (
            self._get_curr_batch(iter(eval_dataloader_info["dataloader"]))
        )

        evaluation_output_directory = os.path.join(self.eval_save_dir, eval_dir_name)
        os.makedirs(evaluation_output_directory, exist_ok=True)

        model.eval()

        eval_graph, eval_state = nnx.split(model)
        vae_graph, vae_state = nnx.split(self.vae_model)
        clip_graph, clip_state = nnx.split(self.clip_model)

        metric_curve = self._evaluate(
            eval_state,
            eval_graph,
            vae_state,
            vae_graph,
            clip_state,
            clip_graph,
            video_unprocessed,
            actions_mouse,
            actions_keyboard,
            real_lengths,
            eval_dir=evaluation_output_directory,
            mesh=self.mesh,
            left_action_padding=self.left_action_padding,
            num_denoising_steps=num_denoising_steps,
        )
        for k, v in metric_curve.items():
            logging.info(f"test_{k}: {v.mean().item()}")
        return metric_curve
