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
from src.runners.base_runner import BaseRunner
from src.utils.config import get_obj_from_str, instantiate_from_config


def tree_l2_norm(tree):
    sq_sum = jax.tree_util.tree_reduce(
        lambda s, x: s + jnp.sum(jnp.square(x)),  # accumulate ∑ x²
        tree,
        initializer=0.0,
    )
    return jnp.sqrt(sq_sum)

def init_checkpoint_manager(checkpoint_config, restore_step):
    """Instantiate ckpt manager and resolve restore step ('latest' supported)."""
    ckpt_mngr = instantiate_from_config(checkpoint_config)
    if restore_step == "latest":
        resolved_restore_step = ckpt_mngr.latest_step()
    else:
        resolved_restore_step = restore_step
    save_every_steps = checkpoint_config.params.save_interval_steps
    return ckpt_mngr, resolved_restore_step, save_every_steps


def switch_model_to_eval(
    optimizer_state,
    optimizer_graph,
):
    """Switch the model to evaluation mode."""
    opt = nnx.merge(optimizer_graph, optimizer_state)
    model = opt.model
    model.eval()
    optimizer_graph, _ = nnx.split(opt)
    return optimizer_graph


class BaseTrainer(BaseRunner):
    test_loss_enabled = True

    def __init__(
        self,
        use_wandb,
        total_steps,
        should_train,
        restore_step,
        train_log_every_steps,
        test_log_every_steps,
        test_num_batches,
        eval_every_steps,
        train_dataset_config,
        test_dataset_config,
        train_dataloader_config,
        test_dataloader_config,
        checkpoint_config,
        eval_num_denoising_steps,
        save_model_state_to=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total_steps = total_steps
        self.use_wandb = use_wandb
        self.should_train = should_train
        self.eval_num_denoising_steps = eval_num_denoising_steps

        self.train_log_every_steps = train_log_every_steps
        self.test_log_every_steps = test_log_every_steps
        self.eval_every_steps = eval_every_steps

        self.ckpt_mngr, self.restore_step, self.save_every_steps = (
            init_checkpoint_manager(checkpoint_config, restore_step)
        )

        self.train_dataloader_config = train_dataloader_config
        if self.should_train:
            self.train_dataset = instantiate_from_config(train_dataset_config)
            self.test_dataset = instantiate_from_config(test_dataset_config)
            self.train_dataloader, _ = instantiate_from_config(
                train_dataloader_config,
                dataset=self.train_dataset,
                seed_offset=self.restore_step,
            )
            self.test_dataloader, _ = instantiate_from_config(
                test_dataloader_config, dataset=self.test_dataset
            )
            self.test_num_batches = test_num_batches
        self.save_model_state_to = save_model_state_to

    def build_optimizer_from_pretrained_model(
        self,
        *,
        pretrained_model_path,
        network_config,
        optimizer_config,
        lr_scheduler,
        rngs,
        mesh,
        repl_sharding,
    ):
        """Create an nnx.Optimizer and an optimizer-state sharding spec."""

        def get_optimizer(rngs_):
            nnx_rngs = nnx.Rngs(params=rngs_)
            model = instantiate_from_config(network_config, rngs=nnx_rngs)
            tx = instantiate_from_config(
                optimizer_config["optimizer"],
                learning_rate=lr_scheduler,
            )
            optimizer = nnx.Optimizer(model=model, tx=tx)
            optimizer.model.train()
            return nnx.split(optimizer)

        _, state_shape = jax.eval_shape(functools.partial(get_optimizer, rngs_=rngs))
        optimizer_sharding = sharding_utils.apply_sharding(state_shape, mesh)
        optimizer_graph, optimizer_state = jax.jit(
            get_optimizer, out_shardings=(repl_sharding, optimizer_sharding)
        )(rngs)
        optimizer = nnx.merge(optimizer_graph, optimizer_state)
        model_graph, model_state = nnx.split(optimizer.model)
        if pretrained_model_path:
            model_state = self._load_pretrained_model(
                pretrained_model_path, model_state
            )
        optimizer.model = nnx.merge(model_graph, model_state)
        optimizer_graph, optimizer_state = nnx.split(optimizer)
        return optimizer_graph, optimizer_state, optimizer_sharding

    def _load_pretrained_model(self, pretrained_model_path, model_state):
        model_state = self.pretrained_checkpointer.restore(
            pretrained_model_path,
            model_state,
        )
        return model_state

    def save_model_state(self, save_model_state_to):
        optimizer = self._get_model_optimizer()
        model_graph, model_state = nnx.split(optimizer.model)
        self.pretrained_checkpointer.save(save_model_state_to, model_state)
        self.pretrained_checkpointer.wait_until_finished()

    @abc.abstractmethod
    def _restore_step(self, restore_step):
        pass


    @abc.abstractmethod
    def _compile_step_functions(self):
        pass

    @abc.abstractmethod
    def _train_step(
        self,
        i,
        vae_graph,
        vae_state,
        clip_graph,
        clip_state,
        video,
        actions_mouse,
        actions_keyboard,
        rngs,
    ):
        pass

    @abc.abstractmethod
    def _test_step(
        self,
        vae_graph,
        vae_state,
        clip_graph,
        clip_state,
        video,
        actions_mouse,
        actions_keyboard,
        rngs,
    ):
        pass

    @abc.abstractmethod
    def _update_metric_summary(self, summary, step):
        pass

    @abc.abstractmethod
    def _save_checkpoint(self, step):
        pass

    @abc.abstractmethod
    def _get_model_optimizer(self):
        pass

    def run(self):
        self.train_num_frames = self.train_dataloader_config.params.num_frames
        if self.restore_step:
            logging.info(f"Restoring from step {self.restore_step}")
            self._restore_step(self.restore_step)
        if self.save_model_state_to is not None and not self.should_train:
            self.save_model_state(self.save_model_state_to)
            logging.info(f"Checkpoint saved to {self.save_model_state_to}")
            exit()

        self._compile_step_functions()

        # instantiate hooks
        step = 0 if self.restore_step is None else self.restore_step
        metrics_history = defaultdict(list)
        metrics_interval = defaultdict(list)
        train_metrics_last_t = time.time()

        if not self.should_train:
            with self.mesh:
                self.run_evals(step)
            return

        train_loader_iter = iter(self.train_dataloader)

        for i in range(step, self.total_steps):
            t = time.time()

            if jax.process_index() == 0:
                metrics_interval["batch_load_time"].append(time.time() - t)

            video, video_unprocessed, actions_mouse, actions_keyboard, real_lengths = (
                self._get_curr_batch(train_loader_iter)
            )

            with self.mesh:
                # Split models before passing to JIT
                vae_graph, vae_state = nnx.split(self.vae_model)
                clip_graph, clip_state = nnx.split(self.clip_model)

                metric_dict, self.rngs = self._train_step(
                    i,
                    vae_graph,
                    vae_state,
                    clip_graph,
                    clip_state,
                    video,
                    actions_mouse,
                    actions_keyboard,
                    self.rngs,
                )

            for k, v in metric_dict.items():
                metrics_interval[k].append(v)

            if (i + 1) % self.train_log_every_steps == 0:
                for k, v in metrics_interval.items():
                    metrics_history[k].append(sum(v) / len(v))
                metrics_interval = defaultdict(list)
                summary = {
                    f"train_{k}": float(v[-1])  # only log the loss from latest interval
                    for k, v in metrics_history.items()
                }
                summary["steps_per_second"] = self.train_log_every_steps / (
                    time.time() - train_metrics_last_t
                )
                self._update_metric_summary(summary, i)
                summary["step"] = i + 1
                if self.use_wandb:
                    wandb_utils.log_copy(summary, step=i + 1)
                # writer.write_scalars(i + 1, summary)
                formatted_summary = {
                    k: f"{v:.4f}" if isinstance(v, float) else v
                    for k, v in summary.items()
                }
                logging.info(f"Step {i + 1}: {formatted_summary}")

                metrics_history = defaultdict(list)
                train_metrics_last_t = time.time()

            if (i + 1) % self.test_log_every_steps == 0 and self.test_loss_enabled:
                logging.info("Evaluating on test set...")
                test_losses = []
                # Iterate over the whole test dataloader
                self.test_dataloader.batch_sampler.reset_rng()
                test_dataloader_iter = iter(self.test_dataloader)

                rngs_test = self.rngs_test

                for _ in tqdm(
                    range(self.test_num_batches),
                    desc="Evaluating test loss...",
                    disable=jax.process_index() != 0,
                ):
                    video, _, actions_mouse, actions_keyboard, _ = self._get_curr_batch(
                        test_dataloader_iter
                    )
                    with self.mesh:

                        loss_val, rngs_test = self._test_step(
                            vae_graph,
                            vae_state,
                            clip_graph,
                            clip_state,
                            video,
                            actions_mouse,
                            actions_keyboard,
                            rngs_test,
                        )

                    test_losses.append(loss_val)

                mean_test_loss = float(sum(test_losses) / len(test_losses))
                # Log the test loss
                if self.use_wandb:
                    wandb_utils.log_copy(
                        {"test_loss": mean_test_loss, "step": i + 1}, step=i + 1
                    )

                logging.info(f"Step {i+1} test_loss: {mean_test_loss:.4f}")

            if (i + 1) % self.save_every_steps == 0 or i + 1 == self.total_steps:
                self._save_checkpoint(i + 1)
                logging.info(f"Checkpoint saved at step {i}")
                if i == 0:
                    logging.info(f"Checkpoint saved at step {i}")

            if (i + 1) % self.eval_every_steps == 0:
                with self.mesh:
                    self.run_evals(i + 1)
        final_save_path = self.save_model_state_to or os.path.join(self.pretrained_model_dir, f"{self.experiment_name}_{self.total_steps}.pt")
        self.save_model_state(final_save_path)
        logging.info(f"Model state saved to {final_save_path}")

    def run_evals(self, step):

        for eval_dataset_name, eval_dataloader_info in self.eval_dataloaders.items():
            logging.info(f"Running evaluation on {eval_dataset_name} at step {step}")
            eval_dir_name = f"step_{step:07d}_{eval_dataset_name}"
            if self.eval_num_denoising_steps is not None:
                eval_dir_name += f"_denoising_steps_{self.eval_num_denoising_steps}"
            model = self._get_model_optimizer().model
            metric_curve = self.run_eval(
                model=model,
                num_denoising_steps=self.eval_num_denoising_steps,
                eval_dataloader_info=eval_dataloader_info,
                eval_dir_name=eval_dir_name,
            )
            if self.use_wandb and jax.process_index() == 0:
                wandb_suffix = f"_{eval_dataset_name}"
                if self.eval_num_denoising_steps is not None:
                    wandb_suffix += f"_denoising_steps_{self.eval_num_denoising_steps}"
                logging.info(
                    f"Logging metrics to wandb... {wandb_suffix}: {metric_curve}"
                )
                for k, v in metric_curve.items():
                    wandb_utils.log_copy(
                        {f"test_{k}_{wandb_suffix}": v.mean().item()},
                        step=step + 1,
                    )
