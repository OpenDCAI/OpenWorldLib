import abc
import functools

import jax
import jax.experimental
import jax.experimental.multihost_utils
import orbax.checkpoint as ocp
from flax import nnx

from src.runners.base_trainer import BaseTrainer, switch_model_to_eval, tree_l2_norm
from src.utils.config import instantiate_from_config


class BaseSSLTrainer(BaseTrainer):
    def __init__(
        self,
        bidirectional,
        noise_type,
        lr_scheduler_config,
        optimizer_config,
        pretrained_model_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bidirectional = bidirectional
        self.noise_type = noise_type
        self.lr_scheduler_config = lr_scheduler_config
        self.optimizer_config = optimizer_config
        self.pretrained_model_path = pretrained_model_path

        self.lr_scheduler = instantiate_from_config(self.lr_scheduler_config)
        self.optimizer_graph, self.optimizer_state, self.optimizer_sharding = (
            self.build_optimizer_from_pretrained_model(
                pretrained_model_path=self.pretrained_model_path,
                network_config=self.network_config,
                optimizer_config=self.optimizer_config,
                lr_scheduler=self.lr_scheduler,
                rngs=self.rngs,
                mesh=self.mesh,
                repl_sharding=self.repl_sharding,
            )
        )

    def _get_model_optimizer(self):
        return nnx.merge(self.optimizer_graph, self.optimizer_state)

    def _restore_step(self, restore_step):
        state_restored = self.ckpt_mngr.restore(
            restore_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(self.optimizer_state),
            ),
        )
        self.optimizer_state = state_restored.state

    def _compile_step_functions(self):
        self.p_train_step = jax.jit(
            functools.partial(
                # Bind positionally so `diffusion_loss_core` is not re-filled by
                # the first positional arg at call time.
                train_loss_step,
                self._diffusion_loss_core,
            ),
            in_shardings=(
                self.optimizer_sharding,  # opt state
                self.repl_sharding,  # vae_state
                self.repl_sharding,
                self.ddp_sharding,
                self.ddp_sharding,
                self.ddp_sharding,
                self.repl_sharding,
            ),
            out_shardings=(
                self.optimizer_sharding,
                self.repl_sharding,
                self.repl_sharding,
            ),
            static_argnames=(
                "vae_graph",
                "clip_graph",
                "opt_graph",
                "noise_type",
                "bidirectional",
                "mesh",
                "left_action_padding",
            ),
            # Donate the optimizer state buffer. Using names is more robust than
            # argnums here because `functools.partial(...)` + signature inference
            # can lead JAX to treat the callable as keyword-only.
            donate_argnames=("opt_state",),
        )
        self.p_test_loss_step = jax.jit(
            functools.partial(
                # Bind positionally so `diffusion_loss_core` is not re-filled by
                # the first positional arg at call time.
                test_loss_step,
                self._diffusion_loss_core,
            ),
            out_shardings=(self.repl_sharding, self.repl_sharding),
            static_argnames=(
                "vae_graph",
                "clip_graph",
                "opt_graph",
                "noise_type",
                "mesh",
                "bidirectional",
                "left_action_padding",
            ),
        )

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
        self.optimizer_state, metric_dict, rngs = self.p_train_step(
            self.optimizer_state,
            self.optimizer_graph,
            vae_state,
            vae_graph,
            clip_state,
            clip_graph,
            video,
            actions_mouse,
            actions_keyboard,
            self.bidirectional,
            self.noise_type,
            self.rngs,
            self.mesh,
            self.left_action_padding,
        )
        return metric_dict, rngs

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
        optimizer_graph_in_eval = switch_model_to_eval(
            self.optimizer_state, self.optimizer_graph
        )
        loss_val, rngs = self.p_test_loss_step(
            self.optimizer_state,
            vae_graph,
            vae_state,
            clip_graph,
            clip_state,
            video,
            actions_mouse,
            actions_keyboard,
            optimizer_graph_in_eval,
            bidirectional=self.bidirectional,
            noise_type=self.noise_type,
            rngs=rngs,
            mesh=self.mesh,
            left_action_padding=self.left_action_padding,
        )
        return loss_val, rngs

    def _update_metric_summary(self, summary, step):
        summary["learning_rate"] = self.lr_scheduler(step)

    def _save_checkpoint(self, step):
        self.ckpt_mngr.save(
            step,
            args=ocp.args.Composite(state=ocp.args.StandardSave(self.optimizer_state)),
        )

    @abc.abstractmethod
    def _diffusion_loss_core(
        self,
        model,
        vae_model,
        clip_model,
        video,
        mouse_actions,
        keyboard_actions,
        *,
        rngs,
        bidirectional,
        noise_type,
        mesh,
        left_action_padding,
    ):
        pass


def train_loss_step(
    diffusion_loss_core,
    opt_state,
    opt_graph,
    vae_state,
    vae_graph,
    clip_state,
    clip_graph,
    video,
    mouse_actions,
    keyboard_actions,
    bidirectional,
    noise_type,
    rngs,
    mesh,
    left_action_padding,
):
    optimizer = nnx.merge(opt_graph, opt_state)
    model = optimizer.model

    vae_model = nnx.merge(vae_graph, vae_state)
    clip_model = nnx.merge(clip_graph, clip_state)

    def loss_fn(m, r):
        return diffusion_loss_core(
            m,
            vae_model,
            clip_model,
            video,
            mouse_actions,
            keyboard_actions,
            rngs=r,
            bidirectional=bidirectional,
            noise_type=noise_type,
            mesh=mesh,
            left_action_padding=left_action_padding,
        )

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    ((loss, rngs), grads) = grad_fn(model, rngs)
    optimizer.update(grads)

    _, new_opt_state = nnx.split(optimizer)
    metric_dict = {
        "loss": loss,
        "grad_norm": tree_l2_norm(grads),
        "param_norm": tree_l2_norm(new_opt_state["model"]),
    }
    return new_opt_state, metric_dict, rngs


def test_loss_step(
    diffusion_loss_core,
    opt_state,
    vae_graph,
    vae_state,
    clip_graph,
    clip_state,
    video,
    mouse_actions,
    keyboard_actions,
    opt_graph,
    bidirectional,
    noise_type,
    rngs,
    mesh,
    left_action_padding,
):
    optimizer = nnx.merge(opt_graph, opt_state)
    model = optimizer.model
    vae_model = nnx.merge(vae_graph, vae_state)
    clip_model = nnx.merge(clip_graph, clip_state)

    loss, rngs = diffusion_loss_core(
        model,
        vae_model,
        clip_model,
        video,
        mouse_actions,
        keyboard_actions,
        rngs=rngs,
        bidirectional=bidirectional,
        noise_type=noise_type,
        mesh=mesh,
        left_action_padding=left_action_padding,
    )
    return loss, rngs
