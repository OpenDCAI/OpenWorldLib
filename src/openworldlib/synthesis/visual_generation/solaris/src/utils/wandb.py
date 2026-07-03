"""File containing helper functions for wandb logging."""

# built-in libs
import hashlib

# external libs
import jax
import wandb


def is_main_process():
    return jax.process_index() == 0


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode("utf-8")).hexdigest(), 16) % 10**8)


def log_copy(x, step=None):
    if is_main_process():
        wandb.log(
            {k: v for k, v in x.items()}, step=step
        )  # Make a copy so garbarge doesn't get inserted
