import importlib.machinery
import importlib.util
import sys
import types

import torch
import pytest

from openworldlib.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.latent_utils import (
    prepare_initial_latents,
)


SHAPE = (2, 4, 3, 8, 8)


def _install_flash_attn_import_stub(monkeypatch):
    """Allow importing the pipeline on CPU-only test environments."""
    try:
        if importlib.util.find_spec("flash_attn") is not None:
            return
    except (ImportError, ValueError):
        pass

    package = types.ModuleType("flash_attn")
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.flash_attn_interface", loader=None
    )
    interface.flash_attn_varlen_func = lambda *args, **kwargs: None
    package.flash_attn_interface = interface
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)


def test_explicit_latents_are_preserved_and_not_sampled():
    provided = torch.full(SHAPE, 3.5, dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)

    result = prepare_initial_latents(
        SHAPE,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
        latents=provided,
        init_noise_sigma=2.0,
    )

    assert result.dtype == torch.float32
    assert result.device == provided.device
    assert torch.equal(result, torch.full(SHAPE, 7.0, dtype=torch.float32))
    # An explicit latent must not consume the supplied generator.
    expected_next = torch.randn((1,), generator=torch.Generator().manual_seed(7))
    actual_next = torch.randn((1,), generator=generator)
    assert torch.equal(actual_next, expected_next)


def test_missing_latents_use_generator_aware_noise_once():
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    result_a = prepare_initial_latents(
        SHAPE,
        dtype=torch.float32,
        device="cpu",
        generator=generator_a,
        init_noise_sigma=0.5,
    )
    result_b = prepare_initial_latents(
        SHAPE,
        dtype=torch.float32,
        device="cpu",
        generator=generator_b,
        init_noise_sigma=0.5,
    )

    assert torch.equal(result_a, result_b)
    # One latent draw should advance the generator by exactly one randn call.
    expected_generator = torch.Generator().manual_seed(123)
    torch.randn(SHAPE, generator=expected_generator)
    expected_next = torch.randn((1,), generator=expected_generator)
    actual_next = torch.randn((1,), generator=generator_a)
    assert torch.equal(actual_next, expected_next)


def test_latent_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Unexpected latent shape"):
        prepare_initial_latents(
            SHAPE,
            dtype=torch.float32,
            device="cpu",
            latents=torch.zeros((1,) + SHAPE[1:]),
        )


def test_non_tensor_latents_are_rejected():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        prepare_initial_latents(
            SHAPE,
            dtype=torch.float32,
            device="cpu",
            latents=[[0.0]],
        )


def test_pipeline_prepare_latents_preserves_explicit_state(monkeypatch):
    """Exercise the pipeline method with a minimal scheduler and pipeline."""
    _install_flash_attn_import_stub(monkeypatch)
    pipeline_module = importlib.import_module(
        "openworldlib.synthesis.visual_generation.hunyuan_world."
        "hunyuan_game_craft.diffusion.pipelines.pipeline_hunyuan_video_game"
    )
    HunyuanVideoGamePipeline = pipeline_module.HunyuanVideoGamePipeline

    class Scheduler:
        order = 1
        init_noise_sigma = 1.5
        timesteps = torch.arange(4)

        def set_begin_index(self, index):
            self.begin_index = index

    pipeline = HunyuanVideoGamePipeline.__new__(HunyuanVideoGamePipeline)
    pipeline.vae_scale_factor = 2
    pipeline.scheduler = Scheduler()
    provided = torch.ones((1, 2, 2, 4, 4), dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)

    result, timesteps = pipeline.prepare_latents(
        1,
        2,
        4,
        8,
        8,
        2,
        torch.float32,
        torch.device("cpu"),
        None,
        generator,
        provided,
        torch.zeros((1, 2, 1, 4, 4)),
    )

    assert result.shape == provided.shape
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.full_like(result, 1.5))
    assert torch.equal(timesteps, torch.arange(4))

    expected_generator = torch.Generator().manual_seed(5)
    assert torch.equal(
        torch.randn(1, generator=generator),
        torch.randn(1, generator=expected_generator),
    )
