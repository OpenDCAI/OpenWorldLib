"""Hunyuan World Play package."""

import importlib

__all__ = [
    "hunyuan_worldplay",
    "HunyuanWorldPlaySynthesis",
    "_HunyuanWorldPlayInternalPipeline",
]


def __getattr__(name):
    if name == "hunyuan_worldplay":
        return importlib.import_module(f"{__name__}.hunyuan_worldplay")
    if name in {"HunyuanWorldPlaySynthesis", "_HunyuanWorldPlayInternalPipeline"}:
        module = importlib.import_module(f"{__name__}.hunyuan_worldplay_synthesis")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
