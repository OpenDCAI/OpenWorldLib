from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ...base_memory import BaseMemory


class RollingForcingMemory(BaseMemory):
    """Short-term video/latent memory for Rolling Forcing streaming."""

    def __init__(self, max_latent_context_frames: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.storage: List[Dict[str, Any]] = []
        self.latest_video = None
        self.latest_latents: Optional[torch.Tensor] = None
        self.max_latent_context_frames = int(max_latent_context_frames)

    def record(self, data, metadata=None, **kwargs):
        metadata = metadata or {}
        item_type = metadata.get("type", "video")
        timestamp = len(self.storage)

        if item_type == "latents" and isinstance(data, torch.Tensor):
            latents = data.detach().cpu()
            self.latest_latents = self._truncate_latents(latents)
        elif item_type == "video":
            self.latest_video = data

        self.storage.append(
            {
                "content": data,
                "type": item_type,
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )

    def select(self, context_query=None, **kwargs):
        item_type = kwargs.get("type")
        if item_type == "latents":
            if self.latest_latents is None:
                return None
            return self.latest_latents.clone() if kwargs.get("copy", True) else self.latest_latents
        if item_type == "video":
            return self.latest_video
        if len(self.storage) == 0:
            return None
        if item_type is None:
            return self.storage[-1]["content"]
        for item in reversed(self.storage):
            if item["type"] == item_type:
                return item["content"]
        return None

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self.latest_video = None
            self.latest_latents = None
        elif action == "set_context_frames":
            self.max_latent_context_frames = int(kwargs["max_latent_context_frames"])
            if self.latest_latents is not None:
                self.latest_latents = self._truncate_latents(self.latest_latents)
        else:
            raise ValueError(f"Unsupported RollingForcing memory action: {action}")

    def _truncate_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.max_latent_context_frames <= 0:
            return latents
        if latents.ndim >= 5 and latents.shape[1] > self.max_latent_context_frames:
            return latents[:, -self.max_latent_context_frames:].contiguous()
        return latents
