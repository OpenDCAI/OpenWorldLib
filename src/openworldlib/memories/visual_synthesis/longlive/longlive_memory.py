from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ...base_memory import BaseMemory


class LongLiveMemory(BaseMemory):
    """Memory records for LongLive streaming prompt/video state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage: List[Dict[str, Any]] = []
        self.latest_video = None
        self.latest_latents: Optional[torch.Tensor] = None

    def record(self, data, metadata=None, **kwargs):
        metadata = metadata or {}
        item_type = metadata.get("type", "video")
        timestamp = len(self.storage)

        if item_type == "latents" and isinstance(data, torch.Tensor):
            self.latest_latents = data.detach().cpu()
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
        if len(self.storage) == 0:
            return None
        item_type = kwargs.get("type")
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
