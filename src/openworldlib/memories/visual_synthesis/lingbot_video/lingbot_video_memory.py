from typing import Any, Dict, Optional

from ...base_memory import BaseMemory


class LingBotVideoMemory(BaseMemory):
    """Simple latest-result memory for LingBot-Video generation pipelines."""

    def __init__(self, capacity: int = 8, **kwargs) -> None:
        super().__init__(capacity=capacity, **kwargs)

    def record(self, data: Any, type: str = "other", metadata: Optional[Dict[str, Any]] = None, **kwargs):
        entry = {
            "content": data,
            "type": type,
            "timestamp": kwargs.get("timestamp"),
            "metadata": metadata or {},
        }
        self.storage.append(entry)
        if self.capacity and len(self.storage) > self.capacity:
            self.storage.pop(0)
        return entry

    def select(self, context_query=None, type: Optional[str] = None, **kwargs):
        for entry in reversed(self.storage):
            if type is None or entry["type"] == type:
                return entry["content"]
        return None

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
        return self.storage
