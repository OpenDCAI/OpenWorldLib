from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...base_memory import BaseMemory

ContextQuery = Union[Dict[str, Any], str, None]


class Ai2ThorMemory(BaseMemory):
    
    TYPE_LIST = {"image", "video", "text", "audio", "action", "other"}
    
    def __init__(self, capacity: Optional[int] = None, **kwargs):
        super().__init__(capacity=capacity, **kwargs)
        self._episode_meta: Dict[str, Any] = {}
        self._tick_count: int = 0
        self._action_count: int = 0
        
        self._interaction_history: List[List[str]] = []

    # ---------------- 1. record (ingestion) ----------------
    def record(self, data, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        if metadata is None:
            metadata = {}

        t = str(metadata.get("type", "other"))
        if t not in self.TYPE_LIST:
            t = "other"
            metadata = dict(metadata)
            metadata["type"] = "other"
            metadata["type_original"] = str(metadata.get("type", ""))

        item = {
            "content": data,
            "type": t,
            "timestamp": time.time(),
            "metadata": metadata,
        }
        self.storage.append(item)

        # capacity FIFO eviction
        if self.capacity is not None and len(self.storage) > int(self.capacity):
            overflow = len(self.storage) - int(self.capacity)
            if overflow > 0:
                self.storage = self.storage[overflow:]

    # ---------------- 2. select (retrieval) ----------------
    def select(self, context_query: ContextQuery = None, **kwargs) -> List[Dict[str, Any]]:
        if len(self.storage) == 0:
            return []

        if context_query is None:
            return list(self.storage)

        if isinstance(context_query, str):
            q = context_query.lower().strip()
            if q == "all":
                return list(self.storage)
            if q == "last_image":
                return self.select({"type": "image", "last_n": 1})
            if q == "last_action":
                return self.select({"type": "action", "last_n": 1})
            return list(self.storage)

        if not isinstance(context_query, dict):
            return list(self.storage)

        items = list(self.storage)

        q_type = context_query.get("type", None)
        if isinstance(q_type, str):
            items = [it for it in items if it.get("type") == q_type.strip()]

        since_time = context_query.get("since_time", None)
        if isinstance(since_time, (int, float)):
            st = float(since_time)
            items = [it for it in items if float(it.get("timestamp", 0.0)) >= st]

        # custom filter
        flt = context_query.get("filter", None)
        if callable(flt):
            items = [it for it in items if bool(flt(it))]

        last_n = context_query.get("last_n", None)
        if isinstance(last_n, (int, float)):
            n = max(0, int(last_n))
            if n > 0:
                items = items[-n:]

        return items

    # ---------------- 3. compress (refinement) ----------------
    def compress(self, memory_items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        return memory_items

    # ---------------- 4. process (adaptation) ----------------
    def process(self, refined_data, target_format: str = "kv_cache", **kwargs):
        if target_format != "export":
            return None

        items = self.select(refined_data) if refined_data is not None else list(self.storage)
        items = self.compress(items)

        frames_rgb = []
        depth_frames = []
        instance_frames = []
        instance_payloads = []

        actions = []

        for it in items:
            t = it.get("type", "other")
            md = it.get("metadata", {}) or {}
            content = it.get("content", None)

            if t == "image":
                if isinstance(content, np.ndarray):
                    frames_rgb.append(content)
                elif isinstance(content, dict):
                    if isinstance(content.get("frame"), np.ndarray):
                        frames_rgb.append(content["frame"])
                    if isinstance(content.get("depth_frame"), np.ndarray):
                        depth_frames.append(content["depth_frame"])
                    if isinstance(content.get("instance_segmentation_frame"), np.ndarray):
                        instance_frames.append(content["instance_segmentation_frame"])

                    # masks/detections2D 可能不是 ndarray（dict/None），单独收集
                    if ("instance_masks" in content) or ("instance_detections2D" in content):
                        instance_payloads.append({
                            "tick": md.get("tick", None),
                            "instance_masks": content.get("instance_masks", None),
                            "instance_detections2D": content.get("instance_detections2D", None),
                        })

            elif t == "action":
                rec = {"timestamp": float(it.get("timestamp", 0.0)), **md}
                if isinstance(content, dict) and "action" not in rec:
                    rec["action"] = content
                actions.append(rec)

        return {
            "frames_rgb": frames_rgb,
            "depth_frames": depth_frames,
            "instance_segmentation_frames": instance_frames,
            "instance_payloads": instance_payloads,
            "actions": actions,
            "meta": dict(self._episode_meta),
        }


    # ---------------- 5. manage (lifecycle) ----------------
    def manage(self, **kwargs):
        action = str(kwargs.get("action", "reset")).lower().strip()

        if action == "reset":
            self.storage = []
            self._episode_meta = {}
            self._tick_count = 0
            self._action_count = 0
            self._interaction_history = []
            return

        if action == "set_meta":
            meta = kwargs.get("meta", None)
            if isinstance(meta, dict):
                self._episode_meta.update(meta)
            return

        if action == "close":
            return

    # ------- counters / histories -------
    def bump_tick(self) -> int:
        self._tick_count += 1
        return self._tick_count

    def bump_action(self) -> int:
        self._action_count += 1
        return self._action_count

    def push_interaction(self, tokens: List[str]):
        self._interaction_history.append([str(t) for t in tokens])

    @property
    def tick_count(self) -> int:
        return int(self._tick_count)

    @property
    def action_count(self) -> int:
        return int(self._action_count)
