from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...base_memory import BaseMemory

ContextQuery = Union[Dict[str, Any], str, None]


class SimWorldMemory(BaseMemory):

    TYPE_LIST = {"image", "video", "text", "audio", "action", "other"}

    def __init__(self, capacity: Optional[int] = None, **kwargs):
        super().__init__(capacity=capacity, **kwargs)
        self._episode_meta: Dict[str, Any] = {}

    # ---------------- 1. record ----------------
    def record(self, data, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        if metadata is None:
            metadata = {}

        t = str(metadata.get("type", "other"))
        if t not in self.TYPE_LIST:
            metadata = dict(metadata)
            metadata["type_original"] = t
            metadata["type"] = "other"
            t = "other"

        item = {
            "content": data,
            "type": t,
            "timestamp": time.time(),
            "metadata": metadata,
        }
        self.storage.append(item)

        if self.capacity is not None and len(self.storage) > int(self.capacity):
            overflow = len(self.storage) - int(self.capacity)
            if overflow > 0:
                self.storage = self.storage[overflow:]

    # ---------------- 2. select ----------------
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

        flt = context_query.get("filter", None)
        if callable(flt):
            items = [it for it in items if bool(flt(it))]

        last_n = context_query.get("last_n", None)
        if isinstance(last_n, (int, float)):
            n = max(0, int(last_n))
            if n > 0:
                items = items[-n:]

        return items

    # ---------------- 3. compress ----------------
    def compress(self, memory_items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        return memory_items

    # ---------------- 4. process ----------------
    def process(self, refined_data, target_format: str = "kv_cache", **kwargs):
        if target_format != "export":
            return None

        items = self.select(refined_data) if refined_data is not None else list(self.storage)
        items = self.compress(items)

        from collections import defaultdict
        agent_frames: dict = defaultdict(list)
        agent_positions: dict = defaultdict(list)
        agent_directions: dict = defaultdict(list)
        agent_collisions: dict = defaultdict(list)
        agent_actions: dict = defaultdict(list)

        for it in items:
            t = it.get("type", "other")
            md = it.get("metadata", {}) or {}
            content = it.get("content", None)
            idx = md.get("agent_idx", 0)

            if t == "image":
                if isinstance(content, dict):
                    rgb = content.get("rgb")
                    if isinstance(rgb, np.ndarray):
                        agent_frames[idx].append(rgb)
                elif isinstance(content, np.ndarray):
                    agent_frames[idx].append(content)

            elif t == "action":
                rec = {"timestamp": float(it.get("timestamp", 0.0)), **md}
                if isinstance(content, dict) and "action" not in rec:
                    rec["action"] = content
                agent_actions[idx].append(rec)

            elif t == "other":
                subtype = md.get("subtype", "")
                if subtype == "position":
                    agent_positions[idx].append({
                        "tick": md.get("tick", None),
                        "timestamp": float(it.get("timestamp", 0.0)),
                        "position": content,
                    })
                if subtype == "direction":
                    agent_directions[idx].append({
                        "tick": md.get("tick", None),
                        "timestamp": float(it.get("timestamp", 0.0)),
                        "direction": content,
                    })
                if subtype == "collision":
                    agent_collisions[idx].append({
                        "tick": md.get("tick", None),
                        "timestamp": float(it.get("timestamp", 0.0)),
                        "collision": content,
                    })

        return {
            # 向后兼容：agent 0 保持原字段名
            "frames_rgb": agent_frames.get(0, []),
            "positions": agent_positions.get(0, []),
            "directions": agent_directions.get(0, []),
            "collisions": agent_collisions.get(0, []),
            "actions": agent_actions.get(0, []),
            # 按 agent 分组
            "agent_frames": dict(agent_frames),
            "agent_positions": dict(agent_positions),
            "agent_directions": dict(agent_directions),
            "agent_collisions": dict(agent_collisions),
            "agent_actions": dict(agent_actions),
            "meta": dict(self._episode_meta),
        }

    # ---------------- 5. manage ----------------
    def manage(self, **kwargs):
        action = str(kwargs.get("action", "reset")).lower().strip()

        if action == "reset":
            self.storage = []
            self._episode_meta = {}
            return

        if action == "set_meta":
            meta = kwargs.get("meta", None)
            if isinstance(meta, dict):
                self._episode_meta.update(meta)
            return

        if action == "close":
            return
        