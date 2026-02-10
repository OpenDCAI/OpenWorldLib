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

    # ---------------- 1. record (ingestion) ----------------
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
            "content": data,                 # 关键：完整保存 content（dict/ndarray/any）
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

        frames_rgb: List[np.ndarray] = []
        depth_frames: List[np.ndarray] = []
        instance_frames: List[np.ndarray] = []
        instance_payloads: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []

        for it in items:
            t = it.get("type", "other")
            md = it.get("metadata", {}) or {}
            content = it.get("content", None)

            if t == "image":
                # 1) content 直接是 ndarray（当 rgb frame）
                if isinstance(content, np.ndarray):
                    frames_rgb.append(content)

                # 2) content 是 dict，按 pipeline 的 payload 结构拆
                elif isinstance(content, dict):
                    fr = content.get("frame", None)
                    if isinstance(fr, np.ndarray):
                        frames_rgb.append(fr)

                    d = content.get("depth_frame", None)
                    if isinstance(d, np.ndarray):
                        depth_frames.append(d)

                    inst = content.get("instance_segmentation_frame", None)
                    if isinstance(inst, np.ndarray):
                        instance_frames.append(inst)

                    # masks/detections2D：不是 ndarray，也要导出去（json-friendly）
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

            elif t == "other":
                if isinstance(md, dict) and md.get("subtype") == "instance_payload":
                    if isinstance(content, dict):
                        instance_payloads.append({"tick": md.get("tick", None), **content})
                    else:
                        instance_payloads.append({"tick": md.get("tick", None), "payload": content})

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
            return

        if action == "set_meta":
            meta = kwargs.get("meta", None)
            if isinstance(meta, dict):
                self._episode_meta.update(meta)
            return

        if action == "close":
            return
        