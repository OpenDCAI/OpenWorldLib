import time
from typing import Any, Dict, List, Optional, Union
from collections import deque

from ...base_memory import BaseMemory


class SpiritV1p5Memory(BaseMemory):
    """
    Memory system for Spirit-VLA pipeline.
    
    Handles storage and retrieval of:
    - Observation history (images, states)
    - Action history
    - Task/interaction history
    
    Supports temporal correlation and importance-based filtering.
    """
    
    ALLOWED_TYPES = ["image", "video", "text", "audio", "action", "state", "task", "other"]
    
    def __init__(
        self,
        capacity: int = 100,
        enable_compression: bool = False,
        **kwargs,
    ):
        """
        Initialize Spirit-VLA Memory.
        
        Args:
            capacity: Maximum number of entries to store
            enable_compression: Whether to enable memory compression
            **kwargs: Additional configuration
        """
        super().__init__(capacity=capacity, **kwargs)
        self.storage: deque = deque(maxlen=capacity)
        self.enable_compression = enable_compression
        self.short_term_buffer: List[Dict] = []
        self.long_term_storage: List[Dict] = []
    
    def check_template(self, entry: Dict) -> bool:
        """
        Validate memory entry format.
        
        Args:
            entry: Memory entry to validate
        
        Returns:
            True if entry is valid
        
        Raises:
            ValueError: If entry format is invalid
        """
        required_keys = ["content", "type", "timestamp"]
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"Memory entry missing required key: {key}")
        
        if entry["type"] not in self.ALLOWED_TYPES:
            raise ValueError(f"Invalid type: {entry['type']}. Allowed: {self.ALLOWED_TYPES}")
        
        return True
    
    def record(
        self,
        data: Any,
        metadata: Optional[Dict] = None,
        data_type: str = "other",
        **kwargs,
    ):
        """
        Record new data into memory.
        
        Args:
            data: Data to record (can be any type)
            metadata: Additional metadata
            data_type: Type of data being recorded
            **kwargs: Additional arguments
        """
        timestamp = time.time()
        
        entry = {
            "content": data,
            "type": data_type,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        
        self.check_template(entry)
        
        # Add to short-term buffer first
        self.short_term_buffer.append(entry)
        
        # Transfer to main storage
        self.storage.append(entry)
    
    def select(
        self,
        context_query: Optional[str] = None,
        data_type: Optional[str] = None,
        time_window: Optional[float] = None,
        max_entries: int = 10,
        **kwargs,
    ) -> List[Dict]:
        """
        Retrieve relevant memory entries.
        
        Args:
            context_query: Text query for semantic matching (not implemented)
            data_type: Filter by data type
            time_window: Only return entries within this time window (seconds)
            max_entries: Maximum number of entries to return
            **kwargs: Additional filters
        
        Returns:
            List of matching memory entries
        """
        results = []
        current_time = time.time()
        
        for entry in reversed(self.storage):
            # Filter by type
            if data_type and entry["type"] != data_type:
                continue
            
            # Filter by time window
            if time_window:
                if current_time - entry["timestamp"] > time_window:
                    continue
            
            results.append(entry)
            
            if len(results) >= max_entries:
                break
        
        return results
    
    def compress(
        self,
        memory_items: List[Dict],
        **kwargs,
    ) -> List[Dict]:
        """
        Compress memory entries to reduce storage.
        
        Currently implements simple downsampling for action sequences.
        
        Args:
            memory_items: Items to compress
            **kwargs: Compression options
        
        Returns:
            Compressed memory items
        """
        if not self.enable_compression:
            return memory_items
        
        compressed = []
        action_buffer = []
        
        for item in memory_items:
            if item["type"] == "action":
                action_buffer.append(item)
            else:
                # Flush action buffer with downsampling
                if action_buffer:
                    # Keep every other action
                    compressed.extend(action_buffer[::2])
                    action_buffer = []
                compressed.append(item)
        
        # Flush remaining actions
        if action_buffer:
            compressed.extend(action_buffer[::2])
        
        return compressed
    
    def process(
        self,
        refined_data: List[Dict],
        target_format: str = "list",
        **kwargs,
    ) -> Any:
        """
        Convert memory entries to target format.
        
        Args:
            refined_data: Memory entries to process
            target_format: Output format ("list", "dict", "tensor")
            **kwargs: Processing options
        
        Returns:
            Processed data in target format
        """
        if target_format == "list":
            return [entry["content"] for entry in refined_data]
        
        elif target_format == "dict":
            result = {}
            for entry in refined_data:
                key = f"{entry['type']}_{entry['timestamp']}"
                result[key] = entry["content"]
            return result
        
        elif target_format == "tensor":
            # For action sequences
            import torch
            actions = [
                entry["content"] 
                for entry in refined_data 
                if entry["type"] == "action"
            ]
            if actions:
                return torch.tensor(actions)
            return None
        
        return refined_data
    
    def manage(self, **kwargs):
        """
        Manage memory lifecycle and consolidation.
        
        Handles:
        - Short-term to long-term memory transfer
        - Memory eviction when capacity exceeded
        - Periodic compression
        """
        # Transfer old short-term memories to long-term
        current_time = time.time()
        stm_threshold = kwargs.get("stm_threshold", 60.0)  # 60 seconds
        
        items_to_transfer = []
        remaining_stm = []
        
        for item in self.short_term_buffer:
            age = current_time - item["timestamp"]
            if age > stm_threshold:
                items_to_transfer.append(item)
            else:
                remaining_stm.append(item)
        
        self.short_term_buffer = remaining_stm
        
        # Compress before transferring to long-term
        if items_to_transfer:
            compressed = self.compress(items_to_transfer)
            self.long_term_storage.extend(compressed)
        
        # Evict old long-term memories if exceeding capacity
        ltm_capacity = kwargs.get("ltm_capacity", self.capacity * 2)
        if len(self.long_term_storage) > ltm_capacity:
            # Keep most recent
            self.long_term_storage = self.long_term_storage[-ltm_capacity:]
    
    def get_action_history(self, max_steps: int = 50) -> List[Any]:
        """
        Retrieve recent action history.
        
        Args:
            max_steps: Maximum number of actions to retrieve
        
        Returns:
            List of recent actions
        """
        action_entries = self.select(data_type="action", max_entries=max_steps)
        return self.process(action_entries, target_format="list")
    
    def get_observation_history(self, max_frames: int = 10) -> List[Dict]:
        """
        Retrieve recent observation history.
        
        Args:
            max_frames: Maximum number of observations to retrieve
        
        Returns:
            List of observation dictionaries
        """
        results = []
        for entry in reversed(self.storage):
            if entry["type"] in ["image", "state"]:
                results.append(entry)
                if len(results) >= max_frames:
                    break
        return results
    
    def clear(self):
        """Clear all memory."""
        self.storage.clear()
        self.short_term_buffer.clear()
        self.long_term_storage.clear()
    
    def __len__(self) -> int:
        """Return total number of stored entries."""
        return len(self.storage)
