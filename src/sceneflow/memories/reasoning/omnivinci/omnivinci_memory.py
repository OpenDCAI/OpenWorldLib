"""
OmniVinciMemory for multimodal reasoning tasks
"""
from ...base_memory import BaseMemory
from typing import Optional, List, Dict, Any


class OmniVinciMemory(BaseMemory):
    """
    Memory module for OmniVinci multimodal reasoning tasks.
    Stores conversation history including text, images, audios, and videos.
    """
    
    def __init__(self, capacity: int = 100, **kwargs):
        """
        Initialize OmniVinciMemory
        
        Args:
            capacity: Maximum number of conversation turns to store
        """
        super().__init__(capacity=capacity, **kwargs)
        self.storage = []
        self.conversation_history = []
    
    def record(self, data: Dict[str, Any], **kwargs):
        """
        Record conversation turn to storage
        
        Args:
            data: Dictionary containing:
                - 'messages': List of message dictionaries
                - 'response': Generated response text
                - 'metadata': Optional metadata
        """
        turn_data = {
            'content': data,
            'type': 'conversation',
            'timestamp': len(self.storage),
            'metadata': data.get('metadata', {})
        }
        
        self.storage.append(turn_data)
        
        # Update conversation history
        if 'messages' in data:
            self.conversation_history.extend(data['messages'])
        if 'response' in data:
            self.conversation_history.append({
                'role': 'assistant',
                'content': data['response']
            })
        
        # Apply capacity limit
        if self.capacity and len(self.storage) > self.capacity:
            removed = self.storage.pop(0)
            # Also remove from conversation_history
            if 'messages' in removed['content']:
                msg_count = len(removed['content']['messages'])
                self.conversation_history = self.conversation_history[msg_count:]
    
    def select(self, num_turns: int = -1, **kwargs) -> List[Dict]:
        """
        Select recent conversation history
        
        Args:
            num_turns: Number of recent turns to retrieve (-1 for all)
            
        Returns:
            List of message dictionaries
        """
        if num_turns == -1 or num_turns >= len(self.conversation_history):
            return self.conversation_history.copy()
        
        return self.conversation_history[-num_turns:] if num_turns > 0 else []
    
    def manage(self, action: str = "reset", **kwargs):
        """
        Manage storage lifecycle
        
        Args:
            action: Management action
                - "reset": Clear all storage
                - "clear_old": Remove oldest turn
        """
        if action == "reset":
            self.storage = []
            self.conversation_history = []
        elif action == "clear_old" and len(self.storage) > 0:
            removed = self.storage.pop(0)
            if 'messages' in removed['content']:
                msg_count = len(removed['content']['messages'])
                self.conversation_history = self.conversation_history[msg_count:]
    
    def get_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history.copy()
