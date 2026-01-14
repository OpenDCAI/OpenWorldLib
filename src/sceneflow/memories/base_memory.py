class BaseMemory(object):
    """
    Generic Multimodal Memory System Template (BaseMemory)
    Designed for VLM, VLA, and Generative/Reasoning tasks.
    """

    def __init__(self, capacity=None, **kwargs):
        """
        Initialize storage and configurations.
        Purpose: Define storage structures (e.g., Vector DB, Ring Buffer, Hierarchical Cache) 
                 and resource constraints.
        """
        self.storage = []
        self.capacity = capacity
        pass

    def check_template(self, **kwargs):
        """
        the template of self.storage is [{'content':..., 'type':<type>, 'timestamp': <timestamp>, 'metadata': <metadata>}, {}, ...]
        """
        key_list = ['content', 'type', 'timestamp', 'metadata']           ## metadata contain corresponding extra information
        type_list = ['image', 'video', 'text', 'audio', 'action', 'other']

    def record(self, data, metadata=None, **kwargs):
        """
        1. Recording (Ingestion)
        Purpose: Ingest raw interaction data (images, actions, text, depth, etc.).
        Logic: Handles the initial entry of data streams and assigns necessary metadata tags.
        """
        pass

    def select(self, context_query, **kwargs):
        """
        2. Selection (Retrieval)
        Purpose: Filter relevant memory snippets based on the current task context or goal.
        Logic: Implements similarity matching, temporal correlation, or importance-based filtering.
        """
        pass

    def compress(self, memory_items, **kwargs):
        """
        3. Compression (Refinement)
        Purpose: Reduce dimensionality or distill selected memories.
        Logic: Summarizes long-form text or extracts key visual features to minimize computational overhead.
        """
        pass

    def process(self, refined_data, target_format="kv_cache", **kwargs):
        """
        4. Processing (Adaptation)
        Purpose: Convert refined memories into model-ready formats.
        Logic: Bridges the gap between memory storage and model input (e.g., KV Cache, Latent tokens).
        """
        pass

    def manage(self, **kwargs):
        """
        5. Management (Lifecycle & Consolidation)
        Purpose: Maintain the long-term health of the memory system.
        Logic: Handles memory merging, eviction of stale data (forgetting), or 
               transferring Short-Term Memory (STM) to Long-Term Memory (LTM).
        """
        pass
