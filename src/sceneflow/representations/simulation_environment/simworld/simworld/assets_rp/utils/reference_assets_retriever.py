"""This class uses Sentence-BERT and FAISS to retrieve the most relevant assets."""

import faiss
from sentence_transformers import SentenceTransformer

from ...utils.load_json import load_json


class ReferenceAssetsRetriever:
    """Retrieve reference assets from a scene graph using description-based similarity."""
    def __init__(self, progen_world_path: str, description_map_path: str, env_description_retrieval_model_name: str):
        """Initialize relevant modules."""
        self.progen_world_path = progen_world_path
        self.model = SentenceTransformer(env_description_retrieval_model_name)
        self.nodes = self._load_nodes()
        self.instance_desc_map = load_json(description_map_path)
        # pre-compute instance_name embedding of every node
        self.embeddings, self.node_ids = self._precompute_embeddings()
        # construct the FAISS index (dimension = d)
        self.d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.d)              # Inner product (cosine similarity) (normalized vectors)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def _load_nodes(self):
        """Load the nodes (assets) from the world graph JSON file.

        Returns:
            A list of dictionaries, each representing an asset node.
        """
        world_data = load_json(self.progen_world_path)
        return world_data.get('nodes', [])

    def _precompute_embeddings(self):
        """Generate and normalize Sentence-BERT embeddings for all node descriptions.

        Returns:
            A tuple of (embeddings, node_ids), where:
                embeddings: 2D NumPy array of normalized vectors.
                node_ids: List of corresponding instance names.
        """
        instance_names = [node.get('instance_name', '') for node in self.nodes]
        descriptions = [self.instance_desc_map.get(name, name) for name in instance_names]
        embeddings = self.model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        return embeddings, instance_names

    def retrieve_reference_assets(self, reference_asset_query: str, top_k: int = 50, similarity_threshold: float = 0.5):
        """Retrieve top-k matching assets from the scene graph based on a query description.

        Args:
            reference_asset_query: The input query to search for (e.g., 'the nearest hospital').
            top_k: Number of top candidates to retrieve from the index.
            similarity_threshold: Minimum similarity score for a result to be accepted.

        Returns:
            A list of (node, similarity_score) tuples for matching assets, or None if none matched.
        """
        query_embedding = self.model.encode([reference_asset_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        sim_score, index = self.index.search(query_embedding, top_k)    # D: similarity score，I: corresponding index

        candidate_nodes = []
        for score, idx in zip(sim_score[0], index[0]):
            if score >= similarity_threshold:
                candidate_nodes.append((self.nodes[idx], float(score)))
        if not candidate_nodes:
            raise ValueError('Cannot find the reference asset')
        return candidate_nodes
