from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class LocalSearchService:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        # paraphrase-multilingual-MiniLM-L12-v2 is excellent for multilingual e-commerce tasks
        # It handles 50+ languages including Vietnamese and English
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a piece of text locally using a Transformer model."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def compute_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1_arr = np.array(v1)
        v2_arr = np.array(v2)
        dot_product = np.dot(v1_arr, v2_arr)
        norm_v1 = np.linalg.norm(v1_arr)
        norm_v2 = np.linalg.norm(v2_arr)
        return dot_product / (norm_v1 * norm_v2)
