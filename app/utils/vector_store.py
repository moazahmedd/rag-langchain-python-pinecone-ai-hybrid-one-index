from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import math
import os
from config.settings import Settings

def hybrid_score_norm(dense: List[float], sparse: Dict[str, List], alpha: float) -> tuple:
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing dense vector
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

class VectorStore:
    def __init__(self, api_key: str, index_name: str):
        """Initialize Pinecone client and index"""
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        
        self.index = self.pc.Index(index_name)
        print(f"Initialized Pinecone index: {index_name}")

    def upload_vectors(self, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 50):
        """Upload vectors to Pinecone in batches"""
        if not vectors:
            return

        if not namespace:
            raise ValueError("Namespace is required for uploading vectors")

        total_batches = math.ceil(len(vectors) / batch_size)
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            self.index.upsert(
                vectors=batch,
                namespace=namespace
            )
            
            print(f"Uploaded batch {current_batch} of {total_batches} to namespace: {namespace}")

    def similarity_search(
        self,
        query_embedding: List[float],
        namespace: str,
        k: int = 3,
        query_sparse_vector: Optional[Dict[str, List]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not namespace:
            raise ValueError("Namespace is required for similarity search")

        # Prepare query parameters
        query_params = {
            "namespace": namespace,
            "top_k": k,
            "include_metadata": True,
            "vector": query_embedding
        }
        
        # If sparse vector is provided, use hybrid search
        if query_sparse_vector is not None:
            # Normalize the hybrid scores
            normalized_dense, normalized_sparse = hybrid_score_norm(
                dense=query_embedding,
                sparse=query_sparse_vector,
                alpha=alpha
            )
            query_params["vector"] = normalized_dense
            query_params["sparse_vector"] = normalized_sparse

        # Perform the search

        print(f"Query parameters: {query_params}")
        results = self.index.query(
            **query_params
        )
        
        return results.matches

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors from a namespace"""
        if not namespace:
            raise ValueError("Namespace is required")
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted namespace: {namespace}")

    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            print(f"Found {len(namespaces)} namespaces: {namespaces}")
            return namespaces
        except Exception as e:
            print(f"Error listing namespaces: {str(e)}")
            raise 