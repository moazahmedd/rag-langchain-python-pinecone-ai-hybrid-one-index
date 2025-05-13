from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from typing import List, Dict, Any
import math
from config.settings import Settings
from .embedding_service import EmbeddingService

class VectorStoreService:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=embedding_service.embeddings
        )

    def upload_vectors(self, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 50):
        """
        Upload vectors to Pinecone in batches
        
        Args:
            vectors: List of vectors to upload
            namespace: Required namespace to upload vectors to
            batch_size: Size of each batch for upload
        """
        if not vectors:
            return

        if not namespace:
            raise ValueError("Namespace is required for uploading vectors")

        print(f"vector_store_service: Uploading vectors to namespace: {namespace}")
        total_batches = math.ceil(len(vectors) / batch_size)
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            # Upload batch to Pinecone using the provided namespace
            self.index.upsert(
                vectors=batch,
                namespace=namespace
            )
            
            print(f"Uploaded batch {current_batch} of {total_batches} to namespace: {namespace}")

    def similarity_search(self, query: str, namespace: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents using the query
        
        Args:
            query: The search query
            namespace: Required namespace to search in
            k: Number of results to return
        """
        if not namespace:
            raise ValueError("Namespace is required for similarity search")

        print(f"vector_store_service: Searching in namespace: {namespace}")
        return self.vectorstore.similarity_search(query, k=k, namespace=namespace)

    def delete_namespace(self, namespace: str) -> None:
        """Delete a namespace from the vector store"""
        if not namespace:
            raise ValueError("Namespace is required")
        self.index.delete(delete_all=True, namespace=namespace)

    def list_namespaces(self) -> List[str]:
        """List all namespaces in the vector store"""
        try:
            # Get index statistics which includes namespace information
            stats = self.index.describe_index_stats()
            # Extract namespaces from the stats
            namespaces = list(stats.get('namespaces', {}).keys())
            return namespaces
        except Exception as e:
            print(f"Error listing namespaces: {str(e)}")
            raise 