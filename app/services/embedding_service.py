from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any
import numpy as np
from config.settings import Settings

class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        return self.embeddings.embed_documents(texts)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        return self.embeddings.embed_query(text)

    def prepare_vectors_for_upload(
        self, 
        documents: List[Document], 
        namespace: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for vector store upload by generating embeddings
        """
        print("embedding_service: Preparing vectors for upload...")
        texts = [doc.page_content for doc in documents]
        embeddings = self.get_embeddings(texts)
        
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector = {
                "id": f"{namespace}#chunk{i+1}",
                "values": embedding,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", None),
                    "page_label": doc.metadata.get("page_label", None),
                }
            }
            vectors.append(vector)
            
        return vectors 