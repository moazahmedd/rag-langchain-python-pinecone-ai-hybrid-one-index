from langchain.chains import create_qa_with_sources_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import Settings
from .llm_service import LLMService
from .vector_store_service import VectorStoreService

class QAService:
    def __init__(self, settings: Settings, llm_service: LLMService, vector_store_service: VectorStoreService):
        self.settings = settings
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service

    def get_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Get answer for a query using the vector store and LLM
        """
        # Get relevant documents
        documents = self.vector_store_service.similarity_search(query, k=k)
        
        # Get structured answer with sources
        response = self.llm_service.get_structured_answer(query, documents)
        
        # Format sources for better readability
        formatted_sources = []
        for source in response["sources"]:
            formatted_source = {
                "source": source.get("source", "Unknown"),
                "page": source.get("page", "N/A"),
                "chunk": source.get("chunk", "N/A")
            }
            formatted_sources.append(formatted_source)
        
        return {
            "answer": response["answer"],
            "sources": formatted_sources
        } 