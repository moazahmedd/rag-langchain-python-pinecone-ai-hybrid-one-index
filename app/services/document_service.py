from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
from config.settings import Settings
from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
import os
from pathlib import Path
import requests
import tempfile

class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunking_service = ChunkingService(settings)
        self.embedding_service = EmbeddingService(settings)
        self.vector_store_service = VectorStoreService(settings, self.embedding_service)
        
        # Ensure data directory exists
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """
        Ensure the data directory exists
        """
        data_dir = Path(self.settings.PDF_PATH).parent
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found at {data_dir}. "
                "Please create the 'data' directory in the project root."
            )

    def process_and_upload_document(self, namespace: str):
        """
        Process PDF document and upload to vector store
        
        Args:
            namespace: Required namespace to upload vectors to
        """
        if not namespace:
            raise ValueError("Namespace is required for document processing")

        print(f"document_service: Processing and uploading document to namespace: {namespace}")
        # Check if PDF exists
        if not os.path.exists(self.settings.PDF_PATH):
            raise FileNotFoundError(
                f"PDF file not found at {self.settings.PDF_PATH}. "
                "Please ensure the PDF file is in the data directory."
            )

        # Load PDF
        loader = PyPDFLoader(self.settings.PDF_PATH)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.chunking_service.chunk_documents(documents)
        
        # Prepare vectors for upload
        vectors = self.embedding_service.prepare_vectors_for_upload(
            chunks,
            namespace
        )
        
        # Upload to vector store
        self.vector_store_service.upload_vectors(vectors, namespace)

    def process_and_upload_url_document(self, url: str, title: str):
        """
        Process PDF document from URL and upload to vector store with custom namespace
        
        Args:
            url: URL of the PDF document
            title: Title to use as namespace
        """
        if not title:
            raise ValueError("Title is required for URL document processing")

        print(f"document_service: Processing and uploading document from URL: {url} to namespace: {title}")
        
        try:
            # Download PDF to temporary file
            response = requests.get(url)
            response.raise_for_status()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                
                # Split into chunks
                chunks = self.chunking_service.chunk_documents(documents)
                
                # Prepare vectors for upload with custom namespace
                vectors = self.embedding_service.prepare_vectors_for_upload(
                    chunks,
                    title
                )
                
                # Upload to vector store
                self.vector_store_service.upload_vectors(vectors, title)
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
            
        except Exception as e:
            raise Exception(f"Failed to process PDF from URL: {str(e)}")

    def similarity_search(self, query: str, namespace: str, k: int = 30) -> List[Document]:
        """
        Search for similar documents
        
        Args: 
            query: The search query
            namespace: Required namespace to search in
            k: Number of results to return
        """
        if not namespace:
            raise ValueError("Namespace is required for similarity search")
            
        return self.vector_store_service.similarity_search(query, namespace, k=k)

    def delete_document(self, namespace: str) -> None:
        """Delete all documents from a namespace"""
        if not namespace:
            raise ValueError("Namespace is required")
        self.vector_store_service.delete_namespace(namespace)

    def list_namespaces(self) -> List[str]:
        """List all namespaces in the vector store"""
        return self.vector_store_service.list_namespaces() 