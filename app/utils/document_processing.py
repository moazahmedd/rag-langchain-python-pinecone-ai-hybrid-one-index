from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os
import tempfile
import requests

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def load_pdf_from_path(file_path: str) -> List[Document]:
    """Load PDF from local path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_pdf_from_url(url: str) -> List[Document]:
    """Load PDF from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            return documents
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        raise Exception(f"Failed to process PDF from URL: {str(e)}") 