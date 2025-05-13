from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from config.settings import Settings

class ChunkingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        """

        print("chunking_service: Chunking documents...")
        return self.text_splitter.split_documents(documents) 