from langchain_openai import OpenAIEmbeddings
from typing import List

def get_embeddings(texts: List[str], model: str, api_key: str) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    return embeddings.embed_documents(texts) 