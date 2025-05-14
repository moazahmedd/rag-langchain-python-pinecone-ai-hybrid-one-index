from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import Settings
from utils.document_processing import chunk_documents, load_pdf_from_path, load_pdf_from_url
from utils.embeddings import get_embeddings
from utils.vector_store import VectorStore
from pinecone_text.sparse import BM25Encoder
import json
import os

class RAGService:
    def __init__(self, settings: Settings):
        print("\n=== Initializing RAG Service ===")
        self.settings = settings
        print(f"Using embedding model: {settings.EMBEDDING_MODEL}")
        print(f"Using LLM model: {settings.LLM_MODEL}")
        
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        self.vector_store = VectorStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME
        )
        
        # Initialize BM25 encoder
        self.bm25 = BM25Encoder.default()
        print("RAG Service initialized successfully\n")

    def _get_bm25_vectors(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate BM25 sparse vectors for texts using Pinecone's BM25Encoder"""
        # Encode texts to sparse vectors
        sparse_vectors = self.bm25.encode_documents(texts)
        return sparse_vectors

    def process_document(self, file_path: str, namespace: str) -> None:
        """Process and upload document to vector store"""
        print(f"\n=== Processing Document ===")
        print(f"File path: {file_path}")
        print(f"Namespace: {namespace}")
        
        # Load document
        print("\n1. Loading document...")
        documents = load_pdf_from_path(file_path)
        print(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        print("\n2. Splitting into chunks...")
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("\n3. Generating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings = get_embeddings(
            texts,
            self.settings.EMBEDDING_MODEL,
            self.settings.OPENAI_API_KEY
        )
        print(f"Generated {len(embeddings)} embeddings")
        
        # Generate BM25 sparse vectors
        print("\n4. Generating BM25 sparse vectors...")
        sparse_vectors = self._get_bm25_vectors(texts)
        print(f"Generated {len(sparse_vectors)} sparse vectors")
        
        # Prepare vectors with embeddings and sparse vectors
        print("\n5. Preparing vectors...")
        vectors = []
        for i, (doc, embedding, sparse_vector) in enumerate(zip(chunks, embeddings, sparse_vectors)):
            vector = {
                "id": f"{namespace}#chunk{i+1}",
                "values": embedding,
                "sparse_values": sparse_vector,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", None),
                    "page_label": doc.metadata.get("page_label", None),
                }
            }
            vectors.append(vector)
        print(f"Prepared {len(vectors)} vectors")
        
        # Upload vectors
        print("\n6. Uploading vectors to Pinecone...")
        self.vector_store.upload_vectors(vectors, namespace)
        print("Document processing completed successfully\n")

    def process_url_document(self, url: str, namespace: str) -> None:
        """Process and upload document from URL to vector store"""
        print(f"\n=== Processing URL Document ===")
        print(f"URL: {url}")
        print(f"Namespace: {namespace}")
        
        # Load document from URL
        print("\n1. Loading document from URL...")
        documents = load_pdf_from_url(url)
        print(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        print("\n2. Splitting into chunks...")
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("\n3. Generating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings = get_embeddings(
            texts,
            self.settings.EMBEDDING_MODEL,
            self.settings.OPENAI_API_KEY
        )
        print(f"Generated {len(embeddings)} embeddings")
        
        # Generate BM25 sparse vectors
        print("\n4. Generating BM25 sparse vectors...")
        sparse_vectors = self._get_bm25_vectors(texts)
        print(f"Generated {len(sparse_vectors)} sparse vectors")
        
        # Prepare vectors with embeddings and sparse vectors
        print("\n5. Preparing vectors...")
        vectors = []
        for i, (doc, embedding, sparse_vector) in enumerate(zip(chunks, embeddings, sparse_vectors)):
            vector = {
                "id": f"{namespace}#chunk{i+1}",
                "values": embedding,
                "sparse_values": sparse_vector,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", None),
                    "page_label": doc.metadata.get("page_label", None),
                }
            }
            vectors.append(vector)
        print(f"Prepared {len(vectors)} vectors")
        
        # Upload vectors
        print("\n6. Uploading vectors to Pinecone...")
        self.vector_store.upload_vectors(vectors, namespace)
        print("URL document processing completed successfully\n")

    def query(self, query: str, namespace: str, k: int = 3, alpha: float = 0.5) -> Dict[str, Any]:
        """Query the document and get answer"""
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        print(f"Namespace: {namespace}")
        print(f"k: {k}")
        print(f"alpha: {alpha}")
        
        # Check if namespace exists
        print("\n1. Checking namespace...")
        available_namespaces = self.list_namespaces()
        print(f"Available namespaces: {available_namespaces}")
        if namespace not in available_namespaces:
            raise ValueError(f"Namespace '{namespace}' does not exist. Available namespaces: {available_namespaces}")
            
        # Generate query embedding
        print("\n2. Generating query embedding...")
        query_embedding = self.embeddings.embed_query(query)
        print("Query embedding generated")
        
        # Generate BM25 sparse vector for query
        print("\n3. Generating BM25 sparse vector...")
        query_sparse_vector = self.bm25.encode_queries([query])[0]
        print(f"Query sparse vector generated")
        
        # Get similar documents
        print("\n4. Searching for similar documents...")
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            query_sparse_vector=query_sparse_vector,
            namespace=namespace,
            k=k,
            alpha=alpha
        )
        print(f"Found {len(results)} similar documents")
        
        if not results:
            raise ValueError(f"No results found in namespace '{namespace}'")
        
        # Prepare context from results
        print("\n5. Preparing context...")
        context = "\n\n".join([match.metadata["text"] for match in results])
        print(f"Context length: {len(context)} characters")
        
        # Generate answer
        print("\n6. Generating answer...")
        response = self.llm.invoke([
            {
                "role": "system",
                "content": """You are a helpful AI assistant that answers questions based ONLY on the provided context. 
                Follow these rules strictly:
                1. ONLY use information from the provided context to answer the question
                2. If the context doesn't contain relevant information to answer the question, try to:
                   a. Look for related concepts or ideas in the context
                   b. Check if the question can be answered from a different angle using the available information
                   c. Only if no relevant information is found, respond with "I cannot answer this question as the provided context does not contain relevant information."
                3. DO NOT make up or add information that is not present in the context
                4. DO NOT provide general knowledge or information outside the context
                5. If the context is about a specific topic (e.g., a book), mention that your answer is based on that specific source
                6. Keep your answer concise and focused on the information from the context
                7. If the question is general, try to find the most relevant information in the context that could help answer it"""
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            }
        ])
        print("Answer generated")
        
        # Prepare response
        print("\n7. Preparing response...")
        response_data = {
            "answer": response.content,
            "sources": [
                {
                    "id": match.id,
                    "content": match.metadata["text"],
                    "page": str(match.metadata.get("page", "N/A")),
                    "page_label": str(match.metadata.get("page_label", "N/A"))
                }
                for match in results
            ]
        }
        print("Response prepared")
        print("\nQuery processing completed successfully\n")
        
        return response_data

    def delete_document(self, namespace: str) -> None:
        """Delete all documents from a namespace"""
        print(f"\n=== Deleting Document ===")
        print(f"Namespace: {namespace}")
        self.vector_store.delete_namespace(namespace)
        print("Document deleted successfully\n")

    def list_namespaces(self) -> List[str]:
        """List all namespaces in the vector store"""
        print("\n=== Listing Namespaces ===")
        namespaces = self.vector_store.list_namespaces()
        print(f"Found {len(namespaces)} namespaces: {namespaces}\n")
        return namespaces 