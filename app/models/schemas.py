from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class Source(BaseModel):
    page: int = Field(..., description="Page number in the document")
    # content: str = Field(..., description="Content snippet from the document")
    id: str = Field(..., description="Unique identifier for the source")

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the document"
    )
    k: int = Field(default=70, ge=1, le=100, description="Number of relevant chunks to retrieve")
    namespace: str = Field(..., description="Required namespace to search in")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()

    @validator('namespace')
    def validate_namespace(cls, v):
        if not v.strip():
            raise ValueError('Namespace cannot be empty or just whitespace')
        return v.strip().lower().replace(' ', '_')

class QueryResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer to the query")
    sources: List[Source] = Field(..., description="List of sources used to generate the answer")
    # timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")

class DocumentResponse(BaseModel):
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")
    document_id: Optional[str] = Field(None, description="ID of the processed document")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error")

class URLDocumentRequest(BaseModel):
    url: str = Field(..., description="URL of the PDF document to process")
    title: str = Field(..., description="Title of the document to use as namespace")
    namespace: str = Field(..., description="Namespace to upload the document to")

    @validator('url')
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL cannot be empty or just whitespace')
        if not v.lower().endswith('.pdf'):
            raise ValueError('URL must point to a PDF file')
        return v.strip()

    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty or just whitespace')
        # Remove any special characters and spaces for namespace
        return v.strip().lower().replace(' ', '_')
    
    @validator('namespace')
    def validate_namespace(cls, v):
        if not v.strip():
            raise ValueError('Namespace cannot be empty or just whitespace')
        return v.strip().lower().replace(' ', '_')

class DeleteDocumentRequest(BaseModel):
    namespace: str = Field(..., description="Required namespace to delete documents from")

    @validator('namespace')
    def validate_namespace(cls, v):
        if not v.strip():
            raise ValueError('Namespace cannot be empty or just whitespace')
        return v.strip().lower().replace(' ', '_')

class UploadDocumentRequest(BaseModel):
    namespace: str = Field(..., description="Namespace to upload the document to")

    @validator('namespace')
    def validate_namespace(cls, v):
        if not v.strip():
            raise ValueError('Namespace cannot be empty or just whitespace')
        return v.strip().lower().replace(' ', '_')

class NamespaceListResponse(BaseModel):
    namespaces: List[str] = Field(..., description="List of namespaces in the index")
    total: int = Field(..., description="Total number of namespaces")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response") 