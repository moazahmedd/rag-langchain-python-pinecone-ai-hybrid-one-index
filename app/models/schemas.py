from pydantic import BaseModel, Field, validator
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the document"
    )
    k: int = Field(default=3, ge=1, le=100, description="Number of relevant chunks to retrieve")
    namespace: str = Field(..., description="Required namespace to search in")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Alpha value for BM25")

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

class Source(BaseModel):
    content: str
    id: str
    page: Optional[str] = "N/A"
    page_label: Optional[str] = "N/A"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

class DocumentResponse(BaseModel):
    message: str

class NamespaceListResponse(BaseModel):
    namespaces: List[str]
    total: int

class ErrorResponse(BaseModel):
    detail: str 