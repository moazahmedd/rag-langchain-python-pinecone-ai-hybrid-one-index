from fastapi import APIRouter, HTTPException, status
from services.rag_service import RAGService
from config.settings import Settings
from models.schemas import QueryRequest, QueryResponse, ErrorResponse
from typing import Union

router = APIRouter(
    prefix="/api/v1/query",
    tags=["query"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
)

# Initialize service
settings = Settings()
rag_service = RAGService(settings)

@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully retrieved answer"},
        400: {"description": "Invalid request"},
        404: {"description": "Namespace not found"},
        500: {"description": "Internal server error"}
    }
)
async def query_document(request: QueryRequest) -> Union[QueryResponse, ErrorResponse]:
    """
    Query the document with a question.
    
    - **query**: The question to ask about the document (3-500 characters)
    - **k**: Number of relevant chunks to retrieve (1-10, default: 3)
    - **namespace**: Required namespace to search in
    """
    try:
        result = rag_service.query(
            query=request.query,
            namespace=request.namespace,
            k=request.k,
            alpha=request.alpha
        )
        return QueryResponse(**result)
    except ValueError as ve:
        # Handle namespace not found or no results found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 