from fastapi import APIRouter, HTTPException, status
from services.document_service import DocumentService
from services.llm_service import LLMService
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

# Initialize services
settings = Settings()
document_service = DocumentService(settings)
llm_service = LLMService(settings)

@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully retrieved answer"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def query_document(request: QueryRequest) -> Union[QueryResponse, ErrorResponse]:
    """
    Query the document with a question.
    
    - **query**: The question to ask about the document (3-500 characters)
    - **k**: Number of relevant chunks to retrieve (1-10, default: 3)
    - **namespace**: Optional namespace to search in (defaults to configured namespace)
    """
    try:
        # Get relevant chunks from vector store
        results = document_service.similarity_search(
            request.query, 
            k=request.k,
            namespace=request.namespace
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant content found for the query"
            )
        
        # Get structured answer with sources
        response = llm_service.get_structured_answer(request.query, results)
        
        return QueryResponse(**response)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 