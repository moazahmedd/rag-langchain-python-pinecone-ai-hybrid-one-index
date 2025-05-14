from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from services.rag_service import RAGService
from config.settings import Settings
from pydantic import BaseModel
import os
from typing import List, Union
from models.schemas import DocumentResponse, ErrorResponse, NamespaceListResponse

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
)

class URLUploadRequest(BaseModel):
    url: str
    namespace: str

def get_rag_service():
    """Dependency to get RAG service instance"""
    settings = Settings()
    return RAGService(settings)

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK
)
async def upload_document(
    file: UploadFile = File(None),
    namespace: str = None,
    rag_service: RAGService = Depends(get_rag_service)
) -> Union[DocumentResponse, ErrorResponse]:
    """Upload a document to process and store in vector database"""
    try:
        if file is None:
            # If no file is provided, use the default document from settings
            settings = Settings()
            default_file = settings.PDF_PATH
            if not os.path.exists(default_file):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Default file not found at {default_file}"
                )
            file_path = default_file
            print(f"Using default file: {file_path}")
        else:
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Save uploaded file
            file_path = f"temp/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            print(f"Using uploaded file: {file_path}")
        
        # Process document
        rag_service.process_document(file_path, namespace or "think-and-grow-rich")
        
        # Clean up if it was an uploaded file
        if file is not None and os.path.exists(file_path):
            os.remove(file_path)
        
        return DocumentResponse(
            message=f"Document processed and uploaded to namespace '{namespace or 'think-and-grow-rich'}' successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post(
    "/upload-url",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK
)
async def upload_url(
    request: URLUploadRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Union[DocumentResponse, ErrorResponse]:
    """Upload a document from URL to process and store in vector database"""
    try:
        rag_service.process_url_document(request.url, request.namespace)
        return DocumentResponse(
            message=f"Document from URL processed and uploaded to namespace '{request.namespace}' successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete(
    "/{namespace}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK
)
async def delete_document(
    namespace: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Union[DocumentResponse, ErrorResponse]:
    """Delete all documents from a namespace"""
    try:
        rag_service.delete_document(namespace)
        return DocumentResponse(
            message=f"Documents in namespace '{namespace}' deleted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get(
    "/namespaces",
    response_model=NamespaceListResponse,
    status_code=status.HTTP_200_OK
)
async def list_namespaces(
    rag_service: RAGService = Depends(get_rag_service)
) -> Union[NamespaceListResponse, ErrorResponse]:
    """List all namespaces in the vector store"""
    try:
        namespaces = rag_service.list_namespaces()
        return NamespaceListResponse(
            namespaces=namespaces,
            total=len(namespaces)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 