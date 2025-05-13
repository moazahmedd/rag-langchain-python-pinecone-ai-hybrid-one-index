from fastapi import APIRouter, HTTPException, status
from services.document_service import DocumentService
from config.settings import Settings
from models.schemas import DocumentResponse, ErrorResponse, URLDocumentRequest, UploadDocumentRequest, NamespaceListResponse
from typing import Union
import json

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
)

# Initialize services
settings = Settings()
document_service = DocumentService(settings)

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Document successfully processed and uploaded"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def upload_document(request: UploadDocumentRequest) -> Union[DocumentResponse, ErrorResponse]:
    """
    Upload and process a PDF document.
    
    The document should be placed in the configured path before calling this endpoint.
    
    Request body must include:
    - namespace: Namespace to upload the document to
    """
    try:
        print(f"document_router: Uploading document to namespace: {request.namespace}")
        document_service.process_and_upload_document(request.namespace)
        return DocumentResponse(
            message=f"Document processed and uploaded successfully to namespace: {request.namespace}",
            document_id=request.namespace
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found in the configured path"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post(
    "/upload-url",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Document successfully processed and uploaded"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def upload_url_document(request: URLDocumentRequest) -> Union[DocumentResponse, ErrorResponse]:
    """
    Upload and process a PDF document from a URL.
    
    The document will be processed and stored in a namespace based on the provided title.
    """
    try:
        print("document_router: Uploading document from URL...")
        document_service.process_and_upload_url_document(request.url, request.title)
        return DocumentResponse(
            message=f"Document processed and uploaded successfully to namespace: {request.title}",
            document_id=request.title
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Documents successfully deleted"},
        404: {"description": "No documents found to delete"},
        500: {"description": "Internal server error"}
    }
)
async def delete_document(namespace: str) -> Union[DocumentResponse, ErrorResponse]:
    """
    Delete all documents from the specified namespace in the vector store.
    
    Args:
        namespace: The namespace to delete documents from
    """
    try:
        print(f"document_router: Deleting documents from namespace: {namespace}")
        document_service.delete_document(namespace)
        return DocumentResponse(
            message=f"Documents in namespace '{namespace}' deleted successfully"
        )
    except Exception as e:
        error_str = str(e)
        try:
            # Try to parse the error message if it contains JSON
            if "HTTP response body:" in error_str:
                json_str = error_str.split("HTTP response body:")[1].strip()
                error_data = json.loads(json_str)
                message = error_data.get("message", "Unknown error")
                details = error_data.get("details", [])
                error_detail = f"{message}. Details: {', '.join(details) if details else 'No additional details'}"
            else:
                error_detail = error_str
        except:
            error_detail = error_str

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

@router.get(
    "/namespaces",
    response_model=NamespaceListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully retrieved list of namespaces"},
        500: {"description": "Internal server error"}
    }
)
async def list_namespaces() -> Union[NamespaceListResponse, ErrorResponse]:
    """
    List all namespaces in the vector store index.
    
    Returns:
        List of namespaces and total count
    """
    try:
        print("document_router: Listing all namespaces")
        namespaces = document_service.list_namespaces()
        return NamespaceListResponse(
            namespaces=namespaces,
            total=len(namespaces)
        )
    except Exception as e:
        error_str = str(e)
        try:
            # Try to parse the error message if it contains JSON
            if "HTTP response body:" in error_str:
                json_str = error_str.split("HTTP response body:")[1].strip()
                error_data = json.loads(json_str)
                message = error_data.get("message", "Unknown error")
                details = error_data.get("details", [])
                error_detail = f"{message}. Details: {', '.join(details) if details else 'No additional details'}"
            else:
                error_detail = error_str
        except:
            error_detail = error_str

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        ) 