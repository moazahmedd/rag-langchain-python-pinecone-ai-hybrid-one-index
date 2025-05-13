from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers import document_router, query_router
from config.settings import get_settings

app = FastAPI(
    title="PDF Q&A API",
    description="API for querying PDF documents using LangChain and Pinecone",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(document_router.router)
app.include_router(query_router.router)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "message": "Service is up and running"
    }

if __name__ == "__main__":
    # Get fresh settings when starting the server
    settings = get_settings()
    print(f"\n🚀 Starting server at http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API Documentation available at http://{settings.HOST}:{settings.PORT}/docs\n")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    ) 