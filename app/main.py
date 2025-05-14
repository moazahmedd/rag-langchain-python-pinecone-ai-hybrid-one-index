from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import document_router, query_router
from config.settings import Settings
import uvicorn

# Initialize settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="PDF Q&A API",
    description="API for querying PDF documents using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router.router)
app.include_router(query_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to PDF Q&A API"} 

if __name__ == "__main__":
    # Get fresh settings when starting the server
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    ) 