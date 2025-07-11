#!/usr/bin/env python3
"""
FastAPI Main Application for RAG Data Pipeline

This FastAPI application provides REST API endpoints for:
- Processing and ingesting files
- Querying the vector database
- Managing the pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path
import uvicorn

from data_pipeline import RAGDataPipeline
from config import Config

# Directory to save uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploaded_files')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Data Pipeline API",
    description="A comprehensive API for processing multi-modal data and building RAG systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline
pipeline = RAGDataPipeline()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    num_results: int = 5

class ProcessRequest(BaseModel):
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class ProcessResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    file_type: str
    chunks_created: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    persist_directory: str

class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    vector_db_accessible: bool

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the API and its dependencies."""
    try:
        # Check if API key is configured
        api_key_configured = bool(Config.GEMINI_API_KEY)
        
        # Check if vector database is accessible
        retriever = pipeline.get_retriever()
        stats = retriever.get_stats()
        vector_db_accessible = bool(stats)
        
        return HealthResponse(
            status="healthy" if api_key_configured and vector_db_accessible else "degraded",
            api_key_configured=api_key_configured,
            vector_db_accessible=vector_db_accessible
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            api_key_configured=bool(Config.GEMINI_API_KEY),
            vector_db_accessible=False
        )
    

# from typing import List, Dict, Optional
# from pydantic import BaseModel

# class CollectionStats(BaseModel):
#     collection_name: str
#     total_documents: int
#     files: List[str] = []

# class StatsResponse(BaseModel):
#     total_collections: int
#     persist_directory: Optional[str] = None
#     collections: List[CollectionStats]


# @app.get("/stats", response_model=StatsResponse)
# async def get_stats():
#     """Get statistics about the vector database."""
#     try:
#         retriever = pipeline.get_retriever()
#         chroma_client = retriever.chroma_client  # Or however you access it
#         persist_directory = retriever.persist_directory

#         collections = chroma_client.list_collections()

#         results = []
#         for col_meta in collections:
#             collection = chroma_client.get_collection(name=col_meta.name)
#             data = collection.get()
#             total_docs = len(data["ids"])
#             files = list({m.get("source") for m in data["metadatas"] if m and "source" in m})

#             results.append(CollectionStats(
#                 collection_name=col_meta.name,
#                 total_documents=total_docs,
#                 files=files
#             ))

#         return StatsResponse(
#             total_collections=len(collections),
#             persist_directory=persist_directory,
#             collections=results
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.get("/stats")
def get_stats():
    try:
        retriever = pipeline.get_retriever()
        stats = retriever.get_stats()
        return stats
    except Exception as e:
        return {"error": f"Error retrieving stats: {e}"}

# Process single file endpoint
@app.post("/process/file", response_model=ProcessResponse)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Process and ingest a single file into the vector database."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 100MB)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        # Save file to working directory
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(file.file, out_file)
        
        # Process the file
        success, documents_count = pipeline.process_and_ingest(
            save_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if success:
            return ProcessResponse(
                success=True,
                message="File processed and ingested successfully",
                file_name=file.filename,
                file_type=pipeline.get_file_type(save_path),
                chunks_created= documents_count
            )
        else:
            return ProcessResponse(
                success=False,
                message="File processing failed",
                file_name=file.filename,
                file_type=pipeline.get_file_type(save_path)
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Process multiple files endpoint
@app.post("/process/files", response_model=List[ProcessResponse])
async def process_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Process and ingest multiple files into the vector database."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        results = []
        
        for file in files:
            if not file.filename:
                continue
            
            # Check file size
            if file.size and file.size > 100 * 1024 * 1024:
                results.append(ProcessResponse(
                    success=False,
                    message="File too large (max 100MB)",
                    file_name=file.filename,
                    file_type="unknown"
                ))
                continue
            
            # Save file to working directory
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(save_path, 'wb') as out_file:
                shutil.copyfileobj(file.file, out_file)
            
            # Process the file
            success = pipeline.process_and_ingest(
                save_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            results.append(ProcessResponse(
                success=success,
                message="File processed successfully" if success else "File processing failed",
                file_name=file.filename,
                file_type=pipeline.get_file_type(save_path)
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Query the vector database for relevant documents."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        retriever = pipeline.get_retriever()
        results = retriever.retrieve(request.query, n_results=request.num_results)
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database: {str(e)}")

# Get supported file types endpoint
@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "text_formats": Config.SUPPORTED_TEXT_FORMATS,
        "image_formats": Config.SUPPORTED_IMAGE_FORMATS,
        "audio_formats": Config.SUPPORTED_AUDIO_FORMATS,
        "video_formats": Config.SUPPORTED_VIDEO_FORMATS,
        "document_formats": Config.SUPPORTED_DOCUMENT_FORMATS,
        "data_formats": Config.SUPPORTED_DATA_FORMATS
    }

# Clear database endpoint
@app.delete("/clear")
async def clear_database():
    """Clear all documents from the vector database."""
    try:
        vector_db = pipeline.vector_db
        success = vector_db.delete_collection()
        
        if success:
            return {"message": "Database cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear database")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Data Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health - Check API health",
            "stats": "GET /stats - Get database statistics",
            "process_file": "POST /process/file - Process single file",
            "process_files": "POST /process/files - Process multiple files",
            "query": "POST /query - Query the database",
            "supported_formats": "GET /supported-formats - Get supported file types",
            "clear": "DELETE /clear - Clear database"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    # Check if API key is configured
    if not Config.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in a .env file or environment variable.")
        print("Some endpoints may not work without the API key.")
    
    # Run the FastAPI application
    try:
        uvicorn.run(
            app,  # Use app directly instead of string
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid multiprocessing issues
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")