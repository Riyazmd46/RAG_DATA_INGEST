#!/usr/bin/env python3
"""
Separate startup script for the FastAPI server
This avoids multiprocessing issues that can occur with uvicorn reload
"""

import uvicorn
from main import app
from config import Config

def main():
    # Check if API key is configured
    if not Config.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in a .env file or environment variable.")
        print("Some endpoints may not work without the API key.")
    
    print("Starting RAG Data Pipeline API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main() 