#!/usr/bin/env python3
"""
Example usage of the RAG Data Pipeline Framework

This script demonstrates how to:
1. Process different types of files
2. Ingest them into a vector database
3. Retrieve relevant documents using the retriever
"""

import os
from data_pipeline import RAGDataPipeline
from config import Config

def main():
    # Initialize the pipeline
    print("Initializing RAG Data Pipeline...")
    pipeline = RAGDataPipeline()
    
    # Example 1: Process a single text file
    print("\n=== Example 1: Processing a text file ===")
    text_file = "sample_data/sample.txt"
    if os.path.exists(text_file):
        success = pipeline.process_and_ingest(
            text_file, 
            chunk_size=500,  # Custom chunk size
            chunk_overlap=50  # Custom overlap
        )
        print(f"Text file processing: {'Success' if success else 'Failed'}")
    
    # Example 2: Process a PDF file
    print("\n=== Example 2: Processing a PDF file ===")
    pdf_file = "sample_data/sample.pdf"
    if os.path.exists(pdf_file):
        success = pipeline.process_and_ingest(pdf_file)
        print(f"PDF file processing: {'Success' if success else 'Failed'}")
    
    # Example 3: Process an image file
    print("\n=== Example 3: Processing an image file ===")
    image_file = "sample_data/sample.jpg"
    if os.path.exists(image_file):
        success = pipeline.process_and_ingest(image_file)
        print(f"Image file processing: {'Success' if success else 'Failed'}")
    
    # Example 4: Process a video file
    print("\n=== Example 4: Processing a video file ===")
    video_file = "sample_data/sample.mp4"
    if os.path.exists(video_file):
        success = pipeline.process_and_ingest(video_file)
        print(f"Video file processing: {'Success' if success else 'Failed'}")
    
    # Example 5: Process an audio file
    print("\n=== Example 5: Processing an audio file ===")
    audio_file = "sample_data/sample.mp3"
    if os.path.exists(audio_file):
        success = pipeline.process_and_ingest(audio_file)
        print(f"Audio file processing: {'Success' if success else 'Failed'}")
    
    # Example 6: Process a CSV file
    print("\n=== Example 6: Processing a CSV file ===")
    csv_file = "sample_data/sample.csv"
    if os.path.exists(csv_file):
        success = pipeline.process_and_ingest(csv_file)
        print(f"CSV file processing: {'Success' if success else 'Failed'}")
    
    # Example 7: Process entire directory
    print("\n=== Example 7: Processing entire directory ===")
    sample_dir = "sample_data"
    if os.path.exists(sample_dir):
        results = pipeline.process_directory(
            sample_dir,
            chunk_size=1000,
            chunk_overlap=200
        )
        print(f"Directory processing results: {results}")
    
    # Example 8: Using the retriever
    print("\n=== Example 8: Using the retriever ===")
    retriever = pipeline.get_retriever()
    
    # Get database stats
    stats = retriever.get_stats()
    print(f"Vector database stats: {stats}")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain data preprocessing",
        "What are the benefits of AI?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Content: {result['content'][:100]}...")
            print(f"  File: {result['metadata'].get('file_name', 'Unknown')}")
            print(f"  Type: {result['metadata'].get('file_type', 'Unknown')}")
            print(f"  Distance: {result['distance']:.4f}")

def create_sample_files():
    """Create sample files for testing if they don't exist."""
    os.makedirs("sample_data", exist_ok=True)
    
    # Create sample text file
    if not os.path.exists("sample_data/sample.txt"):
        with open("sample_data/sample.txt", "w") as f:
            f.write("""
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.

There are three main types of machine learning:
1. Supervised Learning: Uses labeled training data to learn patterns
2. Unsupervised Learning: Finds hidden patterns in unlabeled data
3. Reinforcement Learning: Learns through interaction with an environment

Common applications include:
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles

The process typically involves:
1. Data collection and preprocessing
2. Feature engineering
3. Model selection and training
4. Evaluation and validation
5. Deployment and monitoring
            """)
    
    # Create sample CSV file
    if not os.path.exists("sample_data/sample.csv"):
        with open("sample_data/sample.csv", "w") as f:
            f.write("""name,age,city,occupation
John Doe,30,New York,Engineer
Jane Smith,25,Los Angeles,Designer
Bob Johnson,35,Chicago,Manager
Alice Brown,28,Boston,Developer
Charlie Wilson,32,Seattle,Analyst""")

if __name__ == "__main__":
    # Create sample files for testing
    create_sample_files()
    
    # Check if API key is configured
    if not Config.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in a .env file or environment variable.")
        print("Example .env file content:")
        print("GEMINI_API_KEY=your_api_key_here")
        print("\nContinuing with example (some features may not work without API key)...")
    
    # Run the example
    main() 