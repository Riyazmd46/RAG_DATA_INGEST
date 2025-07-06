#!/usr/bin/env python3
"""
Command Line Interface for RAG Data Pipeline

Usage:
    python cli.py process <file_or_directory> [options]
    python cli.py query <query> [options]
    python cli.py stats
"""

import argparse
import sys
import os
from pathlib import Path
from data_pipeline import RAGDataPipeline
from config import Config

def process_command(args):
    """Process files or directories and ingest into vector database."""
    pipeline = RAGDataPipeline()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{args.path}' does not exist.")
        return 1
    
    if path.is_file():
        print(f"Processing file: {path}")
        success = pipeline.process_and_ingest(
            str(path),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        if success:
            print("File processed successfully!")
        else:
            print("File processing failed!")
            return 1
    else:
        print(f"Processing directory: {path}")
        results = pipeline.process_directory(
            str(path),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if 'error' in results:
            print(f"Directory processing failed: {results['error']}")
            return 1
        
        print(f"Directory processing completed!")
        print(f"   Total files: {results['total_files']}")
        print(f"   Processed: {results['processed_files']}")
        print(f"   Failed: {results['failed_files']}")
    
    return 0

def query_command(args):
    """Query the vector database."""
    pipeline = RAGDataPipeline()
    retriever = pipeline.get_retriever()
    
    print(f"Querying: {args.query}")
    results = retriever.retrieve(args.query, n_results=args.num_results)
    
    if not results:
        print("No results found.")
        return 0
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {result['content'][:200]}...")
        print(f"  File: {result['metadata'].get('file_name', 'Unknown')}")
        print(f"  Type: {result['metadata'].get('file_type', 'Unknown')}")
        print(f"  Similarity: {1 - result['distance']:.4f}")
        print()

def stats_command(args):
    """Show vector database statistics."""
    pipeline = RAGDataPipeline()
    retriever = pipeline.get_retriever()
    
    stats = retriever.get_stats()
    
    if not stats:
        print("Could not retrieve statistics.")
        return 1
    
    print("Vector Database Statistics:")
    print(f"   Collection: {stats.get('collection_name', 'Unknown')}")
    print(f"   Total Documents: {stats.get('total_documents', 0)}")
    print(f"   Storage Path: {stats.get('persist_directory', 'Unknown')}")

def main():
    parser = argparse.ArgumentParser(
        description="RAG Data Pipeline - Process and query multi-modal data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py process document.pdf
  python cli.py process documents/ --chunk-size 500 --chunk-overlap 50
  python cli.py query "What is machine learning?"
  python cli.py query "Explain neural networks" --num-results 10
  python cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process files or directories')
    process_parser.add_argument('path', help='File or directory path to process')
    process_parser.add_argument('--chunk-size', type=int, default=1000,
                               help='Text chunk size (default: 1000)')
    process_parser.add_argument('--chunk-overlap', type=int, default=200,
                               help='Text chunk overlap (default: 200)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the vector database')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--num-results', type=int, default=5,
                             help='Number of results to return (default: 5)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check API key
    if not Config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in a .env file or environment variable.")
        return 1
    
    try:
        if args.command == 'process':
            return process_command(args)
        elif args.command == 'query':
            return query_command(args)
        elif args.command == 'stats':
            return stats_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 