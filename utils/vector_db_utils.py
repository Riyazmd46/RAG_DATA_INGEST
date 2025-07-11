import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
from config import Config
from pathlib import Path

class VectorDBManager:
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Add documents to the vector database."""
        try:
            if not documents or not embeddings:
                print("No documents or embeddings provided")
                return False
            
            # Prepare data for ChromaDB
            ids = [f"{Path(doc.get('metadata', {}).get('file_name', 'file'))}_chunk_{i}" for i, doc in enumerate(documents)]
            texts = [doc['content'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(documents)} documents to vector database")
            return True, len(documents)
            
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
            return False
    
    def search_documents(self, query: str, query_embedding: List[float], 
                        n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector database."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection."""
        try:
            count = self.collection.count()

            # Pull all documents (limit=max to inspect metadata)
            results = self.collection.get(include=["metadatas"], limit=100_000)

            metadatas = results.get("metadatas", [])
            file_paths = [meta.get("file_path") for meta in metadatas if meta.get("file_path")]

            unique_files = set(file_paths)
            chunks_per_file = {}
            for path in file_paths:
                chunks_per_file[path] = chunks_per_file.get(path, 0) + 1

            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_documents": count,
                "total_files": len(unique_files),
                "files": list(unique_files),
                "chunks_per_file": chunks_per_file
            }

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any], 
                       embedding: List[float]) -> bool:
        """Update a specific document in the collection."""
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document from the collection."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False