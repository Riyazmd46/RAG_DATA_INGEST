import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.text_utils import TextProcessor
from utils.image_utils import ImageProcessor
from utils.audio_utils import AudioProcessor
from utils.vector_db_utils import VectorDBManager
from config import Config

class RAGDataPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.text_processor = TextProcessor(self.api_key)
        self.image_processor = ImageProcessor(self.api_key)
        self.audio_processor = AudioProcessor()
        self.vector_db = VectorDBManager()
        
    def get_file_type(self, file_path: str) -> str:
        """Determine the type of file based on extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in Config.SUPPORTED_TEXT_FORMATS:
            return 'text'
        elif ext in Config.SUPPORTED_IMAGE_FORMATS:
            return 'image'
        elif ext in Config.SUPPORTED_AUDIO_FORMATS:
            return 'audio'
        elif ext in Config.SUPPORTED_VIDEO_FORMATS:
            return 'video'
        elif ext in Config.SUPPORTED_DOCUMENT_FORMATS:
            return 'document'
        elif ext in Config.SUPPORTED_DATA_FORMATS:
            return 'data'
        else:
            return 'unknown'
    
    def process_text_file(self, file_path: str, chunk_size: int = None, 
                         chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Process text files and return chunks with metadata."""
        try:
            # Read text content
            text_content = self.text_processor.read_text_file(file_path)
            
            # Chunk the text
            chunks = self.text_processor.chunk_text(
                text_content, chunk_size, chunk_overlap
            )
            
            # Prepare documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'metadata': {
                        'file_path': file_path,
                        'file_name': Path(file_path).name,
                        'file_type': 'text',
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'file_size': os.path.getsize(file_path),
                        'file_extension': Path(file_path).suffix.lower()
                    }
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return []
    
    def process_pdf_file(self, file_path: str, chunk_size: int = None, 
                        chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Process PDF files and return chunks with metadata."""
        try:
            # Extract content from PDF
            pdf_content = self.text_processor.extract_pdf_content(file_path)
            
            documents = []
            
            # Process main text content
            if pdf_content['text']:
                text_chunks = self.text_processor.chunk_text(
                    pdf_content['text'], chunk_size, chunk_overlap
                )
                
                for i, chunk in enumerate(text_chunks):
                    doc = {
                        'content': chunk,
                        'metadata': {
                            'file_path': str(file_path),
                            'file_name': str(Path(file_path).name),
                            'file_type': 'pdf_text',
                            'chunk_index': str(i),
                            'total_chunks': str(len(text_chunks)),
                            'pdf_pages': pdf_content['metadata']['pages'],
                            'pdf_title': pdf_content['metadata']['title'],
                            'pdf_author': pdf_content['metadata']['author'],
                            'pdf_subject': pdf_content['metadata']['subject']
                        }
                    }
                    documents.append(doc)
            
            # Process tables
            for table_idx, table in enumerate(pdf_content['tables']):
                table_summary = self.text_processor.generate_table_summary(
                    table['data'], table['columns']
                )
                
                doc = {
                    'content': table_summary,
                    'metadata': {
                        'file_path': str(file_path),
                        'file_name': str(Path(file_path).name),
                        'file_type': 'pdf_table',
                        'table_index': str(table_idx),
                        'table_columns': str(table['columns']),
                        'pdf_pages': pdf_content['metadata']['pages'],
                        'pdf_title': pdf_content['metadata']['title'],
                        'pdf_author': pdf_content['metadata']['author'],
                        'pdf_subject': pdf_content['metadata']['subject']
                    }
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
            return []
    
    def process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV files with text columns only."""
        try:
            return self.text_processor.process_csv_text_only(file_path)
        except Exception as e:
            print(f"Error processing CSV file {file_path}: {e}")
            return []
    
    def process_image_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process image files and return description with metadata."""
        try:
            image_result = self.image_processor.process_image(file_path)
            
            doc = {
                'content': image_result['description'],
                'metadata': {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': 'image',
                    'file_size': image_result.get('file_size', 0),
                    'image_format': image_result.get('image_format', '')
                }
            }
            
            return [doc]
        except Exception as e:
            print(f"Error processing image file {file_path}: {e}")
            return []
    
    def process_video_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process video files and return frame descriptions and audio transcript."""
        try:
            documents = []
            
            # Process video frames
            video_result = self.image_processor.process_video(file_path)
            if 'frames' in video_result:
                for frame in video_result['frames']:
                    doc = {
                        'content': frame['description'],
                        'metadata': {
                            'file_path': file_path,
                            'file_name': Path(file_path).name,
                            'file_type': 'video_frame',
                            'frame_number': frame['frame_number'],
                            'timestamp': frame['timestamp'],
                            'video_format': video_result.get('video_format', '')
                        }
                    }
                    documents.append(doc)
            
            # Process audio from video
            audio_result = self.audio_processor.process_video_audio(file_path)
            if 'audio_transcription' in audio_result:
                transcript = audio_result['audio_transcription']['transcript']
                doc = {
                    'content': transcript,
                    'metadata': {
                        'file_path': file_path,
                        'file_name': Path(file_path).name,
                        'file_type': 'video_audio',
                        'language': audio_result['audio_transcription'].get('language', 'unknown'),
                        'video_format': audio_result.get('video_format', '')
                    }
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error processing video file {file_path}: {e}")
            return []
    
    def process_audio_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process audio files and return transcription with metadata."""
        try:
            audio_result = self.audio_processor.process_audio_file(file_path)
            
            doc = {
                'content': audio_result['transcription']['transcript'],
                'metadata': {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': 'audio',
                    'language': audio_result['transcription'].get('language', 'unknown'),
                    'file_size': audio_result.get('file_size', 0),
                    'audio_format': audio_result.get('audio_format', '')
                }
            }
            
            return [doc]
        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return []
    
    def process_file(self, file_path: str, chunk_size: int = None, 
                    chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Process a single file based on its type."""
        file_type = self.get_file_type(file_path)
        
        if file_type == 'text':
            return self.process_text_file(file_path, chunk_size, chunk_overlap)
        elif file_type == 'document':
            return self.process_pdf_file(file_path, chunk_size, chunk_overlap)
        elif file_type == 'data':
            return self.process_csv_file(file_path)
        elif file_type == 'image':
            return self.process_image_file(file_path)
        elif file_type == 'video':
            return self.process_video_file(file_path)
        elif file_type == 'audio':
            return self.process_audio_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return []
    
    def ingest_to_vector_db(self, documents: List[Dict[str, Any]]) -> bool:
        """Ingest processed documents into vector database."""
        try:
            if not documents:
                print("No documents to ingest")
                return False
            
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = self.text_processor.get_embeddings(doc['content'])
                if embedding:
                    embeddings.append(embedding)
                else:
                    print(f"Failed to generate embedding for document: {doc.get('metadata', {}).get('file_name', 'unknown')}")
                    return False
            
            # Add to vector database
            success = self.vector_db.add_documents(documents, embeddings)
            return success
            
        except Exception as e:
            print(f"Error ingesting documents to vector database: {e}")
            return False
    
    def process_and_ingest(self, file_path: str, chunk_size: int = None, 
                          chunk_overlap: int = None) -> bool:
        """Process a file and ingest it into the vector database."""
        try:
            print(f"Processing file: {file_path}")
            
            # Process the file
            documents = self.process_file(file_path, chunk_size, chunk_overlap)
            
            if not documents:
                print(f"No documents generated from {file_path}")
                return False
            
            # Ingest to vector database
            success = self.ingest_to_vector_db(documents)
            
            if success:
                print(f"Successfully processed and ingested {file_path}")
            else:
                print(f"Failed to ingest {file_path}")
            
            return success
            
        except Exception as e:
            print(f"Error processing and ingesting {file_path}: {e}")
            return False
    
    def process_directory(self, directory_path: str, chunk_size: int = None, 
                         chunk_overlap: int = None) -> Dict[str, Any]:
        """Process all supported files in a directory."""
        try:
            results = {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'file_results': []
            }
            
            for file_path in Path(directory_path).rglob('*'):
                if file_path.is_file():
                    file_type = self.get_file_type(str(file_path))
                    if file_type != 'unknown':
                        results['total_files'] += 1
                        
                        success = self.process_and_ingest(
                            str(file_path), chunk_size, chunk_overlap
                        )
                        
                        file_result = {
                            'file_path': str(file_path),
                            'file_type': file_type,
                            'success': success
                        }
                        results['file_results'].append(file_result)
                        
                        if success:
                            results['processed_files'] += 1
                        else:
                            results['failed_files'] += 1
            
            return results
            
        except Exception as e:
            print(f"Error processing directory {directory_path}: {e}")
            return {'error': str(e)}
    
    def get_retriever(self):
        """Get a retriever instance for querying the vector database."""
        return RAGRetriever(self.text_processor, self.vector_db)


class RAGRetriever:
    def __init__(self, text_processor: TextProcessor, vector_db: VectorDBManager):
        self.text_processor = text_processor
        self.vector_db = vector_db
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        try:
            # Generate embedding for the query
            query_embedding = self.text_processor.get_embeddings(query)
            
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search in vector database
            results = self.vector_db.search_documents(query, query_embedding, n_results)
            
            return results
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        return self.vector_db.get_collection_stats() 