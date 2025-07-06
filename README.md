# RAG Data Pipeline Framework

A comprehensive, production-ready framework for building Retrieval-Augmented Generation (RAG) systems that can process and ingest multiple data types into a vector database for semantic search and retrieval.

## Features

### Multi-Modal Data Processing
- **Text Files**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.json`
- **Documents**: PDF files with text, image, and table extraction
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`
- **Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv` (1fps frame extraction + audio)
- **Data**: CSV files with text columns

### AI-Powered Processing
- **Gemini Models**: Text generation, image description, table summarization
- **Whisper**: Audio transcription (open-source)
- **Configurable Chunking**: Recursive text splitting with customizable parameters
- **Vector Embeddings**: Gemini embedding model for semantic search

### Vector Database
- **ChromaDB**: Persistent vector storage with cosine similarity
- **Metadata Management**: Rich metadata for each document chunk
- **Efficient Retrieval**: Fast semantic search with configurable results

### Production-Ready Architecture
- **Modular Design**: Separate utilities for different data types
- **Error Handling**: Robust error handling and logging
- **Configuration Management**: Centralized configuration with environment variables
- **Extensible**: Easy to add new data types and processing methods

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio/video processing)

### Install FFmpeg

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Set Up API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here  # Optional
```

## Quick Start

### Basic Usage

```python
from data_pipeline import RAGDataPipeline

# Initialize the pipeline
pipeline = RAGDataPipeline()

# Process a single file
success = pipeline.process_and_ingest("path/to/your/file.pdf")

# Process entire directory
results = pipeline.process_directory("path/to/your/documents/")

# Get retriever for querying
retriever = pipeline.get_retriever()
results = retriever.retrieve("What is machine learning?", n_results=5)
```

### Run Example

```bash
python example.py
```

## Detailed Usage

### Text Processing

```python
# Process text file with custom chunking
documents = pipeline.process_text_file(
    "document.txt",
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

### PDF Processing

```python
# Process PDF with text, images, and tables
documents = pipeline.process_pdf_file(
    "document.pdf",
    chunk_size=800,
    chunk_overlap=150
)
```

### Image Processing

```python
# Process image and generate description
documents = pipeline.process_image_file("image.jpg")
```

### Video Processing

```python
# Process video (extracts frames at 1fps + audio transcription)
documents = pipeline.process_video_file("video.mp4")
```

### Audio Processing

```python
# Process audio file
documents = pipeline.process_audio_file("audio.mp3")
```

### CSV Processing

```python
# Process CSV with text columns (each row as chunk)
documents = pipeline.process_csv_file("data.csv")
```

### Vector Database Operations

```python
from utils.vector_db_utils import VectorDBManager

# Initialize vector database
vector_db = VectorDBManager(
    persist_directory="./my_vector_db",
    collection_name="my_documents"
)

# Add documents
success = vector_db.add_documents(documents, embeddings)

# Search documents
results = vector_db.search_documents(query, query_embedding, n_results=5)

# Get statistics
stats = vector_db.get_collection_stats()
```

## Configuration

### Default Settings

All configuration is managed in `config.py`:

```python
class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Model Settings
    GEMINI_MODEL = "gemini-pro"
    GEMINI_VISION_MODEL = "gemini-pro-vision"
    
    # Vector DB Settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    CHROMA_COLLECTION_NAME = "rag_documents"
    
    # Chunking Default Settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    
    # Video Processing Settings
    VIDEO_FPS = 1  # Extract 1 frame per second
    
    # Audio Processing Settings
    WHISPER_MODEL = "base"
```

### Custom Configuration

```python
# Override default settings
pipeline = RAGDataPipeline(api_key="your_custom_key")
vector_db = VectorDBManager(
    persist_directory="./custom_db",
    collection_name="custom_collection"
)
```

## Architecture

### Project Structure

```
rag-pipeline/
├── config.py                 # Configuration management
├── data_pipeline.py          # Main pipeline orchestrator
├── example.py               # Usage examples
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── utils/
    ├── text_utils.py       # Text and PDF processing
    ├── image_utils.py      # Image and video processing
    ├── audio_utils.py      # Audio processing
    └── vector_db_utils.py  # Vector database operations
```

### Core Components

1. **RAGDataPipeline**: Main orchestrator that coordinates all processing
2. **TextProcessor**: Handles text files, PDFs, and CSV processing
3. **ImageProcessor**: Processes images and extracts video frames
4. **AudioProcessor**: Handles audio files and video audio extraction
5. **VectorDBManager**: Manages ChromaDB operations
6. **RAGRetriever**: Provides retrieval interface for queries

## Advanced Features

### Custom Chunking Strategies

```python
# Custom separators for different content types
custom_separators = ["\n\n", "\n", ". ", " ", ""]
chunks = text_processor.chunk_text(
    text,
    chunk_size=500,
    chunk_overlap=50,
    separators=custom_separators
)
```

### Batch Processing

```python
# Process multiple files with progress tracking
import os
from pathlib import Path

files = list(Path("documents/").glob("**/*.pdf"))
for file_path in files:
    print(f"Processing {file_path}")
    success = pipeline.process_and_ingest(str(file_path))
    print(f"Result: {'Success' if success else 'Failed'}")
```

### Error Handling

```python
try:
    success = pipeline.process_and_ingest("file.pdf")
    if not success:
        print("Processing failed, check logs for details")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Management
- Large files are processed in chunks to manage memory usage
- Temporary files are automatically cleaned up
- Vector embeddings are generated incrementally

### Processing Speed
- Video processing can be slow for large files (1fps extraction)
- Audio transcription depends on file size and Whisper model
- PDF processing speed depends on document complexity

### Optimization Tips
- Use appropriate chunk sizes for your use case
- Process files in batches for better resource management
- Consider using faster Whisper models for audio processing

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
2. **API key errors**: Check your `.env` file and API key validity
3. **Memory issues**: Reduce chunk sizes or process files individually
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
pipeline = RAGDataPipeline()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example code
3. Open an issue on GitHub

## Roadmap

- [ ] Support for more document formats (DOCX, PPTX)
- [ ] Advanced chunking strategies
- [ ] Multi-language support
- [ ] Cloud storage integration
- [ ] Real-time processing capabilities
- [ ] Web interface for management 