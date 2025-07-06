import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Model Settings
    GEMINI_MODEL = "gemini-1.5-flash"  # Faster and more reliable
    GEMINI_VISION_MODEL = "gemini-1.5-flash"  # Supports both text and vision
    GROQ_MODEL = "llama2-70b-4096"
    
    # Vector DB Settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    CHROMA_COLLECTION_NAME = "rag_documents"
    
    # Chunking Default Settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
    
    # Video Processing Settings
    VIDEO_FPS = 1  # Extract 1 frame per second
    
    # Audio Processing Settings
    WHISPER_MODEL = "base"
    
    # File Processing Settings
    SUPPORTED_TEXT_FORMATS = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf']
    SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.xls']
    
    # Output Settings
    MAX_SUMMARY_LENGTH = 500
    MAX_DESCRIPTION_LENGTH = 300 