"""
Configuration file for HaleAI Medical Chatbot
Centralized configuration management with validation
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class with validation"""
    
    # API Keys
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "") #second arg is default value if env var is not set
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = "medical-chatbot"
    PINECONE_DIMENSION: int = 384
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text Splitting Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    
    # Retriever Configuration
    RETRIEVER_K: int = 8  # Initial retrieval count
    RERANK_TOP_K: int = 3  # Final top documents after reranking
    
    # Gemini Configuration
    GEMINI_MODEL: str = "gemini-2.0-flash"  # Latest Gemini Flash model
    GEMINI_TEMPERATURE: float = 0.3  # Lower for more consistent medical responses
    GEMINI_MAX_OUTPUT_TOKENS: int = 1024
    GEMINI_TOP_P: float = 0.8
    GEMINI_TOP_K: int = 40
    
    # Safety Settings
    ENABLE_SAFETY_FILTERS: bool = True
    
    # Data Paths
    DATA_DIR: Path = Path("data")
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        errors = []
        
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY not found in environment variables")
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY not found in environment variables")
        
        # Create data directory if it doesn't exist (don't treat as error)
        if not cls.DATA_DIR.exists():
            try:
                cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created data directory: {cls.DATA_DIR}")
            except Exception as e:
                errors.append(f"Could not create data directory: {cls.DATA_DIR} - {e}")
        
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    @classmethod
    def setup_environment(cls) -> None:
        """Setup environment variables"""
        os.environ["PINECONE_API_KEY"] = cls.PINECONE_API_KEY
        os.environ["GEMINI_API_KEY"] = cls.GEMINI_API_KEY

# Validate configuration on import
Config.validate()
Config.setup_environment()

# Export commonly used values
PINECONE_API_KEY = Config.PINECONE_API_KEY
GEMINI_API_KEY = Config.GEMINI_API_KEY
PINECONE_INDEX_NAME = Config.PINECONE_INDEX_NAME
PINECONE_DIMENSION = Config.PINECONE_DIMENSION
PINECONE_METRIC = Config.PINECONE_METRIC
PINECONE_CLOUD = Config.PINECONE_CLOUD
PINECONE_REGION = Config.PINECONE_REGION
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP
RETRIEVER_K = Config.RETRIEVER_K
DATA_DIR = str(Config.DATA_DIR)