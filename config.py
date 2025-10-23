"""
Configuration file for HaleAI Medical Chatbot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Required for Hugging Face API

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

# Pinecone Configuration
PINECONE_INDEX_NAME = "medical-chatbot"
PINECONE_DIMENSION = 384
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Embeddings Configuration
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Splitting Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retriever Configuration
RETRIEVER_K = 5  # Number of documents to retrieve

# LLM Configuration - Using Hugging Face API
# Using a model that works reliably with free API access
LLM_MODEL = "tiiuae/falcon-7b-instruct"

# Alternative free models that work well:
# "google/flan-t5-large" - Fast, reliable, good for Q&A
# "tiiuae/falcon-7b-instruct" - Good general purpose
# "mistralai/Mistral-7B-Instruct-v0.1" - Older Mistral version (if v0.2 doesn't work)

HF_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.15

# Data Paths
DATA_DIR = "data"