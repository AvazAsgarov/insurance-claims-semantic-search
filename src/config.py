import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global configuration variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "insurance-claims"
EMBEDDING_MODEL = "text-embedding-3-small"

def validate_config():
    """Validate that all required environment variables are set."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from .env file.")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing from .env file.")
    print("Configuration validated successfully.")
