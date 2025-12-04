"""
Embeddings Module
Handles vector embeddings for RAG (Retrieval-Augmented Generation).
Supports multiple embedding providers with fallback options.
"""

import os
import sys
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config


class EmbeddingModel:
    """Manages embedding model initialization and operations"""
    
    def __init__(self, provider: str = "huggingface"):
        """
        Initialize embedding model.
        
        Args:
            provider: Embedding provider ('huggingface', 'openai', or 'google')
        """
        self.provider = provider
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on provider"""
        try:
            if self.provider == "openai":
                self._init_openai_embeddings()
            elif self.provider == "google":
                self._init_google_embeddings()
            else:
                # Default to HuggingFace (free, no API key required)
                self._init_huggingface_embeddings()
        except Exception as e:
            print(f"Error initializing {self.provider} embeddings: {e}")
            # Fallback to HuggingFace
            if self.provider != "huggingface":
                print("Falling back to HuggingFace embeddings...")
                self._init_huggingface_embeddings()
    
    def _init_huggingface_embeddings(self):
        """Initialize HuggingFace embeddings (free, no API key)"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.provider = "huggingface"
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {e}")
    
    def _init_openai_embeddings(self):
        """Initialize OpenAI embeddings"""
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = Config.get_openai_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found")
            self.model = OpenAIEmbeddings(
                api_key=api_key,
                model="text-embedding-3-small"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {e}")
    
    def _init_google_embeddings(self):
        """Initialize Google embeddings"""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = Config.get_gemini_api_key()
            if not api_key:
                raise ValueError("Google API key not found")
            self.model = GoogleGenerativeAIEmbeddings(
                google_api_key=api_key,
                model="models/embedding-001"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google embeddings: {e}")
    
    def get_embeddings(self) -> object:
        """
        Get the initialized embedding model.
        
        Returns:
            Embedding model object for use with vector stores
        """
        if self.model is None:
            raise RuntimeError("Embedding model not initialized")
        return self.model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.model is None:
                raise RuntimeError("Embedding model not initialized")
            return self.model.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to embed documents: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if self.model is None:
                raise RuntimeError("Embedding model not initialized")
            return self.model.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")


def get_embedding_model(provider: str = "huggingface") -> EmbeddingModel:
    """
    Factory function to get embedding model.
    
    Args:
        provider: Embedding provider name
        
    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(provider=provider)
