"""
Configuration Management Module
Handles API keys and application settings via environment variables.
For Streamlit Cloud deployment, set these in the Streamlit Secrets manager.
"""

import os
import streamlit as st

class Config:
    """Configuration class for managing API keys and settings"""
    
    @staticmethod
    def get_api_key(key_name: str) -> str:
        """
        Get API key from environment variables or Streamlit secrets.
        Priority: Streamlit secrets > Environment variables
        
        Args:
            key_name: Name of the API key to retrieve
            
        Returns:
            API key string or empty string if not found
        """
        try:
            # First try Streamlit secrets (for cloud deployment)
            if hasattr(st, 'secrets') and key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
        
        # Fall back to environment variables
        return os.environ.get(key_name, "")
    
    @staticmethod
    def get_groq_api_key() -> str:
        """Get Groq API key"""
        return Config.get_api_key("GROQ_API_KEY")
    
    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key"""
        return Config.get_api_key("OPENAI_API_KEY")
    
    @staticmethod
    def get_gemini_api_key() -> str:
        """Get Google Gemini API key"""
        return Config.get_api_key("GOOGLE_API_KEY")
    
    @staticmethod
    def get_serper_api_key() -> str:
        """Get Serper API key for web search"""
        return Config.get_api_key("SERPER_API_KEY")
    
    @staticmethod
    def get_tavily_api_key() -> str:
        """Get Tavily API key for web search"""
        return Config.get_api_key("TAVILY_API_KEY")


# Default model configurations
DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-1.5-flash"
}

# Available models for each provider
AVAILABLE_MODELS = {
    "groq": [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ]
}

# System prompts for different response modes
SYSTEM_PROMPTS = {
    "concise": """You are a helpful AI assistant. Provide brief, clear, and direct answers. 
Keep responses short and to the point - typically 2-3 sentences unless more detail is absolutely necessary.
Focus on the most important information first.""",
    
    "detailed": """You are a helpful AI assistant. Provide comprehensive, thorough, and well-structured answers.
Include relevant context, examples, and explanations to ensure complete understanding.
Organize information clearly with appropriate structure when helpful."""
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_results": 4
}
