"""
LLM Module
Handles initialization and management of Large Language Models.
Supports Groq, OpenAI, and Google Gemini providers.
"""

import os
import sys
from typing import Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config, DEFAULT_MODELS, AVAILABLE_MODELS


class LLMProvider:
    """Manages LLM initialization and operations"""
    
    def __init__(self, provider: str, model_name: Optional[str] = None, temperature: float = 0.7):
        """
        Initialize LLM provider.
        
        Args:
            provider: LLM provider ('groq', 'openai', or 'gemini')
            model_name: Specific model name (uses default if None)
            temperature: Model temperature for response randomness
        """
        self.provider = provider.lower()
        self.model_name = model_name or DEFAULT_MODELS.get(self.provider)
        self.temperature = temperature
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM based on provider"""
        try:
            if self.provider == "groq":
                self._init_groq()
            elif self.provider == "openai":
                self._init_openai()
            elif self.provider == "gemini":
                self._init_gemini()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.provider} model: {e}")
    
    def _init_groq(self):
        """Initialize Groq model"""
        try:
            from langchain_groq import ChatGroq
            api_key = Config.get_groq_api_key()
            if not api_key:
                raise ValueError("Groq API key not found. Please set GROQ_API_KEY.")
            
            self.model = ChatGroq(
                api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096
            )
        except ImportError:
            raise RuntimeError("langchain-groq not installed. Run: pip install langchain-groq")
    
    def _init_openai(self):
        """Initialize OpenAI model"""
        try:
            from langchain_openai import ChatOpenAI
            api_key = Config.get_openai_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
            
            self.model = ChatOpenAI(
                api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096
            )
        except ImportError:
            raise RuntimeError("langchain-openai not installed. Run: pip install langchain-openai")
    
    def _init_gemini(self):
        """Initialize Google Gemini model"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = Config.get_gemini_api_key()
            if not api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY.")
            
            self.model = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=4096
            )
        except ImportError:
            raise RuntimeError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
    
    def get_model(self) -> Any:
        """
        Get the initialized LLM model.
        
        Returns:
            LLM model object
        """
        if self.model is None:
            raise RuntimeError("LLM model not initialized")
        return self.model
    
    def invoke(self, messages: list) -> str:
        """
        Invoke the model with messages.
        
        Args:
            messages: List of message objects
            
        Returns:
            Model response content
        """
        try:
            if self.model is None:
                raise RuntimeError("LLM model not initialized")
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to get model response: {e}")


def get_chatgroq_model(model_name: Optional[str] = None, temperature: float = 0.7) -> Any:
    """
    Get Groq chat model (backward compatibility function).
    
    Args:
        model_name: Model name (optional)
        temperature: Model temperature
        
    Returns:
        ChatGroq model instance
    """
    try:
        provider = LLMProvider("groq", model_name, temperature)
        return provider.get_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {e}")


def get_openai_model(model_name: Optional[str] = None, temperature: float = 0.7) -> Any:
    """
    Get OpenAI chat model.
    
    Args:
        model_name: Model name (optional)
        temperature: Model temperature
        
    Returns:
        ChatOpenAI model instance
    """
    try:
        provider = LLMProvider("openai", model_name, temperature)
        return provider.get_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {e}")


def get_gemini_model(model_name: Optional[str] = None, temperature: float = 0.7) -> Any:
    """
    Get Google Gemini chat model.
    
    Args:
        model_name: Model name (optional)
        temperature: Model temperature
        
    Returns:
        ChatGoogleGenerativeAI model instance
    """
    try:
        provider = LLMProvider("gemini", model_name, temperature)
        return provider.get_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {e}")


def get_llm_model(provider: str, model_name: Optional[str] = None, temperature: float = 0.7) -> Any:
    """
    Universal function to get any supported LLM model.
    
    Args:
        provider: Provider name ('groq', 'openai', or 'gemini')
        model_name: Model name (optional)
        temperature: Model temperature
        
    Returns:
        LLM model instance
    """
    try:
        llm_provider = LLMProvider(provider, model_name, temperature)
        return llm_provider.get_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {provider} model: {e}")


def check_available_providers() -> dict:
    """
    Check which LLM providers have valid API keys configured.
    
    Returns:
        Dictionary with provider names and availability status
    """
    return {
        "groq": bool(Config.get_groq_api_key()),
        "openai": bool(Config.get_openai_api_key()),
        "gemini": bool(Config.get_gemini_api_key())
    }
