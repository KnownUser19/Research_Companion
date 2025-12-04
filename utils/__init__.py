"""Utils module initialization"""
from .rag_utils import DocumentProcessor, VectorStoreManager, RAGPipeline, format_rag_prompt
from .web_search import WebSearchManager, perform_web_search, format_search_results, format_search_context_for_llm, check_search_availability
