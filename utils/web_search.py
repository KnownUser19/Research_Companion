"""
Web Search Utilities Module
Handles live web search integration for real-time information retrieval.
Supports multiple search providers with fallback options.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config


class WebSearchProvider:
    """Base class for web search providers"""
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        raise NotImplementedError


class SerperSearchProvider(WebSearchProvider):
    """Serper.dev API search provider"""
    
    def __init__(self):
        self.api_key = Config.get_serper_api_key()
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if not self.api_key:
                raise ValueError("Serper API key not configured")
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = json.dumps({
                "q": query,
                "num": num_results
            }).encode('utf-8')
            
            req = urllib.request.Request(
                self.base_url,
                data=data,
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            search_results = []
            
            # Process organic results
            for item in result.get("organic", [])[:num_results]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serper"
                })
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Serper search failed: {e}")


class TavilySearchProvider(WebSearchProvider):
    """Tavily API search provider"""
    
    def __init__(self):
        self.api_key = Config.get_tavily_api_key()
        self.base_url = "https://api.tavily.com/search"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform search using Tavily API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if not self.api_key:
                raise ValueError("Tavily API key not configured")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = json.dumps({
                "api_key": self.api_key,
                "query": query,
                "max_results": num_results,
                "include_answer": True,
                "search_depth": "basic"
            }).encode('utf-8')
            
            req = urllib.request.Request(
                self.base_url,
                data=data,
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            search_results = []
            
            # Add Tavily's generated answer if available
            if result.get("answer"):
                search_results.append({
                    "title": "AI Summary",
                    "link": "",
                    "snippet": result["answer"],
                    "source": "tavily_answer"
                })
            
            # Process search results
            for item in result.get("results", [])[:num_results]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily"
                })
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Tavily search failed: {e}")


class DuckDuckGoSearchProvider(WebSearchProvider):
    """DuckDuckGo search provider (no API key required)"""
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform search using DuckDuckGo.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            from duckduckgo_search import DDGS
            
            search_results = []
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                
                for item in results:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("href", ""),
                        "snippet": item.get("body", ""),
                        "source": "duckduckgo"
                    })
            
            return search_results
            
        except ImportError:
            raise RuntimeError("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo search failed: {e}")


class WebSearchManager:
    """Manages web search operations with multiple providers"""
    
    def __init__(self, preferred_provider: str = "auto"):
        """
        Initialize web search manager.
        
        Args:
            preferred_provider: Preferred search provider ('serper', 'tavily', 'duckduckgo', or 'auto')
        """
        self.preferred_provider = preferred_provider
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, WebSearchProvider]:
        """Initialize available search providers"""
        providers = {}
        
        # Try to initialize each provider
        try:
            serper = SerperSearchProvider()
            if serper.api_key:
                providers["serper"] = serper
        except Exception:
            pass
        
        try:
            tavily = TavilySearchProvider()
            if tavily.api_key:
                providers["tavily"] = tavily
        except Exception:
            pass
        
        # DuckDuckGo is always available (no API key)
        try:
            providers["duckduckgo"] = DuckDuckGoSearchProvider()
        except Exception:
            pass
        
        return providers
    
    def get_available_providers(self) -> List[str]:
        """Get list of available search providers"""
        return list(self.providers.keys())
    
    def search(self, query: str, num_results: int = 5, provider: str = None) -> List[Dict]:
        """
        Perform web search using best available provider.
        
        Args:
            query: Search query
            num_results: Number of results to return
            provider: Specific provider to use (optional)
            
        Returns:
            List of search results
        """
        # Determine which provider to use
        if provider and provider in self.providers:
            selected_provider = provider
        elif self.preferred_provider != "auto" and self.preferred_provider in self.providers:
            selected_provider = self.preferred_provider
        elif self.providers:
            # Auto-select: prefer paid APIs over free
            if "serper" in self.providers:
                selected_provider = "serper"
            elif "tavily" in self.providers:
                selected_provider = "tavily"
            else:
                selected_provider = list(self.providers.keys())[0]
        else:
            raise RuntimeError("No search providers available")
        
        try:
            return self.providers[selected_provider].search(query, num_results)
        except Exception as e:
            # Try fallback to DuckDuckGo
            if selected_provider != "duckduckgo" and "duckduckgo" in self.providers:
                try:
                    return self.providers["duckduckgo"].search(query, num_results)
                except Exception:
                    pass
            raise RuntimeError(f"Web search failed: {e}")


def perform_web_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Convenience function to perform web search.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results
    """
    try:
        manager = WebSearchManager()
        return manager.search(query, num_results)
    except Exception as e:
        raise RuntimeError(f"Web search failed: {e}")


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display or LLM context.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string of search results
    """
    if not results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        entry = f"[{i}] {result['title']}\n"
        entry += f"    {result['snippet']}\n"
        if result['link']:
            entry += f"    Source: {result['link']}"
        formatted.append(entry)
    
    return "\n\n".join(formatted)


def format_search_context_for_llm(query: str, results: List[Dict]) -> str:
    """
    Format search results as context for LLM.
    
    Args:
        query: Original search query
        results: List of search results
        
    Returns:
        Formatted context string
    """
    if not results:
        return f"No web search results found for: {query}"
    
    context = f"Web search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"Source {i}: {result['title']}\n"
        context += f"{result['snippet']}\n"
        if result['link']:
            context += f"URL: {result['link']}\n"
        context += "\n"
    
    return context


def check_search_availability() -> Dict[str, bool]:
    """
    Check which search providers are available.
    
    Returns:
        Dictionary with provider availability status
    """
    return {
        "serper": bool(Config.get_serper_api_key()),
        "tavily": bool(Config.get_tavily_api_key()),
        "duckduckgo": True  # Always available
    }
