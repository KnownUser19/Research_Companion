"""
RAG Utilities Module
Handles document processing, vector store management, and retrieval operations.
"""

import os
import sys
from typing import List, Optional, Any, Tuple
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import RAG_CONFIG


class DocumentProcessor:
    """Handles document loading and text splitting"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or RAG_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or RAG_CONFIG["chunk_overlap"]
        self.text_splitter = self._init_text_splitter()
    
    def _init_text_splitter(self):
        """Initialize the text splitter"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize text splitter: {e}")
    
    def load_pdf(self, file_path: str) -> List[Any]:
        """
        Load and split a PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}")
    
    def load_text(self, file_path: str) -> List[Any]:
        """
        Load and split a text document.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of document chunks
        """
        try:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to load text file: {e}")
    
    def load_docx(self, file_path: str) -> List[Any]:
        """
        Load and split a Word document.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List of document chunks
        """
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to load DOCX file: {e}")
    
    def load_document(self, file_path: str) -> List[Any]:
        """
        Load any supported document type.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of document chunks
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.pdf':
            return self.load_pdf(file_path)
        elif extension == '.txt':
            return self.load_text(file_path)
        elif extension in ['.docx', '.doc']:
            return self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def process_uploaded_file(self, uploaded_file) -> List[Any]:
        """
        Process a Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            List of document chunks
        """
        try:
            # Get file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and process document
            chunks = self.load_document(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"Failed to process uploaded file: {e}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split raw text into chunks.
        
        Args:
            text: Raw text to split
            
        Returns:
            List of text chunks
        """
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            raise RuntimeError(f"Failed to split text: {e}")


class VectorStoreManager:
    """Manages vector store operations for RAG"""
    
    def __init__(self, embedding_model):
        """
        Initialize vector store manager.
        
        Args:
            embedding_model: Embedding model for vectorization
        """
        self.embedding_model = embedding_model
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Any]) -> Any:
        """
        Create vector store from documents.
        
        Args:
            documents: List of document chunks
            
        Returns:
            Vector store object
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            self.vector_store = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            return self.vector_store
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {e}")
    
    def add_documents(self, documents: List[Any]):
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of document chunks to add
        """
        try:
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def similarity_search(self, query: str, k: int = None) -> List[Any]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            if self.vector_store is None:
                raise RuntimeError("Vector store not initialized. Please upload documents first.")
            
            k = k or RAG_CONFIG["top_k_results"]
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Any, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if self.vector_store is None:
                raise RuntimeError("Vector store not initialized. Please upload documents first.")
            
            k = k or RAG_CONFIG["top_k_results"]
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")
    
    def get_retriever(self, k: int = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            k: Number of results to return
            
        Returns:
            Retriever object
        """
        try:
            if self.vector_store is None:
                raise RuntimeError("Vector store not initialized. Please upload documents first.")
            
            k = k or RAG_CONFIG["top_k_results"]
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            raise RuntimeError(f"Failed to create retriever: {e}")


class RAGPipeline:
    """Complete RAG pipeline for document Q&A"""
    
    def __init__(self, embedding_provider: str = "huggingface"):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_provider: Embedding provider to use
        """
        from models.embeddings import get_embedding_model
        
        self.embedding_model = get_embedding_model(embedding_provider)
        self.doc_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager(
            self.embedding_model.get_embeddings()
        )
        self.documents_loaded = False
    
    def process_documents(self, uploaded_files: List[Any]) -> int:
        """
        Process multiple uploaded documents.
        
        Args:
            uploaded_files: List of Streamlit uploaded files
            
        Returns:
            Number of chunks created
        """
        all_chunks = []
        
        for file in uploaded_files:
            try:
                chunks = self.doc_processor.process_uploaded_file(file)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        if all_chunks:
            self.vector_store_manager.create_vector_store(all_chunks)
            self.documents_loaded = True
        
        return len(all_chunks)
    
    def get_relevant_context(self, query: str, k: int = None) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            if not self.documents_loaded:
                return ""
            
            results = self.vector_store_manager.similarity_search(query, k)
            
            if not results:
                return ""
            
            context_parts = []
            for i, doc in enumerate(results, 1):
                context_parts.append(f"[Document {i}]\n{doc.page_content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    def get_relevant_context_with_scores(self, query: str, k: int = None) -> List[dict]:
        """
        Get relevant context with relevance scores.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of dicts with content and scores
        """
        try:
            if not self.documents_loaded:
                return []
            
            results = self.vector_store_manager.similarity_search_with_score(query, k)
            
            context_list = []
            for doc, score in results:
                context_list.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })
            
            return context_list
        except Exception as e:
            print(f"Error getting context: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if RAG pipeline is ready for queries"""
        return self.documents_loaded


def format_rag_prompt(query: str, context: str, response_mode: str = "detailed") -> str:
    """
    Format a prompt with RAG context.
    
    Args:
        query: User query
        context: Retrieved context
        response_mode: 'concise' or 'detailed'
        
    Returns:
        Formatted prompt string
    """
    if response_mode == "concise":
        instruction = "Provide a brief, direct answer based on the context. Keep it to 2-3 sentences."
    else:
        instruction = "Provide a comprehensive answer based on the context. Include relevant details and explanations."
    
    prompt = f"""Based on the following context from uploaded documents, please answer the question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS: {instruction}

If the context doesn't contain relevant information to answer the question, say so clearly and provide what help you can based on your general knowledge."""
    
    return prompt
