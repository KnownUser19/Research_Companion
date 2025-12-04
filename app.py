"""
Smart Research & Knowledge Assistant
A multi-functional AI chatbot with RAG, Web Search, and Response Modes.

Features:
- RAG Integration: Upload documents and ask questions about them
- Live Web Search: Search the web for latest information
- Response Modes: Switch between concise and detailed responses
- Multi-Provider LLM: Support for Groq, OpenAI, and Google Gemini
"""

import streamlit as st
import os
import sys

# Ensure proper path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config.config import Config, AVAILABLE_MODELS, SYSTEM_PROMPTS, DEFAULT_MODELS
from models.llm import get_llm_model, check_available_providers
from utils.rag_utils import RAGPipeline, format_rag_prompt
from utils.web_search import WebSearchManager, format_search_results, format_search_context_for_llm, check_search_availability


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "rag_pipeline": None,
        "documents_uploaded": False,
        "uploaded_file_names": [],
        "response_mode": "detailed",
        "selected_provider": "groq",
        "selected_model": DEFAULT_MODELS["groq"],
        "web_search_enabled": False,
        "chat_initialized": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_chat_model():
    """Get the configured chat model"""
    try:
        provider = st.session_state.selected_provider
        model_name = st.session_state.selected_model
        
        return get_llm_model(provider, model_name)
    except Exception as e:
        st.error(f"Failed to initialize {provider} model: {str(e)}")
        return None


def get_system_prompt() -> str:
    """Get system prompt based on response mode"""
    mode = st.session_state.response_mode
    base_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["detailed"])
    
    # Add RAG context instruction if documents are loaded
    if st.session_state.documents_uploaded:
        base_prompt += "\n\nYou have access to uploaded documents. When relevant context is provided, use it to answer questions accurately."
    
    # Add web search instruction if enabled
    if st.session_state.web_search_enabled:
        base_prompt += "\n\nYou have access to web search results. When search results are provided, use them to provide up-to-date information."
    
    return base_prompt


def process_user_query(query: str, chat_model) -> str:
    """Process user query with RAG and/or web search if applicable"""
    try:
        context_parts = []
        
        # Get RAG context if documents are uploaded
        if st.session_state.documents_uploaded and st.session_state.rag_pipeline:
            try:
                rag_context = st.session_state.rag_pipeline.get_relevant_context(query)
                if rag_context:
                    context_parts.append(f"DOCUMENT CONTEXT:\n{rag_context}")
            except Exception as e:
                st.warning(f"RAG retrieval issue: {str(e)}")
        
        # Get web search results if enabled
        if st.session_state.web_search_enabled:
            try:
                search_manager = WebSearchManager()
                search_results = search_manager.search(query, num_results=3)
                if search_results:
                    search_context = format_search_context_for_llm(query, search_results)
                    context_parts.append(f"WEB SEARCH RESULTS:\n{search_context}")
            except Exception as e:
                st.warning(f"Web search issue: {str(e)}")
        
        # Build the full prompt
        if context_parts:
            full_context = "\n\n---\n\n".join(context_parts)
            enhanced_query = format_rag_prompt(query, full_context, st.session_state.response_mode)
        else:
            enhanced_query = query
        
        # Prepare messages for the model
        system_prompt = get_system_prompt()
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        formatted_messages.append(HumanMessage(content=enhanced_query))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
        
    except Exception as e:
        return f"Error processing query: {str(e)}"


def handle_document_upload(uploaded_files):
    """Handle document upload and RAG initialization"""
    try:
        if not uploaded_files:
            return
        
        with st.spinner("Processing documents..."):
            # Initialize RAG pipeline if not exists
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline(embedding_provider="huggingface")
            
            # Process uploaded files
            num_chunks = st.session_state.rag_pipeline.process_documents(uploaded_files)
            
            # Update session state
            st.session_state.documents_uploaded = True
            st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
            
            st.success(f"✅ Processed {len(uploaded_files)} document(s) into {num_chunks} chunks!")
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def chat_page():
    """Main chat interface page"""
    st.title("🤖 Smart Research Assistant")
    st.markdown("*Your AI companion for document analysis and knowledge discovery*")
    
    # Check for available providers
    available_providers = check_available_providers()
    
    if not any(available_providers.values()):
        st.warning("⚠️ No LLM API keys configured. Please check the **Settings** page to set up your API keys.")
        st.info("You can set API keys in the sidebar or configure them as environment variables.")
        return
    
    # Get chat model
    chat_model = get_chat_model()
    
    if not chat_model:
        st.error("Failed to initialize chat model. Please check your API configuration.")
        return
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.documents_uploaded:
            st.success(f"📄 {len(st.session_state.uploaded_file_names)} doc(s) loaded")
        else:
            st.info("📄 No documents")
    with col2:
        if st.session_state.web_search_enabled:
            st.success("🌐 Web search ON")
        else:
            st.info("🌐 Web search OFF")
    with col3:
        mode_emoji = "📝" if st.session_state.response_mode == "detailed" else "⚡"
        st.info(f"{mode_emoji} {st.session_state.response_mode.capitalize()} mode")
    
    st.divider()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_user_query(prompt, chat_model)
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def documents_page():
    """Document upload and management page"""
    st.title("📄 Document Management")
    st.markdown("Upload documents to enable RAG-powered Q&A")
    
    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Selected {len(uploaded_files)} file(s):**")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")
        
        with col2:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                handle_document_upload(uploaded_files)
    
    st.divider()
    
    # Current documents status
    st.subheader("Current Status")
    if st.session_state.documents_uploaded:
        st.success("✅ Documents loaded and ready for Q&A!")
        st.write("**Loaded documents:**")
        for name in st.session_state.uploaded_file_names:
            st.write(f"- {name}")
        
        if st.button("🗑️ Clear All Documents"):
            st.session_state.rag_pipeline = None
            st.session_state.documents_uploaded = False
            st.session_state.uploaded_file_names = []
            st.rerun()
    else:
        st.info("No documents uploaded yet. Upload documents above to enable RAG functionality.")
    
    st.divider()
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - **PDF files**: Best for reports, papers, and formal documents
    - **TXT files**: Great for plain text content
    - **DOCX files**: Works with Microsoft Word documents
    - Upload multiple files to create a comprehensive knowledge base
    - The chatbot will automatically search uploaded documents when answering questions
    """)


def web_search_page():
    """Web search configuration and testing page"""
    st.title("🌐 Web Search")
    st.markdown("Configure and test live web search functionality")
    
    # Check available providers
    search_availability = check_search_availability()
    
    st.subheader("Search Providers Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if search_availability["serper"]:
            st.success("✅ Serper API")
        else:
            st.warning("❌ Serper API")
    
    with col2:
        if search_availability["tavily"]:
            st.success("✅ Tavily API")
        else:
            st.warning("❌ Tavily API")
    
    with col3:
        st.success("✅ DuckDuckGo (Free)")
    
    st.divider()
    
    # Enable/Disable toggle
    st.subheader("Web Search Settings")
    web_search_toggle = st.toggle(
        "Enable Web Search in Chat",
        value=st.session_state.web_search_enabled,
        help="When enabled, the chatbot will search the web for relevant information"
    )
    
    if web_search_toggle != st.session_state.web_search_enabled:
        st.session_state.web_search_enabled = web_search_toggle
        st.rerun()
    
    st.divider()
    
    # Test search
    st.subheader("Test Web Search")
    test_query = st.text_input("Enter a test query:", placeholder="e.g., Latest news about AI")
    
    if st.button("🔍 Search", type="primary"):
        if test_query:
            with st.spinner("Searching the web..."):
                try:
                    search_manager = WebSearchManager()
                    results = search_manager.search(test_query, num_results=5)
                    
                    if results:
                        st.success(f"Found {len(results)} results!")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"{i}. {result['title']}", expanded=(i == 1)):
                                st.write(result['snippet'])
                                if result['link']:
                                    st.markdown(f"[Read more]({result['link']})")
                    else:
                        st.warning("No results found.")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        else:
            st.warning("Please enter a search query.")
    
    st.divider()
    
    # Info about API keys
    st.subheader("🔑 API Key Setup")
    st.markdown("""
    **For better search results, configure one of these APIs:**
    
    - **Serper API**: Get a free API key at [serper.dev](https://serper.dev)
    - **Tavily API**: Get a free API key at [tavily.com](https://tavily.com)
    
    Set the API keys in your Streamlit secrets or as environment variables:
    ```
    SERPER_API_KEY=your_serper_key
    TAVILY_API_KEY=your_tavily_key
    ```
    
    **Note**: DuckDuckGo search works without any API key but may have rate limits.
    """)


def settings_page():
    """Settings and configuration page"""
    st.title("⚙️ Settings")
    st.markdown("Configure your AI assistant")
    
    # LLM Provider Selection
    st.subheader("🤖 LLM Configuration")
    
    available = check_available_providers()
    available_list = [k for k, v in available.items() if v]
    
    if not available_list:
        st.error("No LLM API keys configured. Please set at least one API key.")
        st.markdown("""
        **Set one of these API keys:**
        - `GROQ_API_KEY` - Get from [Groq Console](https://console.groq.com/keys)
        - `OPENAI_API_KEY` - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
        - `GOOGLE_API_KEY` - Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
        """)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Only show available providers
            if st.session_state.selected_provider not in available_list:
                st.session_state.selected_provider = available_list[0]
            
            provider = st.selectbox(
                "LLM Provider",
                options=available_list,
                index=available_list.index(st.session_state.selected_provider) if st.session_state.selected_provider in available_list else 0,
                format_func=lambda x: x.upper()
            )
            
            if provider != st.session_state.selected_provider:
                st.session_state.selected_provider = provider
                st.session_state.selected_model = DEFAULT_MODELS[provider]
        
        with col2:
            models = AVAILABLE_MODELS.get(st.session_state.selected_provider, [])
            current_model = st.session_state.selected_model
            
            if current_model not in models:
                current_model = models[0] if models else ""
            
            model = st.selectbox(
                "Model",
                options=models,
                index=models.index(current_model) if current_model in models else 0
            )
            
            if model != st.session_state.selected_model:
                st.session_state.selected_model = model
    
    st.divider()
    
    # Response Mode
    st.subheader("📝 Response Mode")
    
    mode = st.radio(
        "Choose response style:",
        options=["detailed", "concise"],
        index=0 if st.session_state.response_mode == "detailed" else 1,
        format_func=lambda x: f"{'📝 Detailed' if x == 'detailed' else '⚡ Concise'} - {SYSTEM_PROMPTS[x][:100]}...",
        horizontal=True
    )
    
    if mode != st.session_state.response_mode:
        st.session_state.response_mode = mode
    
    st.divider()
    
    # Provider Status
    st.subheader("📊 Provider Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if available.get("groq"):
            st.success("✅ Groq")
        else:
            st.error("❌ Groq")
    
    with col2:
        if available.get("openai"):
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    with col3:
        if available.get("gemini"):
            st.success("✅ Gemini")
        else:
            st.error("❌ Gemini")
    
    st.divider()
    
    # Clear Chat
    st.subheader("🗑️ Clear Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
    
    with col2:
        if st.button("Clear All Data", type="secondary", use_container_width=True):
            for key in ["messages", "rag_pipeline", "documents_uploaded", "uploaded_file_names"]:
                if key in st.session_state:
                    if key == "messages":
                        st.session_state[key] = []
                    elif key == "documents_uploaded":
                        st.session_state[key] = False
                    elif key == "uploaded_file_names":
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.success("All data cleared!")
            st.rerun()


def instructions_page():
    """Instructions and help page"""
    st.title("📚 Instructions")
    st.markdown("Welcome to the Smart Research & Knowledge Assistant!")
    
    st.markdown("""
    ## 🎯 What This App Does
    
    This is an intelligent AI chatbot with three powerful features:
    
    ### 1. 📄 RAG (Retrieval-Augmented Generation)
    Upload your documents (PDF, TXT, DOCX) and ask questions about them. The AI will search through your documents to provide accurate, context-aware answers.
    
    ### 2. 🌐 Live Web Search
    Enable web search to get up-to-date information from the internet. Perfect for questions about current events, latest news, or information that may have changed.
    
    ### 3. 📝 Response Modes
    Switch between:
    - **Detailed Mode**: Comprehensive, thorough explanations
    - **Concise Mode**: Quick, to-the-point answers
    
    ---
    
    ## 🚀 Getting Started
    
    ### Step 1: Configure API Keys
    You need at least one LLM API key. Choose from:
    
    | Provider | Get API Key |
    |----------|------------|
    | Groq (Recommended - Free) | [console.groq.com/keys](https://console.groq.com/keys) |
    | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
    | Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
    
    ### Step 2: Set Up Secrets (for Streamlit Cloud)
    In your Streamlit Cloud app settings, add secrets:
    ```toml
    GROQ_API_KEY = "your_groq_api_key"
    OPENAI_API_KEY = "your_openai_api_key"
    GOOGLE_API_KEY = "your_google_api_key"
    SERPER_API_KEY = "your_serper_api_key"
    TAVILY_API_KEY = "your_tavily_api_key"
    ```
    
    ### Step 3: Start Chatting!
    Go to the Chat page and start asking questions.
    
    ---
    
    ## 💡 Tips for Best Results
    
    1. **For Document Q&A**: Upload relevant documents first, then ask specific questions
    2. **For Current Information**: Enable web search in the Web Search page
    3. **For Quick Answers**: Switch to Concise mode in Settings
    4. **For In-depth Explanations**: Use Detailed mode (default)
    
    ---
    
    ## 🔧 Troubleshooting
    
    | Issue | Solution |
    |-------|----------|
    | "No API keys configured" | Add at least one LLM API key in secrets |
    | Document upload fails | Check file format (PDF, TXT, DOCX only) |
    | Web search not working | DuckDuckGo always works; for better results add Serper/Tavily keys |
    | Slow responses | Try a faster model like `llama-3.1-8b-instant` |
    
    ---
    
    Ready to start? Navigate to **Chat** using the sidebar! 🚀
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Smart Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 Research Assistant")
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📄 Documents", "🌐 Web Search", "⚙️ Settings", "📚 Instructions"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick settings in sidebar
        st.subheader("Quick Settings")
        
        # Response mode toggle
        mode_options = ["Detailed", "Concise"]
        current_mode_idx = 0 if st.session_state.response_mode == "detailed" else 1
        mode = st.selectbox(
            "Response Mode",
            mode_options,
            index=current_mode_idx,
            label_visibility="collapsed"
        )
        if mode.lower() != st.session_state.response_mode:
            st.session_state.response_mode = mode.lower()
        
        # Web search toggle
        web_toggle = st.toggle(
            "Web Search",
            value=st.session_state.web_search_enabled
        )
        if web_toggle != st.session_state.web_search_enabled:
            st.session_state.web_search_enabled = web_toggle
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Footer
        st.divider()
        st.caption("Built with ❤️ using Streamlit")
    
    # Route to appropriate page
    if page == "💬 Chat":
        chat_page()
    elif page == "📄 Documents":
        documents_page()
    elif page == "🌐 Web Search":
        web_search_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📚 Instructions":
        instructions_page()


if __name__ == "__main__":
    main()
