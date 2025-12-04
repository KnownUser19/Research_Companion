# 🤖 Smart Research & Knowledge Assistant

An intelligent AI chatbot built with Streamlit that combines **RAG (Retrieval-Augmented Generation)**, **Live Web Search**, and **Multiple Response Modes** to provide comprehensive assistance for research and knowledge discovery.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 1. 📄 RAG Integration
- Upload documents (PDF, TXT, DOCX)
- Automatic text chunking and embedding
- Semantic search through your documents
- Context-aware responses based on uploaded content

### 2. 🌐 Live Web Search
- Real-time web search integration
- Multiple search providers (Serper, Tavily, DuckDuckGo)
- Automatic fallback to free providers
- Up-to-date information retrieval

### 3. 📝 Response Modes
- **Detailed Mode**: Comprehensive, thorough explanations
- **Concise Mode**: Quick, to-the-point answers

### 4. 🔄 Multi-Provider LLM Support
- **Groq** (Recommended - Fast & Free tier)
- **OpenAI** (GPT-4, GPT-3.5)
- **Google Gemini** (Gemini Pro, Gemini Flash)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-research-assistant.git
   cd smart-research-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   # Add other keys as needed
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Fork/Push to GitHub**
   - Push this project to your GitHub repository

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`

3. **Configure Secrets**
   - In app settings, go to "Secrets"
   - Add your API keys:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   OPENAI_API_KEY = "your_openai_api_key"
   GOOGLE_API_KEY = "your_google_api_key"
   SERPER_API_KEY = "your_serper_api_key"
   TAVILY_API_KEY = "your_tavily_api_key"
   ```

## 🔑 API Keys Setup

| Provider | Purpose | Get API Key |
|----------|---------|-------------|
| Groq | LLM (Recommended) | [console.groq.com/keys](https://console.groq.com/keys) |
| OpenAI | LLM | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Google | LLM | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| Serper | Web Search | [serper.dev](https://serper.dev) |
| Tavily | Web Search | [tavily.com](https://tavily.com) |

**Note**: At least one LLM API key is required. Web search works without API keys using DuckDuckGo.

## 📁 Project Structure

```
smart-research-assistant/
├── config/
│   ├── __init__.py
│   └── config.py          # Configuration management
├── models/
│   ├── __init__.py
│   ├── llm.py             # LLM provider handlers
│   └── embeddings.py      # Embedding models for RAG
├── utils/
│   ├── __init__.py
│   ├── rag_utils.py       # RAG functionality
│   └── web_search.py      # Web search integration
├── .streamlit/
│   ├── config.toml        # Streamlit appearance config
│   └── secrets.toml.example
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md
```

## 🎯 Usage Guide

### Chat with AI
1. Navigate to the **Chat** page
2. Type your question in the input box
3. Get instant AI-powered responses

### Document Q&A (RAG)
1. Go to **Documents** page
2. Upload your documents (PDF, TXT, DOCX)
3. Click "Process Documents"
4. Return to **Chat** and ask questions about your documents

### Web Search
1. Go to **Web Search** page
2. Toggle "Enable Web Search in Chat"
3. Now your chat responses will include web search results

### Response Modes
1. Go to **Settings** page
2. Choose between "Detailed" or "Concise" mode
3. Or use the quick toggle in the sidebar

## 🔧 Configuration

### Environment Variables

You can also set API keys as environment variables:

```bash
export GROQ_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
export SERPER_API_KEY="your_key"
export TAVILY_API_KEY="your_key"
```

### Available Models

**Groq:**
- llama-3.1-70b-versatile
- llama-3.1-8b-instant
- mixtral-8x7b-32768
- gemma2-9b-it

**OpenAI:**
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

**Google Gemini:**
- gemini-1.5-pro
- gemini-1.5-flash
- gemini-pro

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No LLM API keys configured" | Add at least one API key in secrets |
| "Failed to initialize model" | Check API key validity and quota |
| Document upload fails | Ensure file format is PDF, TXT, or DOCX |
| Web search not working | DuckDuckGo is used as fallback automatically |
| Slow responses | Try smaller/faster models like `llama-3.1-8b-instant` |
| Import errors | Run `pip install -r requirements.txt` |

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://langchain.com/) - LLM framework
- [FAISS](https://faiss.ai/) - Vector similarity search
- [HuggingFace](https://huggingface.co/) - Embedding models

---

**Built with ❤️ By Tarra Nikhitha**
