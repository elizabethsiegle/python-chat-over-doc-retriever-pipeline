# PDF/Webpage Chat Interface

A retriever pipeline Python web application that allows users to chat with PDFs and web articles using AI. Combines [FAISS vector similarity search](https://github.com/facebookresearch/faiss) and BM25 keyword matching for accurate document retrieval and uses [Voyage AI embeddings](https://www.voyageai.com/) and [LLaMA](https://developers.cloudflare.com/workers-ai/models/llama-3.2-3b-instruct/) hosted on Cloudflare Workers AI for question answering.

## Key Files

### Main Application Files
- `app.py` - The main Flask application that handles routing, file processing, and chat functionality. Integrates all components together and provides the web API endpoints.

- `clients.py` - Manages API keys and environment variables for Voyage AI, Cloudflare, and R2 storage services. Uses dotenv for secure configuration.

### Document Processing
- `get_text.py` - Extracts text content from PDFs and web articles using trafilatura for web scraping.

- `chunk_x_context.py` - Splits extracted text into manageable chunks while preserving context for better question answering.

- `create_index.py` - Creates FAISS and BM25 indexes from text chunks for hybrid search. Combines semantic similarity (FAISS) with keyword matching (BM25) for improved retrieval.

### Frontend
- `templates/index.html` - The web interface featuring a cyberpunk-inspired design. Provides upload functionality for PDFs and URLs, and an interactive chat interface.

## Setup

1. Create a `.env` file with your API keys:
- Cloudflare Account ID
- Cloudflare Auth Token
- Voyage AI API Key
- R2 Access Key ID
- R2 Secret Access Key
- R2 Bucket Name
