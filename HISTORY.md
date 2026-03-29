# AI Ally Light - Development History

## v0.2 (2026-03-28)

### Features
- **Environment Variable Configuration**: Ollama URL and model loaded from `.env` file via python-dotenv
- **Separated Settings UI**: Default Settings (header button) and Project Settings clearly distinguished
- **WSL Compatibility**: Default Ollama URL uses `127.0.0.1` instead of `localhost`
- **Enhanced Chunking**: Sentence-aware chunking preserves complete sentences
- **Rich Document Metadata**: Each chunk tagged with title, section, and position
- **Screen Reader Optimized Sources**: Sources display includes document title, section, and position with ARIA labels
- **File Upload**: Upload documents (.txt, .md) directly from browser
- **File Management**: Select and delete/reindex individual files from the index

### Technical
- `/api/ally/` API prefix for all endpoints
- Default Ollama URL: `http://127.0.0.1:11435`
- Default embedding model: `nomic-embed-text`
- Sentence-aware text chunking (regex-based)
- Markdown heading extraction for section tagging
- Chunk position metadata: "beginning", "middle", "end", or "full"
- Upload and file management API endpoints

---

## v0.1 Alpha (2026-03-28)

### Core Features
- **Multi-Project RAG System**: Isolated document indexes and conversation histories per project
- **Ollama Embeddings**: CPU-friendly `nomic-embed-text` model for document vectorization
- **OpenRouter LLM**: Access to various LLM models (default: openai/gpt-4o-mini)
- **ChromaDB Vector Storage**: Persistent vector database per project
- **Auto-Indexing**: File watcher monitors docs folder for .txt/.md changes
- **Manual Re-indexing**: Button to trigger full document re-indexing
- **Session Logging**: Chat conversations saved as markdown files
- **Configurable Chunking**: Adjustable chunk size, overlap, and top-K results
- **Custom System Prompts**: Per-project or global system prompt configuration

### Accessibility
- Plain HTML interface
- Screen reader compatible (NVDA tested)
- Keyboard navigable
- Skip links, ARIA labels, and semantic HTML
