# AI Ally Light - Development History

## v0.3 (2026-03-30)

### Features
- **Audio Notifications**: Sound alerts for AI responses, indexing completion, and errors
  - Pleasant chime when AI responds to a question
  - Ascending three-tone sequence when indexing completes
  - Warning beep for errors and attention needed
- **Global Sound Toggle**: Enable/disable notification sounds from Default Settings

### Technical
- Web Audio API for sound generation (no external files needed)
- Sounds respect the `sound_enabled` setting

---

## v0.2 (2026-03-28)

### Features
- **Multi-Provider LLM Support**: Chat with OpenRouter, Groq, TogetherAI, Ollama, or custom providers
- **Multi-Provider Embeddings**: Use Ollama, OpenRouter, HuggingFace, or custom endpoints
- **Secure API Key Storage**: API keys stored in `.env` file (not in config.json)
- **Separated Settings UI**: Default Settings (header button) and Project Settings clearly distinguished
- **Enhanced Chunking**: Sentence-aware chunking preserves complete sentences
- **Rich Document Metadata**: Each chunk tagged with title, section, and position
- **Screen Reader Optimized Sources**: Sources display includes document title, section, and position with ARIA labels
- **File Upload**: Upload documents (.txt, .md) directly from browser
- **File Management**: Select and delete/reindex individual files from the index
- **Named Threads**: User-named chat threads with timestamp-based default names
- **Thread Management**: Create, rename, and delete threads from the UI
- **Configurable History**: Limit conversation turns sent to AI (default: 20)
- **Configurable Thread Display**: Show recent N threads in sidebar (default: 10)

### Technical
- `/api/ally/` API prefix for all endpoints
- Modular provider architecture in `providers/` directory
- API keys stored in `.env` file for security
- Chat providers: OpenRouter, Groq, TogetherAI, Ollama, Custom
- Embedding providers: Ollama, OpenRouter, HuggingFace, Custom
- Sentence-aware text chunking (regex-based)
- Markdown heading extraction for section tagging
- Chunk position metadata: "beginning", "middle", "end", or "full"
- Thread storage: JSON for loading, markdown for audit

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
