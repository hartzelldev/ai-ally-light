# AI Ally Light - CHANGELOG

##v0.6.0 Beta (2026-05-23)

### New Features
- Implemented a "Thinking..." status UI indicator and loading spinner to provide real-time user feedback during background processing.
- Swapped the chat input to a multi-line `ui.textarea` component featuring layout `autogrow`.
- Integrated a native client-side event handler to enable `Shift+Enter` for inserting newlines while keeping a standard `Enter` keypress mapped to message submission.

### Changed
- Re-architected the local embedding pipeline in `indexer.py` to utilize the lightweight `ONNXMiniLM_L6_V2` engine, completely dropping the heavy Python PyTorch (`torch`) and `sentence-transformers` frameworks.
- Relocated the file export architecture out of the application root directory and into the native OS user profile `Documents/AI Ally Light Exports/` path for cleaner file management and improved accessibility.

### Fixed
- Fixed an intermittent frontend synchronization issue where finished AI completions would stall on WebSockets and fail to render until the user manually changed active view tabs.
- Resolved an internal event loop starvation bug by wrapping the synchronous ChromaDB `query_vector_db` pipeline inside an asynchronous `run.io_bound()` background worker thread.
- Silenced high-volume, repetitive informational logging from Hugging Face transformers, tokenizers, and hub dependencies in the system console to ensure a clean terminal output.

---

##v 0.5.2 Beta (2026-05-14)

### Bug fixes
- Resolved an issue where the application would fail to find the projects directory when launched via Scoop shims or external symlinks.
- Centralized path management in core/config_manager.py. The application now utilizes absolute path resolution at runtime, ensuring consistent access to configuration, environment variables, and project data across all installation methods (Portable, Scoop, or Source).

---

##v0.5.1 Beta (2026-05-13)

### Bug fixes
- Several syntax changes to be compatible with the Windows executable.
- Fixed ui.run to find an available port rather than being bound to one specific one, like 8080.

---

##v0.5 Beta (2026-05-12)

Version 0.5 is a milestone release as, not only does it move the project into 'Beta' status, but it is a complete frontend overhaul from Flask to NiceGUI. This new framework and UI is more intuitive and much smoother.

### New Features
- **Thread Management:** * Added a Thread History dashboard with sorting by modification date.
- **Auto-Save Engine:** Integrated a background save loop in project_chat.py that triggers after every AI response, ensuring no data loss during creative sessions.
- **Export Tools:** New functionality to export AI responses directly into project sub-folders as Markdown files.
- **Screen Reader Optimization:** * Standardized heading hierarchies (H1 through H3) for easier navigation with NVDA/JAWS, as well as audio cues for errors and other actions (AI response).

---

## v0.4 Alpha (2026-04-06)

### Features
- **Intelligent Browser Dispatcher:** Implemented a platform-aware browser launcher.
- **Dynamic Port Management:** Replaced static port binding with a randomized port allocation strategy

### Technical
- **Flask Blueprint Migration:** Transitioned the application from a single-file structure to a modular Blueprint architecture. Moved all core application routes into routes/main.py to improve maintainability and separation of concerns.
- **Circular Import Resolution:** Optimized the package initialization pattern (create_app) to resolve complex circular dependency issues between the main application entry point and the new route modules.

---

## v0.3 Alpha (2026-03-30)

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

## v0.2 Alpha (2026-03-28)

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
