# AI Ally Light
An accessible, light-weight, local multi-project RAG chat system.

## Features
- Ollama for embeddings (CPU-friendly, no GPU needed)
- OpenRouter for the LLM
- Plain HTML browser interface, fully accessible with screen readers (NVDA tested)
- Environment variable configuration for Ollama settings
- Separate Default and Project-level settings

---

## Requirements

- Python 3.10 or newer
- pip
- Ollama installed and running: https://ollama.com
- An OpenRouter API key: https://openrouter.ai

---

## Installation

### 1. Install Python dependencies

```
pip install flask chromadb requests watchdog python-dotenv werkzeug
```

### 2. Pull the embedding model

```
ollama pull nomic-embed-text
```

This model is ~274 MB and runs on CPU — no GPU needed.

### 3. Configure Ollama (optional)

Create a `.env` file in the project root:

```
OLLAMA_BASE_URL=http://127.0.0.1:11435
OLLAMA_EMBED_MODEL=nomic-embed-text
```

The default values work for Ollama running on the same machine with the default port.

### 4. Run the app

```
python ally.py
```

Open your browser to: http://localhost:5000

---

## First-time setup

1. Click **Default Settings** in the header
2. Enter your OpenRouter API key
3. Choose your preferred model (default: openai/gpt-4o-mini)
4. Click Save Settings

Everything else can be left at defaults to start.

---

## Using the app

### Projects
- Type a name in the "New project name" field and press Enter or click Create Project.
- Each project gets its own isolated document index and conversation history.
- Use Rename, Delete, and Re-index buttons in the project action bar.

### Adding documents
Select a project, then click **Upload Documents** in the right panel to upload
.txt or .md files directly from your browser.

Alternatively, you can manually add files to the project's docs folder. The app
watches the folder and indexes new files automatically.

### Managing indexed files
The right panel shows all indexed files with chunk counts. Use the checkboxes to:
- **Delete Selected** — remove files from the index (keeps the original files)
- **Reindex Selected** — re-process selected files (useful after editing)

### Chatting
- Type in the message box and press Enter to send.
- Shift+Enter inserts a new line.
- Each assistant reply shows which files were used as sources (expandable).
- Clear resets the conversation history but keeps the document index.

### Settings
Two settings dialogs are available:

**Default Settings** (header button):
- OpenRouter API key and model
- Ollama URL and embedding model
- Chunk size, overlap, and top-K results
- System prompt
- These apply to all projects unless overridden

**Project Settings** (project action bar button):
- Override model, top-K, or system prompt for the current project only
- Leave blank to inherit the global default

---

## File structure

```
ai-ally-light/
  ally.py              Main application
  .env                 Ollama configuration (optional)
  config.json          Settings (managed from the UI, also editable by hand)
  projects.json        Project registry
  projects/
    my-project/
      docs/            Put your .txt and .md files here
      config.json      Project overrides (optional)
    other-project/
      docs/
  static/
    index.html         Browser interface
  HISTORY.md           Development history
  README.md            This file
```

---

## Troubleshooting

**Cannot reach Ollama**
Make sure Ollama is running. Open a terminal and run: `ollama serve`

**Embedding model not found**
Run: `ollama pull nomic-embed-text`

**No API key / OpenRouter errors**
Open Default Settings in the header and enter your OpenRouter key.

**Files not being indexed**
Make sure they end in .txt or .md and are in the project's docs folder.
Use the Re-index button if auto-indexing didn't pick them up.

**Port 5000 in use**
The app will automatically try ports 5001-5020 if 5000 is in use.

**Changed embedding model or chunk size?**
Delete the project's `chroma_db/` folder and re-index. The old embeddings are
incompatible with a new model or chunk size.

---

## Accessibility

AI Ally Light is designed for screen reader users:
- Semantic HTML structure
- Keyboard navigable throughout
- Skip links for quick navigation
- ARIA labels and live regions for dynamic content
- No complex visual elements that would confuse assistive technology
