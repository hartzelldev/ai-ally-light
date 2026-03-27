# ai-ally-light
An accessible, light-weight, local multi-project RAG chat system.

## Features
- Ollama for embeddings (CPU-friendly, no GPU needed)
- OpenRouter for the LLM
- Plain HTML browser interface, fully accessible with NVDA

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
pip install flask chromadb requests watchdog
```

### 2. Pull the embedding model

```
ollama pull nomic-embed-text
```

This model is ~274 MB and runs on CPU — no GPU needed.

### 3. Run the app

```
python rag_chat.py
```

Open your browser to: http://localhost:5000

---

## First-time setup

On first run, open Settings (button in the project action bar) and enter:
- Your OpenRouter API key
- Your preferred model (default: openai/gpt-4o-mini)

Everything else can be left at defaults to start.

---

## Using the app

### Projects
- Type a name in the "New project name" field and press Enter or click Create Project.
- Each project gets its own isolated document index and conversation history.
- Use Rename, Delete, and Re-index buttons in the project action bar.

### Adding documents
Each project has its own docs folder. The folder path is shown in the right panel
after you select a project. Drop .txt or .md files there — the app watches the
folder and indexes new files automatically.

To manually trigger a full re-index (e.g. after adding many files), click Re-index.

### Chatting
- Type in the message box and press Enter to send.
- Shift+Enter inserts a new line.
- Each assistant reply shows which files were used as sources (expandable).
- Clear resets the conversation history but keeps the document index.

### Settings
All settings are managed from the Settings dialog in the browser:
- OpenRouter API key and model
- Ollama URL and embedding model
- Chunk size, overlap, and top-K results
- System prompt

---

## File structure

```
rag_chat/
  rag_chat.py         Main application
  config.json         Settings (managed from the UI, also editable by hand)
  projects.json       Project registry
  projects/
    my-project/
      docs/           Put your .txt and .md files here
    other-project/
      docs/
  chroma_db/          Vector database (do not delete)
  static/
    index.html        Browser interface
  README.md           This file
```

---

## Troubleshooting

**Cannot reach Ollama**
Make sure Ollama is running. Open a terminal and run: ollama serve

**Embedding model not found**
Run: ollama pull nomic-embed-text

**No API key / OpenRouter errors**
Open Settings in the browser and enter your OpenRouter key.

**Files not being indexed**
Make sure they end in .txt or .md and are in the project's docs folder.
Use the Re-index button if auto-indexing didn't pick them up.

**Port 5000 in use**
Edit the last line of rag_chat.py and change port=5000 to another port.

**Changed embedding model or chunk size?**
Delete the chroma_db/ folder entirely and re-index. The old embeddings are
incompatible with a new model or chunk size.
