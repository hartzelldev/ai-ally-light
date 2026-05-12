# AI Ally Light (v0.5 Beta)

**AI Ally Light** is a modular, text-first AI development toolbox designed for power users and developers who value accessibility, local data control, and extensible AI workflows. Built with a focus on screen-reader compatibility and a "Safe-by-Design" security posture, it bridges the gap between complex AI agents and the streamlined efficiency of a command-line interface.

## Key Features

* **Accessibility-First UI:** A lean, semantic HTML/CSS interface optimized for screen readers, featuring ARIA-live regions, skip-links, and logical focus management.
* **Intelligent RAG (Retrieval-Augmented Generation):** A sentence-level chunking and vector search pipeline that provides the LLM with local project context for precise coding assistance.
* **Dynamic Project Management:** Create, rename, and manage multiple isolated projects with dedicated document stores, thread histories, and individual configuration overrides.
* **Multi-Provider LLM Integration:** Native support for OpenRouter, Groq, TogetherAI, and local **Ollama** instances for both chat and embeddings.
* **Persistent Session State:** Automatically resumes your last active thread upon project entry, ensuring a seamless transition between configuration, file management, and creative writing.

## Technical Stack

* **Backend:** Python 3.13, NiceGUI (FastAPI based).
* **Vector Database:** ChromaDB for local embedding storage and retrieval.
* **AI Integration:** OpenAI-compatible API support, Ollama for local model hosting.
* **Deployment:** Standalone executable support via PyInstaller, allowing for zero-install portability on Windows.

## Getting Started 

### user

1. Download the Windows portable version from [https://hartzelldev.github.io](https://hartzelldev.github.io).
2. Unzip the archive anywhere (your user space C:\Users\YOUR_USERNAME\ is recommended).
3. Execute ally.exe.

### developers

1.  **Clone & Install:**
    ```bash
    git clone https://github.com/hartzelldev/ai-ally-light.git
    cd ai-ally-light
    pip install -r requirements.txt
    ```

2.  **Environment Setup:**
    Create a `.env` file or use the built-in Settings UI to configure your **OpenRouter** or **Ollama** credentials.

3.  **Run:**
    ```bash
    python main.py
    ```
    
## Accessibility Notes

This project is designed to be fully navigable via keyboard and screen reader.
* **Status Announcements:** All system changes (e.g., "Settings Saved," "Indexing Started") are broadcast via `aria-live` regions.
* **Semantic Landmarks:** Uses `<nav>`, `<main>`, and `<aside>` to allow quick navigation between project lists, chat areas, and file information.
* **Interactive Elements:** Buttons and inputs include clear labels and focus indicators to ensure a predictable user experience.
* **Audio notifications:** Audio alerts can be turned on in 'Settings' to alert on a chat reply, error messages, or any notices.

## Security Posture

* **Local-First:** All document indexing and vector storage happen locally on your machine.
* **Privacy Focused:** Unlike commercial AI wrappers, AI Ally Light does not track user prompts or project metadata; your data stays within your projects/ directory.
* **Path Sanitization:** Uses `secure_filename` and strict path validation to prevent directory traversal vulnerabilities during file uploads and project management.

---

**Developed by an IT Security Professional** with a focus on creating "The Equalizer"—an AI toolbox that works for everyone, regardless of how they interact with their screen.

## Contact

To reach the author, please create an **Issue** through the [GITHub repository](https://github.com/hartzelldev/ai-ally-light) or fill out [this contact form](https://hartzelldev.github.io/contact.html).
                    