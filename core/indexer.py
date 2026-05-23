import chromadb
from pathlib import Path
from typing import List
from core.config_manager import get_active_config, PROJECTS_DIR
import os
import logging

# 1. Tell Hugging Face Hub to remain quiet and disable progress meters
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 2. Silence the high-volume loggers that Chroma dependencies spin up
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def get_vector_db(project_name: str, config: dict):
    """
    Initializes or connects to a local ChromaDB for a specific project.
    Standardizes on the 'embeddings' directory within the project folder.
    """
    db_path = PROJECTS_DIR / project_name / "embeddings"
    
    provider = config.get("embed_provider", "builtin")

    if provider == "builtin":
        # Swapped out heavy SentenceTransformers for the lightweight ONNX engine
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        selected_ef = ONNXMiniLM_L6_V2()
    else:
        try:
            from providers.embeddings import get_remote_embedding_function
            selected_ef = get_remote_embedding_function(config)
        except ImportError:
            # Fallback block updated to look for the same lightweight ONNX engine
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            selected_ef = ONNXMiniLM_L6_V2()

    client = chromadb.PersistentClient(path=str(db_path))
    
    # We name the collection after the project for consistency
    return client.get_or_create_collection(
        name=project_name,
        embedding_function=selected_ef
    )

def query_vector_db(project_name: str, query_text: str, n_results: int = 5):
    """
    Called by chat_router.py to find relevant context for the AI.
    """
    config = get_active_config(project_name)
    # This ensures we use the SAME embedding function used during indexing
    collection = get_vector_db(project_name, config)
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    if results['documents'] and len(results['documents'][0]) > 0:
        return "\n---\n".join(results['documents'][0])
    
    return "No specific project context found."

def chunk_text(text: str, config: dict) -> List[str]:
    """Breaks text into pieces based on the user's project settings."""
    method = config.get("embed_method", "size") 
    
    if method == "full":
        return [text]

    if method == "delimiter":
        delimiter = config.get("delimiter", "\n\n")
        chunks = text.split(delimiter)
        return [c.strip() for c in chunks if c.strip()]

    # Default: Size (Character based)
    size = int(config.get("chunk_size") or 500)
    overlap = int(config.get("chunk_overlap") or 50)
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    
    return chunks

def index_file_chunks(project_name: str, filename: str, chunks: list):
    """Indexes text chunks into the vector database."""
    config = get_active_config(project_name)
    collection = get_vector_db(project_name, config)
    
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def delete_file_from_index(project_name: str, filename: str):
    """Removes a specific file's data from the index."""
    config = get_active_config(project_name)
    collection = get_vector_db(project_name, config)
    collection.delete(where={"source": filename})

def clear_entire_index(project_name: str):
    """Nukes the entire collection for a project."""
    db_path = PROJECTS_DIR / project_name / "embeddings"
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(name=project_name)
    except Exception:
        pass

def reindex_single_file(project_name: str, file_path: Path):
    """Processes a single file: deletes old index, chunks, and re-adds."""
    from core.config_manager import get_active_config
    
    config = get_active_config(project_name)
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    
    # 1. Remove existing entries for this file
    delete_file_from_index(project_name, file_path.name)
    
    # 2. Generate new chunks based on CURRENT settings
    chunks = chunk_text(content, config)
    
    # 3. Add to vector DB
    count = index_file_chunks(project_name, file_path.name, chunks)
    return count

def sync_all_project_files(project_name: str):
    """
    Scans the project's files directory and ensures everything is indexed.
    This is the main 'Update AI Memory' logic.
    """
    from core.config_manager import PROJECTS_DIR
    files_dir = PROJECTS_DIR / project_name / "files"
    
    if not files_dir.exists():
        return 0

    total_chunks = 0
    for file_path in files_dir.iterdir():
        if file_path.is_file():
            total_chunks += reindex_single_file(project_name, file_path)
            
    return total_chunks