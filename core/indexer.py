import chromadb
from pathlib import Path
from typing import List

def get_vector_db(project_name: str, config: dict):
    """
    Initializes or connects to a local ChromaDB for a specific project.
    Lazy-loads the embedding function based on the project configuration.
    """
    project_path = Path("projects") / project_name
    db_path = project_path / "chroma_db"
    
    # 1. Determine which embedding function to use based on config
    provider = config.get("embed_provider", "builtin")

    if provider == "builtin":
        # We import and initialize ONLY if the user is using the local model
        from chromadb.utils import embedding_functions
        selected_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    else:
        # If you've built a cloud provider logic in providers/embeddings.py, call it here.
        # For now, we'll import it locally to keep startup fast.
        try:
            from providers.embeddings import get_remote_embedding_function
            selected_ef = get_remote_embedding_function(config)
        except ImportError:
            # Fallback if the remote provider logic isn't ready yet
            from chromadb.utils import embedding_functions
            selected_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

    # Persistent client saves data to disk inside the project folder
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Use the selected_ef instead of the old global default_ef
    return client.get_or_create_collection(
        name=f"{project_name}_collection",
        embedding_function=selected_ef
    )

def chunk_text(text: str, config: dict) -> List[str]:
    """
    Breaks text into pieces based on the user's project settings.
    """
    method = config.get("embed_method", "Fixed Size")
    
    if method == "Full File":
        return [text]

    if method == "Delimiter":
        # Uses the delimiter from settings, or defaults to double newline
        delimiter = config.get("delimiter", "\n\n")
        chunks = text.split(delimiter)
        return [c.strip() for c in chunks if c.strip()]

    # Default: Fixed Size (Character based)
    size = int(config.get("chunk_size", 500))
    overlap = int(config.get("chunk_overlap", 50))
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    
    return chunks

def index_file_chunks(project_name: str, filename: str, chunks: list):
    """Takes a list of strings and saves them into the vector database."""
    config = get_active_config(project_name)
    
    # Generate unique IDs for every chunk (e.g., "bio.txt_0", "bio.txt_1")
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    return len(chunks)

def delete_file_from_index(project_name: str, filename: str):
    """Removes all chunks associated with a specific filename."""
    config = get_active_config(project_name)
    # We use the metadata 'source' we set during indexing
    collection.delete(where={"source": filename})

def clear_entire_index(project_name: str):
    """Nukes the entire collection for a project."""
    project_path = Path("projects") / project_name
    db_path = project_path / "chroma_db"
    
    # Simple way: connect and delete the collection
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(name=f"{project_name}_collection")
    except Exception:
        pass # Collection might not exist yet
