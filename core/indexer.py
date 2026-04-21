import chromadb
from pathlib import Path
from typing import List
from core.config_manager import get_active_config


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
    """Indexes text chunks into the vector database."""
    # 1. Establish the connection inside this function
    persist_directory = f"projects/{project_name}/embeddings"
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 2. Define 'collection'
    collection = client.get_or_create_collection(name=project_name)
    
    # 3. Now the rest of your indexing logic will work
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def delete_file_from_index(project_name: str, filename: str):
    # 1. Get the configuration (We just moved this to config_manager)
    config = get_active_config(project_name)
    
    # 2. Initialize the Chroma client
    # (Using the path logic you have elsewhere in the project)
    persist_directory = f"projects/{project_name}/embeddings"
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 3. GET the collection (This defines the 'collection' variable)
    collection = client.get_or_create_collection(name=project_name)
    
    # 4. Now 'collection' is defined and you can run the delete
    collection.delete(where={"source": filename})
    print(f"Removed {filename} from vector index.")


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
