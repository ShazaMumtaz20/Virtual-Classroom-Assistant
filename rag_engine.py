# rag_engine.py

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_notes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3  # Number of chunks to retrieve per query

# Load the embedding model once at module load (cached in memory)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single string into a vector using the local model."""
    return embed_model.encode(text).tolist()


def _distance_to_confidence(distance: float | None) -> float:
    if distance is None:
        return 0.0
    try:
        return max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
    except (TypeError, ValueError):
        return 0.0


def retrieve_context_bundle(question: str, top_k: int = TOP_K) -> dict:
    """
    Returns the retrieved context together with a lightweight retrieval
    confidence estimate based on the nearest ChromaDB distance.
    """
    if collection.count() == 0:
        return {
            "context": "No course content has been ingested yet. Please run ingest.py first.",
            "retrieval_confidence": 0.0,
            "chunks": [],
        }

    query_vector = embed_text(question)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances"]
    )

    chunks = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []
    if not chunks:
        return {
            "context": "No relevant content found in the course material.",
            "retrieval_confidence": 0.0,
            "chunks": [],
        }

    closest_distance = min(distances) if distances else None
    return {
        "context": "\n\n---\n\n".join(chunks),
        "retrieval_confidence": _distance_to_confidence(closest_distance),
        "chunks": chunks,
    }


def retrieve_context(question: str, top_k: int = TOP_K) -> str:
    """
    Embeds the question and queries ChromaDB for the top_k most similar chunks.
    Returns retrieved chunks joined as a single context string.
    Returns a fallback message if the vector store is empty.
    """
    return retrieve_context_bundle(question, top_k=top_k)["context"]


def get_collection_count() -> int:
    """Returns the number of documents currently stored in ChromaDB."""
    return collection.count()