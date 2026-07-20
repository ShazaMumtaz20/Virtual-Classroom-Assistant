# ingest.py
# Run: python ingest.py
# Run this once after uploading or placing a course PDF in /data/

import os
import glob
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_notes"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400       # tokens approx
CHUNK_OVERLAP = 50


def _load_pdf_reader_text(filepath: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_txt_files(path: str) -> list[str]:
    texts = []
    for filepath in glob.glob(os.path.join(path, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            texts.append(f.read())
        print(f"  Loaded: {filepath}")
    return texts


def load_pdf_files(path: str) -> list[str]:
    """Requires: pip install pypdf"""
    texts = []
    try:
        for filepath in glob.glob(os.path.join(path, "*.pdf")):
            texts.append(_load_pdf_reader_text(filepath))
            print(f"  Loaded PDF: {filepath}")
    except ImportError:
        print("  [WARN] pypdf not installed. Skipping PDF files. Run: pip install pypdf")
    return texts


def load_pdf_file(filepath: str) -> str:
    """Load a single PDF file from disk and return its extracted text."""
    try:
        text = _load_pdf_reader_text(filepath)
        print(f"  Loaded PDF: {filepath}")
        return text
    except ImportError as exc:
        raise RuntimeError("pypdf is not installed. Install it with: pip install pypdf") from exc


def build_chunks(raw_texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    all_chunks = []
    for text in raw_texts:
        all_chunks.extend(splitter.split_text(text))
    return all_chunks


def store_chunks(all_chunks: list[str]) -> None:
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, embeddings=embeddings, ids=ids)


def ingest_texts(raw_texts: list[str], clear_existing: bool = True) -> int:
    if not raw_texts:
        return 0

    print(f"  Loaded {len(raw_texts)} document(s).\n")
    print("[2] Splitting into chunks ...")
    all_chunks = build_chunks(raw_texts)
    print(f"  Created {len(all_chunks)} chunks.\n")

    print("[3] Loading embedding model ...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[4] Embedding all chunks ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    print("\n[5] Storing in ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if clear_existing:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("  Cleared existing collection.")
        except Exception:
            pass

    collection = client.create_collection(COLLECTION_NAME)
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, embeddings=embeddings, ids=ids)

    print(f"\nIngestion complete. {len(all_chunks)} chunks stored in ChromaDB.")
    print(f"   Vector DB saved to: {CHROMA_PATH}")
    return len(all_chunks)


def ingest_file(filepath: str, clear_existing: bool = True) -> int:
    suffix = Path(filepath).suffix.lower()
    if suffix == ".txt":
        raw_texts = [Path(filepath).read_text(encoding="utf-8")]
    elif suffix == ".pdf":
        raw_texts = [load_pdf_file(filepath)]
    else:
        raise ValueError("Only .txt and .pdf files are supported")

    return ingest_texts(raw_texts, clear_existing=clear_existing)


def main():
    print("=== Virtual Classroom Assistant - Document Ingestion ===\n")

    print("[1] Loading documents from /data ...")
    raw_texts = load_txt_files(DATA_PATH) + load_pdf_files(DATA_PATH)

    if not raw_texts:
        print("ERROR: No .txt or .pdf files found in /data. Upload or place a textbook PDF and re-run.")
        return
    ingest_texts(raw_texts)
    print("\nYou can now start the backend with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()