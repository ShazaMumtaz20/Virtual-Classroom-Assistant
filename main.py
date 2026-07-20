# main.py
# Start with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import os
from pathlib import Path
import shutil
import uuid

load_dotenv()  # Load OPENAI_API_KEY from .env

from rag_engine import retrieve_context_bundle, get_collection_count
from llm_handler import generate_response, get_llm_backend_name
from stt_handler import transcribe_audio
from tts_handler import text_to_speech
from ingest import ingest_file

app = FastAPI(
    title="Virtual Classroom Assistant API",
    description="Backend for AI-powered avatar teacher. Exposes /ask, /transcribe, /tts, /health.",
    version="1.0.0"
)

# Allow Unity (localhost) to call the API without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class HistoryItem(BaseModel):
    role: str           # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default=[], max_length=6)


class AskResponse(BaseModel):
    response: str
    diagram_id: str | None
    visual_asset: dict | None = None
    confidence: float


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    vectordb_docs: int


# --- Routes ---

@app.get("/health")
async def health_check():
    """
    Unity calls this on startup to confirm the backend is running.
    Returns LLM mode and how many chunks are in the vector DB.
    """
    llm_mode = "openai" if os.getenv("OPENAI_API_KEY") else "none"
    return {
        "status": "ok",
        "llm": get_llm_backend_name() if llm_mode != "none" else "ollama",
        "vectordb_docs": get_collection_count()
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Main Q&A endpoint. Runs the full RAG + LLM pipeline.
    Unity sends: { "question": "...", "history": [...] }
    Returns: { "response": "...", "diagram_id": "..." | null, "confidence": 1.0 }
    """
    history = [item.model_dump() for item in request.history]

    # Step 1: Retrieve relevant context from ChromaDB
    retrieval = retrieve_context_bundle(request.question)
    context = retrieval["context"]

    # Step 2: Generate LLM response with context + history
    response_text, diagram_id, visual_asset = await generate_response(
        question=request.question,
        context=context,
        history=history
    )

    confidence = retrieval["retrieval_confidence"]
    if "don't have information" in response_text.lower():
        confidence = min(confidence, 0.35)
    else:
        confidence = min(1.0, round(0.35 + confidence * 0.65, 2))

    return AskResponse(
        response=response_text,
        diagram_id=diagram_id,
        visual_asset=visual_asset,
        confidence=confidence
    )


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    STT endpoint. Unity uploads a WAV file as multipart form data.
    Returns: { "text": "transcribed text", "language": "en" }
    """
    if audio.content_type not in ["audio/wav", "audio/wave", "application/octet-stream"]:
        # Accept any binary type since Unity may not set content-type perfectly
        pass

    audio_bytes = await audio.read()
    result = transcribe_audio(audio_bytes)

    if "error" in result and not result.get("text"):
        return {"text": "", "error": result["error"]}

    return result


@app.post("/tts")
async def tts(request: TTSRequest):
    """
    TTS endpoint. Returns raw WAV audio bytes.
    Unity receives this as binary and loads it into an AudioClip.
    """
    audio_bytes = text_to_speech(request.text)

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"}
    )


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(notes: UploadFile = File(...)):
    """
    Upload a textbook PDF and index it into ChromaDB.
    This is the preferred path for course notes.
    """
    if not notes.filename or not notes.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    uploads_dir = Path("data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid.uuid4().hex}_{Path(notes.filename).name}"
    stored_path = uploads_dir / stored_name

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(notes.file, buffer)

    try:
        chunks_stored = ingest_file(str(stored_path), clear_existing=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {exc}") from exc

    return IngestResponse(
        filename=stored_name,
        chunks_stored=chunks_stored,
        vectordb_docs=get_collection_count(),
    )


# --- Entry Point ---

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)