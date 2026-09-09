# main.py
# Start with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import os
from pathlib import Path
import shutil
import uuid

load_dotenv()

from rag_engine import retrieve_context_bundle, get_collection_count
from llm_handler import generate_response, get_llm_backend_name
from stt_handler import transcribe_audio
from tts_handler import text_to_speech, generate_teaching_audio
from ingest import ingest_file
import media_manager

app = FastAPI(
    title="Virtual Classroom Assistant API",
    description=(
        "Backend for AI-powered avatar teacher. "
        "Endpoints: /ask, /transcribe, /tts, /health, /ingest/pdf, /session/{id}"
    ),
    version="2.0.0"
)

# CORS — allow Unity (and browser debug clients) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve whiteboard images (generated ER diagrams, etc.)
whiteboard_dir = Path("whiteboard")
whiteboard_dir.mkdir(exist_ok=True)
app.mount("/whiteboard", StaticFiles(directory=str(whiteboard_dir)), name="whiteboard")

# Serve per-step audio files for the teaching sequence
audio_dir = Path("whiteboard") / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media/audio", StaticFiles(directory=str(audio_dir)), name="media_audio")


# ---------------------------------------------------------------------------
# Request / Response models (import from models.py for the new types)
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default=[], max_length=6)


class AskResponse(BaseModel):
    # --- Legacy fields (preserved for Unity backward compatibility) ---
    response: str
    diagram_id: str | None = None
    visual_asset: dict | None = None
    confidence: float

    # --- New teaching fields ---
    session_id: str | None = None
    visual_required: bool = False
    visual_type: str | None = None
    visual_spec: dict | None = None      # Structured visual data for Unity
    image_url: str | None = None         # URL to Gemini-generated or rendered image
    teaching_sequence: list[dict] = []   # TeachingStep list with audio_url per step


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    vectordb_docs: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """
    Unity calls this on startup to confirm the backend is up.
    Returns LLM backend name and ChromaDB chunk count.
    """
    return {
        "status": "ok",
        "llm": get_llm_backend_name(),
        "vectordb_docs": get_collection_count(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Main teaching endpoint — runs the full RAG + LLM + visual + TTS pipeline.

    Unity sends: { "question": "...", "history": [...] }

    Returns:
      - response: chat panel answer text
      - teaching_sequence: list of steps, each with visual_action, narration,
        subtitle, and audio_url pointing to a served WAV file
      - visual_spec: structured diagram data for Unity to animate
      - image_url: URL to the rendered PNG (ER diagram, table, etc.)
      - session_id: use with DELETE /session/{id} to interrupt a lesson
    """
    history = [item.model_dump() for item in request.history]

    # 1. Retrieve RAG context
    retrieval = retrieve_context_bundle(request.question)
    context = retrieval["context"]

    # 2. Generate teaching plan (OpenAI structured call → visual + sequence)
    (
        response_text,
        diagram_id,
        visual_asset,
        teaching_plan,
        teaching_sequence,
    ) = await generate_response(
        question=request.question,
        context=context,
        history=history,
    )

    # 3. Confidence scoring (existing logic preserved)
    confidence = retrieval["retrieval_confidence"]
    if "don't have information" in response_text.lower():
        confidence = min(confidence, 0.35)
    else:
        confidence = min(1.0, round(0.35 + confidence * 0.65, 2))

    # 4. Generate per-step TTS audio (concurrent)
    session_id: str | None = None
    if teaching_sequence:
        session_id = uuid.uuid4().hex
        try:
            # Trigger background cleanup of old sessions occasionally
            if hash(session_id) % 20 == 0:
                import asyncio
                asyncio.create_task(media_manager.cleanup_old_sessions_async())

            teaching_sequence = await generate_teaching_audio(teaching_sequence, session_id)
        except Exception as tts_exc:
            print(f"[TTS SEQUENCE ERROR] {tts_exc}")
            # Non-fatal — Unity can still display visuals + subtitles without audio

    return AskResponse(
        # Legacy
        response=response_text,
        diagram_id=diagram_id,
        visual_asset=visual_asset,
        confidence=confidence,
        # New
        session_id=session_id,
        visual_required=teaching_plan.get("visual_required", False),
        visual_type=teaching_plan.get("visual_type"),
        visual_spec=teaching_plan.get("visual_spec"),
        image_url=teaching_plan.get("image_url"),
        teaching_sequence=teaching_sequence,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Interrupt an active lesson: delete the session's audio files.
    Unity should call this before starting a new /ask to stop old audio.
    Unity is responsible for stopping audio playback on its side.
    """
    deleted = media_manager.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    STT endpoint. Upload WAV, WebM/Opus, OGG, or MP3.
    Returns: { "text": "...", "language": "en" }
    """
    audio_bytes = await audio.read()
    result = transcribe_audio(
        audio_bytes,
        content_type=audio.content_type,
        filename=audio.filename,
    )
    if "error" in result and not result.get("text"):
        return {"text": "", "error": result["error"]}
    return result


@app.post("/tts")
async def tts(request: TTSRequest):
    """
    TTS endpoint. Returns raw WAV audio bytes.
    Unity receives this as binary and loads it into an AudioClip.
    Prefer using teaching_sequence[*].audio_url for synchronized teaching.
    """
    audio_bytes = await text_to_speech(request.text)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"},
    )


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(notes: UploadFile = File(...)):
    """Upload a course PDF and index it into ChromaDB."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)