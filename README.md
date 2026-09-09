# Virtual Classroom Assistant Backend

AI-powered backend for a virtual classroom tutor that supports typed questions, retrieval-augmented answers, text-to-speech, and diagram-aware responses for an avatar-style experience.

## Overview

This project provides the backend services for an AI teaching assistant that can:

- receive student questions through a REST API, including browser microphone recordings
- retrieve relevant course content from a local vector database
- generate grounded answers using an LLM
- classify the question and answer with an LLM, then return a diagram identifier and a generated ER diagram image when appropriate
- speak responses back to the user with TTS

## Current Scope

For now, the system is focused on:

- typed chat interaction only
- avatar speech output via text-to-speech
- local RAG-based question answering
- course content ingestion from text or PDF files


## Features

- FastAPI server with health and chat endpoints
- Retrieval-augmented generation (RAG) with ChromaDB and sentence embeddings
- OpenAI GPT support with local Ollama fallback
- Text-to-speech support via pyttsx3
- PDF ingestion for course notes
- LLM-based diagram classification
- Deterministic ER diagram rendering with matplotlib

## Project Structure

- main.py: FastAPI application and API routes
- rag_engine.py: retrieval logic and vector database access
- ingest.py: document ingestion and chunk indexing
- llm_handler.py: LLM response generation and fallback logic
- diagram_renderer.py: deterministic matplotlib renderer for structured ER data
- tts_handler.py: text-to-speech processing
- diagram_keywords.py: legacy keyword detector retained for reference; the live pipeline uses LLM classification in llm_handler.py
- visual_assets.py: mapping of diagram IDs to visual asset metadata
- whiteboard/: fixed assets and generated diagram PNG files
- data/: course content and uploaded files

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the project root with:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
TTS_ENGINE=edge
TTS_VOICE=en-US-JennyNeural
```

If OpenAI credentials are not available, the app can still fall back to a local Ollama instance when configured.

## Dynamic Diagrams

The live diagram flow is not based on keyword matching. After the main answer is generated, a second LLM call classifies the user's intent and answer as one of the supported diagram types:

- `db_er_diagram`
- `db_sql_query`
- `db_normalization`
- `db_transactions`
- `db_indexing`
- `db_relational_model`

ER answers receive an additional structured extraction call. That call returns entities, attributes, relationships, and cardinality as JSON. `diagram_renderer.py` uses the JSON to draw a deterministic PNG with matplotlib. No image-generation model or Graphviz binary is used.

The ER renderer is implemented and tested. The other five types are classified by the LLM but currently use their existing fixed visual assets until their JSON schemas and matplotlib renderers are added.

## Running the Server

Start the backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- http://localhost:8000/health
- http://localhost:8000/docs

## Ingesting Course Content

Place .txt or .pdf files in the data folder, then run:

```bash
python ingest.py
```

You can also upload a PDF through the API endpoint /ingest/pdf.

## API Endpoints

### Health Check

- GET /health

### Ask a Question

- POST /ask
  - accepts a question and optional chat history
  - returns the answer, diagram hint, visual asset info, and confidence
  - for `db_er_diagram`, `visual_asset.image_base64` contains a generated PNG based on the answer
  - generated ER files are also available at `visual_asset.asset_file` under `/whiteboard`
  - other diagram types currently return their fixed asset descriptor

Example request:

```json
{
  "question": "Explain an ER relationship between Customers and Orders",
  "history": []
}
```

Example generated response fields:

```json
{
  "diagram_id": "db_er_diagram",
  "visual_asset": {
    "asset_file": "whiteboard/generated/<generated-id>.png",
    "generated": "true",
    "mime_type": "image/png",
    "image_base64": "<base64 PNG data>"
  }
}
```

Unity can either decode `image_base64` directly or request:

```text
http://<backend-host>:8000/whiteboard/generated/<generated-id>.png
```

### Text-to-Speech

- POST /tts
  - accepts text and returns a WAV audio response using a neural voice by default
  - set `TTS_ENGINE=pyttsx3` for offline fallback speech
  - set `TTS_VOICE` to another Edge neural voice when needed

### Speech-to-Text

- POST /transcribe
  - accepts a multipart field named `audio`
  - accepts WAV, WebM/Opus, OGG, and MP3 recordings
  - normalizes the recording to mono 16 kHz PCM before Whisper transcription
  - send one complete recording blob; do not POST individual `MediaRecorder` timeslice chunks

### PDF Ingestion

- POST /ingest/pdf
  - uploads and indexes a PDF into the vector store

## Testing Dynamic ER Diagrams

Start the server with the project's virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Submit an ER question from another PowerShell window:

```powershell
$body = @{ question = "Explain an ER relationship between Customers and Orders"; history = @() } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body $body
$result | ConvertTo-Json -Depth 5
```

The response should contain `diagram_id: "db_er_diagram"`, `generated: "true"`, and a non-empty `image_base64` value. A valid OpenAI API key must be configured in `.env` for the LLM calls.

## Notes

- The vector database is stored locally in the chroma_db folder.
- Generated diagram files are written to `whiteboard/generated/` at runtime.
- ChromaDB telemetry warnings do not prevent the API or diagram renderer from working.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
