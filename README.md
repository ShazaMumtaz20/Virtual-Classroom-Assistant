# Virtual Classroom Assistant Backend

AI-powered backend for a virtual classroom tutor that supports voice questions, retrieval-augmented answers, text-to-speech, and diagram-aware responses for a Unity frontend.

## Overview

This project provides the backend services for an AI teaching assistant that can:

- receive student questions through a REST API
- transcribe spoken questions with Whisper
- retrieve relevant course content from a local vector database
- generate grounded answers using an LLM
- return a diagram identifier and visual asset hint for frontend rendering
- speak responses back to the user with TTS

## Features

- FastAPI server with health and chat endpoints
- Speech-to-text support using Whisper
- Retrieval-augmented generation (RAG) with ChromaDB and sentence embeddings
- OpenAI GPT support with local Ollama fallback
- Text-to-speech support via pyttsx3
- PDF ingestion for course notes
- CORS enabled for Unity integration

## Project Structure

- main.py: FastAPI application and API routes
- rag_engine.py: retrieval logic and vector database access
- ingest.py: document ingestion and chunk indexing
- llm_handler.py: LLM response generation and fallback logic
- stt_handler.py: speech-to-text processing
- tts_handler.py: text-to-speech processing
- diagram_keywords.py: diagram detection from assistant responses
- visual_assets.py: mapping of diagram IDs to visual asset metadata
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
```

If OpenAI credentials are not available, the app can still fall back to a local Ollama instance when configured.

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

### Speech-to-Text
- POST /transcribe
  - accepts an audio file upload

### Text-to-Speech
- POST /tts
  - accepts text and returns a WAV audio response

### PDF Ingestion
- POST /ingest/pdf
  - uploads and indexes a PDF into the vector store

## Notes

- The vector database is stored locally in the chroma_db folder.
- The project is designed to be connected to a Unity frontend later.
- The current implementation focuses on a practical backend MVP for classroom tutoring and RAG-based assistance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
