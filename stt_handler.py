# stt_handler.py

import whisper
import tempfile
import os

# Load Whisper model once at startup. Use "base" for speed; "small" for better accuracy.
whisper_model = whisper.load_model("base")


def transcribe_audio(audio_bytes: bytes) -> dict:
    """
    Accepts raw WAV audio as bytes.
    Saves to a temp file, runs Whisper transcription, cleans up.
    Returns {"text": str, "language": str} or {"text": "", "error": str}.
    """
    if len(audio_bytes) < 1000:
        return {"text": "", "error": "Audio too short or empty"}

    try:
        # Write audio bytes to a temp WAV file (Whisper requires a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path, fp16=False)
        os.unlink(tmp_path)  # Clean up temp file

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        if not text:
            return {"text": "", "error": "Could not understand audio"}

        return {"text": text, "language": language}

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return {"text": "", "error": str(e)}