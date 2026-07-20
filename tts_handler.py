# tts_handler.py

import os
import pyttsx3
import tempfile

TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")

# Simple cache for common phrases to avoid regenerating audio repeatedly
_tts_cache: dict[str, bytes] = {}


def text_to_speech(text: str) -> bytes:
    """
    Converts text to WAV audio bytes.
    Uses pyttsx3 (offline) by default.
    Switch TTS_ENGINE env var to "elevenlabs" for demo day.
    Returns raw WAV bytes.
    """
    text = text.strip()[:500]  # Cap at 500 characters

    if text in _tts_cache:
        return _tts_cache[text]

    if TTS_ENGINE == "elevenlabs":
        audio_bytes = _elevenlabs_tts(text)
    else:
        audio_bytes = _pyttsx3_tts(text)

    if audio_bytes:
        _tts_cache[text] = audio_bytes

    return audio_bytes


def _pyttsx3_tts(text: str) -> bytes:
    """Offline TTS using pyttsx3. Returns WAV bytes."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)  # Speech speed (words per minute)
        engine.setProperty("volume", 1.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        os.unlink(tmp_path)
        return audio_bytes

    except Exception as e:
        print(f"[TTS pyttsx3 ERROR] {e}")
        return b""


def _elevenlabs_tts(text: str) -> bytes:
    """ElevenLabs TTS (better voice quality for demo day). Returns WAV bytes."""
    import requests

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("[TTS] ElevenLabs key not set. Falling back to pyttsx3.")
        return _pyttsx3_tts(text)

    # Default ElevenLabs voice ID - change this to your preferred voice
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.content  # Returns MP3 bytes (Unity must handle both WAV and MP3)
    except Exception as e:
        print(f"[TTS ElevenLabs ERROR] {e}")
        return _pyttsx3_tts(text)  # Fallback to pyttsx3