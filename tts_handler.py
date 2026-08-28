# tts_handler.py

import asyncio
import os
import pyttsx3
import subprocess
import tempfile

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")

# Simple cache for common phrases to avoid regenerating audio repeatedly
_tts_cache: dict[str, bytes] = {}


async def text_to_speech(text: str) -> bytes:
    """
    Converts text to natural-sounding WAV audio bytes.
    Uses Microsoft Edge's neural voice by default. Set TTS_ENGINE=pyttsx3
    for offline speech or TTS_ENGINE=elevenlabs for ElevenLabs.
    Returns raw WAV bytes.
    """
    text = text.strip()[:500]  # Cap at 500 characters

    if text in _tts_cache:
        return _tts_cache[text]

    if TTS_ENGINE == "edge":
        audio_bytes = await _edge_tts(text)
    elif TTS_ENGINE == "elevenlabs":
        audio_bytes = _elevenlabs_tts(text)
    else:
        audio_bytes = _pyttsx3_tts(text)

    if audio_bytes:
        _tts_cache[text] = audio_bytes

    return audio_bytes


async def _edge_tts(text: str) -> bytes:
    """Generate neural speech and convert it to WAV for the existing API contract."""
    try:
        import edge_tts

        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = os.path.join(temp_dir, "speech.mp3")
            wav_path = os.path.join(temp_dir, "speech.wav")
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            await communicate.save(mp3_path)

            conversion = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", mp3_path,
                    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if conversion.returncode != 0:
                raise RuntimeError(conversion.stderr.strip() or "FFmpeg conversion failed")

            with open(wav_path, "rb") as audio_file:
                return audio_file.read()
    except Exception as error:
        print(f"[TTS Edge ERROR] {error}. Falling back to pyttsx3.")
        return _pyttsx3_tts(text)


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