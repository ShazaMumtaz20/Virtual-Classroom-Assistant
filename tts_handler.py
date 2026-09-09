# tts_handler.py
# Text-to-speech with multiple engine support.
# Engines (set TTS_ENGINE in .env):
#   edge       — Microsoft Edge neural TTS (natural, default)
#   openai     — OpenAI TTS API (highest quality, uses OPENAI_API_KEY)
#   elevenlabs — ElevenLabs API (optional)
#   pyttsx3    — Offline fallback (robotic, use only if no internet)

from __future__ import annotations

import asyncio
import os
import tempfile
import glob
import shutil


def _ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if not local_app_data:
        return
    pattern = os.path.join(
        local_app_data, "Microsoft", "WinGet", "Packages",
        "Gyan.FFmpeg*", "ffmpeg-*-*_build", "bin", "ffmpeg.exe"
    )
    matches = glob.glob(pattern)
    if matches:
        ffmpeg_dir = os.path.dirname(matches[0])
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

_ensure_ffmpeg_on_path()

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")

# OpenAI TTS settings
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1-hd")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")   # nova | shimmer | alloy | echo | fable | onyx

# ElevenLabs settings
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Simple in-memory cache for short repeated phrases
_tts_cache: dict[str, bytes] = {}


async def text_to_speech(text: str) -> bytes:
    """
    Convert text to WAV audio bytes using the configured engine.
    Returns raw WAV bytes (or empty bytes on failure).
    """
    text = text.strip()[:500]
    if not text:
        return b""

    if text in _tts_cache:
        return _tts_cache[text]

    if TTS_ENGINE == "openai":
        audio_bytes = await _openai_tts(text)
    elif TTS_ENGINE == "edge":
        audio_bytes = await _edge_tts(text)
    elif TTS_ENGINE == "elevenlabs":
        audio_bytes = await asyncio.to_thread(_elevenlabs_tts, text)
    else:
        audio_bytes = await asyncio.to_thread(_pyttsx3_tts, text)

    if audio_bytes:
        _tts_cache[text] = audio_bytes
    return audio_bytes


async def generate_teaching_audio(
    teaching_sequence: list[dict],
    session_id: str,
) -> list[dict]:
    """
    Generate one WAV file per teaching step, saved to whiteboard/audio/<session_id>/.
    Returns the sequence with audio_url and duration_hint populated.

    Steps are generated concurrently for faster response times.
    """
    from media_manager import ensure_session_dir, audio_path, audio_url as build_url
    import time

    ensure_session_dir(session_id)

    async def _generate_step(step: dict) -> dict:
        step_copy = dict(step)
        narration = step_copy.get("narration_text", "").strip()
        step_id = step_copy.get("step_id", "step")

        if not narration:
            return step_copy

        try:
            t_start = time.time()
            wav_bytes = await text_to_speech(narration)
            elapsed = time.time() - t_start

            if wav_bytes:
                out_path = audio_path(session_id, step_id)
                out_path.write_bytes(wav_bytes)

                # Estimate duration: WAV = 44-byte header + (samples * 2 bytes)
                # 24 kHz mono → bytes after header / (24000 * 2) = seconds
                audio_data_bytes = max(len(wav_bytes) - 44, 0)
                duration = audio_data_bytes / (24000 * 2)
                step_copy["audio_url"] = build_url(session_id, step_id)
                step_copy["duration_hint"] = round(duration, 2)
        except Exception as exc:
            print(f"[TTS] Step {step_id} failed: {exc}")

        return step_copy

    updated = await asyncio.gather(*[_generate_step(s) for s in teaching_sequence])
    return list(updated)


# ---------------------------------------------------------------------------
# Engine implementations
# ---------------------------------------------------------------------------

async def _openai_tts(text: str) -> bytes:
    """
    OpenAI TTS — highest quality, requires OPENAI_API_KEY.
    Model tts-1-hd with nova voice sounds very natural and teacher-like.
    Returns WAV bytes converted from MP3.
    """
    try:
        from openai import AsyncOpenAI
        import subprocess

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key)
        response = await client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="mp3",
        )
        mp3_bytes = response.content

        # Convert MP3 → WAV for Unity compatibility
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
            f_mp3.write(mp3_bytes)
            mp3_path = f_mp3.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
            wav_path = f_wav.name

        conv = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", mp3_path,
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav_path,
            ],
            capture_output=True, check=False,
        )
        os.unlink(mp3_path)
        if conv.returncode != 0:
            os.unlink(wav_path)
            raise RuntimeError(f"ffmpeg: {conv.stderr.decode()}")

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(wav_path)
        return wav_bytes

    except Exception as exc:
        print(f"[TTS OpenAI ERROR] {exc}. Falling back to edge-tts.")
        return await _edge_tts(text)


async def _edge_tts(text: str) -> bytes:
    """
    Microsoft Edge neural TTS — natural voice, free, requires internet.
    Default voice: en-US-JennyNeural (clear teacher-style speech).
    """
    try:
        import edge_tts
        import subprocess

        # SSML-style rate/pitch adjustments for a clear, measured teacher voice
        communicate = edge_tts.Communicate(
            text,
            TTS_VOICE,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mp3_path = os.path.join(tmp_dir, "speech.mp3")
            wav_path = os.path.join(tmp_dir, "speech.wav")
            await communicate.save(mp3_path)

            conv = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", mp3_path,
                    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav_path,
                ],
                capture_output=True, check=False,
            )
            if conv.returncode != 0:
                raise RuntimeError(conv.stderr.decode().strip() or "FFmpeg failed")

            with open(wav_path, "rb") as f:
                return f.read()

    except Exception as exc:
        print(f"[TTS Edge ERROR] {exc}. Falling back to pyttsx3.")
        return await asyncio.to_thread(_pyttsx3_tts, text)


def _pyttsx3_tts(text: str) -> bytes:
    """Offline TTS fallback using pyttsx3. Clear robotic voice but no network needed."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(tmp_path)
        return wav_bytes
    except Exception as exc:
        print(f"[TTS pyttsx3 ERROR] {exc}")
        return b""


def _elevenlabs_tts(text: str) -> bytes:
    """ElevenLabs TTS — premium voice quality. Requires ELEVENLABS_API_KEY."""
    import subprocess

    if not ELEVENLABS_API_KEY:
        print("[TTS] ElevenLabs key not set. Falling back to pyttsx3.")
        return _pyttsx3_tts(text)

    try:
        import requests
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "style": 0.15},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        mp3_bytes = response.content

        # Convert MP3 → WAV
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
            f_mp3.write(mp3_bytes)
            mp3_path = f_mp3.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
            wav_path = f_wav.name

        conv = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", mp3_path, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav_path],
            capture_output=True, check=False,
        )
        os.unlink(mp3_path)
        if conv.returncode != 0:
            os.unlink(wav_path)
            return _pyttsx3_tts(text)

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(wav_path)
        return wav_bytes

    except Exception as exc:
        print(f"[TTS ElevenLabs ERROR] {exc}. Falling back to pyttsx3.")
        return _pyttsx3_tts(text)