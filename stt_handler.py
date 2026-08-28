# stt_handler.py

import glob
import os
import shutil
import subprocess
import tempfile


def _ensure_ffmpeg_on_path():
    """
    Whisper doesn't decode audio itself; it shells out to the "ffmpeg" binary.
    On Windows, a freshly-installed winget package can be missing from PATH for
    the current process even after the registry is updated (Windows Terminal /
    PowerShell tabs often keep the environment they started with instead of
    re-reading it). Rather than depend on the user restarting their terminal
    correctly, check for ffmpeg here and, if it's not resolvable, look in the
    standard winget install location and add it to THIS process's PATH.
    """
    if shutil.which("ffmpeg"):
        return  # Already resolvable - nothing to do.

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
        print(f"[STT] ffmpeg was not on PATH; added it for this process: {ffmpeg_dir}")


_ensure_ffmpeg_on_path()

try:
    import whisper
except ImportError:
    whisper = None

whisper_model = None


def _get_whisper_model():
    global whisper_model

    if whisper is None:
        raise RuntimeError("openai-whisper is not installed")

    if whisper_model is None:
        whisper_model = whisper.load_model("base")

    return whisper_model


def transcribe_audio(
    audio_bytes: bytes,
    content_type: str | None = None,
    filename: str | None = None,
) -> dict:
    """
    Accepts an uploaded audio recording, normalizes it to mono 16 kHz PCM,
    and runs Whisper transcription.
    Returns {"text": str, "language": str} or {"text": "", "error": str}.
    """
    if len(audio_bytes) < 1000:
        return {"text": "", "error": "Audio too short or empty"}

    source_suffix = ".wav"
    if content_type in {"audio/webm", "video/webm"}:
        source_suffix = ".webm"
    elif content_type in {"audio/ogg", "application/ogg"}:
        source_suffix = ".ogg"
    elif content_type == "audio/mpeg":
        source_suffix = ".mp3"
    elif filename and "." in filename:
        source_suffix = os.path.splitext(filename)[1].lower() or source_suffix

    source_path = None
    normalized_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=source_suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            source_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            normalized_path = tmp.name

        conversion = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", source_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", normalized_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if conversion.returncode != 0:
            detail = conversion.stderr.strip() or "unsupported or incomplete audio"
            return {"text": "", "error": f"Audio conversion failed: {detail}"}

        model = _get_whisper_model()
        result = model.transcribe(normalized_path, fp16=False)

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        if not text:
            return {"text": "", "error": "Could not understand audio"}

        return {"text": text, "language": language}

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return {"text": "", "error": str(e)}
    finally:
        for path in (source_path, normalized_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
