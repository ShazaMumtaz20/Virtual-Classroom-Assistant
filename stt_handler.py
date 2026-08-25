# stt_handler.py

import glob
import os
import shutil
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
        "Gyan.FFmpeg_*", "ffmpeg-*-full_build", "bin", "ffmpeg.exe"
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

        model = _get_whisper_model()
        result = model.transcribe(tmp_path, fp16=False)
        os.unlink(tmp_path)  # Clean up temp file

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        if not text:
            return {"text": "", "error": "Could not understand audio"}

        return {"text": text, "language": language}

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return {"text": "", "error": str(e)}
