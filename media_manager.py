# media_manager.py
# Manages per-session audio directories and TTL-based cleanup.

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

AUDIO_BASE = Path("whiteboard") / "audio"
SESSION_TTL_SECONDS = int(os.getenv("MEDIA_SESSION_TTL_HOURS", "2")) * 3600


def session_dir(session_id: str) -> Path:
    """Return the audio directory path for a session."""
    return AUDIO_BASE / session_id


def ensure_session_dir(session_id: str) -> Path:
    """Create and return the session audio directory."""
    path = session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def audio_path(session_id: str, step_id: str) -> Path:
    """Return the full filesystem path for a step audio file."""
    return session_dir(session_id) / f"{step_id}.wav"


def audio_url(session_id: str, step_id: str) -> str:
    """Return the URL path Unity will use to download this step's audio."""
    return f"/media/audio/{session_id}/{step_id}.wav"


def delete_session(session_id: str) -> bool:
    """Delete all audio files for a session. Returns True if directory existed."""
    path = session_dir(session_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def cleanup_old_sessions() -> int:
    """
    Delete session directories older than SESSION_TTL_SECONDS.
    Returns number of sessions deleted.
    """
    if not AUDIO_BASE.exists():
        return 0

    now = time.time()
    deleted = 0
    for child in AUDIO_BASE.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
            if age > SESSION_TTL_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
                deleted += 1
        except Exception:
            pass
    return deleted


async def cleanup_old_sessions_async() -> int:
    """Async wrapper for cleanup_old_sessions."""
    return await asyncio.to_thread(cleanup_old_sessions)
