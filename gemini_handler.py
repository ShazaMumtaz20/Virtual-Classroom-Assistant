# gemini_handler.py
# Handles Gemini image generation for illustration-type visual content.
# Only called when OpenAI determines visual_type == "generated_image".
# API key never leaves the server.

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from uuid import uuid4

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation")


async def generate_image(prompt: str, save_dir: str = "whiteboard/generated") -> str | None:
    """
    Generate a conceptual illustration using Gemini.
    Returns the URL path (e.g. '/whiteboard/generated/<id>.png') or None on failure.

    Only use this for illustration-type visuals, NOT for precise ER diagrams or tables.
    Precise structured diagrams should use diagram_renderer.py instead.
    """
    if not GEMINI_API_KEY:
        print("[GEMINI] GEMINI_API_KEY not set — skipping image generation.")
        return None

    try:
        return await asyncio.to_thread(_generate_image_sync, prompt, save_dir)
    except Exception as exc:
        print(f"[GEMINI ERROR] {exc}")
        return None


def _generate_image_sync(prompt: str, save_dir: str) -> str | None:
    """Synchronous Gemini image generation using the REST API."""
    import urllib.request
    import json

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4().hex}.png"
    output_path = save_path / filename

    # Use google-genai SDK if available, otherwise use REST
    try:
        return _generate_via_sdk(prompt, output_path)
    except ImportError:
        return _generate_via_rest(prompt, output_path)


def _generate_via_sdk(prompt: str, output_path: Path) -> str | None:
    """Use google-generativeai SDK for image generation."""
    import google.generativeai as genai  # type: ignore
    from PIL import Image  # type: ignore
    import io

    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel(model_name=GEMINI_IMAGE_MODEL)

    enhanced_prompt = (
        f"Educational diagram for a virtual classroom database course. "
        f"Clean, clear, professional illustration. White background. "
        f"Topic: {prompt}"
    )

    response = client.generate_content(
        enhanced_prompt,
        generation_config={"response_modalities": ["IMAGE", "TEXT"]},
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = base64.b64decode(part.inline_data.data)
            output_path.write_bytes(img_bytes)
            return f"/whiteboard/generated/{output_path.name}"

    print("[GEMINI] No image part found in response.")
    return None


def _generate_via_rest(prompt: str, output_path: Path) -> str | None:
    """Fallback: direct REST call to Gemini API for image generation."""
    import urllib.request
    import json

    enhanced_prompt = (
        f"Educational diagram for a virtual classroom database course. "
        f"Clean, clear, professional illustration on a white background. "
        f"Topic: {prompt}"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_IMAGE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    body = json.dumps({
        "contents": [{"parts": [{"text": enhanced_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    candidates = data.get("candidates", [])
    if not candidates:
        print("[GEMINI REST] No candidates returned.")
        return None

    for part in candidates[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData", {})
        if inline.get("mimeType", "").startswith("image/"):
            img_bytes = base64.b64decode(inline["data"])
            output_path.write_bytes(img_bytes)
            return f"/whiteboard/generated/{output_path.name}"

    print("[GEMINI REST] No image found in response parts.")
    return None
