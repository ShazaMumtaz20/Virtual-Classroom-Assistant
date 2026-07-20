# llm_handler.py

import asyncio
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from openai import AsyncOpenAI

from diagram_keywords import detect_diagram
from visual_assets import resolve_visual_asset

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = """You are a friendly, patient AI teacher inside a virtual classroom.
Your job is to explain concepts clearly and concisely to university students.
Answer ONLY from the provided course context. If the context does not contain 
enough information to answer, say: "I don't have information on that in my course notes."
Keep responses under 150 words. Speak directly to the student. Avoid markdown formatting."""

MAX_HISTORY_TURNS = 6  # Max conversation turns to include in context (3 pairs)

FALLBACK_RESPONSE = "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."


def _build_messages(question: str, context: str, history: list[dict] | None = None) -> list[dict]:
    history = history or []
    trimmed_history = history[-MAX_HISTORY_TURNS:]

    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCourse Context:\n{context}"},
        *trimmed_history,
        {"role": "user", "content": question},
    ]


async def _call_openai(messages: list[dict]) -> str:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _call_ollama_sync(messages: list[dict]) -> str:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 300,
            },
        }
    ).encode("utf-8")

    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    message = response_data.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


async def _call_ollama(messages: list[dict]) -> str:
    return await asyncio.to_thread(_call_ollama_sync, messages)


def _backend_status() -> str:
    if client is not None:
        return "openai"
    return "ollama"


async def generate_response(
    question: str,
    context: str,
    history: list[dict] | None = None
) -> tuple[str, str | None, dict[str, str] | None]:
    """
    Builds the LLM prompt with context and history, then tries OpenAI first
    and falls back to local Ollama/Llama 3.1 if the remote call fails.
    Returns (response_text, diagram_id | None, visual_asset | None).
    """
    messages = _build_messages(question, context, history)

    try:
        response_text = await _call_openai(messages)
    except Exception as openai_error:
        print(f"[LLM OPENAI ERROR] {openai_error}")
        try:
            response_text = await _call_ollama(messages)
        except (URLError, TimeoutError, RuntimeError, Exception) as ollama_error:
            print(f"[LLM OLLAMA ERROR] {ollama_error}")
            return FALLBACK_RESPONSE, None, None

    diagram_id = detect_diagram(response_text)
    visual_asset = resolve_visual_asset(diagram_id)
    return response_text, diagram_id, visual_asset


def get_llm_backend_name() -> str:
    """Returns the configured preferred backend name for health reporting."""
    return _backend_status()