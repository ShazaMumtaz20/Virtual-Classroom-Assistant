# llm_handler.py
# Core LLM orchestration for the Virtual Classroom teaching pipeline.
#
# Main flow:
#   1. RAG context retrieved upstream
#   2. Single OpenAI structured call → answer + visual_spec + teaching_sequence
#   3. Visual rendered (matplotlib or Gemini) based on visual_type
#   4. Returns TeachingPlan (Pydantic), which main.py uses to generate TTS

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MAX_HISTORY_TURNS = 6
FALLBACK_RESPONSE = "I'm having trouble connecting right now. Please try again in a moment."

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI teacher inside a virtual classroom teaching DATABASE SYSTEMS.
Your name is Aria. You teach university students clearly and patiently.

When answering a question you MUST return a JSON object with this EXACT structure:
{
  "answer": "The chat-panel text answer. Clear, accurate, under 180 words. No markdown. No bullet points.",
  "visual_required": true | false,
  "visual_type": "er_diagram" | "table" | "normalization" | "comparison" | "flowchart" | "architecture" | "generated_image" | null,
  "visual_spec": { ... },
  "teaching_sequence": [
    {
      "step_id": "step_1",
      "visual_action": { "type": "show", "targets": ["EntityName"], "data": {} },
      "narration_text": "What the teacher says out loud. Natural, conversational, educational. 1-3 sentences.",
      "subtitle_text": "Same or shorter version for on-screen display."
    }
  ]
}

VISUAL TYPE SELECTION (AI decides — student does NOT choose):
- er_diagram: entity-relationship diagrams, table relationships, cardinality, schemas
- table: SQL table structure, row/column display, query result sets
- normalization: 1NF/2NF/3NF transformation steps
- comparison: side-by-side comparisons (INNER JOIN vs LEFT JOIN, PK vs FK, etc.)
- flowchart: process flows (query execution, transaction lifecycle, index lookup)
- architecture: database architecture, system components
- generated_image: conceptual illustrations where a realistic image adds more value than a diagram
- null: purely conceptual explanations that need no visual

VISUAL_SPEC STRUCTURE by type:

er_diagram:
{
  "title": "...",
  "entities": [
    {"name": "TableName", "attributes": [{"name": "col_name", "key": "PK|FK|UK|"}]}
  ],
  "relationships": [
    {"from": "Table1", "to": "Table2", "label": "verb phrase", "cardinality": "1:N|N:M|1:1"}
  ]
}

table:
{
  "title": "...",
  "tables": [
    {"name": "TableName", "columns": [{"name": "col", "type": "INT", "key": "PK|FK|"}], "rows": [["val1","val2"]]}
  ]
}

normalization:
{
  "title": "...",
  "norm_steps": [
    {"label": "Unnormalized", "table": {"name":"...", "columns":[...], "rows":[[...]]}, "explanation": "..."},
    {"label": "1NF", "table": {...}, "explanation": "..."},
    {"label": "2NF", "table": {...}, "explanation": "..."},
    {"label": "3NF", "table": {...}, "explanation": "..."}
  ]
}

comparison:
{
  "title": "...",
  "tables": [
    {"name": "LeftSide (e.g. INNER JOIN)", "columns": [...], "rows": [...]},
    {"name": "RightSide (e.g. LEFT JOIN)", "columns": [...], "rows": [...]}
  ],
  "annotations": ["Key difference: ...", "Use LEFT JOIN when..."]
}

flowchart:
{
  "title": "...",
  "nodes": [{"id": "n1", "label": "Parse SQL", "shape": "box"}, {"id": "n2", "label": "Decision?", "shape": "diamond"}],
  "edges": [{"from": "n1", "to": "n2", "label": "next"}]
}

architecture:
{
  "title": "...",
  "nodes": [...],
  "edges": [...],
  "annotations": [...]
}

generated_image:
{
  "title": "...",
  "prompt": "Detailed image generation prompt describing the visual."
}

TEACHING SEQUENCE RULES:
- Generate 3-8 steps depending on concept complexity.
- Each step reveals ONE new element (one entity, one column, one relationship, one arrow, etc.).
- narration_text: Natural teacher speech. Avoid robotic phrases. Use conversational transitions:
  "Let's start with...", "Now notice...", "Here's the important part...", "Next we have...", "Finally..."
- narration_text must NEVER mention JSON, animations, steps, or internal implementation.
- narration_text must sound like a real teacher explaining at a whiteboard.
- subtitle_text can be the same as narration_text or a slightly shorter version.

VISUAL ACTION TYPES: show, hide, fade_in, fade_out, draw, connect, highlight, emphasize,
reveal, move, replace, transform, zoom, show_label, show_table, show_image, pause, clear

If visual_required is false: return empty visual_spec ({}) and empty teaching_sequence ([]).

ANSWER from course context ONLY. If context is insufficient, say:
"I don't have information on that in my course notes."

Return ONLY the JSON. No markdown fences. No preamble.
"""

# ---------------------------------------------------------------------------
# Ollama fallback (plain-text answer, no visual)
# ---------------------------------------------------------------------------

OLLAMA_SYSTEM_PROMPT = """You are a friendly AI teacher explaining DATABASE SYSTEMS concepts.
Answer clearly in under 150 words. Do not use markdown."""


def _call_ollama_sync(messages: list[dict]) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 300},
    }).encode("utf-8")
    req = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned empty response")
    return content


async def _call_ollama(messages: list[dict]) -> str:
    return await asyncio.to_thread(_call_ollama_sync, messages)


# ---------------------------------------------------------------------------
# Visual rendering dispatch
# ---------------------------------------------------------------------------

async def _render_visual(visual_type: str | None, visual_spec: dict, response_text: str) -> tuple[str | None, dict | None]:
    """
    Dispatch to the correct renderer.
    Returns (image_url | None, updated_visual_spec | None).
    image_url is only set for Gemini-generated images.
    """
    if not visual_type or not visual_spec:
        return None, visual_spec

    from diagram_renderer import (
        render_er_diagram,
        render_table,
        render_normalization,
        render_comparison,
        render_flowchart,
    )

    generated_dir = Path("whiteboard") / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    try:
        if visual_type == "er_diagram":
            png_bytes = render_er_diagram(visual_spec)
        elif visual_type == "table":
            png_bytes = render_table(visual_spec)
        elif visual_type == "normalization":
            png_bytes = render_normalization(visual_spec)
        elif visual_type == "comparison":
            png_bytes = render_comparison(visual_spec)
        elif visual_type in ("flowchart", "architecture"):
            png_bytes = render_flowchart(visual_spec)
        elif visual_type == "generated_image":
            from gemini_handler import generate_image
            prompt = visual_spec.get("prompt", response_text[:300])
            image_url = await generate_image(prompt)
            return image_url, visual_spec
        else:
            return None, visual_spec

        filename = f"{uuid4().hex}.png"
        (generated_dir / filename).write_bytes(png_bytes)
        image_url = f"/whiteboard/generated/{filename}"
        return image_url, visual_spec

    except Exception as exc:
        print(f"[RENDER ERROR] visual_type={visual_type}: {exc}")
        return None, visual_spec


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_response(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> tuple[str, str | None, dict | None, dict | None, list[dict]]:
    """
    Full teaching pipeline.

    Returns:
        answer_text       — chat panel answer
        diagram_id        — legacy field (backward compat)
        visual_asset      — legacy field (backward compat)
        teaching_plan_dict — {visual_required, visual_type, visual_spec, image_url}
        teaching_sequence — list of TeachingStep dicts with narration/subtitle
    """
    history = history or []
    trimmed_history = history[-MAX_HISTORY_TURNS:]

    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nCourse Context:\n{context}"
        },
        *trimmed_history,
        {"role": "user", "content": question},
    ]

    teaching_plan: dict = {}
    response_text = ""

    # --- Try OpenAI structured call ---
    if client is not None:
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.5,
            )
            raw = resp.choices[0].message.content.strip()
            teaching_plan = json.loads(raw)
            response_text = teaching_plan.get("answer", "").strip()
        except Exception as openai_exc:
            print(f"[LLM OpenAI ERROR] {openai_exc}")
            teaching_plan = {}

    # --- Ollama fallback (plain text, no visual) ---
    if not response_text:
        try:
            fallback_messages = [
                {"role": "system", "content": f"{OLLAMA_SYSTEM_PROMPT}\n\nContext:\n{context}"},
                *trimmed_history,
                {"role": "user", "content": question},
            ]
            response_text = await _call_ollama(fallback_messages)
        except Exception as ollama_exc:
            print(f"[LLM Ollama ERROR] {ollama_exc}")
            response_text = FALLBACK_RESPONSE
        # Return plain text with no visual
        return response_text, None, None, {}, []

    # --- Extract fields from teaching plan ---
    visual_required = bool(teaching_plan.get("visual_required", False))
    visual_type = teaching_plan.get("visual_type") if visual_required else None
    visual_spec_raw = teaching_plan.get("visual_spec", {}) if visual_required else {}
    teaching_sequence_raw = teaching_plan.get("teaching_sequence", [])

    # --- Render visual ---
    image_url, visual_spec = await _render_visual(visual_type, visual_spec_raw, response_text)

    # --- Build backward-compatible legacy fields ---
    legacy_diagram_id = _visual_type_to_legacy_id(visual_type)
    legacy_visual_asset: dict | None = None
    if image_url:
        legacy_visual_asset = {
            "asset_id": legacy_diagram_id or "generated",
            "asset_file": image_url.lstrip("/"),
            "generated": "true",
            "mime_type": "image/png",
        }

    teaching_plan_dict = {
        "visual_required": visual_required,
        "visual_type": visual_type,
        "visual_spec": visual_spec,
        "image_url": image_url,
    }

    return (
        response_text,
        legacy_diagram_id,
        legacy_visual_asset,
        teaching_plan_dict,
        _sanitize_sequence(teaching_sequence_raw),
    )


def _visual_type_to_legacy_id(visual_type: str | None) -> str | None:
    """Map new visual_type to old diagram_id for backward compatibility."""
    mapping = {
        "er_diagram": "db_er_diagram",
        "table": "db_relational_model",
        "normalization": "db_normalization",
        "comparison": "db_sql_query",
        "flowchart": "db_sql_query",
        "architecture": "db_relational_model",
    }
    return mapping.get(visual_type or "", None)


def _sanitize_sequence(seq: list) -> list[dict]:
    """Ensure each step has expected fields with safe defaults."""
    sanitized = []
    for i, step in enumerate(seq):
        if not isinstance(step, dict):
            continue
        sanitized.append({
            "step_id": step.get("step_id") or f"step_{i + 1}",
            "visual_action": step.get("visual_action") or {"type": "show", "targets": [], "data": {}},
            "narration_text": str(step.get("narration_text") or ""),
            "subtitle_text": str(step.get("subtitle_text") or step.get("narration_text") or ""),
            "audio_url": "",
            "duration_hint": 0.0,
        })
    return sanitized


def get_llm_backend_name() -> str:
    return "openai" if client is not None else "ollama"