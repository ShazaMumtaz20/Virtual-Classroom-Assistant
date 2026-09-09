# models.py
# Pydantic models for the Virtual Classroom teaching pipeline.

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Incoming request models
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default=[], max_length=6)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Visual specification
# ---------------------------------------------------------------------------

class EntityAttribute(BaseModel):
    name: str
    key: str = ""       # "PK" | "FK" | "UK" | ""

class Entity(BaseModel):
    name: str
    attributes: list[EntityAttribute] = []

class Relationship(BaseModel):
    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    label: str = ""
    cardinality: str = ""   # "1:1" | "1:N" | "N:M"

    class Config:
        populate_by_name = True

class TableColumn(BaseModel):
    name: str
    type: str = ""
    key: str = ""       # "PK" | "FK" | ""

class TableData(BaseModel):
    name: str
    columns: list[TableColumn] = []
    rows: list[list[str]] = []   # optional sample data rows

class NormStep(BaseModel):
    label: str          # e.g. "Unnormalized", "1NF", "2NF", "3NF"
    table: TableData
    explanation: str = ""

class FlowNode(BaseModel):
    id: str
    label: str
    shape: str = "box"  # "box" | "diamond" | "oval"

class FlowEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    label: str = ""

    class Config:
        populate_by_name = True

class VisualSpec(BaseModel):
    """
    Generic visual specification container.
    Only the fields relevant to the chosen visual_type will be populated.
    """
    title: str = ""
    # ER diagram fields
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    # Table visual fields
    tables: list[TableData] = []
    # Normalization fields
    norm_steps: list[NormStep] = []
    # Flowchart fields
    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []
    # Free annotations
    annotations: list[str] = []


# ---------------------------------------------------------------------------
# Animation vocabulary
# ---------------------------------------------------------------------------

VisualActionType = Literal[
    "show", "hide", "fade_in", "fade_out", "draw", "connect",
    "highlight", "emphasize", "reveal", "move", "replace",
    "transform", "zoom", "show_label", "show_table", "show_image",
    "pause", "clear",
]

class VisualAction(BaseModel):
    type: VisualActionType = "show"
    targets: list[str] = []        # entity/node/table names to act on
    data: dict[str, Any] = {}      # extra payload for the action


# ---------------------------------------------------------------------------
# Teaching sequence
# ---------------------------------------------------------------------------

class TeachingStep(BaseModel):
    step_id: str                    # "step_1", "step_2", ...
    visual_action: VisualAction
    narration_text: str             # what the teacher says (spoken, natural)
    subtitle_text: str              # displayed subtitle (can differ from narration)
    audio_url: str = ""             # /media/audio/<session>/<step>.wav — filled after TTS
    duration_hint: float = 0.0     # estimated seconds for Unity timing


# ---------------------------------------------------------------------------
# Full LLM teaching plan (returned by OpenAI structured call)
# ---------------------------------------------------------------------------

class TeachingPlan(BaseModel):
    answer: str                     # chat panel text
    visual_required: bool = False
    visual_type: str | None = None  # "er_diagram" | "table" | "normalization" |
                                    # "comparison" | "flowchart" | "architecture" |
                                    # "generated_image" | None
    visual_spec: VisualSpec | None = None
    teaching_sequence: list[TeachingStep] = []


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class AskResponse(BaseModel):
    # --- legacy fields (backward compatible with existing Unity code) ---
    response: str
    diagram_id: str | None = None
    visual_asset: dict | None = None
    confidence: float

    # --- new teaching fields ---
    session_id: str | None = None
    visual_required: bool = False
    visual_type: str | None = None
    visual_spec: dict | None = None     # serialized VisualSpec
    image_url: str | None = None        # Gemini generated image URL
    teaching_sequence: list[dict] = []  # serialized TeachingStep list


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    vectordb_docs: int
