"""Deterministic matplotlib renderers for LLM-extracted diagram data."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _clean_text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text[:80] or fallback


def _draw_entity(axis: Any, entity: dict[str, Any], x: float, y: float) -> None:
    name = _clean_text(entity.get("name"), "Entity")
    attributes = entity.get("attributes", [])
    if not isinstance(attributes, list):
        attributes = []

    height = 0.72 + 0.22 * min(len(attributes), 8)
    axis.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            2.55,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor="#f7fbff",
            edgecolor="#174a6e",
            linewidth=1.6,
        )
    )
    axis.text(
        x + 1.275,
        y + height / 2 - 0.27,
        name,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#12344d",
    )
    axis.plot([x, x + 2.55], [y + height / 2 - 0.5, y + height / 2 - 0.5], color="#174a6e", linewidth=1)

    for index, attribute in enumerate(attributes[:8]):
        if isinstance(attribute, dict):
            attribute_name = _clean_text(attribute.get("name"), "attribute")
            key = str(attribute.get("key", "")).upper().strip()
            prefix = f"{key}  " if key in {"PK", "FK", "UK"} else ""
        else:
            attribute_name = _clean_text(attribute, "attribute")
            prefix = ""
        axis.text(
            x + 0.16,
            y + height / 2 - 0.72 - index * 0.22,
            f"{prefix}{attribute_name}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#1f2933",
        )


def render_er_diagram(diagram_data: dict[str, Any]) -> bytes:
    """Render an ER diagram payload and return a PNG byte string."""
    entities = diagram_data.get("entities", [])
    relationships = diagram_data.get("relationships", [])
    if not isinstance(entities, list) or not entities:
        raise ValueError("ER diagram must contain at least one entity")
    if not isinstance(relationships, list):
        relationships = []

    entities = [item for item in entities if isinstance(item, dict)][:8]
    if not entities:
        raise ValueError("ER diagram contains no valid entities")

    figure, axis = plt.subplots(figsize=(12, 7), dpi=150)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions: dict[str, tuple[float, float]] = {}
    columns = 2 if len(entities) > 4 else 1
    rows = (len(entities) + columns - 1) // columns

    for index, entity in enumerate(entities):
        name = _clean_text(entity.get("name"), f"Entity {index + 1}")
        column = index // rows
        row = index % rows
        x = 0.75 + column * 6.0
        y = rows - row - 0.5
        positions[name.lower()] = (x, y)
        _draw_entity(axis, entity, x, y)

    for relationship in relationships[:12]:
        if not isinstance(relationship, dict):
            continue
        source = _clean_text(relationship.get("from"), "").lower()
        target = _clean_text(relationship.get("to"), "").lower()
        if source not in positions or target not in positions:
            continue
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        source_center = (source_x + 1.275, source_y)
        target_center = (target_x + 1.275, target_y)
        axis.annotate(
            "",
            xy=target_center,
            xytext=source_center,
            arrowprops={"arrowstyle": "-", "color": "#d97706", "linewidth": 1.5},
        )
        label = _clean_text(relationship.get("label"), "relates to")
        cardinality = _clean_text(relationship.get("cardinality"), "")
        midpoint_x = (source_center[0] + target_center[0]) / 2
        midpoint_y = (source_center[1] + target_center[1]) / 2
        axis.text(
            midpoint_x,
            midpoint_y + 0.14,
            f"{label} ({cardinality})" if cardinality else label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#92400e",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    title = _clean_text(diagram_data.get("title"), "Entity-Relationship Diagram")
    axis.set_title(title, fontsize=15, fontweight="bold", color="#12344d", pad=16)
    axis.set_xlim(0, 3.3 if columns == 1 else 9.3)
    axis.set_ylim(-0.2, rows + 0.45)
    axis.axis("off")
    figure.tight_layout()

    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()
