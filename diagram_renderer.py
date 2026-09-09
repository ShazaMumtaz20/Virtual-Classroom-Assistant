"""Deterministic matplotlib renderers for all LLM-extracted diagram types.

Supported visual types:
  render_er_diagram(data)        — ER diagram with entities and relationships
  render_table(data)             — SQL table(s) with columns and sample rows
  render_normalization(data)     — Step-by-step normalization (UNF → 1NF → 2NF → 3NF)
  render_comparison(data)        — Side-by-side table comparison (JOIN types, PK vs FK, etc.)
  render_flowchart(data)         — Box-and-arrow flowchart / architecture / process diagram
"""

from __future__ import annotations

import textwrap
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _clean(value: Any, fallback: str = "", maxlen: int = 80) -> str:
    text = str(value).strip() if value is not None else ""
    return (text[:maxlen] or fallback)


def _save(figure: Any) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(figure)
    return output.getvalue()


# ---------------------------------------------------------------------------
# ER Diagram
# ---------------------------------------------------------------------------

def _draw_entity(axis: Any, entity: dict[str, Any], x: float, y: float) -> None:
    name = _clean(entity.get("name"), "Entity")
    attributes = entity.get("attributes", [])
    if not isinstance(attributes, list):
        attributes = []

    height = 0.72 + 0.22 * min(len(attributes), 8)
    axis.add_patch(FancyBboxPatch(
        (x, y - height / 2), 2.55, height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor="#f7fbff", edgecolor="#174a6e", linewidth=1.6,
    ))
    axis.text(
        x + 1.275, y + height / 2 - 0.27, name,
        ha="center", va="center", fontsize=11, fontweight="bold", color="#12344d",
    )
    axis.plot([x, x + 2.55], [y + height / 2 - 0.5, y + height / 2 - 0.5],
              color="#174a6e", linewidth=1)

    for idx, attr in enumerate(attributes[:8]):
        if isinstance(attr, dict):
            attr_name = _clean(attr.get("name"), "attribute")
            key = str(attr.get("key", "")).upper().strip()
            prefix = f"{key}  " if key in {"PK", "FK", "UK"} else ""
        else:
            attr_name = _clean(attr, "attribute")
            prefix = ""
        axis.text(
            x + 0.16, y + height / 2 - 0.72 - idx * 0.22,
            f"{prefix}{attr_name}",
            ha="left", va="center", fontsize=8.5, color="#1f2933",
        )


def render_er_diagram(data: dict[str, Any]) -> bytes:
    """Render an ER diagram and return PNG bytes."""
    entities = [e for e in data.get("entities", []) if isinstance(e, dict)][:8]
    if not entities:
        raise ValueError("ER diagram must contain at least one entity")
    relationships = [r for r in data.get("relationships", []) if isinstance(r, dict)][:12]

    cols = min(3, max(1, len(entities)))
    rows = (len(entities) + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions: dict[str, tuple[float, float, float, float]] = {}
    for idx, entity in enumerate(entities):
        name = _clean(entity.get("name"), f"Entity {idx + 1}")
        c, r = idx % cols, idx // cols
        x, y = 0.75 + c * 4.3, rows - r - 0.5
        attrs = entity.get("attributes", [])
        h = 0.72 + 0.22 * min(len(attrs) if isinstance(attrs, list) else 0, 8)
        positions[name.lower()] = (x, y, 2.55, h)
        _draw_entity(ax, entity, x, y)

    for rel in relationships:
        src = _clean(rel.get("from"), "").lower()
        tgt = _clean(rel.get("to"), "").lower()
        if src not in positions or tgt not in positions:
            continue
        sx, sy, sw, _ = positions[src]
        tx, ty, tw, _ = positions[tgt]
        if sx < tx:
            p1, p2 = (sx + sw, sy), (tx, ty)
        elif sx > tx:
            p1, p2 = (sx, sy), (tx + tw, ty)
        elif sy < ty:
            p1, p2 = (sx + sw / 2, sy), (tx + tw / 2, ty)
        else:
            p1, p2 = (sx + sw / 2, sy), (tx + tw / 2, ty)
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops={"arrowstyle": "-", "color": "#d97706", "linewidth": 1.5})
        label = _clean(rel.get("label"), "relates to")
        card = _clean(rel.get("cardinality"), "")
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.14, f"{label} ({card})" if card else label,
                ha="center", va="bottom", fontsize=8, color="#92400e",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    title = _clean(data.get("title"), "Entity-Relationship Diagram")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#12344d", pad=16)
    ax.set_xlim(0, 0.75 + (cols - 1) * 4.3 + 2.55 + 0.5)
    ax.set_ylim(-0.35, rows + 0.45)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig)


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------

def render_table(data: dict[str, Any]) -> bytes:
    """Render one or more SQL tables with column headers and sample rows."""
    tables = data.get("tables", [])
    if not tables:
        tables = [data]  # allow passing a single table dict directly

    n = min(len(tables), 3)
    fig, axes = plt.subplots(1, n, figsize=(max(7, 5 * n), 5))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for i, tbl in enumerate(tables[:3]):
        ax = axes[i]
        ax.axis("off")
        name = _clean(tbl.get("name"), f"Table {i + 1}")
        cols = tbl.get("columns", [])
        rows = tbl.get("rows", [])

        col_names = []
        col_keys = []
        for c in cols:
            if isinstance(c, dict):
                col_names.append(_clean(c.get("name"), "col"))
                col_keys.append(str(c.get("key", "")).upper())
            else:
                col_names.append(str(c))
                col_keys.append("")

        if not col_names:
            ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=12)
            continue

        # Build display rows (header + data)
        header = [f"🔑 {n}" if k == "PK" else f"🔗 {n}" if k == "FK" else n
                  for n, k in zip(col_names, col_keys)]
        display_rows = [header] + [[str(v) for v in r[:len(col_names)]] for r in rows[:6]]

        colors_header = ["#174a6e"] * len(col_names)
        fc = [[colors_header[j] if r_idx == 0 else ("#f0f4f8" if r_idx % 2 == 0 else "white")
               for j in range(len(col_names))]
              for r_idx, _ in enumerate(display_rows)]

        tbl_obj = ax.table(
            cellText=display_rows,
            cellLoc="center",
            loc="center",
            cellColours=fc,
        )
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(9)
        tbl_obj.scale(1, 1.5)

        # White text for header row
        for j in range(len(col_names)):
            tbl_obj[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title(name, fontsize=12, fontweight="bold", color="#12344d", pad=8)

    title = _clean(data.get("title"), "Table View")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#12344d", y=1.02)
    fig.tight_layout()
    return _save(fig)


# ---------------------------------------------------------------------------
# Normalization renderer
# ---------------------------------------------------------------------------

def render_normalization(data: dict[str, Any]) -> bytes:
    """Render normalization steps as progressive table transformations."""
    steps = data.get("norm_steps", [])
    if not steps:
        raise ValueError("No normalization steps provided")

    n = min(len(steps), 4)
    fig, axes = plt.subplots(1, n, figsize=(max(8, 5.5 * n), 6))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    header_colors = ["#174a6e", "#1a6e3e", "#5a2a8a", "#8a3a0a"]

    for i, step in enumerate(steps[:4]):
        ax = axes[i]
        ax.axis("off")
        label = _clean(step.get("label"), f"Step {i + 1}")
        explanation = _clean(step.get("explanation"), "", maxlen=120)
        tbl = step.get("table", {})

        cols = tbl.get("columns", [])
        rows = tbl.get("rows", [])
        tbl_name = _clean(tbl.get("name"), label)

        col_names = []
        col_keys = []
        for c in cols:
            if isinstance(c, dict):
                col_names.append(_clean(c.get("name"), "col"))
                col_keys.append(str(c.get("key", "")).upper())
            else:
                col_names.append(str(c))
                col_keys.append("")

        if not col_names:
            ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
            continue

        header = [f"🔑{n}" if k == "PK" else f"🔗{n}" if k == "FK" else n
                  for n, k in zip(col_names, col_keys)]
        display_rows = [header] + [[str(v) for v in r[:len(col_names)]] for r in rows[:5]]

        hc = header_colors[i % len(header_colors)]
        fc = [[hc if r_idx == 0 else ("#f0f4f8" if r_idx % 2 == 1 else "white")
               for _ in range(len(col_names))]
              for r_idx in range(len(display_rows))]

        tbl_obj = ax.table(
            cellText=display_rows, cellLoc="center", loc="center", cellColours=fc,
        )
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(8.5)
        tbl_obj.scale(1, 1.4)
        for j in range(len(col_names)):
            tbl_obj[0, j].set_text_props(color="white", fontweight="bold")

        step_label = f"◆ {label}"
        ax.set_title(step_label, fontsize=11, fontweight="bold", color=hc, pad=6)
        if explanation:
            wrapped = "\n".join(textwrap.wrap(explanation, width=30))
            ax.text(0.5, -0.08, wrapped, ha="center", va="top", fontsize=7.5,
                    color="#555", transform=ax.transAxes, style="italic")

    title = _clean(data.get("title"), "Normalization Steps")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#12344d")
    fig.tight_layout()
    return _save(fig)


# ---------------------------------------------------------------------------
# Comparison renderer (e.g. INNER JOIN vs LEFT JOIN)
# ---------------------------------------------------------------------------

def render_comparison(data: dict[str, Any]) -> bytes:
    """Render two tables side-by-side for comparison, with optional annotations."""
    tables = data.get("tables", [])
    if len(tables) < 2:
        # Can fall back to single table
        return render_table(data)

    annotations = data.get("annotations", [])
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(2 if annotations else 1, 2, hspace=0.5)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    colors = ["#174a6e", "#8a3a0a"]

    def _draw_table(ax: Any, tbl: dict, color: str) -> None:
        ax.axis("off")
        name = _clean(tbl.get("name"), "Table")
        cols = tbl.get("columns", [])
        rows = tbl.get("rows", [])
        col_names = []
        col_keys = []
        for c in cols:
            if isinstance(c, dict):
                col_names.append(_clean(c.get("name"), "col"))
                col_keys.append(str(c.get("key", "")).upper())
            else:
                col_names.append(str(c))
                col_keys.append("")
        if not col_names:
            ax.text(0.5, 0.5, name, ha="center", va="center")
            return
        header = [f"🔑{n}" if k == "PK" else f"🔗{n}" if k == "FK" else n
                  for n, k in zip(col_names, col_keys)]
        display = [header] + [[str(v) for v in r[:len(col_names)]] for r in rows[:6]]
        fc = [[color if r_i == 0 else "white" for _ in range(len(col_names))]
              for r_i in range(len(display))]
        tbl_obj = ax.table(cellText=display, cellLoc="center", loc="center", cellColours=fc)
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(9)
        tbl_obj.scale(1, 1.5)
        for j in range(len(col_names)):
            tbl_obj[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title(name, fontsize=12, fontweight="bold", color=color, pad=8)

    _draw_table(ax_left, tables[0], colors[0])
    _draw_table(ax_right, tables[1], colors[1])

    if annotations:
        ax_ann = fig.add_subplot(gs[1, :])
        ax_ann.axis("off")
        ann_text = "\n".join(f"• {a}" for a in annotations[:5])
        ax_ann.text(0.05, 0.9, ann_text, ha="left", va="top", fontsize=10,
                    color="#1f2933", transform=ax_ann.transAxes,
                    bbox={"facecolor": "#f0f4f8", "edgecolor": "#aab", "pad": 10})

    title = _clean(data.get("title"), "Comparison")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#12344d")
    fig.tight_layout()
    return _save(fig)


# ---------------------------------------------------------------------------
# Flowchart / Architecture renderer
# ---------------------------------------------------------------------------

def render_flowchart(data: dict[str, Any]) -> bytes:
    """Render a directed flowchart / architecture diagram."""
    nodes = [n for n in data.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in data.get("edges", []) if isinstance(e, dict)]
    annotations = data.get("annotations", [])

    if not nodes:
        raise ValueError("Flowchart must contain at least one node")

    n_nodes = len(nodes)
    per_row = min(4, n_nodes)
    n_rows = (n_nodes + per_row - 1) // per_row

    fig_w = max(10, per_row * 3.5)
    fig_h = max(5, n_rows * 2.5 + (1.5 if annotations else 0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    node_pos: dict[str, tuple[float, float]] = {}
    box_w, box_h = 2.8, 0.9

    for idx, node in enumerate(nodes):
        nid = _clean(node.get("id"), f"n{idx}")
        label = _clean(node.get("label"), nid)
        shape = str(node.get("shape", "box")).lower()
        col = idx % per_row
        row = idx // per_row
        x = col * 3.5 + 1.0
        y = (n_rows - row - 1) * 2.2 + 1.0
        node_pos[nid] = (x + box_w / 2, y + box_h / 2)

        if shape == "diamond":
            corners = [
                (x + box_w / 2, y + box_h),
                (x + box_w, y + box_h / 2),
                (x + box_w / 2, y),
                (x, y + box_h / 2),
            ]
            poly = plt.Polygon(corners, facecolor="#fff3cd", edgecolor="#e67e22", linewidth=1.5)
            ax.add_patch(poly)
        elif shape == "oval":
            ellipse = mpatches.Ellipse(
                (x + box_w / 2, y + box_h / 2), box_w, box_h,
                facecolor="#d4edda", edgecolor="#155724", linewidth=1.5,
            )
            ax.add_patch(ellipse)
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.06",
                facecolor="#dbeafe", edgecolor="#174a6e", linewidth=1.5,
            ))

        wrapped = "\n".join(textwrap.wrap(label, width=18))
        ax.text(x + box_w / 2, y + box_h / 2, wrapped,
                ha="center", va="center", fontsize=9, fontweight="bold", color="#12344d")

    for edge in edges:
        src_id = _clean(edge.get("from"), "")
        tgt_id = _clean(edge.get("to"), "")
        if src_id not in node_pos or tgt_id not in node_pos:
            continue
        sx, sy = node_pos[src_id]
        tx, ty = node_pos[tgt_id]
        edge_label = _clean(edge.get("label"), "")
        ax.annotate(
            "", xy=(tx, ty), xytext=(sx, sy),
            arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.4,
                        "connectionstyle": "arc3,rad=0.05"},
        )
        if edge_label:
            mx, my = (sx + tx) / 2, (sy + ty) / 2
            ax.text(mx + 0.1, my + 0.1, edge_label, fontsize=8, color="#555",
                    ha="left", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    if annotations:
        ann_y = -0.1
        for ann in annotations[:3]:
            ax.text(0.01, ann_y, f"• {ann}", transform=ax.transAxes,
                    fontsize=9, color="#444", va="top", style="italic")
            ann_y -= 0.06

    ax.set_xlim(-0.3, per_row * 3.5 + 0.5)
    ax.set_ylim(-0.5, n_rows * 2.2 + 0.8)

    title = _clean(data.get("title"), "Process Diagram")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#12344d", pad=12)
    fig.tight_layout()
    return _save(fig)
