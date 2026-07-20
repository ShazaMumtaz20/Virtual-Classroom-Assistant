"""Visual asset metadata for Unity whiteboard rendering.

The backend does not render the assets itself. It returns a stable asset
descriptor that Unity can map to its own sprites, prefabs, or scene objects.
"""

VISUAL_ASSET_MAP = {
    "db_er_diagram": {
        "asset_id": "db_er_diagram",
        "label": "ER Diagram",
        "asset_file": "whiteboard/db_er_diagram.png",
        "description": "Use for entities, relationships, cardinality, and table design.",
    },
    "db_sql_query": {
        "asset_id": "db_sql_query",
        "label": "SQL Query Flow",
        "asset_file": "whiteboard/db_sql_query.png",
        "description": "Use for SELECT, JOIN, filtering, grouping, and query examples.",
    },
    "db_normalization": {
        "asset_id": "db_normalization",
        "label": "Normalization Diagram",
        "asset_file": "whiteboard/db_normalization.png",
        "description": "Use for 1NF, 2NF, 3NF, redundancy, and anomaly explanations.",
    },
    "db_transactions": {
        "asset_id": "db_transactions",
        "label": "Transaction / ACID Diagram",
        "asset_file": "whiteboard/db_transactions.png",
        "description": "Use for ACID, commit, rollback, concurrency, and isolation.",
    },
    "db_indexing": {
        "asset_id": "db_indexing",
        "label": "Index / B-Tree Diagram",
        "asset_file": "whiteboard/db_indexing.png",
        "description": "Use for indexes, B-trees, lookup speed, and disk access reduction.",
    },
    "db_relational_model": {
        "asset_id": "db_relational_model",
        "label": "Relational Model Diagram",
        "asset_file": "whiteboard/db_relational_model.png",
        "description": "Use for tables, rows, columns, keys, and relational schema basics.",
    },
}


def resolve_visual_asset(diagram_id: str | None) -> dict[str, str] | None:
    if not diagram_id:
        return None
    asset = VISUAL_ASSET_MAP.get(diagram_id)
    return dict(asset) if asset else None