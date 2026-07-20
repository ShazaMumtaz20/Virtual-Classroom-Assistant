# diagram_keywords.py
# Maps keywords (found in LLM response text) to diagram IDs.
# The backend now focuses on Database Systems topics.

DIAGRAM_KEYWORD_MAP = {
    "db_er_diagram": ["er diagram", "entity-relationship", "entity relationship", "cardinality", "many-to-many", "one-to-many", "entity", "relationship"],
    "db_sql_query": ["sql", "select", "insert", "update", "delete", "join", "where", "order by", "group by", "having"],
    "db_normalization": ["normalization", "1nf", "2nf", "3nf", "functional dependency", "transitive dependency", "anomaly"],
    "db_transactions": ["transaction", "acid", "commit", "rollback", "atomicity", "consistency", "isolation", "durability", "concurrency"],
    "db_indexing": ["index", "indexes", "b-tree", "btree", "balanced tree", "query performance", "disk read"],
    "db_relational_model": ["relational model", "table", "relation", "tuple", "attribute", "primary key", "foreign key"],
}


def detect_diagram(response_text: str) -> str | None:
    """
    Scans LLM response text for known keywords.
    Returns the first matching diagram_id, or None if no match.
    Keyword matching is case-insensitive.
    """
    text_lower = response_text.lower()
    for diagram_id, keywords in DIAGRAM_KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return diagram_id
    return None