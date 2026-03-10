import re

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "merge", "exec", "execute", "create", "grant", "revoke"
]


def is_safe_select_query(sql: str):
    if not sql or not sql.strip():
        return False, "SQL is empty."

    normalized = sql.strip().lower()

    if ";" in normalized[:-1]:
        return False, "Multiple statements are not allowed."

    if not normalized.startswith("select"):
        return False, "Only SELECT statements are allowed."

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return False, f"Forbidden keyword detected: {keyword}"

    return True, "Safe"