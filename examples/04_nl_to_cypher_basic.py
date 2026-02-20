"""
Example 04 — NLToCypher: basic usage with the real GLiNER2 relation-extraction model.

Run:
    pip install gliner2
    python examples/04_nl_to_cypher_basic.py
"""
from cypher_validator import NLToCypher, Schema

schema = Schema(
    nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
    relationships={
        "WORKS_FOR": ("Person", "Company", []),
        "LIVES_IN":  ("Person", "City",    []),
        "ACQUIRED":  ("Person", "Company", []),
    },
)

# Load the real GLiNER2 relation-extraction model once and reuse it.
print("Loading fastino/gliner2-large-v1 …")
pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
print("Model ready.\n")

# ── 1. CREATE mode ────────────────────────────────────────────────────────────
print("=== mode='create' ===")
cypher = pipeline(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    mode="create",
)
print(cypher)

# ── 2. MERGE mode ─────────────────────────────────────────────────────────────
print("\n=== mode='merge' ===")
cypher = pipeline(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    mode="merge",
)
print(cypher)

# ── 3. MATCH mode ─────────────────────────────────────────────────────────────
print("\n=== mode='match' ===")
cypher = pipeline(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    mode="match",
)
print(cypher)

# ── 4. extract_and_convert — get raw extraction + Cypher ─────────────────────
print("\n=== extract_and_convert ===")
relations, cypher = pipeline.extract_and_convert(
    "Bob acquired TechCorp in 2019.",
    ["acquired"],
    mode="create",
)
print("Relations dict:", relations)
print("Cypher:\n" + cypher)

# ── 5. No schema → no labels ──────────────────────────────────────────────────
print("\n=== No schema → no labels ===")
pipeline_no_schema = NLToCypher.from_pretrained("fastino/gliner2-large-v1")
cypher = pipeline_no_schema(
    "Alice works for Acme Corp.",
    ["works_for"],
    mode="create",
)
print(cypher)

# ── 6. Custom return clause ───────────────────────────────────────────────────
print("\n=== Custom return clause ===")
cypher = pipeline(
    "John works for Apple Inc.",
    ["works_for"],
    mode="create",
    return_clause="RETURN *",
)
print(cypher)
