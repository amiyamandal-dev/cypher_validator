"""
Example 05 — db_aware=True with the real GLiNER2 model.

The example runs two rounds against the live DB to show all key scenarios:

  Round 1  (empty DB expected)
    → All entities are new → full inline CREATE

  Round 2  (entities from Round 1 now exist)
    → Both entities already in DB → MATCH both, CREATE only the edge

  Round 3  (multi-relation, builds on Round 2 state)
    → Shows variable reuse when one entity appears in multiple relations

Pre-condition:
    Reset the DB to empty before running:
        MATCH (n) DETACH DELETE n

Run:
    pip install "cypher_validator[neo4j]" gliner2
    python examples/05_db_aware_all_cases.py
"""
from cypher_validator import Neo4jDatabase, NLToCypher, Schema

DB_URI  = "neo4j://192.168.0.222:7687"
DB_USER = "neo4j"
DB_PASS = "qaz123!@#WSX"

schema = Schema(
    nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
    relationships={
        "WORKS_FOR": ("Person", "Company", []),
        "LIVES_IN":  ("Person", "City",    []),
    },
)

DIVIDER = "-" * 60

def snapshot(db, label="DB"):
    nodes = db.execute("MATCH (n) RETURN labels(n) AS lbl, n.name AS name ORDER BY lbl, name")
    rels  = db.execute("MATCH (a)-[r]->(b) RETURN a.name AS a, type(r) AS t, b.name AS b")
    print(f"  [{label}]")
    for n in nodes: print(f"    {n['lbl']} name={n['name']!r}")
    for r in rels:  print(f"    ({r['a']})-[:{r['t']}]->({r['b']})")
    if not nodes:   print("    (empty)")

db = Neo4jDatabase(DB_URI, DB_USER, DB_PASS)

print("Loading fastino/gliner2-large-v1 …")
pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema, db=db)
print("Model ready.\n")

TEXT_SINGLE = "John works for Apple Inc."
TEXT_MULTI  = "John works for Apple Inc. and lives in San Francisco."

# ── Round 1: entities are new → full CREATE ───────────────────────────────────
print(f"\n{DIVIDER}")
print("Round 1 — entities not in DB → CREATE both inline")
print(DIVIDER)
snapshot(db, "BEFORE")

cypher, records = pipeline(TEXT_SINGLE, ["works_for"], db_aware=True, execute=True)
print(f"\n  Cypher:")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"  Records: {len(records)}")
snapshot(db, "AFTER")

# ── Round 2: both entities now exist → MATCH both, CREATE only the edge ───────
print(f"\n{DIVIDER}")
print("Round 2 — John + Apple Inc. now in DB → MATCH both, CREATE edge only")
print(DIVIDER)
snapshot(db, "BEFORE")

cypher, records = pipeline(TEXT_SINGLE, ["works_for"], db_aware=True, execute=True)
print(f"\n  Cypher:")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"  Records: {len(records)}")
snapshot(db, "AFTER")

# ── Round 3: multi-relation — John is shared across both relations ─────────────
print(f"\n{DIVIDER}")
print("Round 3 — multi-relation: John shared across works_for + lives_in")
print(DIVIDER)
snapshot(db, "BEFORE")

cypher, records = pipeline(TEXT_MULTI, ["works_for", "lives_in"], db_aware=True, execute=True)
print(f"\n  Cypher:")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"  Records: {len(records)}")
snapshot(db, "AFTER")

db.close()
print(f"\n{'=' * 60}\nAll rounds complete.\n{'=' * 60}")
