"""
Example 07 — db_aware=True + EntityNERExtractor with real GLiNER2 + HuggingFace NER.

The NER model enriches entity-label resolution: entities that don't appear in
a known schema relation-type still get the correct label from NER, which makes
the DB lookup MATCH query more precise.

Pre-condition:
    Reset the DB to empty before running:
        MATCH (n) DETACH DELETE n

Run:
    pip install "cypher_validator[neo4j,ner-transformers]" gliner2
    python examples/07_db_aware_with_ner.py
"""
import sys, io, warnings
warnings.filterwarnings("ignore")

from cypher_validator import (
    Neo4jDatabase, NLToCypher, EntityNERExtractor, Schema
)

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

DIVIDER = "=" * 65

# ── Load NER model ────────────────────────────────────────────────────────────
print("Loading HuggingFace NER model …")
old_stderr = sys.stderr; sys.stderr = io.StringIO()
ner = EntityNERExtractor.from_transformers(
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    label_map={"PER": "Person", "ORG": "Company", "LOC": "City"},
)
sys.stderr = old_stderr
print("NER model ready.\n")

# ── Standalone NER demo ───────────────────────────────────────────────────────
print(DIVIDER)
print("NER extraction demo")
print(DIVIDER)
for text in [
    "John works for Apple Inc. and lives in San Francisco.",
    "Alice manages the Engineering team at Microsoft.",
    "Bob acquired TechCorp in 2019.",
]:
    ents = ner.extract(text)
    print(f"\n  Input : {text!r}")
    for e in ents:
        print(f"    {e['text']:28s}  → {e['label']}")

# ── Load GLiNER2 + build pipeline ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("Loading fastino/gliner2-large-v1 …")
db = Neo4jDatabase(DB_URI, DB_USER, DB_PASS)
pipeline = NLToCypher.from_pretrained(
    "fastino/gliner2-large-v1",
    schema=schema,
    db=db,
    ner_extractor=ner,
)
print("GLiNER2 model ready.\n")

def snapshot(db, label=""):
    nodes = db.execute("MATCH (n) RETURN labels(n) AS lbl, n.name AS name ORDER BY lbl, name")
    rels  = db.execute("MATCH (a)-[r]->(b) RETURN a.name AS a, type(r) AS t, b.name AS b")
    print(f"\n  [{label}]")
    for n in nodes: print(f"    {n['lbl']} name={n['name']!r}")
    for r in rels:  print(f"    ({r['a']})-[:{r['t']}]->({r['b']})")
    if not nodes:   print("    (empty)")

# ── Round 1: empty DB → all entities new ─────────────────────────────────────
print(f"{DIVIDER}")
print("Round 1 — empty DB: NER labels drive inline CREATE")
print(DIVIDER)

TEXT = "John works for Apple Inc. and lives in San Francisco."
snapshot(db, "BEFORE")

print(f"\n  NER labels for lookup:")
for e in ner.extract(TEXT):
    print(f"    {e['text']:28s} → {e['label']}")

cypher, records = pipeline(TEXT, ["works_for", "lives_in"], db_aware=True, execute=True)
print(f"\n  Generated Cypher:")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"\n  Returned {len(records)} record(s)")
snapshot(db, "AFTER")

# ── Round 2: all nodes exist → MATCH all, CREATE only edges ───────────────────
print(f"\n{DIVIDER}")
print("Round 2 — all nodes now exist: NER labels used for MATCH lookup")
print(DIVIDER)
snapshot(db, "BEFORE")

cypher, records = pipeline(TEXT, ["works_for", "lives_in"], db_aware=True, execute=True)
print(f"\n  Generated Cypher:")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"\n  Returned {len(records)} record(s)")
snapshot(db, "AFTER — zero new nodes, edges created or already exist")

# ── Round 3: generate only (no execute) ──────────────────────────────────────
print(f"\n{DIVIDER}")
print("Round 3 — generate Cypher only (execute=False, default)")
print(DIVIDER)

TEXT2 = "Alice manages the Engineering team at Microsoft."
print(f"\n  Input: {TEXT2!r}")
print(f"\n  NER labels:")
for e in ner.extract(TEXT2):
    print(f"    {e['text']:28s} → {e['label']}")

cypher = pipeline(TEXT2, ["works_for"], db_aware=True)
print(f"\n  Generated Cypher (not executed):")
for line in cypher.strip().splitlines():
    print(f"    {line}")

db.close()
print(f"\n{DIVIDER}\nAll db_aware + NER examples complete.\n{DIVIDER}")
