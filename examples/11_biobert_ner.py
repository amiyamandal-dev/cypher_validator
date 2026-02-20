"""
Example 11 — BioBERT NER with EntityNERExtractor and the real GLiNER2 pipeline.

IMPORTANT — BioBERT output format
----------------------------------
`dmis-lab/biobert-v1.1` is a *pre-trained language model*, not a fine-tuned NER
model.  When loaded as a token-classification pipeline it outputs GENERIC labels:

    LABEL_0  — the "O" (outside) class  (pre-training artifact)
    LABEL_1  — the "I" (inside) class

These carry NO semantic meaning on their own.  Two approaches are shown:

  A) Use dmis-lab/biobert-v1.1 directly with a custom label_map
     → maps LABEL_0 / LABEL_1 to a single "BioEntity" type (candidate spans)

  B) Use d4data/biomedical-ner-all — fine-tuned on biomedical text
     → outputs semantic labels: Medication, Disease_disorder, Sign_symptom, …
     → recommended for production

Pre-condition:
    Reset the DB to empty before running:
        MATCH (n) DETACH DELETE n

Run:
    pip install "cypher_validator[neo4j,ner-transformers]" gliner2
    python examples/11_biobert_ner.py
"""

import sys, io, warnings
warnings.filterwarnings("ignore")

from transformers import pipeline as hf_pipeline
from cypher_validator import EntityNERExtractor, Schema, Neo4jDatabase, NLToCypher

DIVIDER = "=" * 65

DB_URI  = "neo4j://192.168.0.222:7687"
DB_USER = "neo4j"
DB_PASS = "qaz123!@#WSX"

SENTENCES = [
    "Aspirin inhibits COX-1 and reduces inflammation.",
    "BRCA1 mutations are associated with breast cancer.",
    "Ibuprofen is used to treat fever and arthritis.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Raw BioBERT output
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("Part 1 — Raw dmis-lab/biobert-v1.1 output  (LABEL_0 / LABEL_1)")
print(DIVIDER)

print("\nLoading dmis-lab/biobert-v1.1 …")
old_stderr = sys.stderr; sys.stderr = io.StringIO()
raw_pipe = hf_pipeline(
    "token-classification",
    model="dmis-lab/biobert-v1.1",
    aggregation_strategy="simple",
)
sys.stderr = old_stderr
print("Model ready.\n")

for sent in SENTENCES:
    ents = raw_pipe(sent)
    print(f"  Input : {sent!r}")
    if ents:
        for e in ents:
            print(f"    group={e['entity_group']:8s}  word={e['word']!r:40s}  score={e['score']:.3f}")
    else:
        print("    (no entities detected)")
    print()

print(
    "  Note: LABEL_0 = 'O' class, LABEL_1 = 'I' class (pre-training artefacts).\n"
    "  These carry NO semantic meaning. See Part 2 for proper biomedical NER.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Part 2A — Approach A: custom label_map with BioBERT
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("Part 2A — Approach A: dmis-lab/biobert-v1.1 + custom label_map")
print(DIVIDER)

# Re-use the already-loaded pipeline object — wrap with custom label_map
ner_biobert = EntityNERExtractor(
    raw_pipe,
    backend="transformers",
    label_map={
        "LABEL_0": "BioEntity",
        "LABEL_1": "BioEntity",
    },
)

print()
for sent in SENTENCES:
    ents = ner_biobert.extract(sent)
    print(f"  Input : {sent!r}")
    for e in ents:
        print(f"    {e['text']:40s} → {e['label']}")
    if not ents:
        print("    (no entities detected)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Part 2B — Approach B: fine-tuned biomedical NER
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("Part 2B — Approach B: d4data/biomedical-ner-all  (fine-tuned)")
print(DIVIDER)

print("\nLoading d4data/biomedical-ner-all …")
old_stderr = sys.stderr; sys.stderr = io.StringIO()
ner_bio = EntityNERExtractor.from_transformers(
    "d4data/biomedical-ner-all",
    label_map={
        "Medication":           "Drug",
        "Disease_disorder":     "Disease",
        "Sign_symptom":         "Symptom",
        "Biological_structure": "Anatomy",
        "Diagnostic_procedure": "Procedure",
        "Lab_value":            "LabResult",
        "Coreference":          "Entity",
    },
    # "first" avoids subword fragments for this model (e.g. "as" + "##pirin")
    aggregation_strategy="first",
)
sys.stderr = old_stderr
print("Model ready.\n")

BIO_SENTENCES = [
    "Aspirin inhibits COX-1 and reduces fever in patients with arthritis.",
    "BRCA1 mutations are found in hereditary breast cancer.",
    "Metformin is used to treat type 2 diabetes.",
    "HIV causes AIDS and damages the immune system.",
]

for sent in BIO_SENTENCES:
    ents = ner_bio.extract(sent)
    print(f"  Input : {sent!r}")
    for e in ents:
        print(f"    {e['text']:35s} → {e['label']}")
    if not ents:
        print("    (no entities detected)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Full db_aware pipeline: real GLiNER2 + biomedical NER + Neo4j
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("Part 3 — db_aware pipeline: real GLiNER2 + fine-tuned NER + Neo4j")
print(DIVIDER)

bio_schema = Schema(
    nodes={
        "Drug":    ["name"],
        "Disease": ["name"],
        "Symptom": ["name"],
    },
    relationships={
        "TREATS":   ("Drug",    "Disease", ["evidence"]),
        "CAUSES":   ("Disease", "Symptom", []),
        "RELIEVES": ("Drug",    "Symptom", []),
    },
)

db = Neo4jDatabase(DB_URI, DB_USER, DB_PASS)

print("\nLoading fastino/gliner2-large-v1 …")
pipeline = NLToCypher.from_pretrained(
    "fastino/gliner2-large-v1",
    schema=bio_schema,
    db=db,
    ner_extractor=ner_bio,
)
print("GLiNER2 model ready.\n")

def snapshot(db, label=""):
    nodes = db.execute(
        "MATCH (n) RETURN labels(n) AS lbl, n.name AS name ORDER BY lbl, name"
    )
    rels = db.execute(
        "MATCH (a)-[r]->(b) RETURN a.name AS a, type(r) AS t, b.name AS b"
    )
    print(f"\n  [{label}]")
    for n in nodes:
        print(f"    {n['lbl']} name={n['name']!r}")
    for r in rels:
        print(f"    ({r['a']})-[:{r['t']}]->({r['b']})")
    if not nodes:
        print("    (empty)")

TEXT = "Aspirin is used to treat fever and relieve inflammation."
snapshot(db, "BEFORE")

print(f"\n  Input : {TEXT!r}")
print("\n  NER labels detected:")
for e in ner_bio.extract(TEXT):
    print(f"    {e['text']:30s} → {e['label']}")

cypher, records = pipeline(TEXT, ["treats", "relieves"], db_aware=True, execute=True)

print("\n  Generated Cypher (db_aware=True):")
for line in cypher.strip().splitlines():
    print(f"    {line}")
print(f"\n  Returned {len(records)} record(s)")

snapshot(db, "AFTER")
db.close()

print(f"\n{DIVIDER}")
print("Summary")
print(DIVIDER)
print("""
  Approach A — dmis-lab/biobert-v1.1 + custom label_map
    + No separate fine-tuned model needed
    + All detected spans become a single user-defined label (BioEntity)
    - No type differentiation (Disease vs Drug vs Gene)
    - Pre-training artefact labels (LABEL_0 / LABEL_1)

  Approach B — d4data/biomedical-ner-all (recommended)
    + Fine-tuned on biomedical text → semantic labels out of the box
    + Outputs: Medication, Disease_disorder, Sign_symptom, …
    + Custom label_map bridges model labels to your graph schema
    + aggregation_strategy overridable via from_transformers(**kwargs)

  from_transformers() tip:
    All kwargs beyond model_name and label_map are forwarded to
    transformers.pipeline() — use them to set aggregation_strategy,
    device, batch_size, max_length, etc.
""")
