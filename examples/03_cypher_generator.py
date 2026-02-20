"""
Example 03 — CypherGenerator: generate all 13 query types and batch generation.

Run:
    python examples/03_cypher_generator.py
"""
from cypher_validator import Schema, CypherGenerator, CypherValidator

schema = Schema(
    nodes={
        "Person": ["name", "age"],
        "Movie":  ["title", "year"],
    },
    relationships={
        "ACTED_IN": ("Person", "Movie", ["role"]),
        "DIRECTED": ("Person", "Movie", []),
    },
)

gen       = CypherGenerator(schema, seed=42)
validator = CypherValidator(schema)

# ── 1. Generate one of each type ──────────────────────────────────────────────
print("=== All 13 query types ===\n")
for qtype in CypherGenerator.supported_types():
    q = gen.generate(qtype)
    r = validator.validate(q)
    status = "✓" if r.is_valid else f"✗ {r.errors}"
    print(f"  [{qtype:25s}] {status}")
    print(f"    {q}\n")

# ── 2. Batch generation ───────────────────────────────────────────────────────
batch = gen.generate_batch("match_return", 500)
results = validator.validate_batch(batch)
all_valid = all(r.is_valid for r in results)
print(f"=== Batch: 500 × match_return, all valid: {all_valid} ===")
