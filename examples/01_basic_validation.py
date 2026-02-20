"""
Example 01 — Schema definition, validation, and "did you mean?" suggestions.

Run:
    python examples/01_basic_validation.py
"""
from cypher_validator import Schema, CypherValidator

# ── 1. Define schema ──────────────────────────────────────────────────────────
schema = Schema(
    nodes={
        "Person":  ["name", "age", "email"],
        "Company": ["name", "founded"],
        "City":    ["name", "population"],
    },
    relationships={
        "WORKS_FOR": ("Person", "Company", ["since", "role"]),
        "LIVES_IN":  ("Person", "City",    []),
        "FOUNDED":   ("Person", "Company", ["year"]),
    },
)

validator = CypherValidator(schema)

# ── 2. Valid queries ──────────────────────────────────────────────────────────
valid_queries = [
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name",
    "MATCH (p:Person {age: 30})-[:LIVES_IN]->(city:City) RETURN city.name",
    "CREATE (p:Person {name: 'Alice', age: 25})-[:WORKS_FOR]->(c:Company {name: 'Acme'}) RETURN p, c",
    "MATCH (p:Person) WHERE p.age > 18 AND p.name = 'Bob' RETURN p",
]

print("=== Valid queries ===")
for q in valid_queries:
    r = validator.validate(q)
    print(f"  {'✓' if r.is_valid else '✗'}  {q[:70]}")

# ── 3. Invalid queries — "did you mean?" ─────────────────────────────────────
invalid_queries = [
    "MATCH (p:Preson)-[:WORKS_FOR]->(c:Company) RETURN p",       # typo: Preson
    "MATCH (p:Person)-[:WORKFOR]->(c:Company) RETURN p",          # typo: WORKFOR
    "MATCH (p:Person) RETURN p.salary",                           # unknown property
    "MATCH (m:Company)-[:WORKS_FOR]->(p:Person) RETURN m",        # wrong direction
    "MATCH (n:Person) RETURN x",                                   # unbound variable
]

print("\n=== Invalid queries (with suggestions) ===")
for q in invalid_queries:
    r = validator.validate(q)
    print(f"\n  Query : {q}")
    for err in r.errors:
        print(f"  Error : {err}")

# ── 4. Batch validation ───────────────────────────────────────────────────────
all_queries = valid_queries + invalid_queries
results = validator.validate_batch(all_queries)

print(f"\n=== Batch validation: {len(results)} queries ===")
valid_count = sum(1 for r in results if r.is_valid)
print(f"  Valid:   {valid_count}")
print(f"  Invalid: {len(results) - valid_count}")
