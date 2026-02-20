"""
Example 02 — Schema serialization (to_dict, from_dict, to_json, from_json, merge).

Run:
    python examples/02_schema_serialization.py
"""
import json
from cypher_validator import Schema

schema = Schema(
    nodes={
        "Person":  ["name", "age"],
        "Company": ["name", "founded"],
    },
    relationships={
        "WORKS_FOR": ("Person", "Company", ["since", "role"]),
        "KNOWS":     ("Person", "Person",  []),
    },
)

# ── 1. to_dict / from_dict ────────────────────────────────────────────────────
d = schema.to_dict()
print("=== to_dict() ===")
print(json.dumps(d, indent=2))

schema2 = Schema.from_dict(d)
print(f"\nRound-trip OK: {schema2.node_labels() == schema.node_labels()}")

# ── 2. to_json / from_json ────────────────────────────────────────────────────
json_str = schema.to_json()
print(f"\n=== to_json() ===\n{json_str}")

schema3 = Schema.from_json(json_str)
print(f"\nRound-trip OK: {schema3.rel_types() == schema.rel_types()}")

# ── 3. Schema.merge — union of two schemas ────────────────────────────────────
extra = Schema(
    nodes={"Movie": ["title", "year"], "Person": ["email"]},  # Person gets extra prop
    relationships={"DIRECTED": ("Person", "Movie", [])},
)

merged = schema.merge(extra)
print("\n=== merge() ===")
print(f"Node labels:          {merged.node_labels()}")
print(f"Person properties:    {merged.node_properties('Person')}")  # union
print(f"Relationship types:   {merged.rel_types()}")

# ── 4. Schema prompt helpers ──────────────────────────────────────────────────
print("\n=== to_prompt() ===")
print(schema.to_prompt())

print("\n=== to_cypher_context() ===")
print(schema.to_cypher_context())
