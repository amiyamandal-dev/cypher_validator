"""
Example 09 — LLM utility functions: extract_cypher_from_text, format_records,
             repair_cypher, cypher_tool_spec, few_shot_examples.

No real LLM or DB needed — uses mock functions.

Run:
    python examples/09_llm_utils.py
"""
from cypher_validator import (
    Schema, CypherValidator, CypherGenerator,
    extract_cypher_from_text, format_records,
    repair_cypher, cypher_tool_spec, few_shot_examples,
)

schema = Schema(
    nodes={"Person": ["name", "age"], "Company": ["name"]},
    relationships={"WORKS_FOR": ("Person", "Company", ["since"])},
)
validator = CypherValidator(schema)

# ── 1. extract_cypher_from_text ───────────────────────────────────────────────
print("=== extract_cypher_from_text ===\n")

examples = [
    # fenced block
    "Sure! Here is the query:\n```cypher\nMATCH (p:Person) RETURN p\n```",
    # inline backtick
    "Run `MATCH (n:Person) RETURN n` against your DB.",
    # plain text
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name",
    # any fenced block with Cypher keywords
    "```sql\nMATCH (n) RETURN n\n```",
]
for raw in examples:
    extracted = extract_cypher_from_text(raw)
    print(f"  Input : {raw[:60]!r}")
    print(f"  Output: {extracted!r}\n")

# ── 2. format_records ─────────────────────────────────────────────────────────
print("=== format_records ===\n")
records = [
    {"p.name": "Alice", "p.age": 30, "c.name": "Acme Corp"},
    {"p.name": "Bob",   "p.age": 25, "c.name": "TechStart"},
]

for fmt in ("markdown", "csv", "json", "text"):
    print(f"  [{fmt}]")
    print(format_records(records, format=fmt))
    print()

# ── 3. repair_cypher ──────────────────────────────────────────────────────────
print("=== repair_cypher ===\n")

bad_query = "MATCH (p:Persn)-[:WORKFOR]->(c:Company) RETURN p"
attempt_log = []

def mock_llm(query: str, errors: list) -> str:
    """Simulated LLM that fixes the query on the first attempt."""
    attempt_log.append({"query": query, "errors": errors})
    # Return the corrected query
    return "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p"

fixed, result = repair_cypher(validator, bad_query, mock_llm, max_retries=3)
print(f"  Original : {bad_query}")
print(f"  Fixed    : {fixed}")
print(f"  Valid    : {result.is_valid}")
print(f"  Attempts : {len(attempt_log)}")
print(f"  Errors shown to LLM: {attempt_log[0]['errors']}")

# ── 4. cypher_tool_spec ───────────────────────────────────────────────────────
print("\n=== cypher_tool_spec (Anthropic format) ===\n")
import json
tool = cypher_tool_spec(schema, db_description="HR knowledge graph", format="anthropic")
print(json.dumps(tool, indent=2)[:600] + "\n  ...")

# ── 5. few_shot_examples ──────────────────────────────────────────────────────
print("\n=== few_shot_examples ===\n")
gen      = CypherGenerator(schema, seed=0)
examples = few_shot_examples(gen, n=6)
for description, cypher in examples:
    print(f"  Q: {description}")
    print(f"  A: {cypher}\n")
