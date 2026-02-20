"""
Example 10 — GraphRAGPipeline: Graph RAG loop with schema injection,
             Cypher generation, validation, DB execution, and result formatting.

Pre-condition:
    DB should contain Person and Company nodes.  If starting fresh, seed manually:

        CREATE (a:Person {name: 'Alice', age: 30})-[:WORKS_FOR {since: 2020}]->(c:Company {name: 'Acme Corp'})
        CREATE (b:Person {name: 'Bob',   age: 25})-[:WORKS_FOR {since: 2022}]->(c)
        CREATE (d:Person {name: 'Diana', age: 35})-[:WORKS_FOR {since: 2018}]->(c2:Company {name: 'TechStart'})

Run:
    pip install "cypher_validator[neo4j]"
    python examples/10_graph_rag_pipeline.py
"""
from cypher_validator import Neo4jDatabase, Schema, CypherValidator
from cypher_validator.llm_utils import extract_cypher_from_text, format_records

DB_URI  = "neo4j://192.168.0.222:7687"
DB_USER = "neo4j"
DB_PASS = "qaz123!@#WSX"

schema = Schema(
    nodes={"Person": ["name", "age"], "Company": ["name"]},
    relationships={"WORKS_FOR": ("Person", "Company", ["since"])},
)
validator = CypherValidator(schema)

db = Neo4jDatabase(DB_URI, DB_USER, DB_PASS)

# ── Auto-discover schema from live DB ─────────────────────────────────────────
print("=== Auto-discovered schema via introspect_schema() ===\n")
live_schema = db.introspect_schema()
print(f"  Labels   : {live_schema.node_labels()}")
print(f"  Rel types: {live_schema.rel_types()}")

# ── Run and validate Cypher queries directly ──────────────────────────────────
queries = [
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name AS person, p.age AS age, c.name AS company",
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company {name: 'Acme Corp'}) RETURN p.name AS person, p.age AS age",
    "MATCH (p:Person) RETURN p.name AS person, p.age AS age ORDER BY p.age DESC",
]

for q in queries:
    result = validator.validate(q)
    records = db.execute(q)
    print(f"\n--- Query ---")
    print(f"  {q}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Results:")
    print(format_records(records, format="markdown"))

# ── execute_and_format shortcut ────────────────────────────────────────────────
print("\n=== execute_and_format (markdown) ===\n")
print(db.execute_and_format(
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, p.age, c.name"
))

print("\n=== execute_and_format (csv) ===\n")
print(db.execute_and_format(
    "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age",
    format="csv",
))

# ── extract_cypher_from_text — parse Cypher out of LLM-style text ─────────────
print("\n=== extract_cypher_from_text ===\n")
llm_style_response = (
    "Here is the query to find all employees:\n"
    "```cypher\n"
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name\n"
    "```"
)
cypher = extract_cypher_from_text(llm_style_response)
print(f"  Extracted: {cypher!r}")
result = validator.validate(cypher)
print(f"  Valid    : {result.is_valid}")
records = db.execute(cypher)
print(f"  Records  : {records}")

db.close()
