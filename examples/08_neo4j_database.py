"""
Example 08 — Neo4jDatabase: execute, execute_many, execute_and_format,
             introspect_schema, context manager.

Pre-condition:
    DB should contain some nodes.  If starting fresh, run example 05 first
    (which uses db_aware=True to populate Person / Company nodes), or seed
    manually:

        CREATE (:Person {name: 'Alice', age: 30})-[:WORKS_FOR]->(c:Company {name: 'Acme'})
        CREATE (:Person {name: 'Bob',   age: 25})-[:WORKS_FOR]->(c)
        CREATE (:City   {name: 'Berlin'})

Run:
    pip install "cypher_validator[neo4j]"
    python examples/08_neo4j_database.py
"""
from cypher_validator import Neo4jDatabase

DB_URI  = "neo4j://192.168.0.222:7687"
DB_USER = "neo4j"
DB_PASS = "qaz123!@#WSX"

# ── 1. Context manager ────────────────────────────────────────────────────────
print("=== Context manager ===")
with Neo4jDatabase(DB_URI, DB_USER, DB_PASS) as db:

    # ── 2. execute ────────────────────────────────────────────────────────────
    print("\n=== execute ===")
    rows = db.execute(
        "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name AS person, c.name AS company"
    )
    for row in rows:
        print(f"  {row}")

    # ── 3. execute with parameters ────────────────────────────────────────────
    print("\n=== execute with parameters ===")
    # Use whatever Person name exists in your DB — change 'Alice' if needed.
    rows = db.execute(
        "MATCH (p:Person {name: $name}) RETURN p.name, p.age",
        {"name": "Alice"},
    )
    print(f"  {rows}")

    # ── 4. execute_many ───────────────────────────────────────────────────────
    print("\n=== execute_many ===")
    all_results = db.execute_many([
        "MATCH (n:Person)  RETURN count(n) AS cnt",
        "MATCH (n:Company) RETURN count(n) AS cnt",
        "MATCH (n:City)    RETURN count(n) AS cnt",
    ])
    labels = ["Person", "Company", "City"]
    for label, result in zip(labels, all_results):
        print(f"  {label}: {result[0]['cnt']}")

    # ── 5. execute_and_format ─────────────────────────────────────────────────
    print("\n=== execute_and_format (markdown) ===")
    table = db.execute_and_format(
        "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, p.age, c.name"
    )
    print(table)

    print("\n=== execute_and_format (csv) ===")
    csv = db.execute_and_format(
        "MATCH (p:Person) RETURN p.name, p.age",
        format="csv",
    )
    print(csv)

    # ── 6. introspect_schema ──────────────────────────────────────────────────
    print("\n=== introspect_schema ===")
    schema = db.introspect_schema()
    print(f"  Node labels:    {schema.node_labels()}")
    print(f"  Rel types:      {schema.rel_types()}")
    print(f"  Person props:   {schema.node_properties('Person')}")
