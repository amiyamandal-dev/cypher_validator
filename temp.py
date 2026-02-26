from cypher_validator import Schema, NLToCypher, Neo4jDatabase

db = Neo4jDatabase("neo4j://192.168.0.222:7687", "neo4j", "qaz123!@#WSX")
pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1",db=db)

# execute=True → returns (cypher, records) instead of just cypher
cypher, records = pipeline(
    "John works for Apple Inc.",
    ["works_for"],
    mode="create",
    execute=True,
)
print(cypher)

