# cypher_validator

A fast, schema-aware **Cypher query validator and generator** with optional **GLiNER2 relation-extraction** support for natural-language-to-Cypher pipelines.

The core parser and validator are written in **Rust** (via [pyo3](https://pyo3.rs/) and [maturin](https://github.com/PyO3/maturin)) for performance. The GLiNER2 integration layer is pure Python and is an optional add-on.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core API](#core-api)
  - [Schema](#schema)
  - [CypherValidator](#cyphervalidator)
  - [ValidationResult](#validationresult)
  - [CypherGenerator](#cyphergenerator)
  - [parse\_query / QueryInfo](#parse_query--queryinfo)
- [GLiNER2 integration](#gliner2-integration)
  - [RelationToCypherConverter](#relationtocypherconverter)
  - [GLiNER2RelationExtractor](#gliner2relationextractor)
  - [NLToCypher](#nltocypher)
- [What the validator checks](#what-the-validator-checks)
- [Generated query types](#generated-query-types)
- [Performance](#performance)
- [Type stubs and IDE support](#type-stubs-and-ide-support)
- [Project structure](#project-structure)
- [Development](#development)

---

## Features

| Capability | Description |
|---|---|
| **Syntax validation** | Parses Cypher with a hand-written PEG grammar (Pest) and surfaces clear syntax errors |
| **Semantic validation** | Checks node labels, relationship types, properties, endpoint labels, relationship direction, and unbound variables against a user-supplied schema |
| **"Did you mean?" suggestions** | Typos in labels and relationship types produce helpful suggestions (e.g. `:Preson → did you mean :Person?`) via capped Levenshtein edit-distance |
| **Batch validation** | `validate_batch()` validates many queries in parallel using Rayon, releasing the Python GIL for the duration |
| **Query generation** | Generates syntactically correct and schema-valid Cypher queries for 13 common patterns |
| **Schema-free parsing** | Extracts labels, relationship types, and property keys from any query without requiring a schema |
| **Schema serialization** | `Schema.to_dict()` and `Schema.from_dict()` enable round-trip JSON serialization |
| **NL → Cypher** | Converts GLiNER2 relation-extraction output to MATCH / MERGE / CREATE queries with automatic deduplication |
| **Zero-shot RE** | Wraps the `gliner2` model for natural-language relation extraction (optional) |
| **Type stubs** | Full `.pyi` stub files for IDE autocompletion and mypy / pyright type checking |

---

## Installation

### Prerequisites

- Python ≥ 3.8
- Rust toolchain (for building from source — `rustup.rs`)
- `maturin` (install via `pip install maturin`)

### From source

```bash
# Clone the repository
git clone <repo-url>
cd cypher_validator

# Build and install in editable/development mode
maturin develop

# Or build an optimised release wheel
maturin build --release
pip install dist/cypher_validator-*.whl
```

---

## Quick start

```python
from cypher_validator import Schema, CypherValidator, CypherGenerator, parse_query

# 1. Define your graph schema
schema = Schema(
    nodes={
        "Person": ["name", "age"],
        "Movie":  ["title", "year"],
    },
    relationships={
        # rel_type: (source_label, target_label, [properties])
        "ACTED_IN": ("Person", "Movie", ["role"]),
        "DIRECTED": ("Person", "Movie", []),
    },
)

# 2. Validate a single query
validator = CypherValidator(schema)
result = validator.validate("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title")
print(result.is_valid)   # True

# 3. Validate with errors — get "did you mean?" suggestions
result = validator.validate("MATCH (p:Preson)-[:ACTEDIN]->(m:Movie) RETURN p")
print(result.is_valid)   # False
print(result.errors)
# ["Unknown node label: :Preson — did you mean :Person?",
#  "Unknown relationship type: :ACTEDIN — did you mean :ACTED_IN?"]

# 4. Validate multiple queries in parallel
results = validator.validate_batch([
    "MATCH (p:Person) RETURN p",
    "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name",
    "MATCH (x:BadLabel) RETURN x",
])
for r in results:
    print(r.is_valid, r.errors)

# 5. Round-trip schema serialization
d = schema.to_dict()
schema2 = Schema.from_dict(d)

# … or as a JSON string
json_str = schema.to_json()
schema3  = Schema.from_json(json_str)

# 6. Merge two schemas
s_extra = Schema({"Director": ["name"]}, {"DIRECTED": ("Director", "Movie", [])})
merged  = schema.merge(s_extra)   # union of labels, types, and properties

# 7. Generate random valid queries
gen = CypherGenerator(schema, seed=42)
print(gen.generate("match_relationship"))
# MATCH (a:Person)-[r:ACTED_IN]->(b:Movie) RETURN a, r, b

# Generate many queries at once (avoids per-call Python overhead)
batch = gen.generate_batch("match_return", 100)

# 8. Parse without a schema — also extracts property keys
info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.year")
print(info.is_valid)        # True
print(info.labels_used)     # ['Movie', 'Person']
print(info.rel_types_used)  # ['ACTED_IN']
print(info.properties_used) # ['name', 'year']
```

---

## Core API

### Schema

Describes the graph model: node labels with their allowed properties, and relationship types with their source label, target label, and allowed properties.

```python
from cypher_validator import Schema

schema = Schema(
    nodes={
        "Person":  ["name", "age", "email"],
        "Company": ["name", "founded"],
        "City":    ["name", "population"],
    },
    relationships={
        # "REL_TYPE": ("SourceLabel", "TargetLabel", ["prop1", "prop2"])
        "WORKS_FOR":        ("Person",  "Company", ["since", "role"]),
        "LIVES_IN":         ("Person",  "City",    []),
        "HEADQUARTERED_IN": ("Company", "City",    []),
    },
)
```

**Schema inspection methods:**

```python
schema.node_labels()               # ["City", "Company", "Person"]
schema.rel_types()                 # ["HEADQUARTERED_IN", "LIVES_IN", "WORKS_FOR"]
schema.has_node_label("Person")    # True
schema.has_rel_type("WORKS_FOR")   # True

schema.node_properties("Person")   # ["name", "age", "email"]
schema.rel_properties("WORKS_FOR") # ["since", "role"]
schema.rel_endpoints("WORKS_FOR")  # ("Person", "Company")
```

**Round-trip serialization:**

```python
# Export to a plain Python dict (JSON-serializable)
d = schema.to_dict()
# {
#   "nodes": {"Person": ["name", "age", "email"], ...},
#   "relationships": {"WORKS_FOR": ["Person", "Company", ["since", "role"]], ...}
# }

# Reconstruct from a dict (e.g. loaded from JSON)
import json
with open("schema.json") as f:
    schema2 = Schema.from_dict(json.load(f))

# or directly
schema2 = Schema.from_dict(d)
```

**JSON serialization (`to_json` / `from_json`):**

```python
# Serialise to a compact JSON string
json_str = schema.to_json()
# '{"nodes":{"Person":["age","email","name"],...},...}'

# Restore from the JSON string
schema2 = Schema.from_json(json_str)

# Store/load via a file
with open("schema.json", "w") as f:
    f.write(schema.to_json())

with open("schema.json") as f:
    schema3 = Schema.from_json(f.read())
```

**Merging schemas (`merge`):**

```python
s1 = Schema(
    nodes={"Person": ["name", "age"]},
    relationships={"KNOWS": ("Person", "Person", [])},
)
s2 = Schema(
    nodes={"Movie": ["title"], "Person": ["email"]},   # Person gets extra property
    relationships={"ACTED_IN": ("Person", "Movie", ["role"])},
)
merged = s1.merge(s2)
merged.node_labels()               # ["Movie", "Person"]
merged.node_properties("Person")   # ["age", "email", "name"]  ← union
merged.rel_types()                 # ["ACTED_IN", "KNOWS"]
```

`Schema.from_dict()` is the preferred way to restore a schema from a plain dict produced by `to_dict()`.

---

### CypherValidator

Validates Cypher queries against a `Schema` in two phases:

1. **Syntax** — Parses the query with the Cypher PEG grammar.
2. **Semantic** — Checks labels, types, properties, directions, and variable scopes against the schema. Typos in labels and relationship types trigger a "did you mean?" suggestion using capped Levenshtein edit-distance.

```python
from cypher_validator import CypherValidator

validator = CypherValidator(schema)

# Single query
result = validator.validate("MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name")
print(result.is_valid)   # True

# Batch validation — parallel Rayon execution, GIL released for the duration
results = validator.validate_batch([
    "MATCH (p:Person) RETURN p.name",
    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p, c",
    "MATCH (x:Employe) RETURN x",   # typo
])
for r in results:
    print(r.is_valid, r.errors)
# True  []
# True  []
# False ["Unknown node label: :Employe — did you mean :Employee?"]
```

**`validate_batch()` notes:**

- Accepts a `list[str]` and returns a `list[ValidationResult]` in the same order.
- Validation is done in parallel on the Rust side using [Rayon](https://github.com/rayon-rs/rayon).
- The Python GIL is released for the entire batch, so other Python threads (e.g. an asyncio event loop) are not blocked.
- There is no minimum batch size — it is efficient even for a single query, though the overhead is negligible for small lists.

---

### ValidationResult

Returned by both `CypherValidator.validate()` and each element of `CypherValidator.validate_batch()`.

| Attribute | Type | Description |
|---|---|---|
| `is_valid` | `bool` | `True` when no errors were found |
| `errors` | `list[str]` | All errors combined (syntax + semantic) |
| `syntax_errors` | `list[str]` | Parse / grammar errors only |
| `semantic_errors` | `list[str]` | Schema-level errors only |

Also supports `bool(result)` and `len(result)`:

```python
result = validator.validate(query)

if result:
    print("Valid!")
else:
    print(f"{len(result)} error(s):")
    for err in result.errors:
        print(" -", err)

# Categorised errors
print(result.syntax_errors)   # e.g. ["Parse error: …"]
print(result.semantic_errors) # e.g. ["Unknown node label: :Foo — did you mean :Bar?"]
```

**Example error messages:**

```
# Unknown label — with suggestion
"Unknown node label: :Preson — did you mean :Person?"

# Unknown label — no close match
"Unknown node label: :Actor"

# Unknown relationship type — with suggestion
"Unknown relationship type: :ACTEDIN — did you mean :ACTED_IN?"

# Property not in schema
"Unknown property 'salary' for node label :Person"

# Wrong relationship direction / endpoints
"Relationship :ACTED_IN expects source label :Person, but node has label(s): :Movie"
"Relationship :ACTED_IN expects target label :Movie, but node has label(s): :Person"

# Unbound variable
"Variable 'x' is not bound in this scope"

# WITH scope reset
"Variable 'n' is not bound in this scope"  # used after WITH that didn't project it

# Label used in SET/REMOVE
"Unknown node label: :Managr — did you mean :Manager?"
```

"Did you mean?" suggestions appear when a misspelled label or relationship type has a Levenshtein edit-distance of ≤ 2 from a known schema entry (case-insensitive comparison).

---

### CypherGenerator

Generates syntactically correct, schema-valid Cypher queries for rapid prototyping, testing, and dataset creation.

```python
from cypher_validator import CypherGenerator

gen = CypherGenerator(schema)           # random seed each run
gen = CypherGenerator(schema, seed=42)  # deterministic / reproducible output

query = gen.generate("match_return")
# "MATCH (n:Person) RETURN n LIMIT 17"

query = gen.generate("order_by")
# "MATCH (n:Movie) RETURN n ORDER BY n.year DESC LIMIT 5"

query = gen.generate("distinct_return")
# "MATCH (n:Person) RETURN DISTINCT n.name"

query = gen.generate("unwind")
# "MATCH (n:Person) UNWIND n.name AS item RETURN item"

# List all supported patterns
CypherGenerator.supported_types()
# ['match_return', 'match_where_return', 'create', 'merge', 'aggregation',
#  'match_relationship', 'create_relationship', 'match_set', 'match_delete',
#  'with_chain', 'distinct_return', 'order_by', 'unwind']

# Generate many queries in one call (avoids per-call Python overhead)
queries = gen.generate_batch("match_return", 500)  # list[str], len == 500
```

**Supported query types (13 total):**

| Type | Example output |
|---|---|
| `match_return` | `MATCH (n:Movie) RETURN n` |
| `match_where_return` | `MATCH (n:Person) WHERE n.name = "Alice" RETURN n.name` |
| `create` | `CREATE (n:Person {name: "Bob", age: 42}) RETURN n` |
| `merge` | `MERGE (n:Movie {title: $value}) RETURN n` |
| `aggregation` | `MATCH (n:Person) RETURN count(n.age) AS result` |
| `match_relationship` | `OPTIONAL MATCH (a:Person)-[r:ACTED_IN]->(b:Movie) RETURN a, r, b` |
| `create_relationship` | `MATCH (a:Person),(b:Movie) CREATE (a)-[r:ACTED_IN]->(b) RETURN r` |
| `match_set` | `MATCH (n:Person) SET n.name = "Carol" RETURN n` |
| `match_delete` | `MATCH (n:Movie) DETACH DELETE n` |
| `with_chain` | `MATCH (n:Person) WITH n.name AS val RETURN count(*)` |
| `distinct_return` | `MATCH (n:Person) RETURN DISTINCT n.name LIMIT 10` |
| `order_by` | `MATCH (n:Movie) RETURN n ORDER BY n.year DESC LIMIT 25` |
| `unwind` | `MATCH (n:Person) UNWIND n.age AS item RETURN item` |

Generated scalar values cycle through string literals (`"Alice"`), integers (`42`), booleans (`true`/`false`), and parameters (`$name`). `OPTIONAL MATCH` and `LIMIT` clauses are randomly included. All generated queries are guaranteed to pass `CypherValidator` with the same schema.

---

### parse\_query / QueryInfo

Parse a Cypher string and extract structural information **without needing a schema**.

```python
from cypher_validator import parse_query

info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.year")

info.is_valid        # True
info.errors          # []
info.labels_used     # ["Movie", "Person"]   (sorted, deduplicated)
info.rel_types_used  # ["ACTED_IN"]          (sorted, deduplicated)
info.properties_used # ["name", "year"]      (sorted, deduplicated)

bool(info)           # True  (same as is_valid)

# Invalid query
info = parse_query("THIS IS NOT CYPHER")
info.is_valid   # False
info.errors     # ["Parse error: …"]
```

`QueryInfo` attributes:

| Attribute | Type | Description |
|---|---|---|
| `is_valid` | `bool` | Syntax check result |
| `errors` | `list[str]` | Syntax error messages (empty when valid) |
| `labels_used` | `list[str]` | Sorted, deduplicated node labels referenced in the query |
| `rel_types_used` | `list[str]` | Sorted, deduplicated relationship types referenced in the query |
| `properties_used` | `list[str]` | Sorted, deduplicated property keys accessed anywhere in the query |

`properties_used` collects every property key that appears in the query — in `RETURN`, `WHERE`, `SET`, inline node/relationship maps, `WITH`, `ORDER BY`, list comprehensions, etc. This is useful for dependency analysis, schema introspection, and query auditing without a schema.

```python
# Complex query — all property accesses are captured
info = parse_query("""
    MATCH (p:Person {age: 30})-[r:WORKS_FOR {since: 2020}]->(c:Company)
    WHERE p.name = "Alice" AND c.founded > 2000
    WITH p, p.email AS contact
    RETURN contact, c.name
    ORDER BY c.name
""")
info.properties_used
# ['age', 'email', 'founded', 'name', 'since']
```

---

## GLiNER2 integration

The GLiNER2 integration converts relation-extraction results into Cypher queries. It consists of three classes, each usable independently.

### RelationToCypherConverter

A **pure-Python** class that converts any dict matching the GLiNER2 output format into a Cypher query. No ML model required.

```python
from cypher_validator import RelationToCypherConverter, Schema

# GLiNER2 extraction result
results = {
    "relation_extraction": {
        "works_for": [("John", "Apple Inc.")],
        "lives_in":  [("John", "San Francisco")],
        "founded":   [],   # requested but not found in text
    }
}

# Without schema (no node labels)
converter = RelationToCypherConverter()

# ── MATCH mode (find existing data) ────────────────────────────────────────
print(converter.to_match_query(results))
# MATCH (a0 {name: "John"})-[:WORKS_FOR]->(b0 {name: "Apple Inc."})
# MATCH (a1 {name: "John"})-[:LIVES_IN]->(b1 {name: "San Francisco"})
# RETURN a0, b0, a1, b1

# ── MERGE mode (upsert) ────────────────────────────────────────────────────
print(converter.to_merge_query(results))
# MERGE (a0 {name: "John"})-[:WORKS_FOR]->(b0 {name: "Apple Inc."})
# MERGE (a1 {name: "John"})-[:LIVES_IN]->(b1 {name: "San Francisco"})
# RETURN a0, b0, a1, b1

# ── CREATE mode (insert new) ───────────────────────────────────────────────
print(converter.to_create_query(results))
# CREATE (a0 {name: "John"})-[:WORKS_FOR]->(b0 {name: "Apple Inc."})
# CREATE (a1 {name: "John"})-[:LIVES_IN]->(b1 {name: "San Francisco"})
# RETURN a0, b0, a1, b1

# ── Unified dispatcher ─────────────────────────────────────────────────────
cypher = converter.convert(results, mode="merge")
```

**With a schema** — node labels are added automatically:

```python
schema = Schema(
    nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
    relationships={
        "WORKS_FOR": ("Person", "Company", []),
        "LIVES_IN":  ("Person", "City",    []),
    },
)
converter = RelationToCypherConverter(schema=schema)
print(converter.to_merge_query(results))
# MERGE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Apple Inc."})
# MERGE (a1:Person {name: "John"})-[:LIVES_IN]->(b1:City {name: "San Francisco"})
# RETURN a0, b0, a1, b1
```

**Multiple pairs of the same relation type:**

```python
results = {
    "relation_extraction": {
        "works_for": [
            ("John",  "Microsoft"),
            ("Mary",  "Google"),
            ("Bob",   "Apple"),
        ],
    }
}
print(converter.to_merge_query(results))
# MERGE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Microsoft"})
# MERGE (a1:Person {name: "Mary"})-[:WORKS_FOR]->(b1:Company {name: "Google"})
# MERGE (a2:Person {name: "Bob"})-[:WORKS_FOR]->(b2:Company {name: "Apple"})
# RETURN a0, b0, a1, b1, a2, b2
```

**Automatic deduplication:**

If the same `(subject, object)` pair for a given relation type appears more than once in the extraction output (which can happen with overlapping model detections), the converter silently deduplicates them — each unique triple `(subject, relation, object)` appears exactly once in the generated Cypher:

```python
results = {
    "relation_extraction": {
        "works_for": [
            ("John", "Apple"),   # first occurrence
            ("John", "Apple"),   # duplicate — skipped
            ("Mary", "Apple"),   # different subject — kept
        ]
    }
}
# Only two MERGE clauses are emitted, not three
```

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `schema` | `None` | `cypher_validator.Schema` for label-aware generation |
| `name_property` | `"name"` | Node property key used for entity text spans |

**`convert()` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `relations` | required | GLiNER2 output dict |
| `mode` | `"match"` | `"match"`, `"merge"`, or `"create"` |
| `return_clause` | auto | Custom `RETURN …` tail (e.g. `"RETURN *"`) |

---

### GLiNER2RelationExtractor

Wraps a loaded `gliner2.GLiNER2` model and normalises its output to the standard format.

```python
from cypher_validator import GLiNER2RelationExtractor

# Load from HuggingFace Hub
extractor = GLiNER2RelationExtractor.from_pretrained("fastino/gliner2-large-v1")

# Or set a custom threshold
extractor = GLiNER2RelationExtractor.from_pretrained(
    "fastino/gliner2-large-v1",
    threshold=0.7,
)

text = "John works for Apple Inc. and lives in San Francisco."

results = extractor.extract_relations(
    text,
    relation_types=["works_for", "lives_in", "founded"],
)
# {
#     "relation_extraction": {
#         "works_for": [("John", "Apple Inc.")],
#         "lives_in":  [("John", "San Francisco")],
#         "founded":   [],   # requested but not found
#     }
# }

# Override threshold for a single call
results = extractor.extract_relations(
    text,
    ["works_for"],
    threshold=0.85,   # high precision
)
```

**Key behaviours:**
- Every requested relation type is **always present** in the output — missing types get an empty list.
- Works with both the wrapped (`{"relation_extraction": {...}}`) and flat (`{rel_type: [...]}`) model output formats.
- `threshold` set at construction becomes the default; it can be overridden per-call.

| Method / attribute | Description |
|---|---|
| `GLiNER2RelationExtractor.from_pretrained(model_name, threshold=0.5)` | Load from HuggingFace Hub or local path |
| `extractor.extract_relations(text, relation_types, threshold=None)` | Extract relations from text |
| `extractor.threshold` | Instance-level default confidence threshold |
| `GLiNER2RelationExtractor.DEFAULT_MODEL` | `"fastino/gliner2-large-v1"` |

---

### NLToCypher

End-to-end pipeline combining `GLiNER2RelationExtractor` and `RelationToCypherConverter`.

```python
from cypher_validator import NLToCypher, Schema

schema = Schema(
    nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
    relationships={
        "WORKS_FOR": ("Person", "Company", []),
        "LIVES_IN":  ("Person", "City",    []),
    },
)

pipeline = NLToCypher.from_pretrained(
    "fastino/gliner2-large-v1",
    schema=schema,       # optional: enables label-aware generation
    threshold=0.5,
)

# Single sentence → Cypher
cypher = pipeline(
    "John works for Apple Inc. and lives in San Francisco.",
    relation_types=["works_for", "lives_in"],
    mode="merge",
)
# MERGE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Apple Inc."})
# MERGE (a1:Person {name: "John"})-[:LIVES_IN]->(b1:City {name: "San Francisco"})
# RETURN a0, b0, a1, b1

# Get both the raw extraction dict and the Cypher string
relations, cypher = pipeline.extract_and_convert(
    "Alice manages the Engineering team.",
    ["manages", "reports_to"],
    mode="match",
)
print(relations)
# {"relation_extraction": {"manages": [("Alice", "Engineering team")], "reports_to": []}}
print(cypher)
# MATCH (a0 {name: "Alice"})-[:MANAGES]->(b0 {name: "Engineering team"})
# RETURN a0, b0

# High-precision extraction
cypher = pipeline(
    "Bob acquired TechCorp in 2019.",
    ["acquired", "merged_with"],
    mode="merge",
    threshold=0.85,
)
```

**Database-aware execution (`execute=True`):**

Pass a `Neo4jDatabase` to execute the generated query directly and receive both the Cypher string and the Neo4j records:

```python
from cypher_validator import NLToCypher, Neo4jDatabase

db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema, db=db)

# execute=True → returns (cypher, records) instead of just cypher
cypher, records = pipeline(
    "John works for Apple Inc.",
    ["works_for"],
    mode="create",
    execute=True,
)
# cypher  → 'CREATE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Apple Inc."})\nRETURN a0, b0'
# records → [{"a0": {...}, "b0": {...}}]  (Neo4j driver records as dicts)
```

**Credentials from environment variables (`from_env`):**

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j        # optional, defaults to "neo4j"
export NEO4J_PASSWORD=secret
```

```python
# Reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD automatically
pipeline = NLToCypher.from_env("fastino/gliner2-large-v1", schema=schema)

cypher, records = pipeline(
    "John works for Apple Inc.",
    ["works_for"],
    mode="create",
    execute=True,
)
```

**`from_pretrained()` / `from_env()` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"fastino/gliner2-large-v1"` | HuggingFace model ID or local path |
| `schema` | `None` | Optional schema for label-aware Cypher |
| `threshold` | `0.5` | Confidence threshold for relation extraction |
| `name_property` | `"name"` | Node property key for entity text |
| `db` | `None` | `Neo4jDatabase` connection (`from_pretrained` only) |
| `database` | `"neo4j"` | Neo4j database name (`from_env` only) |

**`__call__()` / `extract_and_convert()` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `text` | required | Input sentence or passage |
| `relation_types` | required | Relation labels to extract |
| `mode` | `"match"` | Cypher generation mode (`"match"`, `"merge"`, `"create"`) |
| `threshold` | `None` | Override instance threshold |
| `execute` | `False` | When `True`, run the query against the DB and return `(cypher, records)` |
| `return_clause` | auto | Custom `RETURN …` tail |

---

### Neo4jDatabase

Thin wrapper around the official [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/) for executing Cypher queries. Requires `pip install "cypher_validator[neo4j]"`.

```python
from cypher_validator import Neo4jDatabase

# Direct instantiation
db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password", database="neo4j")

# Context manager — driver is closed on exit
with Neo4jDatabase("bolt://localhost:7687", "neo4j", "password") as db:
    results = db.execute("MATCH (n:Person) RETURN n.name LIMIT 5")
    # [{"n.name": "Alice"}, {"n.name": "Bob"}, ...]

# With parameters
results = db.execute(
    "MATCH (n:Person {name: $name}) RETURN n",
    {"name": "Alice"},
)

# Run multiple queries in one call
queries = [
    "MATCH (n:Person) RETURN count(n)",
    "MATCH (n:Movie) RETURN count(n)",
]
all_results = db.execute_many(queries)
# [[{"count(n)": 42}], [{"count(n)": 17}]]

# With per-query parameters
all_results = db.execute_many(
    ["MATCH (n {name: $x}) RETURN n", "MATCH (n {name: $x}) RETURN n"],
    [{"x": "Alice"}, {"x": "Bob"}],
)
```

| Method | Description |
|---|---|
| `execute(cypher, parameters=None)` | Run one query, return `list[dict]` |
| `execute_many(queries, parameters_list=None)` | Run multiple queries, return `list[list[dict]]` |
| `close()` | Release driver connections |

---

## What the validator checks

The semantic validator performs the following checks **against the provided schema**:

### Node labels
- Every node label used (`:Person`, `:Movie`, …) must exist in the schema.
- Properties accessed on a labelled node (e.g. `n.salary` on `:Person`) must be declared for that label.
- Typos trigger a "did you mean?" suggestion when a schema label is within edit-distance 2.

### Relationship types
- Every relationship type used (`[:ACTED_IN]`, …) must exist in the schema.
- Properties accessed on a labelled relationship (e.g. `r.since` on `:WORKS_FOR`) must be declared for that type.
- Typos trigger a "did you mean?" suggestion when a schema type is within edit-distance 2.

### Relationship direction and endpoints
- For **directed** relationships (`-->` or `<--`), the source and target node labels are checked against the schema's declared endpoints.
- **Undirected** patterns (`--`) skip the endpoint-direction check.
- Nodes **without labels** are skipped (open-world assumption).

```python
# Schema: ACTED_IN: (Person → Movie)
validator.validate("MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN m")
# Error: "Relationship :ACTED_IN expects source label :Person, but node has label(s): :Movie"
# Error: "Relationship :ACTED_IN expects target label :Movie, but node has label(s): :Person"

# Undirected — no direction error
validator.validate("MATCH (m:Movie)-[:ACTED_IN]-(p:Person) RETURN m")  # valid
```

### Variable scope and unbound variables
- Variables in `RETURN`, `WHERE`, `SET`, `DELETE`, etc. must have been bound in a preceding `MATCH`, `CREATE`, `MERGE`, or `UNWIND`.
- `WITH` enforces a **scope reset**: only variables explicitly projected through `WITH` remain accessible afterwards.

```python
validator.validate("MATCH (n:Person) RETURN m")
# Error: "Variable 'm' is not bound in this scope"

validator.validate("MATCH (n:Person) WITH n.name AS nm RETURN n")
# Error: "Variable 'n' is not bound in this scope"  (n not projected through WITH)

validator.validate("MATCH (n:Person) WITH n.name AS nm RETURN nm")
# valid — nm was projected
```

### WHERE clause boolean operators
- `AND`, `OR`, `XOR`, and `NOT` in `WHERE` clauses are fully supported, including combinations and precedence.

```python
validator.validate(
    "MATCH (p:Person) WHERE p.age > 30 AND p.name = 'Alice' OR p.name = 'Bob' RETURN p"
)
# valid

validator.validate(
    "MATCH (p:Person) WHERE NOT p.age < 18 AND p.name = 'Alice' RETURN p"
)
# valid
```

### List comprehensions and quantifiers
- Variables introduced by `[x IN list | ...]` and `ALL(x IN list WHERE ...)` are locally scoped and don't leak.

### Open-world assumption
- Nodes and relationships **without labels/types** are never flagged (e.g. `MATCH (n) RETURN n` is always valid).
- Property access on variables without known labels is not checked.

---

## Generated query types

`CypherGenerator` supports **13** query patterns. All generated queries are guaranteed to pass `CypherValidator` with the same schema.

```python
gen = CypherGenerator(schema, seed=0)
for query_type in CypherGenerator.supported_types():
    print(f"{query_type}: {gen.generate(query_type)}")
```

Generated property values cycle through four value types: string literals (`"Alice"`), integers (`42`), booleans (`true`/`false`), and parameters (`$name`). `OPTIONAL MATCH` and `LIMIT` clauses appear randomly. `ORDER BY` queries randomly include `DESC`.

---

## Performance

The Rust core is designed for low-latency, high-throughput validation:

### Parallel batch validation

`validate_batch()` uses [Rayon](https://github.com/rayon-rs/rayon) to validate queries across all available CPU cores simultaneously. The Python GIL is released for the entire batch, so asyncio and other Python threads remain unblocked:

```python
import time

queries = ["MATCH (n:Person) RETURN n"] * 10_000

start = time.perf_counter()
results = validator.validate_batch(queries)
elapsed = time.perf_counter() - start

print(f"Validated {len(results)} queries in {elapsed:.3f}s")
# Validated 10000 queries in ~0.05s  (hardware-dependent)
```

### Capped Levenshtein

"Did you mean?" suggestions use a 1D rolling-array Levenshtein implementation with two early-exit optimisations:

1. **Length-difference short-circuit** — if `|len(a) - len(b)| > cap`, return immediately without running the algorithm.
2. **Row-minimum early exit** — if the minimum value in the current DP row already exceeds `cap`, no future row can produce a distance ≤ cap, so we exit early.

This keeps suggestion lookup fast even when the schema has many labels.

### O(1) property lookup

Node and relationship property sets are stored as Rust `HashSet<String>` internally, so `node_has_property` and `rel_has_property` are O(1) regardless of how many properties a label declares.

### Construction-time caching

`CypherGenerator` and `SemanticValidator` precompute their working data sets (label lists, relationship-type lists, per-label property maps) once at construction. Repeated calls to `generate()` / `generate_batch()` and `validate()` avoid redundant allocations.

---

## Type stubs and IDE support

Full `.pyi` stub files are included in the package, providing:

- **IDE autocompletion** for all Rust-backed classes (`Schema`, `CypherValidator`, `ValidationResult`, `CypherGenerator`, `QueryInfo`, `parse_query`)
- **mypy / pyright type checking** — all methods, attributes, and return types are fully annotated
- **Inline docstrings** accessible via IDE hover / `help()`

The stubs are automatically discovered by type checkers when the package is installed. No additional configuration is needed.

```python
# mypy will verify this correctly
from cypher_validator import Schema, CypherValidator

schema: Schema = Schema(nodes={"Person": ["name"]}, relationships={})
validator: CypherValidator = CypherValidator(schema)
result = validator.validate("MATCH (p:Person) RETURN p")
reveal_type(result.is_valid)  # Revealed type is "bool"
reveal_type(result.errors)    # Revealed type is "list[str]"
```

---

## Project structure

```
cypher_validator/
├── Cargo.toml                        # Rust crate (lib name: _cypher_validator)
├── Cargo.lock                        # Locked dependency versions
├── pyproject.toml                    # maturin mixed-package config
│
├── src/                              # Rust source
│   ├── lib.rs                        # PyO3 module registration
│   ├── error.rs                      # CypherError enum
│   ├── grammar/
│   │   └── cypher.pest               # PEG grammar (Pest)
│   ├── parser/
│   │   ├── mod.rs                    # parse() entry point
│   │   ├── ast.rs                    # AST types
│   │   └── builder.rs                # Pest → AST builder
│   ├── schema/
│   │   └── mod.rs                    # Schema struct
│   ├── validator/
│   │   ├── mod.rs                    # CypherValidator, ValidationResult
│   │   └── semantic.rs               # SemanticValidator (labels, props, scope, "did you mean")
│   ├── generator/
│   │   └── mod.rs                    # CypherGenerator (13 query types)
│   └── bindings/
│       ├── mod.rs
│       ├── py_schema.rs              # Python Schema wrapper (incl. from_dict)
│       ├── py_validator.rs           # Python CypherValidator / ValidationResult (incl. validate_batch)
│       ├── py_generator.rs           # Python CypherGenerator
│       └── py_parser.rs              # Python parse_query / QueryInfo (incl. properties_used)
│
├── python/
│   └── cypher_validator/             # Python package (maturin python-source)
│       ├── __init__.py               # Re-exports Rust core + GLiNER2 classes
│       ├── __init__.pyi              # Package-level type stubs
│       ├── _cypher_validator.pyi     # Rust extension type stubs
│       └── gliner2_integration.py   # RelationToCypherConverter, GLiNER2RelationExtractor, NLToCypher
│
└── tests/                            # 261 tests total
    ├── test_syntax.py                # PEG grammar / syntax tests
    ├── test_schema.py                # Schema API tests
    ├── test_validator.py             # Validator smoke tests
    ├── test_new_features.py          # Direction, scope, error-category, parse_query tests
    ├── test_generator.py             # CypherGenerator tests (all 13 types)
    ├── test_roundtrip.py             # Generator output validated by validator
    ├── test_gliner2_integration.py   # GLiNER2 integration tests (no ML required)
    ├── test_task2_features.py        # AND/OR grammar fix, Schema.from_dict, validate_batch, properties_used
    └── test_task3_features.py        # "Did you mean", new generator types, deduplication
```

---

## Development

### Building

```bash
# Development build (installs editable into the active venv)
maturin develop

# Optimised release build
maturin develop --release

# Build a distributable wheel
maturin build --release
```

### Running tests

```bash
# All 283 tests
pytest tests/

# Specific test modules
pytest tests/test_task2_features.py -v   # Schema.from_dict, validate_batch, properties_used
pytest tests/test_task3_features.py -v   # "did you mean", new generator types, dedup

# GLiNER2 integration only (no gliner2 package needed — uses mocks)
pytest tests/test_gliner2_integration.py -v

# With coverage
pytest tests/ --cov=cypher_validator
```

The GLiNER2 tests use `unittest.mock` to simulate the model, so the full test suite runs without installing `gliner2`.

### Dependency management

Python dependencies are managed with `uv`:

```bash
uv pip install maturin pytest
uv pip install gliner2   # optional, for NLToCypher
```

### CI

GitHub Actions (`.github/workflows/CI.yml`) builds wheels for Linux (x86\_64, x86, aarch64, armv7, s390x, ppc64le), Linux musl, Windows (x64, x86, arm64), and macOS (x86\_64, aarch64) on every push. Releases to PyPI are triggered by pushing a version tag.
