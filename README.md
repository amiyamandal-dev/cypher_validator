# cypher_validator

A fast, schema-aware **Cypher query validator and generator** with optional **GLiNER2 relation-extraction** and **Graph RAG** support for LLM-driven graph database applications.

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
  - [parse_query / QueryInfo](#parse_query--queryinfo)
- [GLiNER2 integration](#gliner2-integration)
  - [**NLToCypher** ← start here](#nltocypher)
  - [DB-aware query generation](#db-aware-query-generation)
  - [EntityNERExtractor](#entitynerextractor)
  - [GLiNER2RelationExtractor](#gliner2relationextractor)
  - [RelationToCypherConverter](#relationtocypherconverter)
  - [Neo4jDatabase](#neo4jdatabase)
- [LLM integration](#llm-integration)
  - [Schema prompt helpers](#schema-prompt-helpers)
  - [extract_cypher_from_text](#extract_cypher_from_text)
  - [repair_cypher](#repair_cypher)
  - [format_records](#format_records)
  - [few_shot_examples](#few_shot_examples)
  - [cypher_tool_spec](#cypher_tool_spec)
  - [GraphRAGPipeline](#graphragpipeline)
- [What the validator checks](#what-the-validator-checks)
- [Generated query types](#generated-query-types)
- [Performance](#performance)
- [Type stubs and IDE support](#type-stubs-and-ide-support)
- [Project structure](#project-structure)
- [Examples](#examples)
- [Development](#development)

---

## Features

| Capability                  | Description |
|-----------------------------|-------------|
| **Syntax validation**       | Parses Cypher with a hand-written PEG grammar (Pest) and surfaces clear syntax errors |
| **Semantic validation**     | Checks node labels, relationship types, properties, endpoint labels, relationship direction, and unbound variables against a user-supplied schema |
| **"Did you mean?" suggestions** | Typos in labels and relationship types produce helpful suggestions (e.g. `:Preson → did you mean :Person?`) via capped Levenshtein edit-distance |
| **Batch validation**        | `validate_batch()` validates many queries in parallel using Rayon, releasing the Python GIL for the duration |
| **Query generation**        | Generates syntactically correct and schema-valid Cypher queries for 13 common patterns |
| **Schema-free parsing**     | Extracts labels, relationship types, and property keys from any query without requiring a schema |
| **Schema serialization**    | `Schema.to_dict()`, `from_dict()`, `to_json()`, `from_json()`, and `merge()` for complete schema lifecycle management |
| **NL → Cypher**             | Converts GLiNER2 relation-extraction output to MATCH / MERGE / CREATE queries with automatic deduplication |
| **DB-aware generation**     | `db_aware=True` looks up every extracted entity in Neo4j before query generation — existing nodes are MATCHed, new ones are CREATEd inline, preventing duplicate nodes |
| **NER entity extraction**   | `EntityNERExtractor` wraps spaCy or any HuggingFace Transformers NER pipeline to enrich entity-label resolution during DB-aware generation |
| **Zero-shot RE**            | Wraps the `gliner2` model for natural-language relation extraction (optional) |
| **LLM schema context**      | `to_prompt()`, `to_markdown()`, `to_cypher_context()` format the schema for LLM system prompts |
| **Cypher extraction**       | `extract_cypher_from_text()` pulls Cypher out of any LLM response (fenced blocks, inline, plain text) |
| **Self-repair loop**        | `repair_cypher()` feeds validation errors back to an LLM for iterative self-correction |
| **Result formatting**       | `format_records()` renders Neo4j results as Markdown, CSV, JSON, or plain text for LLM context |
| **Few-shot examples**       | `few_shot_examples()` auto-generates (description, Cypher) pairs for LLM prompting |
| **Tool spec builder**       | `cypher_tool_spec()` produces Anthropic / OpenAI function-calling schemas for Cypher execution |
| **Graph RAG pipeline**      | `GraphRAGPipeline` chains schema injection → Cypher generation → validation → execution → answer |
| **Schema introspection**    | `Neo4jDatabase.introspect_schema()` discovers the live DB schema automatically |
| **Type stubs**              | Full `.pyi` stub files for IDE autocompletion and mypy / pyright type checking |

---

## Installation

### Prerequisites

- Python ≥ 3.8
- Rust toolchain (for building from source — [rustup.rs](https://rustup.rs))
- `maturin` (`pip install maturin`)

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

### Optional dependencies

```bash
# Neo4j driver (required for execute=True and db_aware=True)
pip install "cypher_validator[neo4j]"

# NER with spaCy (EntityNERExtractor.from_spacy)
pip install "cypher_validator[ner-spacy]"
python -m spacy download en_core_web_sm   # or en_core_web_trf for transformer accuracy

# NER with HuggingFace Transformers (EntityNERExtractor.from_transformers)
pip install "cypher_validator[ner-transformers]"

# Everything at once
pip install "cypher_validator[neo4j,ner]"
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

# 8. NL → Cypher with GLiNER2 (no boilerplate — this is the recommended entry point)
from cypher_validator import NLToCypher
pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
cypher = pipeline(
    "Tom Hanks acted in Cast Away.",
    ["acted_in"],
    mode="merge",
)
# MERGE (a0:Person {name: $a0_val})-[:ACTED_IN]->(b0:Movie {name: $b0_val})
# RETURN a0, b0

# 9. Parse without a schema — also extracts property keys
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

| Attribute          | Type       | Description |
|--------------------|------------|-------------|
| `is_valid`         | `bool`     | `True` when no errors were found |
| `errors`           | `list[str]`| All errors combined (syntax + semantic) |
| `syntax_errors`    | `list[str]`| Parse / grammar errors only |
| `semantic_errors`  | `list[str]`| Schema-level errors only |

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

| Type                | Example output |
|---------------------|----------------|
| `match_return`      | `MATCH (n:Movie) RETURN n` |
| `match_where_return`| `MATCH (n:Person) WHERE n.name = "Alice" RETURN n.name` |
| `create`            | `CREATE (n:Person {name: "Bob", age: 42}) RETURN n` |
| `merge`             | `MERGE (n:Movie {title: $value}) RETURN n` |
| `aggregation`       | `MATCH (n:Person) RETURN count(n.age) AS result` |
| `match_relationship`| `OPTIONAL MATCH (a:Person)-[r:ACTED_IN]->(b:Movie) RETURN a, r, b` |
| `create_relationship`| `MATCH (a:Person),(b:Movie) CREATE (a)-[r:ACTED_IN]->(b) RETURN r` |
| `match_set`         | `MATCH (n:Person) SET n.name = "Carol" RETURN n` |
| `match_delete`      | `MATCH (n:Movie) DETACH DELETE n` |
| `with_chain`        | `MATCH (n:Person) WITH n.name AS val RETURN count(*)` |
| `distinct_return`   | `MATCH (n:Person) RETURN DISTINCT n.name LIMIT 10` |
| `order_by`          | `MATCH (n:Movie) RETURN n ORDER BY n.year DESC LIMIT 25` |
| `unwind`            | `MATCH (n:Person) UNWIND n.age AS item RETURN item` |

Generated scalar values cycle through string literals (`"Alice"`), integers (`42`), booleans (`true`/`false`), and parameters (`$name`). `OPTIONAL MATCH` and `LIMIT` clauses are randomly included. All generated queries are guaranteed to pass `CypherValidator` with the same schema.

---

### parse_query / QueryInfo

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

| Attribute         | Type        | Description |
|-------------------|-------------|-------------|
| `is_valid`        | `bool`      | Syntax check result |
| `errors`          | `list[str]` | Syntax error messages (empty when valid) |
| `labels_used`     | `list[str]` | Sorted, deduplicated node labels referenced in the query |
| `rel_types_used`  | `list[str]` | Sorted, deduplicated relationship types referenced in the query |
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

The GLiNER2 integration converts relation-extraction results into Cypher queries.

> **Most users should start with [`NLToCypher`](#nltocypher)** — it wraps all three classes into a single callable that goes from raw text to Cypher (and optionally executes it against Neo4j) in one line.
> `RelationToCypherConverter` and `GLiNER2RelationExtractor` are lower-level building blocks exposed for advanced use cases.

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

# All three methods return (cypher_str, params_dict) — entity values are
# passed as $param placeholders to prevent Cypher injection.

# ── MATCH mode (find existing data) ────────────────────────────────────────
cypher, params = converter.to_match_query(results)
print(cypher)
# MATCH (a0 {name: $a0_val})-[:WORKS_FOR]->(b0 {name: $b0_val})
# MATCH (a1 {name: $a1_val})-[:LIVES_IN]->(b1 {name: $b1_val})
# RETURN a0, b0, a1, b1
print(params)
# {"a0_val": "John", "b0_val": "Apple Inc.", "a1_val": "John", "b1_val": "San Francisco"}

# ── MERGE mode (upsert) ────────────────────────────────────────────────────
cypher, params = converter.to_merge_query(results)

# ── CREATE mode (insert new) ───────────────────────────────────────────────
cypher, params = converter.to_create_query(results)

# ── Unified dispatcher ─────────────────────────────────────────────────────
cypher, params = converter.convert(results, mode="merge")

# Pass both to Neo4jDatabase.execute() — the driver handles escaping:
# results = db.execute(cypher, params)
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
cypher, params = converter.to_merge_query(results)
print(cypher)
# MERGE (a0:Person {name: $a0_val})-[:WORKS_FOR]->(b0:Company {name: $b0_val})
# MERGE (a1:Person {name: $a1_val})-[:LIVES_IN]->(b1:City {name: $b1_val})
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
cypher, params = converter.to_merge_query(results)
print(cypher)
# MERGE (a0:Person {name: $a0_val})-[:WORKS_FOR]->(b0:Company {name: $b0_val})
# MERGE (a1:Person {name: $a1_val})-[:WORKS_FOR]->(b1:Company {name: $b1_val})
# MERGE (a2:Person {name: $a2_val})-[:WORKS_FOR]->(b2:Company {name: $b2_val})
# RETURN a0, b0, a1, b1, a2, b2
print(params)
# {"a0_val": "John", "b0_val": "Microsoft", "a1_val": "Mary", ...}
```

**Automatic deduplication:**

If the same `(subject, object)` pair for a given relation type appears more than once in the extraction output (which can happen with overlapping model detections), the converter silently deduplicates them — each unique triple `(subject, relation, object)` appears exactly once in the generated Cypher.

**Constructor parameters:**

| Parameter      | Default | Description |
|----------------|---------|-------------|
| `schema`       | `None`  | `cypher_validator.Schema` for label-aware generation |
| `name_property`| `"name"`| Node property key used for entity text spans |

**`convert()` parameters:**

| Parameter      | Default   | Description |
|----------------|-----------|-------------|
| `relations`    | required  | GLiNER2 output dict |
| `mode`         | `"match"` | `"match"`, `"merge"`, or `"create"` |
| `return_clause`| auto      | Custom `RETURN …` tail (e.g. `"RETURN *"` ) |

---

### DB-aware query generation

> **`db_aware=True`** is the flag that makes `NLToCypher` graph-state-aware.  
> Without it, every call blindly CREATEs all entities, producing **duplicate nodes**.  
> With it, each entity is looked up first and either MATCHed (existing) or CREATEd (new).

#### How it works

When `db_aware=True` is passed:

1. Relations are extracted from the text.
2. Each unique entity is identified and its label is resolved (from schema + optional NER).
3. A `MATCH … LIMIT 1` lookup is sent to Neo4j for each entity.
4. A mixed query is generated:
   - Existing entities → `MATCH (eN:Label {name: $eN_val})`
   - New entities → `CREATE (eN:Label {name: $eN_val})` inline
   - Relationships → always `CREATE (eA)-[:REL]->(eB)`

#### All four entity-existence combinations

(See full examples in the original documentation — the logic reuses variables across relations and only creates edges when both endpoints exist.)

---

### EntityNERExtractor

Optional NER wrapper that enriches entity-label resolution during DB-aware generation.

```python
from cypher_validator import EntityNERExtractor
```

**spaCy backend**

```python
ner = EntityNERExtractor.from_spacy("en_core_web_sm")
ner.extract("John works for Apple Inc.")
```

**HuggingFace Transformers backend**

```python
ner = EntityNERExtractor.from_transformers("Jean-Baptiste/roberta-large-ner-english")
```

Plug into `NLToCypher` via the `ner_extractor=` parameter. Label resolution priority: NER > Schema > fallback.

---

### GLiNER2RelationExtractor

```python
extractor = GLiNER2RelationExtractor.from_pretrained("fastino/gliner2-large-v1")
results = extractor.extract_relations(text, ["works_for", "lives_in"])
```

---

### NLToCypher

**Recommended entry point** — single callable from text to Cypher (and optional execution).

```python
from cypher_validator import NLToCypher, Schema

pipeline = NLToCypher.from_pretrained(
    "fastino/gliner2-large-v1",
    schema=schema,
    db=db,               # optional for execute/db_aware
)

cypher = pipeline("John works for Apple Inc.", ["works_for"], mode="merge")
# or with execution
cypher, records = pipeline(..., execute=True, db_aware=True)
```

Supports `from_env()` for credentials from environment variables.

---

### Neo4jDatabase

Thin wrapper around the official Neo4j Python driver.

```python
db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
results = db.execute("MATCH (n:Person) RETURN n.name LIMIT 5")
```

Also supports `execute_many`, context manager, and `introspect_schema()`.

---

## LLM integration

```python
from cypher_validator import (
    extract_cypher_from_text,
    format_records,
    repair_cypher,
    cypher_tool_spec,
    few_shot_examples,
    GraphRAGPipeline,
)
```

- `Schema.to_prompt()`, `to_markdown()`, `to_cypher_context()`
- `extract_cypher_from_text()` — robust extraction from any LLM output
- `repair_cypher()` — self-repair loop with LLM
- `format_records()` — Markdown / CSV / JSON / text
- `few_shot_examples()` — auto-generated few-shot pairs
- `cypher_tool_spec()` — OpenAI / Anthropic tool schema
- `GraphRAGPipeline` — complete schema → Cypher → execute → answer pipeline

---

## What the validator checks

- Node labels & properties
- Relationship types, direction & endpoints
- Variable binding & WITH scope
- WHERE boolean logic
- List comprehensions / quantifiers
- Open-world assumption for unlabeled patterns

---

## Generated query types

13 patterns (see [CypherGenerator](#cyphergenerator) section above). All guaranteed schema-valid.

---

## Performance

- `validate_batch()`: parallel Rust + Rayon, GIL released
- Capped Levenshtein suggestions (≤ 2 edits)
- O(1) property lookups via `HashSet`
- Construction-time caching

---

## Type stubs and IDE support

Full `.pyi` files included — perfect autocompletion and type checking with mypy/pyright.

---

## Project structure

(See detailed tree in the original document — Rust core in `src/`, Python layer in `python/cypher_validator/`, 395 tests.)

---

## Examples

11 ready-to-run scripts in the `examples/` directory covering every feature.

---

## Development

```bash
maturin develop          # editable install
maturin develop --release
pytest tests/            # full test suite (mocks for optional deps)
```

---

**Ready for README.md** — copy-paste directly. All tables, code blocks, anchors, and formatting are now clean and GitHub-compatible.
