# Examples

Self-contained scripts demonstrating every major feature of `cypher_validator`.
Each file runs independently and can be used as a starting point for your own code.

## Prerequisites

```bash
# Core (always required — built by maturin)
maturin develop

# Optional extras
pip install "cypher_validator[neo4j]"           # for examples 05, 07, 08, 10
pip install "cypher_validator[ner-transformers]" # for examples 06, 07
pip install "cypher_validator[ner-spacy]"        # for example 06 (Python < 3.14 only)
python -m spacy download en_core_web_sm          # if using spaCy
```

## Examples

| File | Topic | Requires |
|------|-------|----------|
| [`01_basic_validation.py`](01_basic_validation.py) | Schema definition, single/batch validation, "did you mean?" | core |
| [`02_schema_serialization.py`](02_schema_serialization.py) | `to_dict`, `from_dict`, `to_json`, `from_json`, `merge`, prompt helpers | core |
| [`03_cypher_generator.py`](03_cypher_generator.py) | All 13 `CypherGenerator` query types, batch generation | core |
| [`04_nl_to_cypher_basic.py`](04_nl_to_cypher_basic.py) | `NLToCypher` — CREATE / MERGE / MATCH modes, mocked extractor | core |
| [`05_db_aware_all_cases.py`](05_db_aware_all_cases.py) | `db_aware=True` — all 4 entity-existence combinations + multi-relation | neo4j |
| [`06_ner_extractors.py`](06_ner_extractors.py) | `EntityNERExtractor` — spaCy and HuggingFace backends, custom `label_map` | ner |
| [`07_db_aware_with_ner.py`](07_db_aware_with_ner.py) | `db_aware=True` + `EntityNERExtractor` against a live Neo4j DB | neo4j + ner |
| [`08_neo4j_database.py`](08_neo4j_database.py) | `Neo4jDatabase` — execute, execute_many, execute_and_format, introspect_schema | neo4j |
| [`09_llm_utils.py`](09_llm_utils.py) | `extract_cypher_from_text`, `format_records`, `repair_cypher`, `cypher_tool_spec`, `few_shot_examples` | core |
| [`10_graph_rag_pipeline.py`](10_graph_rag_pipeline.py) | `GraphRAGPipeline` — full RAG loop with mock LLM | neo4j |
| [`11_biobert_ner.py`](11_biobert_ner.py) | BioBERT NER — raw output explanation, `label_map` adapter, fine-tuned biomedical NER, db_aware biomedical pipeline | neo4j + ner |

## Quick run (all core examples, no extras needed)

```bash
for f in 01 02 03 04 09; do
    echo "=== examples/${f}_*.py ===" && python examples/0${f}_*.py
done
```

## DB-aware mode summary

`db_aware=True` prevents duplicate nodes by checking DB existence before generating:

| John in DB | Apple in DB | Generated Cypher |
|:-----------:|:-----------:|-----------------|
| ✗ | ✗ | `CREATE (e0:Person {…})-[:WORKS_FOR]->(e1:Company {…})` |
| ✓ | ✗ | `MATCH (e0:Person {…})` → `CREATE (e0)-[:WORKS_FOR]->(e1:Company {…})` |
| ✗ | ✓ | `MATCH (e1:Company {…})` → `CREATE (e0:Person {…})-[:WORKS_FOR]->(e1)` |
| ✓ | ✓ | `MATCH (e0:Person {…})` → `MATCH (e1:Company {…})` → `CREATE (e0)-[:WORKS_FOR]->(e1)` |
