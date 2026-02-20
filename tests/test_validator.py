"""Tests for schema-aware semantic validation."""
import pytest
from cypher_validator import Schema, CypherValidator


@pytest.fixture
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )


@pytest.fixture
def v(schema):
    return CypherValidator(schema)


# ===== Valid queries =====

def test_valid_match_known_label(v):
    r = v.validate("MATCH (p:Person) RETURN p")
    assert r.is_valid
    assert r.errors == []


def test_valid_known_relationship(v):
    r = v.validate("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m")
    assert r.is_valid


def test_valid_property_access(v):
    r = v.validate("MATCH (p:Person) RETURN p.name")
    assert r.is_valid


def test_valid_where_property(v):
    r = v.validate("MATCH (p:Person) WHERE p.age > 18 RETURN p")
    assert r.is_valid


def test_valid_return_multiple_props(v):
    r = v.validate("MATCH (m:Movie) RETURN m.title, m.year")
    assert r.is_valid


def test_valid_unlabeled_node(v):
    """Unlabeled nodes skip property checks (open-world)."""
    r = v.validate("MATCH (n) RETURN n.anything")
    assert r.is_valid


def test_valid_full_pattern(v):
    r = v.validate(
        "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) RETURN p.name, r, m.title"
    )
    assert r.is_valid


# ===== Invalid queries — semantic errors =====

def test_unknown_node_label(v):
    r = v.validate("MATCH (n:UnknownLabel) RETURN n")
    assert not r.is_valid
    assert any("UnknownLabel" in e for e in r.errors)


def test_unknown_relationship_type(v):
    r = v.validate("MATCH (a)-[:UNKNOWN_REL]->(b) RETURN a")
    assert not r.is_valid
    assert any("UNKNOWN_REL" in e for e in r.errors)


def test_unknown_property_on_node(v):
    r = v.validate("MATCH (p:Person) RETURN p.nonexistent")
    assert not r.is_valid
    assert any("nonexistent" in e for e in r.errors)


def test_unknown_property_in_where(v):
    r = v.validate("MATCH (p:Person) WHERE p.badprop = 'x' RETURN p")
    assert not r.is_valid


def test_multiple_errors_collected(v):
    r = v.validate("MATCH (n:BadLabel)-[:BadRel]->(m:AnotherBad) RETURN n")
    assert not r.is_valid
    assert len(r.errors) >= 2


def test_wrong_property_in_create(v):
    r = v.validate("CREATE (p:Person {badprop: 'x'})")
    assert not r.is_valid


def test_valid_property_in_create(v):
    r = v.validate("CREATE (p:Person {name: 'Alice', age: 30})")
    assert r.is_valid


def test_validation_result_attributes(v):
    r = v.validate("MATCH (p:Person) RETURN p")
    assert isinstance(r.is_valid, bool)
    assert isinstance(r.errors, list)


def test_syntax_error_gives_invalid(v):
    r = v.validate("THIS IS NOT CYPHER !!!!")
    assert not r.is_valid
    assert len(r.errors) > 0


# ===== v0.7.0: Gap 1 — Aggregate type checking =====

def test_sum_string_literal_is_error(v):
    r = v.validate("MATCH (p:Person) RETURN SUM('hello')")
    assert not r.is_valid
    assert any("SUM" in e for e in r.errors)


def test_avg_string_literal_is_error(v):
    r = v.validate("MATCH (p:Person) RETURN AVG('hello')")
    assert not r.is_valid
    assert any("AVG" in e for e in r.errors)


def test_sum_numeric_is_valid(v):
    r = v.validate("MATCH (p:Person) RETURN SUM(p.age)")
    assert r.is_valid


# ===== v0.7.0: Gap 2 — Cartesian product warning =====

def test_cartesian_product_warning(v):
    r = v.validate("MATCH (a:Person), (b:Movie) RETURN a, b")
    assert r.is_valid   # still valid, just warned
    assert any("Cartesian" in w for w in r.warnings)


def test_connected_match_no_cartesian_warning(v):
    r = v.validate("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m")
    assert r.is_valid
    assert not any("Cartesian" in w for w in r.warnings)


# ===== v0.7.0: Gap 3 — Duplicate RETURN aliases =====

def test_duplicate_return_alias(v):
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name AS x, m.title AS x"
    )
    assert not r.is_valid
    assert any("Duplicate" in e and "x" in e for e in r.errors)


def test_unique_return_aliases_valid(v):
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name AS name, m.title AS title"
    )
    assert r.is_valid


# ===== v0.7.0: Gap 4 — REDUCE() syntax =====

def test_reduce_valid(v):
    r = v.validate(
        "MATCH (p:Person) RETURN reduce(total = 0, x IN [1,2,3] | total + x)"
    )
    assert r.is_valid, r.errors


def test_reduce_case_insensitive(v):
    r = v.validate(
        "MATCH (p:Person) RETURN REDUCE(acc = 0, x IN [1,2,3] | acc + x)"
    )
    assert r.is_valid, r.errors


# ===== v0.7.0: Gap 5 — Complexity heuristics =====

def test_unlabeled_node_warning(v):
    r = v.validate("MATCH (n) RETURN n")
    assert r.is_valid
    assert any("unlabeled" in w.lower() for w in r.warnings)


def test_labeled_node_no_scan_warning(v):
    r = v.validate("MATCH (p:Person) RETURN p")
    assert r.is_valid
    assert not any("unlabeled" in w.lower() for w in r.warnings)


def test_unbounded_varlength_warning(v):
    r = v.validate("MATCH (p:Person)-[*]->(m:Movie) RETURN p, m")
    assert r.is_valid
    assert any("unbounded" in w.lower() for w in r.warnings)


def test_bounded_varlength_no_warning(v):
    r = v.validate("MATCH (p:Person)-[*1..3]->(m:Movie) RETURN p, m")
    assert r.is_valid
    assert not any("unbounded" in w.lower() for w in r.warnings)


def test_warnings_field_exists(v):
    r = v.validate("MATCH (p:Person) RETURN p")
    assert isinstance(r.warnings, list)


# ===== v0.8.0: shortestPath / allShortestPaths =====

def test_shortest_path_valid(v):
    r = v.validate(
        "MATCH (a:Person), (b:Movie), "
        "p = shortestPath((a)-[:ACTED_IN*]->(b)) RETURN p"
    )
    assert r.is_valid, r.errors


def test_all_shortest_paths_valid(v):
    r = v.validate(
        "MATCH (a:Person), (b:Movie), "
        "p = allShortestPaths((a)-[:ACTED_IN*]->(b)) RETURN p"
    )
    assert r.is_valid, r.errors


def test_shortest_path_bad_label_errors(v):
    r = v.validate(
        "MATCH p = shortestPath((a:Typo)-[:ACTED_IN*]->(b:Movie)) RETURN p"
    )
    assert not r.is_valid
    assert any("Typo" in e for e in r.errors)


# ===== v0.8.0: FOREACH =====

def test_foreach_set_valid(v):
    r = v.validate(
        "MATCH (p:Person) "
        "FOREACH (x IN [1, 2, 3] | SET p.age = x)"
    )
    assert r.is_valid, r.errors


def test_foreach_create_valid(v):
    r = v.validate(
        "MATCH (p:Person) "
        "FOREACH (name IN ['Alice', 'Bob'] | CREATE (q:Person {name: name}))"
    )
    assert r.is_valid, r.errors


def test_foreach_bad_label_errors(v):
    r = v.validate(
        "MATCH (p:Person) "
        "FOREACH (x IN [1] | CREATE (q:Typo))"
    )
    assert not r.is_valid
    assert any("Typo" in e for e in r.errors)


# ===== v0.8.0: aggregate context errors =====

def test_aggregate_in_where_is_error(v):
    r = v.validate("MATCH (p:Person) WHERE COUNT(p) > 5 RETURN p")
    assert not r.is_valid
    assert any("aggregate" in e.lower() for e in r.errors)


def test_aggregate_in_set_is_error(v):
    r = v.validate("MATCH (p:Person) SET p.age = SUM(p.age) RETURN p")
    assert not r.is_valid
    assert any("aggregate" in e.lower() for e in r.errors)


def test_count_star_in_where_is_error(v):
    r = v.validate("MATCH (p:Person) WHERE COUNT(*) > 0 RETURN p")
    assert not r.is_valid
    assert any("aggregate" in e.lower() for e in r.errors)


def test_aggregate_in_return_is_valid(v):
    r = v.validate("MATCH (p:Person) RETURN COUNT(p)")
    assert r.is_valid


def test_aggregate_in_with_is_valid(v):
    """COUNT() in a WITH projection is valid (aggregation context)."""
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "WITH p, COUNT(m) AS movie_count RETURN p, movie_count"
    )
    assert r.is_valid, r.errors


# ===== v0.9.0: WITH alias scope in WHERE / ORDER BY =====

def test_with_alias_visible_in_where(v):
    """WITH aliases must be usable in the WITH WHERE clause (projected scope)."""
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "WITH p, COUNT(m) AS movie_count WHERE movie_count > 2 RETURN p"
    )
    assert r.is_valid, r.errors


def test_with_alias_visible_in_order(v):
    """WITH aliases must be usable in ORDER BY after WITH (projected scope)."""
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "WITH p, COUNT(m) AS movie_count ORDER BY movie_count RETURN p, movie_count"
    )
    assert r.is_valid, r.errors


def test_with_old_var_not_visible_after_scope_reset(v):
    """Variables not projected through WITH must not be accessible afterwards."""
    r = v.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "WITH p RETURN m"  # m was not projected through WITH
    )
    assert not r.is_valid
    assert any("m" in e for e in r.errors)


# ===== v0.8.0: LIMIT / SKIP type checks =====

def test_limit_string_is_error(v):
    r = v.validate("MATCH (p:Person) RETURN p LIMIT 'five'")
    assert not r.is_valid
    assert any("LIMIT" in e for e in r.errors)


def test_skip_boolean_is_error(v):
    r = v.validate("MATCH (p:Person) RETURN p SKIP true")
    assert not r.is_valid
    assert any("SKIP" in e for e in r.errors)


def test_limit_integer_is_valid(v):
    r = v.validate("MATCH (p:Person) RETURN p LIMIT 10")
    assert r.is_valid


def test_limit_parameter_is_valid(v):
    r = v.validate("MATCH (p:Person) RETURN p LIMIT $limit")
    assert r.is_valid


# ===== v0.8.0: result.to_dict / to_json / validator.schema =====

def test_result_to_dict(v):
    r = v.validate("MATCH (p:Person) RETURN p")
    d = r.to_dict()
    assert isinstance(d, dict)
    assert "is_valid" in d
    assert "errors" in d
    assert "warnings" in d
    assert "syntax_errors" in d
    assert "semantic_errors" in d
    assert d["is_valid"] is True


def test_result_to_dict_invalid(v):
    r = v.validate("MATCH (p:Typo) RETURN p")
    d = r.to_dict()
    assert d["is_valid"] is False
    assert len(d["errors"]) > 0


def test_result_to_json(v):
    import json
    r = v.validate("MATCH (p:Person) RETURN p")
    j = r.to_json()
    assert isinstance(j, str)
    parsed = json.loads(j)
    assert parsed["is_valid"] is True
    assert "errors" in parsed
    assert "warnings" in parsed


def test_validator_schema_property(v, schema):
    s = v.schema
    assert s.node_labels() == schema.node_labels()
    assert s.rel_types() == schema.rel_types()


def test_validator_schema_to_cypher_context(v):
    ctx = v.schema.to_cypher_context()
    assert "Person" in ctx
    assert "Movie" in ctx
    assert "ACTED_IN" in ctx


def test_llm_retry_loop_pattern(v):
    """Demonstrates the full LLM retry loop pattern."""
    import json
    result = v.validate("MATCH (p:Persn) RETURN p")
    assert not result.is_valid
    # Build retry payload
    payload = {
        "errors": result.to_dict()["errors"],
        "schema": v.schema.to_cypher_context(),
    }
    assert "Persn" in json.dumps(payload)
    assert "Person" in payload["schema"]
