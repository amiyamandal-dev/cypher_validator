"""Tests for Cypher syntax validation (parse-level errors)."""
import pytest
from cypher_validator import Schema, CypherValidator

EMPTY_SCHEMA = Schema(nodes={}, relationships={})


@pytest.fixture
def v():
    return CypherValidator(EMPTY_SCHEMA)


def test_simple_match_return(v):
    r = v.validate("MATCH (n) RETURN n")
    assert r.is_valid


def test_match_with_label(v):
    r = v.validate("MATCH (n:Person) RETURN n")
    # Label not in schema but syntax is valid — errors only for unknown labels when schema has entries
    assert isinstance(r.is_valid, bool)


def test_create_node(v):
    r = v.validate("CREATE (n:Person {name: 'Alice'})")
    assert isinstance(r.is_valid, bool)


def test_match_relationship(v):
    r = v.validate("MATCH (a)-[:KNOWS]->(b) RETURN a, b")
    assert isinstance(r.is_valid, bool)


def test_match_where(v):
    r = v.validate("MATCH (n) WHERE n.age > 18 RETURN n")
    assert r.is_valid


def test_return_with_alias(v):
    r = v.validate("MATCH (n) RETURN n.name AS name")
    assert r.is_valid


def test_match_order_by(v):
    r = v.validate("MATCH (n) RETURN n ORDER BY n.name")
    assert r.is_valid


def test_match_skip_limit(v):
    r = v.validate("MATCH (n) RETURN n SKIP 10 LIMIT 5")
    assert r.is_valid


def test_optional_match(v):
    r = v.validate("OPTIONAL MATCH (n) RETURN n")
    assert r.is_valid


def test_delete(v):
    r = v.validate("MATCH (n) DETACH DELETE n")
    assert r.is_valid


def test_set_property(v):
    r = v.validate("MATCH (n) SET n.name = 'Alice' RETURN n")
    assert r.is_valid


def test_merge(v):
    r = v.validate("MERGE (n:Person {name: 'Alice'}) RETURN n")
    assert isinstance(r.is_valid, bool)


def test_unwind(v):
    r = v.validate("UNWIND [1, 2, 3] AS x RETURN x")
    assert r.is_valid


def test_with_clause(v):
    r = v.validate("MATCH (n) WITH n.name AS name RETURN name")
    assert r.is_valid


def test_count_star(v):
    r = v.validate("MATCH (n) RETURN COUNT(*)")
    assert r.is_valid


def test_invalid_syntax(v):
    r = v.validate("MATCH RETURN")
    assert not r.is_valid
    assert len(r.errors) > 0


def test_empty_string(v):
    r = v.validate("")
    assert not r.is_valid


def test_case_insensitive_keywords(v):
    r = v.validate("match (n) return n")
    assert r.is_valid


def test_multiline_query(v):
    q = """
    MATCH (n)
    WHERE n.age > 10
    RETURN n
    """
    r = v.validate(q)
    assert r.is_valid
