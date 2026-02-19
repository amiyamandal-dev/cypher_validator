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
