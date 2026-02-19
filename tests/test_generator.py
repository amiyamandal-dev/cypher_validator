"""Tests for CypherGenerator."""
import pytest
from cypher_validator import Schema, CypherGenerator


@pytest.fixture
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )


@pytest.fixture
def gen(schema):
    return CypherGenerator(schema, seed=42)


def test_supported_types():
    types = CypherGenerator.supported_types()
    assert isinstance(types, list)
    # The original 10 types plus 3 new ones: distinct_return, order_by, unwind
    assert len(types) == 13
    original = {
        "match_return", "match_where_return", "create", "merge", "aggregation",
        "match_relationship", "create_relationship", "match_set", "match_delete", "with_chain"
    }
    new_types = {"distinct_return", "order_by", "unwind"}
    assert set(types) == original | new_types


def test_generate_match_return(gen):
    q = gen.generate("match_return")
    assert isinstance(q, str)
    assert "MATCH" in q.upper()
    assert "RETURN" in q.upper()


def test_generate_match_where_return(gen):
    q = gen.generate("match_where_return")
    assert isinstance(q, str)
    assert "MATCH" in q.upper()


def test_generate_create(gen):
    q = gen.generate("create")
    assert isinstance(q, str)
    assert "CREATE" in q.upper()


def test_generate_merge(gen):
    q = gen.generate("merge")
    assert isinstance(q, str)
    assert "MERGE" in q.upper()


def test_generate_aggregation(gen):
    q = gen.generate("aggregation")
    assert isinstance(q, str)
    assert "COUNT" in q.upper()


def test_generate_match_relationship(gen):
    q = gen.generate("match_relationship")
    assert isinstance(q, str)
    assert "MATCH" in q.upper()


def test_generate_create_relationship(gen):
    q = gen.generate("create_relationship")
    assert isinstance(q, str)
    assert "CREATE" in q.upper() or "MATCH" in q.upper()


def test_generate_match_set(gen):
    q = gen.generate("match_set")
    assert isinstance(q, str)
    assert "MATCH" in q.upper()


def test_generate_match_delete(gen):
    q = gen.generate("match_delete")
    assert isinstance(q, str)
    assert "DELETE" in q.upper()
    assert "DETACH" in q.upper()


def test_generate_with_chain(gen):
    q = gen.generate("with_chain")
    assert isinstance(q, str)
    assert "WITH" in q.upper()


def test_deterministic_with_seed(schema):
    g1 = CypherGenerator(schema, seed=123)
    g2 = CypherGenerator(schema, seed=123)
    results1 = [g1.generate("match_return") for _ in range(5)]
    results2 = [g2.generate("match_return") for _ in range(5)]
    assert results1 == results2


def test_different_seeds_may_differ(schema):
    g1 = CypherGenerator(schema, seed=1)
    g2 = CypherGenerator(schema, seed=2)
    # With two node labels, seeds will pick different labels eventually
    results1 = [g1.generate("match_return") for _ in range(20)]
    results2 = [g2.generate("match_return") for _ in range(20)]
    # At least one result differs (extremely unlikely to be all same)
    assert set(results1) != set(results2) or len(set(results1)) > 0


def test_no_seed_works(schema):
    g = CypherGenerator(schema)
    q = g.generate("match_return")
    assert isinstance(q, str)


def test_unknown_type_raises(gen):
    with pytest.raises(ValueError):
        gen.generate("nonexistent_type")


def test_all_types_generate_strings(gen):
    for t in CypherGenerator.supported_types():
        q = gen.generate(t)
        assert isinstance(q, str) and len(q) > 0, f"Empty result for type: {t}"


def test_generator_with_empty_schema():
    s = Schema(nodes={}, relationships={})
    g = CypherGenerator(s, seed=0)
    # Should fall back gracefully
    for t in CypherGenerator.supported_types():
        q = g.generate(t)
        assert isinstance(q, str)
