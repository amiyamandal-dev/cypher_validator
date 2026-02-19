"""Roundtrip tests: generate a query then validate it."""
import pytest
from cypher_validator import Schema, CypherValidator, CypherGenerator


@pytest.fixture
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )


@pytest.fixture
def validator(schema):
    return CypherValidator(schema)


@pytest.fixture
def generator(schema):
    return CypherGenerator(schema, seed=99)


def test_match_return_roundtrip(validator, generator):
    q = generator.generate("match_return")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_match_where_return_roundtrip(validator, generator):
    q = generator.generate("match_where_return")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_create_roundtrip(validator, generator):
    q = generator.generate("create")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_merge_roundtrip(validator, generator):
    q = generator.generate("merge")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_aggregation_roundtrip(validator, generator):
    q = generator.generate("aggregation")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_match_relationship_roundtrip(validator, generator):
    q = generator.generate("match_relationship")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_create_relationship_roundtrip(validator, generator):
    q = generator.generate("create_relationship")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_match_set_roundtrip(validator, generator):
    q = generator.generate("match_set")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_match_delete_roundtrip(validator, generator):
    q = generator.generate("match_delete")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_with_chain_roundtrip(validator, generator):
    q = generator.generate("with_chain")
    r = validator.validate(q)
    assert r.is_valid, f"Query failed validation: {q!r}\nErrors: {r.errors}"


def test_all_types_roundtrip(validator, schema):
    """All supported types should generate valid queries (over many seeds)."""
    for seed in range(10):
        g = CypherGenerator(schema, seed=seed)
        for t in CypherGenerator.supported_types():
            q = g.generate(t)
            r = validator.validate(q)
            assert r.is_valid, (
                f"seed={seed}, type={t!r}: {q!r}\nErrors: {r.errors}"
            )
