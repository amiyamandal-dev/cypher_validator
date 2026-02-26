"""Tests for Schema.from_neo4j() convenience wrapper."""
from unittest.mock import MagicMock, patch

import pytest

from cypher_validator import Schema


@pytest.fixture
def mock_schema():
    """Return a real Schema for mock introspection to return."""
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])},
    )


@patch("cypher_validator.Neo4jDatabase")
def test_from_neo4j_returns_schema(MockDB, mock_schema):
    """Schema.from_neo4j() should return the schema from introspect_schema()."""
    instance = MockDB.return_value
    instance.introspect_schema.return_value = mock_schema

    result = Schema.from_neo4j("bolt://localhost:7687", "neo4j", "password")

    assert isinstance(result, Schema)
    assert "Person" in result.node_labels()
    assert "Movie" in result.node_labels()
    assert "ACTED_IN" in result.rel_types()


@patch("cypher_validator.Neo4jDatabase")
def test_from_neo4j_passes_arguments(MockDB, mock_schema):
    """Arguments should be forwarded to Neo4jDatabase and introspect_schema."""
    instance = MockDB.return_value
    instance.introspect_schema.return_value = mock_schema

    Schema.from_neo4j(
        "neo4j://example.com:7687", "admin", "secret",
        database="mydb", sample_limit=500,
    )

    MockDB.assert_called_once_with(
        "neo4j://example.com:7687", "admin", "secret", database="mydb",
    )
    instance.introspect_schema.assert_called_once_with(sample_limit=500)


@patch("cypher_validator.Neo4jDatabase")
def test_from_neo4j_closes_connection(MockDB, mock_schema):
    """Connection must be closed even on success."""
    instance = MockDB.return_value
    instance.introspect_schema.return_value = mock_schema

    Schema.from_neo4j("bolt://localhost:7687", "neo4j", "password")

    instance.close.assert_called_once()


@patch("cypher_validator.Neo4jDatabase")
def test_from_neo4j_closes_on_error(MockDB):
    """Connection must be closed even when introspect_schema raises."""
    instance = MockDB.return_value
    instance.introspect_schema.side_effect = RuntimeError("introspection failed")

    with pytest.raises(RuntimeError, match="introspection failed"):
        Schema.from_neo4j("bolt://localhost:7687", "neo4j", "password")

    instance.close.assert_called_once()


@patch("cypher_validator.Neo4jDatabase")
def test_from_neo4j_default_arguments(MockDB, mock_schema):
    """Default database='neo4j' and sample_limit=1000."""
    instance = MockDB.return_value
    instance.introspect_schema.return_value = mock_schema

    Schema.from_neo4j("bolt://localhost:7687", "neo4j", "password")

    MockDB.assert_called_once_with(
        "bolt://localhost:7687", "neo4j", "password", database="neo4j",
    )
    instance.introspect_schema.assert_called_once_with(sample_limit=1000)
