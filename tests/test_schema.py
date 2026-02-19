"""Tests for Schema construction and inspection."""
import pytest
from cypher_validator import Schema


def test_empty_schema():
    s = Schema(nodes={}, relationships={})
    assert s.node_labels() == []
    assert s.rel_types() == []


def test_node_labels():
    s = Schema(nodes={"Person": ["name", "age"], "Movie": ["title"]}, relationships={})
    labels = set(s.node_labels())
    assert labels == {"Person", "Movie"}


def test_rel_types():
    s = Schema(
        nodes={"Person": ["name"], "Movie": ["title"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )
    assert s.rel_types() == ["ACTED_IN"]


def test_has_node_label_true():
    s = Schema(nodes={"Person": ["name"]}, relationships={})
    assert s.has_node_label("Person")


def test_has_node_label_false():
    s = Schema(nodes={"Person": ["name"]}, relationships={})
    assert not s.has_node_label("Movie")


def test_has_rel_type_true():
    s = Schema(
        nodes={"Person": ["name"], "Movie": ["title"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )
    assert s.has_rel_type("ACTED_IN")


def test_has_rel_type_false():
    s = Schema(nodes={}, relationships={})
    assert not s.has_rel_type("ACTED_IN")


def test_schema_repr():
    s = Schema(nodes={"Person": ["name"]}, relationships={})
    r = repr(s)
    assert "Schema" in r


def test_invalid_node_props_type():
    with pytest.raises(TypeError):
        Schema(nodes={"Person": "not_a_list"}, relationships={})


def test_invalid_rel_tuple_wrong_length():
    with pytest.raises((TypeError, ValueError)):
        Schema(
            nodes={},
            relationships={"R": ("A", "B")}  # missing props list
        )


def test_multiple_node_labels():
    nodes = {f"Label{i}": [f"prop{i}"] for i in range(10)}
    s = Schema(nodes=nodes, relationships={})
    assert len(s.node_labels()) == 10


def test_rel_with_no_properties():
    s = Schema(
        nodes={"A": [], "B": []},
        relationships={"REL": ("A", "B", [])}
    )
    assert s.has_rel_type("REL")
