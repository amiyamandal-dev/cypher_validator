"""
Tests for Task #2 features:
  - AND / OR / XOR / NOT in WHERE clauses (grammar bug fix)
  - Schema.from_dict() classmethod
  - CypherValidator.validate_batch()
  - QueryInfo.properties_used
"""
import pytest
from cypher_validator import (
    Schema, CypherValidator, parse_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])},
    )


@pytest.fixture
def v(schema):
    return CypherValidator(schema)


# ===========================================================================
# 1. AND / OR / XOR / NOT in WHERE clauses
# ===========================================================================

class TestBooleanOperatorsInWhere:
    """Regression tests for the Pest keyword-rule whitespace bug."""

    def test_and_in_where(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" AND p.age > 18 RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_or_in_where(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" OR p.name = "Bob" RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_xor_in_where(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" XOR p.age > 30 RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_not_in_where(self):
        info = parse_query(
            'MATCH (p:Person) WHERE NOT p.name = "Alice" RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_chained_and_conditions(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" AND p.age > 18 AND p.age < 60 RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_mixed_and_or(self):
        info = parse_query(
            'MATCH (p:Person) WHERE (p.name = "Alice" AND p.age > 18) OR p.name = "Bob" RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_not_and_combined(self):
        info = parse_query(
            'MATCH (p:Person) WHERE NOT p.name = "Alice" AND p.age > 18 RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_and_with_validator(self, v):
        """AND in WHERE should pass semantic validation too."""
        r = v.validate(
            'MATCH (p:Person)-[:ACTED_IN]->(m:Movie) '
            'WHERE p.name = "Alice" AND m.year > 2000 RETURN p, m'
        )
        assert r.is_valid, r.errors

    def test_or_with_validator(self, v):
        r = v.validate(
            'MATCH (p:Person) WHERE p.name = "Alice" OR p.age > 30 RETURN p'
        )
        assert r.is_valid, r.errors

    def test_where_with_function_and_and(self):
        info = parse_query(
            'MATCH (p:Person) WHERE toLower(p.name) = "alice" AND p.age > 0 RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"

    def test_complex_nested_boolean(self):
        info = parse_query(
            'MATCH (p:Person) '
            'WHERE (p.age > 18 AND p.age < 60) OR (NOT p.name = "bot" AND p.active = true) '
            'RETURN p'
        )
        assert info.is_valid, f"Expected valid but got errors: {info.errors}"


# ===========================================================================
# 2. Schema.from_dict()
# ===========================================================================

class TestSchemaFromDict:
    def test_round_trip_through_dict(self, schema):
        d = schema.to_dict()
        restored = Schema.from_dict(d)
        assert set(restored.node_labels()) == set(schema.node_labels())
        assert set(restored.rel_types()) == set(schema.rel_types())

    def test_node_properties_preserved(self, schema):
        restored = Schema.from_dict(schema.to_dict())
        assert set(restored.node_properties("Person")) == {"name", "age"}
        assert set(restored.node_properties("Movie")) == {"title", "year"}

    def test_rel_properties_preserved(self, schema):
        restored = Schema.from_dict(schema.to_dict())
        assert set(restored.rel_properties("ACTED_IN")) == {"role"}

    def test_rel_endpoints_preserved(self, schema):
        restored = Schema.from_dict(schema.to_dict())
        assert restored.rel_endpoints("ACTED_IN") == ("Person", "Movie")

    def test_from_plain_dict(self):
        d = {
            "nodes": {"City": ["name"], "Country": ["name", "iso"]},
            "relationships": {"LOCATED_IN": ("City", "Country", [])},
        }
        s = Schema.from_dict(d)
        assert s.has_node_label("City")
        assert s.has_node_label("Country")
        assert s.has_rel_type("LOCATED_IN")

    def test_missing_nodes_key_raises(self):
        with pytest.raises(KeyError):
            Schema.from_dict({"relationships": {}})

    def test_missing_relationships_key_raises(self):
        with pytest.raises(KeyError):
            Schema.from_dict({"nodes": {}})

    def test_wrong_type_for_nodes_raises(self):
        with pytest.raises(TypeError):
            Schema.from_dict({"nodes": [1, 2, 3], "relationships": {}})

    def test_restored_schema_validates_correctly(self, schema, v):
        restored = Schema.from_dict(schema.to_dict())
        v2 = CypherValidator(restored)
        r = v2.validate("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m")
        assert r.is_valid, r.errors

    def test_restored_schema_rejects_bad_label(self, schema):
        restored = Schema.from_dict(schema.to_dict())
        v2 = CypherValidator(restored)
        r = v2.validate("MATCH (x:Unknown) RETURN x")
        assert not r.is_valid


# ===========================================================================
# 3. validate_batch()
# ===========================================================================

class TestValidateBatch:
    def test_empty_list(self, v):
        results = v.validate_batch([])
        assert results == []

    def test_single_valid_query(self, v):
        results = v.validate_batch(["MATCH (p:Person) RETURN p"])
        assert len(results) == 1
        assert results[0].is_valid

    def test_single_invalid_query(self, v):
        results = v.validate_batch(["MATCH (x:BadLabel) RETURN x"])
        assert len(results) == 1
        assert not results[0].is_valid

    def test_mixed_valid_and_invalid(self, v):
        queries = [
            "MATCH (p:Person) RETURN p",        # valid
            "MATCH (x:BadLabel) RETURN x",       # invalid: bad label
            "MATCH (m:Movie) RETURN m",           # valid
        ]
        results = v.validate_batch(queries)
        assert len(results) == 3
        assert results[0].is_valid
        assert not results[1].is_valid
        assert results[2].is_valid

    def test_syntax_error_in_batch(self, v):
        results = v.validate_batch(["NOT VALID CYPHER !!!"])
        assert len(results) == 1
        assert not results[0].is_valid
        assert len(results[0].syntax_errors) > 0

    def test_all_valid(self, v):
        queries = [
            "MATCH (p:Person) RETURN p",
            "MATCH (m:Movie) RETURN m",
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m",
        ]
        results = v.validate_batch(queries)
        assert all(r.is_valid for r in results)

    def test_results_have_correct_errors(self, v):
        results = v.validate_batch(["MATCH (x:Ghost) RETURN x"])
        assert not results[0].is_valid
        assert any("Ghost" in e for e in results[0].errors)

    def test_order_preserved(self, v):
        """Result order must match query order."""
        queries = [
            "MATCH (x:BadA) RETURN x",
            "MATCH (p:Person) RETURN p",
            "MATCH (x:BadB) RETURN x",
        ]
        results = v.validate_batch(queries)
        assert not results[0].is_valid
        assert results[1].is_valid
        assert not results[2].is_valid
        assert any("BadA" in e for e in results[0].errors)
        assert any("BadB" in e for e in results[2].errors)

    def test_result_types_match_validate(self, v):
        """validate_batch results should be equivalent to repeated validate() calls."""
        queries = [
            "MATCH (p:Person) RETURN p",
            "MATCH (x:NoSuchLabel) RETURN x",
        ]
        batch = v.validate_batch(queries)
        singles = [v.validate(q) for q in queries]
        for b, s in zip(batch, singles):
            assert b.is_valid == s.is_valid
            assert b.errors == s.errors
            assert b.syntax_errors == s.syntax_errors
            assert b.semantic_errors == s.semantic_errors


# ===========================================================================
# 4. QueryInfo.properties_used
# ===========================================================================

class TestPropertiesUsed:
    def test_simple_property_in_return(self):
        info = parse_query("MATCH (p:Person) RETURN p.name")
        assert "name" in info.properties_used

    def test_multiple_properties(self):
        info = parse_query("MATCH (p:Person) RETURN p.name, p.age")
        assert "name" in info.properties_used
        assert "age" in info.properties_used

    def test_property_in_where(self):
        info = parse_query('MATCH (p:Person) WHERE p.name = "Alice" RETURN p')
        assert "name" in info.properties_used

    def test_properties_from_both_match_and_where(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.age > 18 RETURN p.name'
        )
        assert "age" in info.properties_used
        assert "name" in info.properties_used

    def test_inline_map_property(self):
        info = parse_query('MATCH (p:Person {name: "Alice"}) RETURN p')
        assert "name" in info.properties_used

    def test_properties_deduplicated(self):
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" RETURN p.name'
        )
        # name appears twice but should not be duplicated
        assert info.properties_used.count("name") == 1

    def test_no_properties_wildcard_return(self):
        info = parse_query("MATCH (p:Person) RETURN *")
        assert info.properties_used == []

    def test_no_properties_variable_only_return(self):
        info = parse_query("MATCH (p:Person) RETURN p")
        assert info.properties_used == []

    def test_properties_sorted(self):
        info = parse_query("MATCH (p:Person) RETURN p.title, p.age, p.name")
        assert info.properties_used == sorted(info.properties_used)

    def test_property_in_function_arg(self):
        info = parse_query("MATCH (p:Person) RETURN toLower(p.name)")
        assert "name" in info.properties_used

    def test_relationship_property(self):
        info = parse_query(
            "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) RETURN r.role"
        )
        assert "role" in info.properties_used

    def test_empty_on_parse_error(self):
        info = parse_query("NOT VALID !!!")
        assert not info.is_valid
        assert info.properties_used == []

    def test_and_where_properties(self):
        """Specifically tests property extraction with AND (the grammar bug fix)."""
        info = parse_query(
            'MATCH (p:Person) WHERE p.name = "Alice" AND p.age > 18 RETURN p'
        )
        assert info.is_valid, f"Parse failed: {info.errors}"
        assert "name" in info.properties_used
        assert "age" in info.properties_used
