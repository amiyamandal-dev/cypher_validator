"""
Tests for all newly implemented features:
  - Relationship direction validation
  - Relationship endpoint label validation
  - Unbound variable detection
  - WITH scope reset
  - Error categorization (syntax_errors / semantic_errors)
  - ValidationResult __bool__ / __len__
  - Schema inspection API (node_properties, rel_properties, rel_endpoints, to_dict)
  - parse_query() / QueryInfo
  - Generator improvements (mixed value types, OPTIONAL MATCH, LIMIT)
"""
import pytest
from cypher_validator import (
    Schema, CypherValidator, CypherGenerator,
    QueryInfo, parse_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
        relationships={"ACTED_IN": ("Person", "Movie", ["role"])}
    )


@pytest.fixture
def v(schema):
    return CypherValidator(schema)


# ===========================================================================
# 1. Relationship direction validation
# ===========================================================================

class TestDirectionValidation:
    def test_correct_outgoing_direction(self, v):
        r = v.validate("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m")
        assert r.is_valid, r.errors

    def test_wrong_outgoing_direction(self, v):
        """Direction reversed: Movie → Person should fail."""
        r = v.validate("MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN m, p")
        assert not r.is_valid
        # Should report endpoint mismatch, not just unknown rel
        assert any("ACTED_IN" in e for e in r.errors)

    def test_correct_incoming_direction(self, v):
        r = v.validate("MATCH (m:Movie)<-[:ACTED_IN]-(p:Person) RETURN m, p")
        assert r.is_valid, r.errors

    def test_wrong_incoming_direction(self, v):
        """Incoming arrow reversal: Person should be source, not target."""
        r = v.validate("MATCH (p:Person)<-[:ACTED_IN]-(m:Movie) RETURN p, m")
        assert not r.is_valid
        assert any("ACTED_IN" in e for e in r.errors)

    def test_undirected_skips_check(self, v):
        """Undirected pattern should not error on endpoint order."""
        r = v.validate("MATCH (m:Movie)-[:ACTED_IN]-(p:Person) RETURN m, p")
        assert r.is_valid, r.errors

    def test_unlabeled_endpoints_skip_check(self, v):
        """If endpoints have no labels, direction check is skipped."""
        r = v.validate("MATCH ()-[:ACTED_IN]->() RETURN 1")
        assert r.is_valid, r.errors

    def test_unknown_rel_skips_endpoint_check(self, v):
        """Unknown rel type → error, but no duplicate endpoint errors."""
        r = v.validate("MATCH (m:Movie)-[:UNKNOWN]->(p:Person) RETURN 1")
        assert not r.is_valid
        assert any("UNKNOWN" in e for e in r.errors)


# ===========================================================================
# 2. Relationship endpoint label validation
# ===========================================================================

class TestEndpointValidation:
    def test_correct_endpoints(self, v):
        r = v.validate("MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) RETURN r")
        assert r.is_valid, r.errors

    def test_wrong_source_label(self, v):
        """Source is Movie, but schema says source should be Person."""
        r = v.validate("MATCH (m:Movie)-[:ACTED_IN]->(n:Movie) RETURN m")
        assert not r.is_valid
        assert any("source" in e.lower() for e in r.errors)

    def test_wrong_target_label(self, v):
        """Target is Person, but schema says target should be Movie."""
        r = v.validate("MATCH (p:Person)-[:ACTED_IN]->(q:Person) RETURN p")
        assert not r.is_valid
        assert any("target" in e.lower() for e in r.errors)

    def test_both_endpoints_wrong(self, v):
        r = v.validate("MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN m")
        assert not r.is_valid
        errors = r.errors
        # Both source and target errors
        assert len(errors) >= 2


# ===========================================================================
# 3. Unbound variable detection
# ===========================================================================

class TestUnboundVariables:
    def test_bound_variable_ok(self, v):
        r = v.validate("MATCH (n:Person) RETURN n")
        assert r.is_valid, r.errors

    def test_unbound_in_return(self, v):
        r = v.validate("MATCH (n:Person) RETURN m")
        assert not r.is_valid
        assert any("'m'" in e or "m" in e for e in r.errors)

    def test_unbound_in_where(self, v):
        r = v.validate("MATCH (n:Person) WHERE x.age > 10 RETURN n")
        assert not r.is_valid
        assert any("'x'" in e or "x" in e for e in r.errors)

    def test_unwind_binds_variable(self, v):
        r = v.validate("UNWIND [1, 2, 3] AS x RETURN x")
        assert r.is_valid, r.errors

    def test_list_comprehension_local_var(self, schema):
        v = CypherValidator(Schema(nodes={}, relationships={}))
        r = v.validate("MATCH (n) RETURN [x IN n.tags | x]")
        assert r.is_valid, r.errors

    def test_with_alias_binds_variable(self, v):
        r = v.validate("MATCH (n:Person) WITH n.name AS nm RETURN nm")
        assert r.is_valid, r.errors

    def test_unbound_after_with_scope_reset(self, v):
        """Variable 'n' is not projected through WITH, so it's unbound after."""
        r = v.validate("MATCH (n:Person) WITH n.name AS nm RETURN n")
        assert not r.is_valid
        assert any("'n'" in e or "n" in e for e in r.errors)


# ===========================================================================
# 4. WITH scope reset
# ===========================================================================

class TestWithScopeReset:
    def test_projected_var_accessible(self, v):
        r = v.validate("MATCH (n:Person) WITH n AS p RETURN p")
        assert r.is_valid, r.errors

    def test_non_projected_var_inaccessible(self, v):
        r = v.validate("MATCH (n:Person)-[r:ACTED_IN]->(m:Movie) WITH n RETURN m")
        assert not r.is_valid
        assert any("'m'" in e or "m" in e for e in r.errors)

    def test_labels_carried_through_with(self, v):
        """Variable projected via WITH carries its labels for property validation."""
        r = v.validate("MATCH (n:Person) WITH n RETURN n.name")
        assert r.is_valid, r.errors

    def test_alias_carries_no_labels(self, v):
        """Projected alias from expression (not bare variable) has no label info."""
        r = v.validate("MATCH (n:Person) WITH n.name AS nm RETURN nm")
        assert r.is_valid, r.errors

    def test_multi_with_chain(self, v):
        r = v.validate(
            "MATCH (n:Person) WITH n "
            "MATCH (n)-[:ACTED_IN]->(m:Movie) WITH m "
            "RETURN m.title"
        )
        assert r.is_valid, r.errors


# ===========================================================================
# 5. Error categorization
# ===========================================================================

class TestErrorCategorization:
    def test_syntax_error_goes_to_syntax_errors(self, v):
        r = v.validate("NOT VALID CYPHER !!!")
        assert not r.is_valid
        assert len(r.syntax_errors) > 0
        assert len(r.semantic_errors) == 0

    def test_semantic_error_goes_to_semantic_errors(self, v):
        r = v.validate("MATCH (n:UnknownLabel) RETURN n")
        assert not r.is_valid
        assert len(r.semantic_errors) > 0
        assert len(r.syntax_errors) == 0

    def test_valid_query_has_no_errors(self, v):
        r = v.validate("MATCH (p:Person) RETURN p")
        assert r.is_valid
        assert r.syntax_errors == []
        assert r.semantic_errors == []
        assert r.errors == []

    def test_combined_errors_list(self, v):
        """r.errors combines both categories."""
        r = v.validate("MATCH (n:BadLabel) RETURN n")
        assert r.errors == r.semantic_errors


# ===========================================================================
# 6. ValidationResult __bool__ / __len__
# ===========================================================================

class TestValidationResultProtocol:
    def test_bool_true_when_valid(self, v):
        r = v.validate("MATCH (n:Person) RETURN n")
        assert bool(r) is True

    def test_bool_false_when_invalid(self, v):
        r = v.validate("MATCH (n:BadLabel) RETURN n")
        assert bool(r) is False

    def test_len_zero_when_valid(self, v):
        r = v.validate("MATCH (n:Person) RETURN n")
        assert len(r) == 0

    def test_len_nonzero_when_invalid(self, v):
        r = v.validate("MATCH (n:BadLabel) RETURN n")
        assert len(r) > 0

    def test_len_matches_errors_list(self, v):
        r = v.validate("MATCH (n:BadLabel)-[:BadRel]->(m:AnotherBad) RETURN n")
        assert len(r) == len(r.errors)


# ===========================================================================
# 7. Schema inspection API
# ===========================================================================

class TestSchemaInspection:
    def test_node_properties(self, schema):
        props = schema.node_properties("Person")
        assert set(props) == {"name", "age"}

    def test_node_properties_unknown_label(self, schema):
        props = schema.node_properties("Unknown")
        assert props == []

    def test_rel_properties(self, schema):
        props = schema.rel_properties("ACTED_IN")
        assert props == ["role"]

    def test_rel_properties_unknown_type(self, schema):
        props = schema.rel_properties("UNKNOWN")
        assert props == []

    def test_rel_endpoints(self, schema):
        endpoints = schema.rel_endpoints("ACTED_IN")
        assert endpoints == ("Person", "Movie")

    def test_rel_endpoints_unknown(self, schema):
        assert schema.rel_endpoints("UNKNOWN") is None

    def test_to_dict_roundtrip(self, schema):
        d = schema.to_dict()
        assert "nodes" in d
        assert "relationships" in d
        # Reconstruct from dict
        s2 = Schema(nodes=d["nodes"], relationships=d["relationships"])
        assert set(s2.node_labels()) == set(schema.node_labels())
        assert set(s2.rel_types()) == set(schema.rel_types())

    def test_to_dict_node_props_preserved(self, schema):
        d = schema.to_dict()
        assert set(d["nodes"]["Person"]) == {"name", "age"}

    def test_to_dict_rel_preserved(self, schema):
        d = schema.to_dict()
        src, tgt, props = d["relationships"]["ACTED_IN"]
        assert src == "Person"
        assert tgt == "Movie"
        assert props == ["role"]


# ===========================================================================
# 8. parse_query() / QueryInfo
# ===========================================================================

class TestParseQuery:
    def test_valid_query_is_valid(self):
        info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name")
        assert info.is_valid
        assert info.errors == []

    def test_invalid_query_is_invalid(self):
        info = parse_query("THIS IS NOT CYPHER !!!")
        assert not info.is_valid
        assert len(info.errors) > 0

    def test_labels_extracted(self):
        info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p")
        assert set(info.labels_used) == {"Person", "Movie"}

    def test_rel_types_extracted(self):
        info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p")
        assert info.rel_types_used == ["ACTED_IN"]

    def test_no_labels_in_labelless_query(self):
        info = parse_query("MATCH (n) RETURN n")
        assert info.labels_used == []
        assert info.rel_types_used == []

    def test_multiple_labels(self):
        info = parse_query("MATCH (p:Person), (m:Movie) RETURN p, m")
        assert set(info.labels_used) == {"Person", "Movie"}

    def test_bool_true_when_valid(self):
        info = parse_query("MATCH (n) RETURN n")
        assert bool(info) is True

    def test_bool_false_when_invalid(self):
        info = parse_query("")
        assert bool(info) is False

    def test_parse_query_no_schema_needed(self):
        """parse_query works without any schema — pure syntax check."""
        info = parse_query("MATCH (a:Foo)-[:BAR]->(b:Baz) RETURN a")
        assert info.is_valid
        assert "Foo" in info.labels_used
        assert "Baz" in info.labels_used
        assert "BAR" in info.rel_types_used

    def test_create_query_labels(self):
        info = parse_query("CREATE (n:Person {name: 'Alice'}) RETURN n")
        assert "Person" in info.labels_used

    def test_labels_sorted(self):
        info = parse_query("MATCH (z:Zebra), (a:Antelope) RETURN z, a")
        assert info.labels_used == sorted(info.labels_used)


# ===========================================================================
# 9. Generator improvements
# ===========================================================================

class TestGeneratorImprovements:
    def test_generated_values_include_integers(self, schema):
        """Run many generations and confirm integer values appear."""
        g = CypherGenerator(schema, seed=7)
        queries = [g.generate("create") for _ in range(40)]
        has_int = any(
            any(c.isdigit() for c in q.split("{")[1].split("}")[0])
            for q in queries if "{" in q
        )
        assert has_int, "Expected at least one integer value in generated property maps"

    def test_generated_values_include_booleans(self, schema):
        g = CypherGenerator(schema, seed=3)
        queries = [g.generate("merge") for _ in range(40)]
        has_bool = any("true" in q or "false" in q for q in queries)
        assert has_bool, "Expected at least one boolean value in generated property maps"

    def test_generated_values_include_params(self, schema):
        g = CypherGenerator(schema, seed=4)
        queries = [g.generate("create") for _ in range(40)]
        has_param = any("$" in q for q in queries)
        assert has_param, "Expected at least one $param value in generated property maps"

    def test_optional_match_generated(self, schema):
        g = CypherGenerator(schema, seed=2)
        queries = [g.generate("match_relationship") for _ in range(30)]
        has_optional = any("OPTIONAL" in q for q in queries)
        assert has_optional, "Expected at least one OPTIONAL MATCH query"

    def test_limit_generated(self, schema):
        g = CypherGenerator(schema, seed=6)
        queries = [g.generate("match_return") for _ in range(30)]
        has_limit = any("LIMIT" in q for q in queries)
        assert has_limit, "Expected at least one LIMIT clause"

    def test_all_generated_queries_are_valid(self, schema):
        """Generator output should always pass the validator (over many seeds)."""
        v = CypherValidator(schema)
        for seed in range(20):
            g = CypherGenerator(schema, seed=seed)
            for t in CypherGenerator.supported_types():
                q = g.generate(t)
                r = v.validate(q)
                assert r.is_valid, (
                    f"seed={seed} type={t!r}: {q!r}\nErrors: {r.errors}"
                )
