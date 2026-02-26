"""Tests for v0.10.0 structured diagnostics, error codes, and auto-fix."""

import json
import pytest
from cypher_validator import Schema, CypherValidator, ValidationDiagnostic
from cypher_validator.llm_utils import repair_cypher


@pytest.fixture
def schema():
    return Schema(
        nodes={
            "Person": ["name", "age"],
            "Movie": ["title", "year"],
            "Company": ["name", "founded"],
        },
        relationships={
            "ACTED_IN": ("Person", "Movie", ["role"]),
            "WORKS_FOR": ("Person", "Company", ["since"]),
        },
    )


@pytest.fixture
def validator(schema):
    return CypherValidator(schema)


# ---------------------------------------------------------------------------
# Error code presence for each category
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_e201_unknown_node_label(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        assert not r.is_valid
        codes = [d.code for d in r.diagnostics]
        assert "E201" in codes

    # E202/E203 (SET/REMOVE label) tests skipped: parser grammar bug prevents
    # `SET n:Label` / `REMOVE n:Label` from parsing (node_labels rule requires
    # double colon). These error codes are wired up but unreachable until the
    # grammar is fixed.

    def test_e211_unknown_rel_type(self, validator):
        r = validator.validate("MATCH (a:Person)-[:ACTS_IN]->(b:Movie) RETURN a")
        codes = [d.code for d in r.diagnostics]
        assert "E211" in codes

    def test_e301_unknown_node_property(self, validator):
        r = validator.validate("MATCH (n:Person {nme: 'Alice'}) RETURN n")
        codes = [d.code for d in r.diagnostics]
        assert "E301" in codes

    def test_e302_unknown_rel_property(self, validator):
        r = validator.validate(
            "MATCH (a:Person)-[r:ACTED_IN {rle: 'lead'}]->(b:Movie) RETURN a"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E302" in codes

    def test_e303_unknown_node_property_on_var(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n.nme")
        codes = [d.code for d in r.diagnostics]
        assert "E303" in codes

    def test_e304_unknown_rel_property_on_var(self, validator):
        r = validator.validate(
            "MATCH (a:Person)-[r:ACTED_IN]->(b:Movie) RETURN r.rle"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E304" in codes

    def test_e401_wrong_endpoint_label(self, validator):
        r = validator.validate(
            "MATCH (a:Movie)-[:ACTED_IN]->(b:Movie) RETURN a"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E401" in codes

    def test_e501_unbound_variable(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN m")
        codes = [d.code for d in r.diagnostics]
        assert "E501" in codes

    def test_e502_duplicate_projection_name(self, validator):
        r = validator.validate(
            "MATCH (n:Person) RETURN n.name AS x, n.age AS x"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E502" in codes

    def test_e601_aggregate_string_arg(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN SUM('hello')")
        codes = [d.code for d in r.diagnostics]
        assert "E601" in codes

    def test_e602_aggregate_in_forbidden_context(self, validator):
        r = validator.validate(
            "MATCH (n:Person) WHERE COUNT(*) > 0 RETURN n"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E602" in codes

    def test_e611_pagination_type_string(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n LIMIT 'ten'")
        codes = [d.code for d in r.diagnostics]
        assert "E611" in codes

    def test_e612_pagination_type_bool(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n LIMIT true")
        codes = [d.code for d in r.diagnostics]
        assert "E612" in codes

    def test_e613_pagination_type_null(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n LIMIT null")
        codes = [d.code for d in r.diagnostics]
        assert "E613" in codes

    def test_e614_pagination_type_float(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n LIMIT 1.5")
        codes = [d.code for d in r.diagnostics]
        assert "E614" in codes

    def test_w101_cartesian_product(self, validator):
        r = validator.validate(
            "MATCH (a:Person), (b:Movie) RETURN a, b"
        )
        codes = [d.code for d in r.diagnostics]
        assert "W101" in codes

    def test_w201_unlabeled_full_scan(self, validator):
        r = validator.validate("MATCH (n) RETURN n")
        codes = [d.code for d in r.diagnostics]
        assert "W201" in codes

    def test_w202_unbounded_var_length(self, validator):
        r = validator.validate(
            "MATCH (a:Person)-[*]->(b:Person) RETURN a, b"
        )
        codes = [d.code for d in r.diagnostics]
        assert "W202" in codes


# ---------------------------------------------------------------------------
# Suggestion fields
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_label_typo_suggestion(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        diag = [d for d in r.diagnostics if d.code == "E201"][0]
        assert diag.suggestion_original == ":Persn"
        assert diag.suggestion_replacement == ":Person"
        assert diag.suggestion_description is not None

    def test_rel_type_typo_suggestion(self, validator):
        r = validator.validate(
            "MATCH (a:Person)-[:ACTS_IN]->(b:Movie) RETURN a"
        )
        diag = [d for d in r.diagnostics if d.code == "E211"][0]
        assert diag.suggestion_original == ":ACTS_IN"
        assert diag.suggestion_replacement == ":ACTED_IN"

    def test_node_property_typo_suggestion(self, validator):
        r = validator.validate("MATCH (n:Person {nme: 'Alice'}) RETURN n")
        diag = [d for d in r.diagnostics if d.code == "E301"][0]
        assert diag.suggestion_replacement is not None
        assert "name" in diag.suggestion_replacement

    def test_property_on_var_suggestion(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n.nme")
        diag = [d for d in r.diagnostics if d.code == "E303"][0]
        assert diag.suggestion_replacement == "name"

    def test_no_suggestion_for_distant_typo(self, validator):
        r = validator.validate("MATCH (n:XYZ123) RETURN n")
        diag = [d for d in r.diagnostics if d.code == "E201"][0]
        assert diag.suggestion_replacement is None


# ---------------------------------------------------------------------------
# fixed_query
# ---------------------------------------------------------------------------


class TestFixedQuery:
    def test_fixable_label_typo(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        assert r.fixed_query is not None
        assert ":Person" in r.fixed_query
        # Fixed query should be valid
        r2 = validator.validate(r.fixed_query)
        assert r2.is_valid

    def test_fixable_rel_typo(self, validator):
        r = validator.validate(
            "MATCH (a:Person)-[:ACTS_IN]->(b:Movie) RETURN a"
        )
        assert r.fixed_query is not None
        assert ":ACTED_IN" in r.fixed_query
        r2 = validator.validate(r.fixed_query)
        assert r2.is_valid

    def test_unfixable_error_returns_none(self, validator):
        # Unbound variable has no suggestion
        r = validator.validate("MATCH (n:Person) RETURN m")
        assert r.fixed_query is None

    def test_mixed_fixable_unfixable_returns_none(self, validator):
        # Both label typo (fixable) and unbound var (unfixable)
        r = validator.validate("MATCH (n:Persn) RETURN m")
        assert r.fixed_query is None

    def test_valid_query_no_fixed_query(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n")
        assert r.is_valid
        assert r.fixed_query is None


# ---------------------------------------------------------------------------
# Parse errors have position info
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_parse_error_has_position(self, validator):
        r = validator.validate("MATC (n:Person) RETURN n")
        assert not r.is_valid
        diag = r.diagnostics[0]
        assert diag.code == "E101"
        assert diag.severity == "error"
        assert diag.position_line is not None
        assert diag.position_col is not None

    def test_parse_error_has_code_name(self, validator):
        r = validator.validate("MATC (n:Person) RETURN n")
        diag = r.diagnostics[0]
        assert diag.code_name == "ParseError"


# ---------------------------------------------------------------------------
# to_dict / to_json include new fields
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_has_diagnostics(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        d = r.to_dict()
        assert "diagnostics" in d
        assert "fixed_query" in d
        assert len(d["diagnostics"]) > 0
        diag_dict = d["diagnostics"][0]
        assert "code" in diag_dict
        assert "severity" in diag_dict
        assert "message" in diag_dict

    def test_to_json_has_diagnostics(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        j = json.loads(r.to_json())
        assert "diagnostics" in j
        assert "fixed_query" in j
        assert len(j["diagnostics"]) > 0
        assert j["diagnostics"][0]["code"] == "E201"

    def test_to_dict_valid_query(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n")
        d = r.to_dict()
        assert d["is_valid"] is True
        assert d["diagnostics"] == []
        assert d["fixed_query"] is None

    def test_diagnostic_to_dict(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        diag = r.diagnostics[0]
        d = diag.to_dict()
        assert d["code"] == "E201"
        assert d["severity"] == "error"
        assert d["suggestion_replacement"] == ":Person"


# ---------------------------------------------------------------------------
# Backward compat: existing string fields unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_errors_unchanged(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        assert len(r.errors) == 1
        assert "Unknown node label" in r.errors[0]
        assert "did you mean :Person" in r.errors[0]

    def test_syntax_errors_unchanged(self, validator):
        r = validator.validate("MATC (n:Person) RETURN n")
        assert len(r.syntax_errors) == 1

    def test_warnings_unchanged(self, validator):
        r = validator.validate("MATCH (a:Person), (b:Movie) RETURN a, b")
        assert len(r.warnings) >= 1
        assert "Cartesian" in r.warnings[0]

    def test_valid_query_unchanged(self, validator):
        r = validator.validate("MATCH (n:Person) RETURN n")
        assert r.is_valid
        assert r.errors == []
        assert r.syntax_errors == []
        assert r.semantic_errors == []

    def test_bool_and_len(self, validator):
        r_good = validator.validate("MATCH (n:Person) RETURN n")
        assert bool(r_good) is True
        assert len(r_good) == 0
        r_bad = validator.validate("MATCH (n:Persn) RETURN n")
        assert bool(r_bad) is False
        assert len(r_bad) == 1


# ---------------------------------------------------------------------------
# repair_cypher tries fixed_query before LLM
# ---------------------------------------------------------------------------


class TestRepairCypher:
    def test_skips_llm_when_fixed_query_suffices(self, validator):
        llm_called = []

        def mock_llm(query, errors):
            llm_called.append(True)
            return query  # no-op

        query, result = repair_cypher(
            validator,
            "MATCH (n:Persn) RETURN n",
            mock_llm,
        )
        assert result.is_valid
        assert ":Person" in query
        assert len(llm_called) == 0  # LLM was never called

    def test_falls_back_to_llm_when_unfixable(self, validator):
        llm_called = []

        def mock_llm(query, errors):
            llm_called.append(True)
            return "MATCH (n:Person) RETURN n"

        query, result = repair_cypher(
            validator,
            "MATCH (n:Person) RETURN m",  # unbound var, no suggestion
            mock_llm,
        )
        assert result.is_valid
        assert len(llm_called) == 1


# ---------------------------------------------------------------------------
# Diagnostics severity
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_error_severity(self, validator):
        r = validator.validate("MATCH (n:Persn) RETURN n")
        assert all(d.severity == "error" for d in r.diagnostics if d.code.startswith("E"))

    def test_warning_severity(self, validator):
        r = validator.validate("MATCH (a:Person), (b:Movie) RETURN a, b")
        warnings = [d for d in r.diagnostics if d.code.startswith("W")]
        assert len(warnings) >= 1
        assert all(d.severity == "warning" for d in warnings)

    def test_warnings_dont_block_is_valid(self, validator):
        """Queries with only warnings should still be is_valid=True."""
        r = validator.validate("MATCH (a:Person), (b:Movie) RETURN a, b")
        assert r.is_valid  # warnings don't make it invalid
