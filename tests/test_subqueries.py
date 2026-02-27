"""Tests for CALL subqueries, pattern comprehension, and COUNT/COLLECT subqueries."""

import pytest
from cypher_validator import Schema, CypherValidator


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
            "KNOWS": ("Person", "Person", []),
            "WORKS_FOR": ("Person", "Company", ["since"]),
        },
    )


@pytest.fixture
def validator(schema):
    return CypherValidator(schema)


# ---------------------------------------------------------------------------
# CALL subqueries  (Neo4j 4.1+)
# ---------------------------------------------------------------------------


class TestCallSubquery:
    def test_valid_call_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) CALL { MATCH (m:Movie) RETURN m } RETURN p, m"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_unknown_label_in_call_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) CALL { MATCH (m:Moovie) RETURN m } RETURN p"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E201" in codes

    def test_outer_scope_visible_in_call_subquery(self, validator):
        """Outer-scope variables should be accessible inside the subquery."""
        r = validator.validate(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m } RETURN p, m"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_procedure_call_still_works(self, validator):
        """Ensure traditional CALL proc() syntax isn't broken."""
        r = validator.validate("CALL db.labels()")
        assert r.syntax_errors == []


# ---------------------------------------------------------------------------
# Pattern comprehension
# ---------------------------------------------------------------------------


class TestPatternComprehension:
    def test_valid_pattern_comprehension(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN [(p)-[:ACTED_IN]->(m:Movie) | m.title]"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_pattern_comprehension_with_where(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN [(p)-[:ACTED_IN]->(m:Movie) WHERE m.year > 2000 | m.title]"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_unknown_label_in_pattern_comprehension(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN [(p)-[:ACTED_IN]->(m:Moovie) | m.title]"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E201" in codes

    def test_unknown_rel_in_pattern_comprehension(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN [(p)-[:ACTS_IN]->(m:Movie) | m.title]"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E211" in codes

    def test_pattern_comprehension_bindings_dont_leak(self, validator):
        """Variables bound inside the pattern comprehension should NOT be
        visible in the outer scope."""
        r = validator.validate(
            "RETURN [(n:Person)-[:KNOWS]->(m:Person) | m.name], m"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E501" in codes


# ---------------------------------------------------------------------------
# COUNT / COLLECT subqueries (Neo4j 5.x)
# ---------------------------------------------------------------------------


class TestCountSubquery:
    def test_valid_count_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) WHERE COUNT { MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m } > 0 RETURN p"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_unknown_label_in_count_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) WHERE COUNT { MATCH (p)-[:ACTED_IN]->(m:Moovie) RETURN m } > 0 RETURN p"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E201" in codes

    def test_count_bindings_dont_leak(self, validator):
        r = validator.validate(
            "MATCH (p:Person) WHERE COUNT { MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m } > 0 RETURN m"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E501" in codes


class TestCollectSubquery:
    def test_valid_collect_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN COLLECT { MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m.title }"
        )
        error_codes = [d.code for d in r.diagnostics if d.code.startswith("E")]
        assert error_codes == [], f"Unexpected errors: {error_codes}"

    def test_unknown_label_in_collect_subquery(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN COLLECT { MATCH (p)-[:ACTED_IN]->(m:Moovie) RETURN m.title }"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E201" in codes

    def test_collect_bindings_dont_leak(self, validator):
        r = validator.validate(
            "MATCH (p:Person) RETURN COLLECT { MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m.title }, m"
        )
        codes = [d.code for d in r.diagnostics]
        assert "E501" in codes
