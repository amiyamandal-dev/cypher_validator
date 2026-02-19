"""
Tests for Task #3 features:
  - Error suggestions ("did you mean") for unknown node labels and rel types
  - New generator types: distinct_return, order_by, unwind
  - GLiNER2 RelationToCypherConverter deduplication
"""
import pytest
from cypher_validator import (
    Schema, CypherValidator, CypherGenerator,
    RelationToCypherConverter, parse_query,
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


@pytest.fixture
def gen(schema):
    return CypherGenerator(schema, seed=99)


# ===========================================================================
# 1. "Did you mean" suggestions in error messages
# ===========================================================================

class TestDidYouMean:
    def test_typo_in_node_label_suggests_correction(self, v):
        """'Preson' is one edit from 'Person'."""
        r = v.validate("MATCH (p:Preson) RETURN p")
        assert not r.is_valid
        # Error message should contain a suggestion
        full_errors = " ".join(r.errors)
        assert "Person" in full_errors  # suggestion present

    def test_exactly_wrong_label_no_suggestion_when_far(self, v):
        """'XyzAbc123' is far from any known label — no suggestion needed."""
        r = v.validate("MATCH (x:XyzAbc123) RETURN x")
        assert not r.is_valid
        # "did you mean" should NOT appear (too dissimilar)
        full_errors = " ".join(r.errors)
        assert "did you mean" not in full_errors.lower()

    def test_typo_in_rel_type_suggests_correction(self, v):
        """'ACTED_ON' is close to 'ACTED_IN'."""
        r = v.validate("MATCH (p:Person)-[:ACTED_ON]->(m:Movie) RETURN p")
        assert not r.is_valid
        full_errors = " ".join(r.errors)
        assert "ACTED_IN" in full_errors

    def test_suggestion_uses_correct_label(self, v):
        """'Moive' → suggests 'Movie'."""
        r = v.validate("MATCH (m:Moive) RETURN m")
        assert not r.is_valid
        full_errors = " ".join(r.errors)
        assert "Movie" in full_errors

    def test_case_insensitive_suggestion(self, v):
        """'person' (lowercase) → should suggest 'Person'."""
        r = v.validate("MATCH (p:person) RETURN p")
        assert not r.is_valid
        full_errors = " ".join(r.errors)
        assert "Person" in full_errors

    def test_error_message_still_includes_unknown_label(self, v):
        """Error message should still report the unknown label."""
        r = v.validate("MATCH (p:Preson) RETURN p")
        full_errors = " ".join(r.errors)
        assert "Preson" in full_errors

    def test_valid_query_no_suggestion(self, v):
        """No error, no suggestion for correct queries."""
        r = v.validate("MATCH (p:Person) RETURN p")
        assert r.is_valid
        assert r.errors == []


# ===========================================================================
# 2. New generator types: distinct_return, order_by, unwind
# ===========================================================================

class TestNewGeneratorTypes:
    # --- distinct_return ---

    def test_distinct_return_in_supported_types(self):
        assert "distinct_return" in CypherGenerator.supported_types()

    def test_distinct_return_contains_distinct(self, gen):
        q = gen.generate("distinct_return")
        assert "DISTINCT" in q

    def test_distinct_return_contains_match(self, gen):
        q = gen.generate("distinct_return")
        assert "MATCH" in q

    def test_distinct_return_contains_return(self, gen):
        q = gen.generate("distinct_return")
        assert "RETURN" in q

    def test_distinct_return_is_valid_cypher(self, gen):
        for _ in range(5):
            q = gen.generate("distinct_return")
            info = parse_query(q)
            assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    # --- order_by ---

    def test_order_by_in_supported_types(self):
        assert "order_by" in CypherGenerator.supported_types()

    def test_order_by_contains_order_by(self, gen):
        q = gen.generate("order_by")
        assert "ORDER BY" in q

    def test_order_by_contains_return(self, gen):
        q = gen.generate("order_by")
        assert "RETURN" in q

    def test_order_by_is_valid_cypher(self, gen):
        for _ in range(5):
            q = gen.generate("order_by")
            info = parse_query(q)
            assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_order_by_optionally_desc(self, schema):
        """Run many seeds — DESC should appear sometimes."""
        results = []
        for seed in range(50):
            g = CypherGenerator(schema, seed=seed)
            q = g.generate("order_by")
            results.append(q)
        assert any("DESC" in q for q in results), "Expected DESC in at least one generated query"

    # --- unwind ---

    def test_unwind_in_supported_types(self):
        assert "unwind" in CypherGenerator.supported_types()

    def test_unwind_contains_return(self, gen):
        q = gen.generate("unwind")
        assert "RETURN" in q

    def test_unwind_is_valid_cypher(self, gen):
        for _ in range(5):
            q = gen.generate("unwind")
            info = parse_query(q)
            assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_unknown_type_raises(self, gen):
        with pytest.raises(Exception):
            gen.generate("no_such_type")

    def test_all_new_types_generate_valid_cypher(self, gen):
        for qt in ("distinct_return", "order_by", "unwind"):
            q = gen.generate(qt)
            info = parse_query(q)
            assert info.is_valid, f"[{qt}] Invalid: {q}\nErrors: {info.errors}"


# ===========================================================================
# 3. GLiNER2 RelationToCypherConverter deduplication
# ===========================================================================

DUPED_RELATIONS = {
    "relation_extraction": {
        "works_for": [
            ("John", "Apple"),
            ("John", "Apple"),   # exact duplicate
            ("Mary", "Google"),
        ]
    }
}

CROSS_TYPE_DUPE = {
    "relation_extraction": {
        "works_for": [("John", "Apple"), ("John", "Apple")],
        "lives_in":  [("John", "NYC"),   ("John", "NYC")],
    }
}


class TestConverterDeduplication:
    def test_match_deduplicates_exact_pairs(self):
        conv = RelationToCypherConverter()
        q, params = conv.to_match_query(DUPED_RELATIONS)
        # John→Apple appears twice in input but only once in output after dedup
        assert list(params.values()).count("John") == 1   # deduped: John-Apple only once
        assert list(params.values()).count("Mary") == 1   # Mary-Google is a unique pair
        assert q.count("MATCH") == 2    # two distinct pairs (John-Apple + Mary-Google)
        # Values must be in params, NOT embedded as literals in the query string
        assert '"John"' not in q
        assert '"Mary"' not in q

    def test_merge_deduplicates_exact_pairs(self):
        conv = RelationToCypherConverter()
        q, params = conv.to_merge_query(DUPED_RELATIONS)
        assert q.count("MERGE") == 2    # John-Apple once, Mary-Google once

    def test_create_deduplicates_exact_pairs(self):
        conv = RelationToCypherConverter()
        q, params = conv.to_create_query(DUPED_RELATIONS)
        assert q.count("CREATE") == 2

    def test_cross_type_deduplication_independent(self):
        """Duplicates within each relation type are deduplicated independently."""
        conv = RelationToCypherConverter()
        q, params = conv.to_match_query(CROSS_TYPE_DUPE)
        # John-Apple (WORKS_FOR) and John-NYC (LIVES_IN) each appear once
        assert q.count("MATCH") == 2

    def test_different_pairs_not_deduplicated(self):
        """Different pairs with the same subject or object are NOT removed."""
        relations = {
            "relation_extraction": {
                "works_for": [
                    ("Alice", "Apple"),
                    ("Alice", "Google"),    # same subject, different object
                    ("Bob",   "Apple"),     # same object, different subject
                ]
            }
        }
        conv = RelationToCypherConverter()
        q, params = conv.to_match_query(relations)
        assert q.count("MATCH") == 3

    def test_dedup_per_rel_type(self):
        """Same (subj, obj) with different rel types are NOT deduplicated."""
        relations = {
            "relation_extraction": {
                "works_for": [("John", "Apple")],
                "founded":   [("John", "Apple")],   # same pair, different type
            }
        }
        conv = RelationToCypherConverter()
        q, params = conv.to_match_query(relations)
        assert q.count("MATCH") == 2    # one per distinct (subj, obj, rel_type)

    def test_empty_after_dedup_returns_empty(self):
        """If all pairs are duplicates, the output is empty."""
        all_dupes = {
            "relation_extraction": {
                "works_for": [
                    ("John", "Apple"),
                    ("John", "Apple"),
                ]
            }
        }
        conv = RelationToCypherConverter()
        # After dedup: only one pair remains, so output is NOT empty
        q, params = conv.to_match_query(all_dupes)
        assert q != ""
        assert q.count("MATCH") == 1

    def test_no_injection_via_entity_name(self):
        """Adversarial entity names must not alter the query structure."""
        malicious = {
            "relation_extraction": {
                "works_for": [
                    ('Alice" OR 1=1 OR "x', "Apple"),
                    ("Bob", 'Acme"})-[:ADMIN]->(x'),
                ]
            }
        }
        conv = RelationToCypherConverter()
        q, params = conv.to_match_query(malicious)
        # Malicious strings land in params, never in the query template
        assert 'OR 1=1' not in q
        assert 'ADMIN' not in q
        assert any("OR 1=1" in v for v in params.values())
        assert any("ADMIN" in v for v in params.values())
        # Query must only contain $ placeholders, not the raw values
        assert "$" in q
        assert q.count("MATCH") == 2
