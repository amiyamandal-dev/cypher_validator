"""
Tests for GLiNER2 integration (RelationToCypherConverter, GLiNER2RelationExtractor,
NLToCypher).

The Cypher-conversion tests run without any ML model.
The GLiNER2RelationExtractor and NLToCypher tests use unittest.mock to simulate
the model, so they also run without the optional ``gliner2`` package installed.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from cypher_validator import (
    Schema,
    GLiNER2RelationExtractor,
    RelationToCypherConverter,
    NLToCypher,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return Schema(
        nodes={
            "Person":  ["name", "age"],
            "Company": ["name"],
            "City":    ["name"],
        },
        relationships={
            "WORKS_FOR": ("Person", "Company", []),
            "LIVES_IN":  ("Person", "City", []),
        },
    )


@pytest.fixture
def converter():
    return RelationToCypherConverter()


@pytest.fixture
def converter_with_schema(schema):
    return RelationToCypherConverter(schema=schema)


# Sample extraction results
SINGLE_PAIR = {
    "relation_extraction": {
        "works_for": [("John", "Apple Inc.")],
    }
}

TWO_TYPES = {
    "relation_extraction": {
        "works_for": [("John", "Apple Inc.")],
        "lives_in":  [("John", "San Francisco")],
    }
}

MULTI_PAIRS = {
    "relation_extraction": {
        "works_for": [
            ("John",  "Microsoft"),
            ("Mary",  "Google"),
            ("Bob",   "Apple"),
        ],
    }
}

EMPTY = {
    "relation_extraction": {
        "works_for": [],
        "lives_in":  [],
    }
}


# ===========================================================================
# 1. RelationToCypherConverter — MATCH
# ===========================================================================

class TestMatchQuery:
    def test_keywords_present(self, converter):
        q, params = converter.to_match_query(SINGLE_PAIR)
        assert "MATCH" in q
        assert ":WORKS_FOR" in q
        assert "RETURN" in q

    def test_entities_appear(self, converter):
        # Entity values are in params (parameterized query), not embedded in q
        q, params = converter.to_match_query(SINGLE_PAIR)
        assert "John" in params.values()
        assert "Apple Inc." in params.values()

    def test_rel_type_uppercased(self, converter):
        q, params = converter.to_match_query(SINGLE_PAIR)
        assert ":WORKS_FOR" in q
        assert ":works_for" not in q

    def test_two_types_give_two_match_clauses(self, converter):
        q, params = converter.to_match_query(TWO_TYPES)
        assert q.count("MATCH") == 2
        assert ":WORKS_FOR" in q
        assert ":LIVES_IN" in q

    def test_multiple_pairs_same_type_give_multiple_clauses(self, converter):
        q, params = converter.to_match_query(MULTI_PAIRS)
        # One MATCH clause per pair
        assert q.count("MATCH") == 3
        assert "John" in params.values()
        assert "Mary" in params.values()
        assert "Bob" in params.values()

    def test_empty_relations_return_empty_string(self, converter):
        assert converter.to_match_query(EMPTY) == ("", {})

    def test_custom_return_clause(self, converter):
        q, params = converter.to_match_query(SINGLE_PAIR, return_clause="RETURN *")
        assert q.endswith("RETURN *")

    def test_auto_return_vars(self, converter):
        q, params = converter.to_match_query(SINGLE_PAIR)
        # auto return should mention a0 and b0
        assert "a0" in q
        assert "b0" in q

    def test_inline_property_map_used(self, converter):
        # MATCH uses {name: $param} inline props, not WHERE conditions
        q, params = converter.to_match_query(SINGLE_PAIR)
        assert "WHERE" not in q
        assert "{name:" in q or '{"name"' in q or "name:" in q

    def test_missing_relation_extraction_key(self, converter):
        # Top-level key absent → treated as empty
        assert converter.to_match_query({}) == ("", {})


# ===========================================================================
# 2. RelationToCypherConverter — MERGE
# ===========================================================================

class TestMergeQuery:
    def test_merge_keyword(self, converter):
        q, params = converter.to_merge_query(SINGLE_PAIR)
        assert "MERGE" in q

    def test_entities_and_rel(self, converter):
        q, params = converter.to_merge_query(SINGLE_PAIR)
        assert "John" in params.values()
        assert "Apple Inc." in params.values()
        assert ":WORKS_FOR" in q

    def test_schema_adds_node_labels(self, converter_with_schema):
        q, params = converter_with_schema.to_merge_query(SINGLE_PAIR)
        assert ":Person" in q
        assert ":Company" in q

    def test_no_schema_no_node_labels(self, converter):
        q, params = converter.to_merge_query(SINGLE_PAIR)
        assert ":Person" not in q
        assert ":Company" not in q

    def test_multiple_pairs_give_multiple_merges(self, converter):
        q, params = converter.to_merge_query(MULTI_PAIRS)
        assert q.count("MERGE") == 3

    def test_empty_returns_empty_string(self, converter):
        assert converter.to_merge_query(EMPTY) == ("", {})

    def test_custom_return_clause(self, converter):
        q, params = converter.to_merge_query(SINGLE_PAIR, return_clause="RETURN *")
        assert q.endswith("RETURN *")


# ===========================================================================
# 3. RelationToCypherConverter — CREATE
# ===========================================================================

class TestCreateQuery:
    def test_create_keyword(self, converter):
        q, params = converter.to_create_query(SINGLE_PAIR)
        assert "CREATE" in q

    def test_schema_adds_labels(self, converter_with_schema):
        q, params = converter_with_schema.to_create_query(SINGLE_PAIR)
        assert ":Person" in q
        assert ":Company" in q

    def test_empty_returns_empty_string(self, converter):
        assert converter.to_create_query(EMPTY) == ("", {})

    def test_multiple_pairs_give_multiple_creates(self, converter):
        q, params = converter.to_create_query(MULTI_PAIRS)
        assert q.count("CREATE") == 3


# ===========================================================================
# 4. RelationToCypherConverter — convert() dispatcher
# ===========================================================================

class TestConvertDispatcher:
    def test_match_mode(self, converter):
        q, params = converter.convert(SINGLE_PAIR, mode="match")
        assert "MATCH" in q

    def test_merge_mode(self, converter):
        q, params = converter.convert(SINGLE_PAIR, mode="merge")
        assert "MERGE" in q

    def test_create_mode(self, converter):
        q, params = converter.convert(SINGLE_PAIR, mode="create")
        assert "CREATE" in q

    def test_unknown_mode_raises_value_error(self, converter):
        with pytest.raises(ValueError, match="Unknown mode"):
            converter.convert(SINGLE_PAIR, mode="upsert")

    def test_kwargs_forwarded_to_generator(self, converter):
        q, params = converter.convert(SINGLE_PAIR, mode="match", return_clause="RETURN *")
        assert q.endswith("RETURN *")


# ===========================================================================
# 5. Custom name_property
# ===========================================================================

class TestNameProperty:
    def test_custom_property_in_match(self):
        conv = RelationToCypherConverter(name_property="title")
        q, params = conv.to_match_query(SINGLE_PAIR)
        assert "title:" in q  # inline property map uses {title: $param}
        assert "name:" not in q

    def test_custom_property_in_merge(self):
        conv = RelationToCypherConverter(name_property="label")
        q, params = conv.to_merge_query(SINGLE_PAIR)
        assert "label:" in q or "{label:" in q
        assert "name:" not in q


# ===========================================================================
# 6. Two relation types with schema (integration-style)
# ===========================================================================

class TestSchemaLabelResolution:
    def test_merge_two_types_correct_labels(self, converter_with_schema):
        q, params = converter_with_schema.to_merge_query(TWO_TYPES)
        assert ":Person" in q
        assert ":Company" in q
        assert ":City" in q

    def test_unknown_rel_type_no_labels(self, converter_with_schema):
        rels = {"relation_extraction": {"founded": [("Elon", "Tesla")]}}
        q, params = converter_with_schema.to_merge_query(rels)
        # FOUNDED not in schema → no label annotation
        assert ":FOUNDED" in q  # the rel type is still there
        # No spurious node labels
        lines = [l for l in q.splitlines() if "MERGE" in l]
        assert all(":Person" not in l and ":Company" not in l for l in lines)


# ===========================================================================
# 7. GLiNER2RelationExtractor (mocked model)
# ===========================================================================

def _make_extractor(raw_output: dict, threshold: float = 0.5) -> GLiNER2RelationExtractor:
    mock_model = MagicMock()
    mock_model.extract_relations.return_value = raw_output
    return GLiNER2RelationExtractor(mock_model, threshold=threshold)


class TestGLiNER2RelationExtractor:
    def test_output_has_relation_extraction_key(self):
        ext = _make_extractor({"relation_extraction": {"works_for": [("John", "Apple")]}})
        result = ext.extract_relations("text", ["works_for"])
        assert "relation_extraction" in result

    def test_all_requested_types_present(self):
        raw = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        ext = _make_extractor(raw)
        result = ext.extract_relations("text", ["works_for", "lives_in", "founded"])
        rel = result["relation_extraction"]
        assert set(rel.keys()) == {"works_for", "lives_in", "founded"}

    def test_missing_type_gets_empty_list(self):
        raw = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        ext = _make_extractor(raw)
        result = ext.extract_relations("text", ["works_for", "lives_in"])
        assert result["relation_extraction"]["lives_in"] == []

    def test_found_type_preserved(self):
        raw = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        ext = _make_extractor(raw)
        result = ext.extract_relations("text", ["works_for"])
        assert result["relation_extraction"]["works_for"] == [("John", "Apple")]

    def test_threshold_override_passed_to_model(self):
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {"relation_extraction": {}}
        ext = GLiNER2RelationExtractor(mock_model, threshold=0.5)
        ext.extract_relations("text", ["rel"], threshold=0.9)
        mock_model.extract_relations.assert_called_once_with("text", ["rel"], threshold=0.9)

    def test_instance_threshold_used_when_not_overridden(self):
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {"relation_extraction": {}}
        ext = GLiNER2RelationExtractor(mock_model, threshold=0.7)
        ext.extract_relations("text", ["rel"])
        mock_model.extract_relations.assert_called_once_with("text", ["rel"], threshold=0.7)

    def test_raw_dict_without_wrapper_key_still_works(self):
        # Some model versions return the dict directly without the wrapper key
        raw = {"works_for": [("John", "Apple")]}
        ext = _make_extractor(raw)
        result = ext.extract_relations("text", ["works_for"])
        assert result["relation_extraction"]["works_for"] == [("John", "Apple")]

    def test_from_pretrained_raises_import_error_when_gliner2_missing(self):
        # Temporarily hide the gliner2 module
        with patch.dict(sys.modules, {"gliner2": None}):
            with pytest.raises(ImportError, match="gliner2"):
                GLiNER2RelationExtractor.from_pretrained("fastino/gliner2-large-v1")

    def test_default_model_name_constant(self):
        assert GLiNER2RelationExtractor.DEFAULT_MODEL == "fastino/gliner2-large-v1"


# ===========================================================================
# 8. NLToCypher (mocked extractor)
# ===========================================================================

def _make_pipeline(raw_output: dict, schema=None) -> NLToCypher:
    mock_model = MagicMock()
    mock_model.extract_relations.return_value = raw_output
    extractor = GLiNER2RelationExtractor(mock_model)
    return NLToCypher(extractor, schema=schema)


class TestNLToCypher:
    def test_call_default_match_mode(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": [("John", "Apple")]}})
        cypher = pipeline("John works for Apple.", ["works_for"])
        assert "MATCH" in cypher
        assert ":WORKS_FOR" in cypher

    def test_call_merge_mode(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": [("John", "Apple")]}})
        cypher = pipeline("text", ["works_for"], mode="merge")
        assert "MERGE" in cypher

    def test_call_create_mode(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": [("John", "Apple")]}})
        cypher = pipeline("text", ["works_for"], mode="create")
        assert "CREATE" in cypher

    def test_schema_adds_labels_in_merge(self):
        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        pipeline = _make_pipeline(
            {"relation_extraction": {"works_for": [("John", "Apple")]}},
            schema=schema,
        )
        cypher = pipeline("text", ["works_for"], mode="merge")
        assert ":Person" in cypher
        assert ":Company" in cypher

    def test_empty_extraction_returns_empty_string(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": []}})
        assert pipeline("text", ["works_for"]) == ""

    def test_extract_and_convert_returns_tuple(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": [("John", "Apple")]}})
        relations, cypher = pipeline.extract_and_convert("text", ["works_for"])
        assert "relation_extraction" in relations
        assert isinstance(cypher, str)
        assert "MATCH" in cypher

    def test_threshold_forwarded_to_extractor(self):
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {"relation_extraction": {}}
        extractor = GLiNER2RelationExtractor(mock_model, threshold=0.5)
        pipeline = NLToCypher(extractor)
        pipeline("text", ["rel"], threshold=0.9)
        mock_model.extract_relations.assert_called_once_with("text", ["rel"], threshold=0.9)

    def test_kwargs_forwarded_to_converter(self):
        pipeline = _make_pipeline({"relation_extraction": {"works_for": [("John", "Apple")]}})
        cypher = pipeline("text", ["works_for"], mode="match", return_clause="RETURN *")
        assert cypher.endswith("RETURN *")

    def test_from_pretrained_raises_if_gliner2_missing(self):
        with patch.dict(sys.modules, {"gliner2": None}):
            with pytest.raises(ImportError, match="gliner2"):
                NLToCypher.from_pretrained("fastino/gliner2-large-v1")

    def test_repr(self):
        pipeline = _make_pipeline({"relation_extraction": {}})
        assert "NLToCypher" in repr(pipeline)


# ===========================================================================
# 9. Cypher validity — generated queries pass parse_query
# ===========================================================================

from cypher_validator import parse_query


class TestGeneratedCypherValidity:
    """Spot-check that converter output is syntactically valid Cypher."""

    def test_match_query_parses(self, converter):
        q, params = converter.to_match_query(SINGLE_PAIR)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_match_two_types_parses(self, converter):
        q, params = converter.to_match_query(TWO_TYPES)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_match_multi_pairs_parses(self, converter):
        q, params = converter.to_match_query(MULTI_PAIRS)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_merge_query_parses(self, converter):
        q, params = converter.to_merge_query(SINGLE_PAIR)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_merge_with_schema_parses(self, converter_with_schema):
        q, params = converter_with_schema.to_merge_query(TWO_TYPES)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_create_query_parses(self, converter):
        q, params = converter.to_create_query(SINGLE_PAIR)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"

    def test_create_multi_pairs_parses(self, converter):
        q, params = converter.to_create_query(MULTI_PAIRS)
        info = parse_query(q)
        assert info.is_valid, f"Invalid Cypher: {q}\nErrors: {info.errors}"
