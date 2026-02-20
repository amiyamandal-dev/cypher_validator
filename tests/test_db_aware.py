"""
Tests for the DB-aware query generation in NLToCypher and RelationToCypherConverter.

All tests mock the Neo4j driver and GLiNER2 model — no real Neo4j instance
or ML model is required.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(relations: dict):
    """Return a mock GLiNER2RelationExtractor that returns *relations*."""
    from cypher_validator import GLiNER2RelationExtractor
    extractor = MagicMock(spec=GLiNER2RelationExtractor)
    extractor.extract_relations.return_value = {"relation_extraction": relations}
    return extractor


def _make_db(records_map: dict = None):
    """Return a mock Neo4jDatabase.

    *records_map* maps query substrings to return values, enabling per-entity
    lookup mocking.  If *None*, ``execute()`` always returns ``[]``.
    """
    from cypher_validator import Neo4jDatabase
    db = MagicMock(spec=Neo4jDatabase)

    if records_map is None:
        db.execute.return_value = []
    else:
        def _side_effect(cypher, params=None):
            val = (params or {}).get("val", "")
            for key, result in records_map.items():
                if key == val:
                    return result
            return []
        db.execute.side_effect = _side_effect

    return db


# ---------------------------------------------------------------------------
# RelationToCypherConverter.to_db_aware_query — unit tests
# ---------------------------------------------------------------------------

class TestToDbAwareQuery:
    """Unit-test the converter method directly with pre-built entity_status."""

    def _make_status(self, entries):
        """Build an entity_status dict from a list of (name, var, label, found)."""
        return {
            name: {
                "var": var,
                "label": label,
                "param_key": f"{var}_val",
                "found": found,
                "introduced": False,
            }
            for name, var, label, found in entries
        }

    def test_both_new_entities(self):
        """Neither entity exists → single CREATE clause with both inline."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = self._make_status([
            ("John", "e0", "Person", False),
            ("Apple", "e1", "Company", False),
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert "MATCH" not in cypher
        assert "CREATE (e0:Person {name: $e0_val})-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher
        assert "RETURN e0, e1" in cypher
        assert params["e0_val"] == "John"
        assert params["e1_val"] == "Apple"

    def test_subject_exists_object_new(self):
        """Subject found in DB → MATCHed; object is new → inline CREATE."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = self._make_status([
            ("John", "e0", "Person", True),    # exists
            ("Apple", "e1", "Company", False),  # new
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert "MATCH (e0:Person {name: $e0_val})" in cypher
        assert "CREATE (e0)-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher
        assert "RETURN e0, e1" in cypher

    def test_object_exists_subject_new(self):
        """Object found in DB → MATCHed; subject is new → inline CREATE."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": [("Alice", "Acme")]}}
        status = self._make_status([
            ("Alice", "e0", "Person", False),  # new
            ("Acme", "e1", "Company", True),   # exists
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert "MATCH (e1:Company {name: $e1_val})" in cypher
        assert "CREATE (e0:Person {name: $e0_val})-[:WORKS_FOR]->(e1)" in cypher
        assert "RETURN e1, e0" in cypher

    def test_both_entities_exist(self):
        """Both found in DB → two MATCH clauses, CREATE only the relationship."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = self._make_status([
            ("John", "e0", "Person", True),
            ("Apple", "e1", "Company", True),
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert "MATCH (e0:Person {name: $e0_val})" in cypher
        assert "MATCH (e1:Company {name: $e1_val})" in cypher
        assert "CREATE (e0)-[:WORKS_FOR]->(e1)" in cypher
        # No inline property pattern for existing vars
        assert "e0:Person {name: $e0_val})-[:WORKS_FOR]" not in cypher

    def test_shared_subject_across_relations(self):
        """Same subject in multiple relations → introduced once, reused."""
        from cypher_validator import RelationToCypherConverter, Schema
        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
            relationships={
                "WORKS_FOR": ("Person", "Company", []),
                "LIVES_IN": ("Person", "City", []),
            },
        )
        converter = RelationToCypherConverter(schema=schema)
        relations = {"relation_extraction": {
            "works_for": [("John", "Apple")],
            "lives_in":  [("John", "NYC")],
        }}
        # John exists, Apple and NYC are new
        status = self._make_status([
            ("John",  "e0", "Person",  True),
            ("Apple", "e1", "Company", False),
            ("NYC",   "e2", "City",    False),
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        # John is MATCHed once at the top
        assert cypher.count("MATCH (e0") == 1
        # Two CREATE clauses reusing e0
        assert "CREATE (e0)-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher
        assert "CREATE (e0)-[:LIVES_IN]->(e2:City {name: $e2_val})" in cypher

    def test_shared_new_subject_across_relations(self):
        """New subject in multiple relations → inline in first CREATE, reused in second."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {
            "works_for": [("John", "Apple")],
            "lives_in":  [("John", "NYC")],
        }}
        # All entities are new
        status = self._make_status([
            ("John",  "e0", "Person",  False),
            ("Apple", "e1", "Company", False),
            ("NYC",   "e2", "City",    False),
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert "MATCH" not in cypher
        # John appears inline in the first CREATE
        assert "CREATE (e0:Person {name: $e0_val})-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher
        # John is reused (just variable) in the second
        assert "CREATE (e0)-[:LIVES_IN]->(e2:City {name: $e2_val})" in cypher

    def test_empty_relations_returns_empty(self):
        """No relation pairs → empty string and empty params."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": []}}
        status = {}
        cypher, params = converter.to_db_aware_query(relations, status)
        assert cypher == ""
        assert params == {}

    def test_no_labels_without_schema(self):
        """Without schema, labels are empty strings → no label in pattern."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()  # no schema
        relations = {"relation_extraction": {"acquired": [("Bob", "TechCorp")]}}
        status = self._make_status([
            ("Bob",     "e0", "", False),
            ("TechCorp","e1", "", False),
        ])
        cypher, params = converter.to_db_aware_query(relations, status)

        assert ":Person" not in cypher
        assert "CREATE (e0 {name: $e0_val})-[:ACQUIRED]->(e1 {name: $e1_val})" in cypher

    def test_custom_return_clause(self):
        """return_clause kwarg overrides the auto-generated RETURN."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {"works_for": [("A", "B")]}}
        status = self._make_status([("A", "e0", "", False), ("B", "e1", "", False)])
        cypher, _ = converter.to_db_aware_query(
            relations, status, return_clause="RETURN *"
        )
        assert cypher.endswith("RETURN *")

    def test_deduplication_same_triple(self):
        """Duplicate (subject, object, rel) pairs generate only one clause."""
        from cypher_validator import RelationToCypherConverter
        converter = RelationToCypherConverter()
        relations = {"relation_extraction": {
            "works_for": [("John", "Apple"), ("John", "Apple")]
        }}
        status = self._make_status([
            ("John",  "e0", "", False),
            ("Apple", "e1", "", False),
        ])
        cypher, _ = converter.to_db_aware_query(relations, status)
        assert cypher.count("CREATE") == 1


# ---------------------------------------------------------------------------
# NLToCypher._collect_entity_status — unit tests
# ---------------------------------------------------------------------------

class TestCollectEntityStatus:
    def test_entities_assigned_unique_vars(self):
        """Each unique entity gets its own eN variable."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({})  # not used in this test
        db = _make_db(records_map={})  # none found
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        relations = {"relation_extraction": {
            "works_for": [("John", "Apple"), ("Alice", "Apple")]
        }}
        status = pipeline._collect_entity_status("...", relations, db)

        vars_ = [info["var"] for info in status.values()]
        assert len(set(vars_)) == len(vars_), "Variable names must be unique"
        # 3 unique entities: John, Apple, Alice
        assert len(status) == 3

    def test_schema_labels_resolved(self):
        """Entity labels are populated from schema rel endpoints."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({})
        db = _make_db()
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = pipeline._collect_entity_status("John works for Apple", relations, db)

        assert status["John"]["label"] == "Person"
        assert status["Apple"]["label"] == "Company"

    def test_found_true_when_entity_exists(self):
        """found=True when DB returns a non-empty result for the entity."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({})
        # John exists; Apple does not
        db = _make_db(records_map={"John": [{"id": "abc"}], "Apple": []})
        pipeline = NLToCypher(extractor, db=db)

        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = pipeline._collect_entity_status("...", relations, db)

        assert status["John"]["found"] is True
        assert status["Apple"]["found"] is False

    def test_db_error_marks_not_found(self):
        """If DB raises during lookup, entity is conservatively marked not found."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({})
        db = MagicMock()
        db.execute.side_effect = Exception("connection error")
        pipeline = NLToCypher(extractor, db=db)

        relations = {"relation_extraction": {"works_for": [("John", "Apple")]}}
        status = pipeline._collect_entity_status("...", relations, db)

        assert status["John"]["found"] is False
        assert status["Apple"]["found"] is False

    def test_ner_extractor_enriches_labels(self):
        """NER extractor labels override schema-resolved labels."""
        from cypher_validator import NLToCypher, EntityNERExtractor

        # Mock NER extractor
        ner = MagicMock(spec=EntityNERExtractor)
        ner.extract.return_value = [
            {"text": "John", "label": "Employee"},
            {"text": "Acme Corp", "label": "Employer"},
        ]

        extractor = _make_extractor({})
        db = _make_db()
        pipeline = NLToCypher(extractor, db=db, ner_extractor=ner)

        relations = {"relation_extraction": {"works_for": [("John", "Acme Corp")]}}
        status = pipeline._collect_entity_status("John works for Acme Corp", relations, db)

        assert status["John"]["label"] == "Employee"
        assert status["Acme Corp"]["label"] == "Employer"
        ner.extract.assert_called_once_with("John works for Acme Corp")


# ---------------------------------------------------------------------------
# NLToCypher.__call__ with db_aware=True
# ---------------------------------------------------------------------------

class TestNLToCypherDbAware:
    def test_db_aware_no_db_raises(self):
        """RuntimeError when db_aware=True but no db provided."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        pipeline = NLToCypher(extractor, db=None)

        with pytest.raises(RuntimeError, match="db_aware=True requires a database"):
            pipeline("text", ["works_for"], db_aware=True)

    def test_db_aware_returns_string_by_default(self):
        """db_aware=True without execute=True returns a str."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        db = _make_db()
        pipeline = NLToCypher(extractor, db=db)

        result = pipeline("John works for Apple", ["works_for"], db_aware=True)
        assert isinstance(result, str)
        assert "CREATE" in result or "MATCH" in result

    def test_db_aware_execute_returns_tuple(self):
        """db_aware=True + execute=True returns (str, list)."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        db = _make_db(records_map={"John": [{"id": "x"}], "Apple": []})
        db.execute.side_effect = None  # reset side_effect, use return_value
        db.execute.return_value = []

        # Re-mock with controlled side effect
        call_count = [0]
        def _side(cypher, params=None):
            call_count[0] += 1
            val = (params or {}).get("val", "")
            if val == "John":
                return [{"id": "x"}]
            return []

        db.execute.side_effect = _side
        pipeline = NLToCypher(extractor, db=db)

        result = pipeline("John works for Apple", ["works_for"],
                          db_aware=True, execute=True)
        assert isinstance(result, tuple)
        cypher, records = result
        assert isinstance(cypher, str)
        assert isinstance(records, list)

    def test_john_exists_apple_new(self):
        """John exists → MATCHed; Apple is new → CREATEd inline."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({"works_for": [("John", "Apple Inc.")]})

        def _db_lookup(cypher, params=None):
            val = (params or {}).get("val", "")
            if val == "John":
                return [{"id": "elem-99"}]
            return []

        db = MagicMock()
        db.execute.side_effect = _db_lookup
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        cypher = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            db_aware=True,
        )

        assert "MATCH (e0:Person {name: $e0_val})" in cypher
        assert "CREATE (e0)-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher
        assert "RETURN e0, e1" in cypher

    def test_john_and_apple_both_new(self):
        """Neither exists → single CREATE with both inline."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({"works_for": [("John", "Apple Inc.")]})
        db = _make_db()  # nothing found
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        cypher = pipeline("John works for Apple Inc.", ["works_for"], db_aware=True)

        assert "MATCH" not in cypher
        assert "CREATE (e0:Person {name: $e0_val})-[:WORKS_FOR]->(e1:Company {name: $e1_val})" in cypher

    def test_both_entities_exist(self):
        """Both in DB → two MATCH clauses, CREATE only the edge."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({"works_for": [("John", "Apple")]})

        def _db_lookup(cypher, params=None):
            return [{"id": "x"}]  # all found

        db = MagicMock()
        db.execute.side_effect = _db_lookup
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        cypher = pipeline("John works for Apple", ["works_for"], db_aware=True)

        assert "MATCH (e0:Person" in cypher
        assert "MATCH (e1:Company" in cypher
        assert "CREATE (e0)-[:WORKS_FOR]->(e1)" in cypher
        # Existing vars must not have inline properties in the CREATE
        assert "e0:Person {name:" not in cypher.split("CREATE")[1]

    def test_multiple_relations_shared_subject(self):
        """Reproduce the user's example: John exists, Apple and SF are new."""
        from cypher_validator import NLToCypher, Schema

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"], "City": ["name"]},
            relationships={
                "WORKS_FOR": ("Person", "Company", []),
                "LIVES_IN":  ("Person", "City", []),
            },
        )
        extractor = _make_extractor({
            "works_for": [("John", "Apple Inc.")],
            "lives_in":  [("John", "San Francisco")],
        })

        def _db_lookup(cypher, params=None):
            val = (params or {}).get("val", "")
            return [{"id": "elem-99"}] if val == "John" else []

        db = MagicMock()
        db.execute.side_effect = _db_lookup
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        cypher = pipeline(
            "John works for Apple Inc. and lives in San Francisco.",
            ["works_for", "lives_in"],
            db_aware=True,
        )

        # John is MATCHed (exists) — exactly once
        assert cypher.count("MATCH (e0:Person") == 1
        # Relationships reference e0 without re-declaring
        assert "CREATE (e0)-[:WORKS_FOR]->" in cypher
        assert "CREATE (e0)-[:LIVES_IN]->" in cypher
        # Apple and SF are created inline
        assert "Company {name: $" in cypher
        assert "City {name: $" in cypher

    def test_db_aware_empty_extraction(self):
        """No relations extracted → returns empty string."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": []})
        db = _make_db()
        pipeline = NLToCypher(extractor, db=db)

        cypher = pipeline("text", ["works_for"], db_aware=True)
        assert cypher == ""

    def test_db_aware_no_schema_no_labels(self):
        """Without schema, labels are empty — query still generated."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"acquired": [("Bob", "TechCorp")]})
        db = _make_db()  # nothing found
        pipeline = NLToCypher(extractor, db=db)

        cypher = pipeline("Bob acquired TechCorp", ["acquired"], db_aware=True)

        assert "CREATE" in cypher
        assert ":ACQUIRED" in cypher
        # No labels since no schema
        assert "Person" not in cypher
        assert "Company" not in cypher


# ---------------------------------------------------------------------------
# NLToCypher.extract_and_convert with db_aware
# ---------------------------------------------------------------------------

class TestExtractAndConvertDbAware:
    def test_db_aware_false_execute_false(self):
        """db_aware=False, execute=False → (dict, str) unchanged."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("A", "B")]})
        pipeline = NLToCypher(extractor)

        out = pipeline.extract_and_convert("text", ["works_for"], mode="create")
        assert len(out) == 2
        relations, cypher = out
        assert "CREATE" in cypher

    def test_db_aware_true_execute_false_returns_two_tuple(self):
        """db_aware=True, execute=False → (dict, str) with MATCH/CREATE query."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        db = _make_db(records_map={"John": [{"id": "x"}]})

        def _db_lookup(cypher, params=None):
            val = (params or {}).get("val", "")
            return [{"id": "x"}] if val == "John" else []

        db.execute.side_effect = _db_lookup
        pipeline = NLToCypher(extractor, db=db)

        out = pipeline.extract_and_convert("text", ["works_for"], db_aware=True)
        assert len(out) == 2
        relations, cypher = out
        assert "MATCH" in cypher
        assert "CREATE" in cypher

    def test_db_aware_true_execute_true_returns_three_tuple(self):
        """db_aware=True, execute=True → (dict, str, list)."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        db = MagicMock()
        db.execute.return_value = []
        pipeline = NLToCypher(extractor, db=db)

        out = pipeline.extract_and_convert(
            "text", ["works_for"], db_aware=True, execute=True
        )
        assert len(out) == 3
        relations, cypher, records = out
        assert "relation_extraction" in relations
        assert isinstance(cypher, str)
        assert isinstance(records, list)

    def test_db_aware_no_db_raises(self):
        """RuntimeError in extract_and_convert when db_aware=True but no db."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"works_for": [("A", "B")]})
        pipeline = NLToCypher(extractor, db=None)

        with pytest.raises(RuntimeError, match="db_aware=True requires a database"):
            pipeline.extract_and_convert("text", ["works_for"], db_aware=True)


# ---------------------------------------------------------------------------
# EntityNERExtractor
# ---------------------------------------------------------------------------

class TestEntityNERExtractorSpacy:
    def test_from_spacy_import_error(self):
        """ImportError when spacy is not installed."""
        import sys
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("No module named 'spacy'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from cypher_validator.gliner2_integration import EntityNERExtractor as _NER
            with pytest.raises(ImportError, match="spacy"):
                _NER.from_spacy("en_core_web_sm")

    def test_extract_spacy_maps_labels(self):
        """Spacy entities are mapped through label_map to graph labels."""
        from cypher_validator import EntityNERExtractor

        # Mock spaCy model and doc
        mock_ent1 = MagicMock()
        mock_ent1.text = "John"
        mock_ent1.label_ = "PERSON"
        mock_ent2 = MagicMock()
        mock_ent2.text = "Apple Inc."
        mock_ent2.label_ = "ORG"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2]

        mock_nlp = MagicMock(return_value=mock_doc)

        ner = EntityNERExtractor(mock_nlp, backend="spacy",
                                 label_map=EntityNERExtractor._SPACY_DEFAULTS)
        results = ner.extract("John works for Apple Inc.")

        assert results[0] == {"text": "John", "label": "Person"}
        assert results[1] == {"text": "Apple Inc.", "label": "Organization"}

    def test_extract_unknown_label_capitalized(self):
        """Unknown labels are capitalized and returned as-is."""
        from cypher_validator import EntityNERExtractor

        mock_ent = MagicMock()
        mock_ent.text = "Aspirin"
        mock_ent.label_ = "CHEMICAL"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)
        ner = EntityNERExtractor(mock_nlp, backend="spacy", label_map={})
        results = ner.extract("Aspirin is a drug.")
        assert results[0]["label"] == "Chemical"

    def test_repr(self):
        from cypher_validator import EntityNERExtractor
        ner = EntityNERExtractor(MagicMock(), backend="spacy")
        assert "spacy" in repr(ner)


class TestEntityNERExtractorTransformers:
    def test_from_transformers_import_error(self):
        """ImportError when transformers is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from cypher_validator.gliner2_integration import EntityNERExtractor as _NER
            with pytest.raises(ImportError, match="transformers"):
                _NER.from_transformers("some-model")

    def test_extract_transformers_maps_labels(self):
        """HuggingFace NER output is mapped through label_map."""
        from cypher_validator import EntityNERExtractor

        mock_pipe = MagicMock(return_value=[
            {"entity_group": "PER", "word": "John", "score": 0.99},
            {"entity_group": "ORG", "word": "Apple", "score": 0.98},
        ])
        ner = EntityNERExtractor(
            mock_pipe,
            backend="transformers",
            label_map=EntityNERExtractor._HF_DEFAULTS,
        )
        results = ner.extract("John works for Apple.")
        assert results[0] == {"text": "John", "label": "Person"}
        assert results[1] == {"text": "Apple", "label": "Organization"}

    def test_unknown_backend_raises(self):
        """ValueError for unrecognised backend."""
        from cypher_validator import EntityNERExtractor
        ner = EntityNERExtractor(MagicMock(), backend="unknown")
        with pytest.raises(ValueError, match="Unknown backend"):
            ner.extract("text")


# ---------------------------------------------------------------------------
# NLToCypher constructor / factory with ner_extractor
# ---------------------------------------------------------------------------

class TestNLToCypherNerExtractor:
    def test_init_stores_ner_extractor(self):
        """ner_extractor is stored on the pipeline instance."""
        from cypher_validator import NLToCypher, EntityNERExtractor
        extractor = _make_extractor({})
        ner = MagicMock(spec=EntityNERExtractor)
        pipeline = NLToCypher(extractor, ner_extractor=ner)
        assert pipeline.ner_extractor is ner

    def test_init_ner_extractor_none_by_default(self):
        """ner_extractor defaults to None."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({})
        pipeline = NLToCypher(extractor)
        assert pipeline.ner_extractor is None

    def test_from_pretrained_stores_ner_extractor(self):
        """from_pretrained forwards ner_extractor to the instance."""
        from cypher_validator import NLToCypher, EntityNERExtractor, GLiNER2RelationExtractor
        ner = MagicMock(spec=EntityNERExtractor)

        with patch(
            "cypher_validator.gliner2_integration.GLiNER2RelationExtractor.from_pretrained"
        ) as mock_fp:
            mock_fp.return_value = MagicMock(spec=GLiNER2RelationExtractor)
            pipeline = NLToCypher.from_pretrained(
                "fastino/gliner2-large-v1",
                ner_extractor=ner,
            )
        assert pipeline.ner_extractor is ner

    def test_repr_includes_ner_extractor(self):
        """__repr__ mentions ner_extractor."""
        from cypher_validator import NLToCypher, EntityNERExtractor
        extractor = _make_extractor({})
        ner = MagicMock(spec=EntityNERExtractor)
        ner.__repr__ = lambda s: "EntityNERExtractor(backend='spacy')"
        pipeline = NLToCypher(extractor, ner_extractor=ner)
        assert "ner_extractor" in repr(pipeline)
