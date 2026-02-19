"""
Tests for Neo4j database-aware execution in NLToCypher.

All tests mock the neo4j driver and the GLiNER2 model — no real
Neo4j instance or ML model is required.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(relations: dict):
    """Return a mock GLiNER2RelationExtractor that returns *relations*."""
    from cypher_validator import GLiNER2RelationExtractor
    extractor = MagicMock(spec=GLiNER2RelationExtractor)
    extractor.extract_relations.return_value = {"relation_extraction": relations}
    return extractor


def _make_db(records: list = None):
    """Return a mock Neo4jDatabase whose execute() returns *records*."""
    from cypher_validator import Neo4jDatabase
    db = MagicMock(spec=Neo4jDatabase)
    db.execute.return_value = records if records is not None else []
    return db


# ---------------------------------------------------------------------------
# Neo4jDatabase construction
# ---------------------------------------------------------------------------

class TestNeo4jDatabaseConstruction:
    def test_raises_import_error_without_neo4j(self):
        """ImportError when the neo4j package is not installed."""
        import sys
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "neo4j":
                raise ImportError("No module named 'neo4j'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from cypher_validator.gliner2_integration import Neo4jDatabase as _DB
            with pytest.raises(ImportError, match="neo4j"):
                _DB("bolt://localhost:7687", "neo4j", "password")

    def test_construction_with_mock_driver(self):
        """Constructor stores database name and creates driver."""
        mock_driver_instance = MagicMock()
        with patch("neo4j.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = mock_driver_instance
            with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
                from cypher_validator.gliner2_integration import Neo4jDatabase as _DB
                db = _DB("bolt://localhost:7687", "neo4j", "secret", database="mydb")
                assert db._database == "mydb"

    def test_repr(self):
        db = _make_db()
        # MagicMock spec repr is not the real repr, so test on a real instance via repr check
        from cypher_validator import Neo4jDatabase
        db2 = MagicMock(spec=Neo4jDatabase)
        db2.__repr__ = lambda self: "Neo4jDatabase(database='neo4j')"
        assert "Neo4jDatabase" in repr(db2)


# ---------------------------------------------------------------------------
# Neo4jDatabase.execute
# ---------------------------------------------------------------------------

class TestNeo4jDatabaseExecute:
    def test_execute_returns_list_of_dicts(self):
        """execute() returns the records from the session."""
        db = _make_db(records=[{"n": {"name": "Alice"}}])
        result = db.execute("MATCH (n:Person) RETURN n")
        assert result == [{"n": {"name": "Alice"}}]

    def test_execute_with_parameters(self):
        """execute() forwards parameters to the underlying driver."""
        db = _make_db(records=[])
        db.execute("MATCH (n:Person {name: $name}) RETURN n", {"name": "Alice"})
        db.execute.assert_called_once_with(
            "MATCH (n:Person {name: $name}) RETURN n", {"name": "Alice"}
        )

    def test_execute_empty_result(self):
        """execute() returns [] for queries with no RETURN rows."""
        db = _make_db(records=[])
        result = db.execute("CREATE (n:Person {name: 'Bob'})")
        assert result == []

    def test_context_manager(self):
        """Neo4jDatabase.__exit__ calls close() on a real instance."""
        mock_driver = MagicMock()
        mock_neo4j = MagicMock()
        mock_neo4j.GraphDatabase.driver.return_value = mock_driver

        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            # Re-import inside the patch so GraphDatabase resolves to the mock
            import importlib
            import cypher_validator.gliner2_integration as _mod
            importlib.reload(_mod)
            db = _mod.Neo4jDatabase("bolt://localhost:7687", "neo4j", "secret")
            with db:
                pass
            mock_driver.close.assert_called_once()
            # Restore module to original state
            importlib.reload(_mod)


# ---------------------------------------------------------------------------
# NLToCypher without execute flag (backward compatibility)
# ---------------------------------------------------------------------------

class TestNLToCypherWithoutExecute:
    def test_call_returns_string(self):
        """__call__ returns str when execute=False (default)."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        pipeline = NLToCypher(extractor)
        result = pipeline("John works for Apple.", ["works_for"], mode="create")
        assert isinstance(result, str)
        assert "CREATE" in result
        assert ":WORKS_FOR" in result

    def test_call_explicit_execute_false(self):
        """execute=False explicitly still returns str."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        pipeline = NLToCypher(extractor)
        result = pipeline("text", ["works_for"], mode="create", execute=False)
        assert isinstance(result, str)

    def test_extract_and_convert_returns_two_tuple(self):
        """extract_and_convert returns (dict, str) when execute=False."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("John", "Apple")]})
        pipeline = NLToCypher(extractor)
        out = pipeline.extract_and_convert("text", ["works_for"], mode="create")
        assert len(out) == 2
        relations, cypher = out
        assert "relation_extraction" in relations
        assert isinstance(cypher, str)

    def test_db_not_required_when_execute_false(self):
        """No db needed when execute=False."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"lives_in": [("Bob", "Paris")]})
        pipeline = NLToCypher(extractor, db=None)
        result = pipeline("text", ["lives_in"], mode="merge")
        assert "MERGE" in result


# ---------------------------------------------------------------------------
# NLToCypher with execute=True (db-aware mode)
# ---------------------------------------------------------------------------

class TestNLToCypherWithExecute:
    def test_execute_true_returns_tuple(self):
        """__call__ with execute=True returns (str, list)."""
        from cypher_validator import NLToCypher, Schema
        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        extractor = _make_extractor({"works_for": [("John", "Apple Inc.")]})
        db = _make_db(records=[{"a0": {"name": "John"}, "b0": {"name": "Apple Inc."}}])
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        result = pipeline("John works for Apple Inc.", ["works_for"],
                          mode="create", execute=True)

        assert isinstance(result, tuple)
        cypher, records = result
        assert isinstance(cypher, str)
        assert isinstance(records, list)
        assert "CREATE" in cypher
        assert ":Person" in cypher
        assert ":WORKS_FOR" in cypher

    def test_execute_true_calls_db_with_generated_query(self):
        """The exact generated Cypher is passed to db.execute()."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"lives_in": [("John", "San Francisco")]})
        db = _make_db(records=[])
        pipeline = NLToCypher(extractor, db=db)

        cypher, _ = pipeline("text", ["lives_in"], mode="create", execute=True)
        db.execute.assert_called_once()
        assert db.execute.call_args[0][0] == cypher

    def test_execute_true_no_db_raises(self):
        """RuntimeError when execute=True but db is None."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("Alice", "Corp")]})
        pipeline = NLToCypher(extractor, db=None)

        with pytest.raises(RuntimeError, match="database connection"):
            pipeline("text", ["works_for"], mode="create", execute=True)

    def test_extract_and_convert_execute_true_returns_three_tuple(self):
        """extract_and_convert returns (dict, str, list) when execute=True."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"acquired": [("Bob", "TechCorp")]})
        db = _make_db(records=[])
        pipeline = NLToCypher(extractor, db=db)

        out = pipeline.extract_and_convert("text", ["acquired"],
                                           mode="create", execute=True)
        assert len(out) == 3
        relations, cypher, records = out
        assert "relation_extraction" in relations
        assert "CREATE" in cypher
        assert isinstance(records, list)

    def test_extract_and_convert_execute_true_no_db_raises(self):
        """RuntimeError in extract_and_convert when execute=True but no db."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": [("Alice", "Corp")]})
        pipeline = NLToCypher(extractor, db=None)

        with pytest.raises(RuntimeError, match="database connection"):
            pipeline.extract_and_convert("text", ["works_for"],
                                         mode="create", execute=True)

    def test_empty_extraction_returns_empty_results(self):
        """When no relations are extracted, db.execute is skipped."""
        from cypher_validator import NLToCypher
        extractor = _make_extractor({"works_for": []})   # nothing found
        db = _make_db(records=[])
        pipeline = NLToCypher(extractor, db=db)

        cypher, records = pipeline("text", ["works_for"], mode="create", execute=True)
        assert cypher == ""
        assert records == []
        db.execute.assert_not_called()   # empty query → skip execution


# ---------------------------------------------------------------------------
# NLToCypher.from_pretrained passes db through
# ---------------------------------------------------------------------------

class TestNLToCypherFromPretrainedWithDb:
    def test_from_pretrained_stores_db(self):
        """db kwarg is stored on the pipeline instance."""
        from cypher_validator import NLToCypher, Neo4jDatabase
        mock_model = MagicMock()
        db = _make_db()

        with patch(
            "cypher_validator.gliner2_integration.GLiNER2RelationExtractor.from_pretrained"
        ) as mock_fp:
            from cypher_validator import GLiNER2RelationExtractor
            mock_fp.return_value = MagicMock(spec=GLiNER2RelationExtractor)
            pipeline = NLToCypher.from_pretrained(
                "fastino/gliner2-large-v1",
                db=db,
            )
            assert pipeline.db is db

    def test_from_pretrained_db_none_by_default(self):
        """db defaults to None when not provided."""
        with patch(
            "cypher_validator.gliner2_integration.GLiNER2RelationExtractor.from_pretrained"
        ) as mock_fp:
            from cypher_validator import GLiNER2RelationExtractor, NLToCypher
            mock_fp.return_value = MagicMock(spec=GLiNER2RelationExtractor)
            pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1")
            assert pipeline.db is None


# ---------------------------------------------------------------------------
# Realistic end-to-end scenarios (all mocked)
# ---------------------------------------------------------------------------

class TestRealisticScenarios:
    def test_john_works_for_apple_lives_in_sf(self):
        """Reproduce the example from the task description."""
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
        db = _make_db(records=[])
        pipeline = NLToCypher(extractor, schema=schema, db=db)

        cypher, results = pipeline(
            "John works for Apple Inc. and lives in San Francisco.",
            ["works_for", "lives_in"],
            mode="create",
            execute=True,
        )

        assert 'CREATE (a0:Person {name: $a0_val})-[:WORKS_FOR]->(b0:Company {name: $b0_val})' in cypher
        assert 'CREATE (a1:Person {name: $a1_val})-[:LIVES_IN]->(b1:City {name: $b1_val})' in cypher
        assert "RETURN a0, b0, a1, b1" in cypher
        db.execute.assert_called_once()
        call_args = db.execute.call_args[0]
        assert call_args[0] == cypher
        assert call_args[1]["a0_val"] == "John"
        assert call_args[1]["b0_val"] == "Apple Inc."

    def test_bob_acquired_techcorp_no_schema(self):
        """No schema → no labels, but query is still generated and executed."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"acquired": [("Bob", "TechCorp")]})
        db = _make_db(records=[])
        pipeline = NLToCypher(extractor, db=db)   # no schema

        cypher, results = pipeline(
            "Bob acquired TechCorp in 2019.",
            ["acquired"],
            mode="create",
            execute=True,
        )

        assert 'CREATE (a0 {name: $a0_val})-[:ACQUIRED]->(b0 {name: $b0_val})' in cypher
        assert "RETURN a0, b0" in cypher

    def test_merge_mode_upserts(self):
        """mode='merge' generates MERGE clauses and executes them."""
        from cypher_validator import NLToCypher

        extractor = _make_extractor({"friends_with": [("Alice", "Bob")]})
        db = _make_db(records=[{"a0": {}, "b0": {}}])
        pipeline = NLToCypher(extractor, db=db)

        cypher, records = pipeline("Alice is friends with Bob.",
                                   ["friends_with"], mode="merge", execute=True)

        assert "MERGE" in cypher
        assert len(records) == 1
