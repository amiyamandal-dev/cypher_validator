"""
Tests for GraphRAGPipeline — all LLM and Neo4j calls are mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

from cypher_validator import Schema, GraphRAGPipeline, Neo4jDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Company": ["name"]},
        relationships={"WORKS_FOR": ("Person", "Company", ["since"])},
    )


def _make_db(records=None):
    db = MagicMock(spec=Neo4jDatabase)
    db.execute.return_value = records if records is not None else []
    return db


def _make_llm(*responses):
    """Return an LLM mock that cycles through *responses* on successive calls."""
    iterator = iter(responses)
    return lambda prompt: next(iterator)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGraphRAGPipelineConstruction:
    def test_stores_attributes(self):
        schema = _make_schema()
        db = _make_db()
        llm = lambda p: ""
        pipeline = GraphRAGPipeline(schema=schema, db=db, llm_fn=llm)
        assert pipeline.schema is schema
        assert pipeline.db is db
        assert pipeline.llm_fn is llm
        assert pipeline.max_repair_retries == 2
        assert pipeline.result_format == "markdown"

    def test_custom_retries_and_format(self):
        pipeline = GraphRAGPipeline(
            schema=_make_schema(),
            db=_make_db(),
            llm_fn=lambda p: "",
            max_repair_retries=5,
            result_format="json",
        )
        assert pipeline.max_repair_retries == 5
        assert pipeline.result_format == "json"

    def test_custom_system_prompts(self):
        pipeline = GraphRAGPipeline(
            schema=_make_schema(),
            db=_make_db(),
            llm_fn=lambda p: "",
            cypher_system_prompt="Custom cypher prompt",
            answer_system_prompt="Custom answer prompt",
        )
        assert pipeline._cypher_system == "Custom cypher prompt"
        assert pipeline._answer_system == "Custom answer prompt"

    def test_repr(self):
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=_make_db(), llm_fn=lambda p: "")
        assert "GraphRAGPipeline" in repr(pipeline)

    def test_default_cypher_prompt_contains_schema(self):
        schema = _make_schema()
        pipeline = GraphRAGPipeline(schema=schema, db=_make_db(), llm_fn=lambda p: "")
        assert "Person" in pipeline._cypher_system
        assert "WORKS_FOR" in pipeline._cypher_system


# ---------------------------------------------------------------------------
# query() — happy path
# ---------------------------------------------------------------------------

class TestGraphRAGPipelineQuery:
    def test_returns_string(self):
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n\n```",   # Cypher generation
            "Here are the people.",                          # answer generation
        )
        pipeline = GraphRAGPipeline(
            schema=_make_schema(),
            db=_make_db(records=[{"n": {"name": "Alice"}}]),
            llm_fn=llm,
        )
        answer = pipeline.query("List all people")
        assert isinstance(answer, str)
        assert answer == "Here are the people."

    def test_llm_called_twice(self):
        calls = []
        def llm(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return "```cypher\nMATCH (n:Person) RETURN n\n```"
            return "Final answer."

        pipeline = GraphRAGPipeline(schema=_make_schema(), db=_make_db(), llm_fn=llm)
        pipeline.query("Who are the people?")
        assert len(calls) == 2

    def test_db_execute_called_with_extracted_cypher(self):
        db = _make_db(records=[])
        llm = _make_llm(
            "```cypher\nMATCH (p:Person) RETURN p\n```",
            "No one.",
        )
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=db, llm_fn=llm)
        pipeline.query("Find people")
        db.execute.assert_called_once()
        called_cypher = db.execute.call_args[0][0]
        assert "MATCH" in called_cypher
        assert "Person" in called_cypher

    def test_empty_records_handled(self):
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n\n```",
            "Nobody found.",
        )
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=_make_db([]), llm_fn=llm)
        ctx = pipeline.query_with_context("Find anyone")
        assert ctx["records"] == []
        assert ctx["formatted_results"] == ""

    def test_query_with_context_keys(self):
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n\n```",
            "Alice works there.",
        )
        pipeline = GraphRAGPipeline(
            schema=_make_schema(),
            db=_make_db([{"name": "Alice"}]),
            llm_fn=llm,
        )
        ctx = pipeline.query_with_context("Who is Alice?")
        expected_keys = {
            "question", "cypher", "is_valid", "validation_errors",
            "repair_attempts", "records", "formatted_results",
            "answer", "execution_error",
        }
        assert expected_keys <= set(ctx.keys())

    def test_cypher_extracted_from_fence(self):
        llm = _make_llm(
            "Sure:\n```cypher\nMATCH (n:Person) RETURN n\n```\nThat's it.",
            "Done.",
        )
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=_make_db(), llm_fn=llm)
        ctx = pipeline.query_with_context("test")
        assert ctx["cypher"] == "MATCH (n:Person) RETURN n"


# ---------------------------------------------------------------------------
# Validation + repair loop
# ---------------------------------------------------------------------------

class TestGraphRAGRepairLoop:
    def test_valid_query_zero_repair_attempts(self):
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n\n```",
            "Answer.",
        )
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=_make_db(), llm_fn=llm)
        ctx = pipeline.query_with_context("test")
        assert ctx["repair_attempts"] == 0
        assert ctx["is_valid"] is True

    def test_invalid_query_triggers_repair(self):
        calls = []
        def llm(prompt):
            calls.append(prompt)
            n = len(calls)
            if n == 1:
                return "MATCH (n:NonExistent) RETURN n"    # invalid
            if n == 2:
                return "```cypher\nMATCH (n:Person) RETURN n\n```"  # repaired
            return "Final answer."

        pipeline = GraphRAGPipeline(
            schema=_make_schema(),
            db=_make_db(),
            llm_fn=llm,
            max_repair_retries=2,
        )
        ctx = pipeline.query_with_context("test")
        assert ctx["repair_attempts"] == 1
        assert ctx["is_valid"] is True

    def test_repair_prompt_contains_errors(self):
        prompts = []
        call_n = [0]

        def llm(prompt):
            prompts.append(prompt)
            call_n[0] += 1
            if call_n[0] == 1:
                return "MATCH (n:Ghost) RETURN n"
            if call_n[0] == 2:
                return "```cypher\nMATCH (n:Person) RETURN n\n```"
            return "Answer."

        pipeline = GraphRAGPipeline(
            schema=_make_schema(), db=_make_db(), llm_fn=llm, max_repair_retries=1
        )
        pipeline.query_with_context("test")
        # Second prompt (repair) should contain the error message
        repair_prompt = prompts[1]
        assert "Ghost" in repair_prompt or "error" in repair_prompt.lower()

    def test_repair_respects_max_retries(self):
        calls = [0]

        def llm(prompt):
            calls[0] += 1
            if calls[0] <= 1:
                return "MATCH (n:Bad1) RETURN n"   # always invalid label
            if calls[0] <= 2:
                return "MATCH (n:Bad2) RETURN n"
            return "Answer."

        pipeline = GraphRAGPipeline(
            schema=_make_schema(), db=_make_db(), llm_fn=llm, max_repair_retries=1
        )
        ctx = pipeline.query_with_context("test")
        assert ctx["repair_attempts"] == 1
        # Still executed (even if invalid) — execution may or may not error

    def test_no_repair_when_max_retries_zero(self):
        call_n = [0]

        def llm(prompt):
            call_n[0] += 1
            if call_n[0] == 1:
                return "MATCH (n:Fake) RETURN n"
            return "Answer."

        pipeline = GraphRAGPipeline(
            schema=_make_schema(), db=_make_db(), llm_fn=llm, max_repair_retries=0
        )
        ctx = pipeline.query_with_context("test")
        assert ctx["repair_attempts"] == 0


# ---------------------------------------------------------------------------
# Execution error handling
# ---------------------------------------------------------------------------

class TestGraphRAGExecutionError:
    def test_execution_error_captured(self):
        db = MagicMock(spec=Neo4jDatabase)
        db.execute.side_effect = RuntimeError("Connection refused")

        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n\n```",
            "There was an error.",
        )
        pipeline = GraphRAGPipeline(schema=_make_schema(), db=db, llm_fn=llm)
        ctx = pipeline.query_with_context("test")
        assert ctx["execution_error"] is not None
        assert "Connection refused" in ctx["execution_error"]
        assert ctx["records"] == []

    def test_execution_error_mentioned_in_answer_prompt(self):
        db = MagicMock(spec=Neo4jDatabase)
        db.execute.side_effect = RuntimeError("DB is down")

        prompts = []
        call_n = [0]

        def llm(prompt):
            prompts.append(prompt)
            call_n[0] += 1
            if call_n[0] == 1:
                return "```cypher\nMATCH (n:Person) RETURN n\n```"
            return "Error occurred."

        pipeline = GraphRAGPipeline(schema=_make_schema(), db=db, llm_fn=llm)
        pipeline.query_with_context("test")
        answer_prompt = prompts[-1]
        assert "DB is down" in answer_prompt or "failed" in answer_prompt.lower()


# ---------------------------------------------------------------------------
# Result formats
# ---------------------------------------------------------------------------

class TestGraphRAGResultFormats:
    def test_markdown_format_in_context(self):
        db = _make_db(records=[{"name": "Alice"}, {"name": "Bob"}])
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n.name AS name\n```",
            "Answer.",
        )
        pipeline = GraphRAGPipeline(
            schema=_make_schema(), db=db, llm_fn=llm, result_format="markdown"
        )
        ctx = pipeline.query_with_context("test")
        assert "|" in ctx["formatted_results"]

    def test_json_format_in_context(self):
        db = _make_db(records=[{"name": "Alice"}])
        llm = _make_llm(
            "```cypher\nMATCH (n:Person) RETURN n.name AS name\n```",
            "Answer.",
        )
        pipeline = GraphRAGPipeline(
            schema=_make_schema(), db=db, llm_fn=llm, result_format="json"
        )
        ctx = pipeline.query_with_context("test")
        import json
        parsed = json.loads(ctx["formatted_results"])
        assert parsed == [{"name": "Alice"}]

    def test_results_injected_into_answer_prompt(self):
        db = _make_db(records=[{"name": "Alice"}])
        prompts = []
        call_n = [0]

        def llm(prompt):
            prompts.append(prompt)
            call_n[0] += 1
            if call_n[0] == 1:
                return "```cypher\nMATCH (n:Person) RETURN n.name AS name\n```"
            return "Alice."

        pipeline = GraphRAGPipeline(schema=_make_schema(), db=db, llm_fn=llm)
        pipeline.query_with_context("test")
        answer_prompt = prompts[-1]
        assert "Alice" in answer_prompt


# ---------------------------------------------------------------------------
# introspect_schema and execute_and_format (Neo4jDatabase extensions)
# ---------------------------------------------------------------------------

class TestNeo4jDatabaseExtensions:
    def _make_real_db(self, node_rows=None, rel_rows=None):
        """Create a Neo4jDatabase mock with controlled execute() responses."""
        from unittest.mock import patch, MagicMock

        mock_neo4j = MagicMock()
        mock_neo4j.GraphDatabase.driver.return_value = MagicMock()

        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            import importlib
            import cypher_validator.gliner2_integration as mod
            importlib.reload(mod)
            db = mod.Neo4jDatabase("bolt://localhost", "neo4j", "pw")
            # Provide scripted execute() responses
            responses = []
            if node_rows is not None:
                responses.append(node_rows)
            if rel_rows is not None:
                responses.append(rel_rows)
            db.execute = MagicMock(side_effect=responses if responses else [[], []])
            importlib.reload(mod)
            return db

    def test_execute_and_format_markdown(self):
        from cypher_validator import Neo4jDatabase
        db = MagicMock(spec=Neo4jDatabase)
        db.execute.return_value = [{"name": "Alice"}, {"name": "Bob"}]

        from cypher_validator.llm_utils import format_records
        records = db.execute("MATCH (n) RETURN n.name AS name")
        result = format_records(records, format="markdown")
        assert "|" in result
        assert "Alice" in result

    def test_execute_and_format_empty(self):
        from cypher_validator import Neo4jDatabase
        db = MagicMock(spec=Neo4jDatabase)
        db.execute.return_value = []

        from cypher_validator.llm_utils import format_records
        records = db.execute("MATCH (n:NoSuchLabel) RETURN n")
        result = format_records(records)
        assert result == ""
