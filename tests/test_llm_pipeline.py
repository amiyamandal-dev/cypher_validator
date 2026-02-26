"""Tests for LLMNLToCypher — all LLM calls are mocked."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from cypher_validator import Schema
from cypher_validator.llm_pipeline import (
    LLMNLToCypher,
    _build_langchain_fn,
    _build_openai_fn,
    _split_prompt_to_messages,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Company": ["name"]},
        relationships={"WORKS_FOR": ("Person", "Company", [])},
    )


@pytest.fixture()
def mock_llm_create():
    """LLM that returns a CREATE Cypher in a code fence."""
    def _llm(prompt: str) -> str:
        return (
            "Sure! Here's the query:\n\n"
            "```cypher\n"
            "CREATE (p:Person {name: 'John'})-[:WORKS_FOR]->(c:Company {name: 'Apple'})\n"
            "```"
        )
    return _llm


@pytest.fixture()
def mock_llm_infer():
    """LLM that returns inferred schema + Cypher (Mode B)."""
    def _llm(prompt: str) -> str:
        schema_json = json.dumps({
            "inferred_schema": {
                "nodes": {
                    "Person": ["name"],
                    "City": ["name"],
                },
                "relationships": {
                    "LIVES_IN": ["Person", "City", []],
                },
            }
        }, indent=2)
        return (
            f"```json\n{schema_json}\n```\n\n"
            "```cypher\n"
            "CREATE (p:Person {name: 'Alice'})-[:LIVES_IN]->(c:City {name: 'NYC'})\n"
            "```"
        )
    return _llm


# ---------------------------------------------------------------------------
# Constructor / factory tests
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_requires_llm_fn_or_model(self):
        with pytest.raises(ValueError, match="llm_fn.*model"):
            LLMNLToCypher()

    def test_llm_fn_only(self, schema):
        fn = lambda prompt: "MATCH (n) RETURN n"
        pipe = LLMNLToCypher(llm_fn=fn, schema=schema)
        assert pipe.schema is schema
        assert pipe.db is None

    def test_from_openai(self, schema):
        with patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_openai(
                "gpt-4o", api_key="sk-test", schema=schema
            )
            mock_build.assert_called_once()
            assert pipe.schema is schema

    def test_from_deepseek(self, schema):
        with patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_deepseek(api_key="sk-test", schema=schema)
            call_kwargs = mock_build.call_args
            assert call_kwargs.kwargs.get("base_url") or call_kwargs[1].get("base_url") or \
                   (len(call_kwargs[0]) > 1 and "deepseek" in str(call_kwargs))

    def test_from_anthropic(self, schema):
        with patch("cypher_validator.llm_pipeline._build_anthropic_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_anthropic(api_key="sk-test", schema=schema)
            mock_build.assert_called_once()
            assert pipe.schema is schema

    def test_from_env_deepseek(self, schema):
        env = {"DEEPSEEK_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False), \
             patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_env(schema=schema)
            assert pipe.schema is schema

    def test_from_env_openai(self, schema):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False), \
             patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_env(schema=schema)
            assert pipe.schema is schema

    def test_from_env_anthropic(self, schema):
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False), \
             patch("cypher_validator.llm_pipeline._build_anthropic_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            pipe = LLMNLToCypher.from_env(schema=schema)
            assert pipe.schema is schema

    def test_from_env_no_keys(self):
        env = {}
        with patch.dict(os.environ, env, clear=True), \
             pytest.raises(KeyError, match="ANTHROPIC_API_KEY|DEEPSEEK_API_KEY|OPENAI_API_KEY"):
            LLMNLToCypher.from_env()


# ---------------------------------------------------------------------------
# #1 — System/user message separation
# ---------------------------------------------------------------------------

class TestMessageSplitting:
    def test_split_with_mode_instruction(self):
        prompt = "You are a Cypher expert.\nSchema: ...\n\nGenerate a CREATE query.\n\nText: Hello"
        msgs = _split_prompt_to_messages(prompt)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "Cypher expert" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "Generate a CREATE" in msgs[1]["content"]

    def test_split_fallback_single_message(self):
        prompt = "Just some text with no markers"
        msgs = _split_prompt_to_messages(prompt)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_build_prompt_returns_system_and_user(self, schema):
        pipe = LLMNLToCypher(llm_fn=lambda p: "", schema=schema)
        system, user = pipe._build_prompt("test text", "create")
        assert "Schema" in system
        assert "Text: test text" in user
        assert "CREATE" in user


# ---------------------------------------------------------------------------
# __call__ with schema (Mode A)
# ---------------------------------------------------------------------------

class TestModeA:
    def test_create_with_schema(self, schema, mock_llm_create):
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema)
        result = pipe("John works for Apple.", mode="create")
        assert isinstance(result, str)
        assert "CREATE" in result
        assert "Person" in result

    def test_merge_mode(self, schema):
        def llm(prompt):
            assert "MERGE" in prompt.upper() or "merge" in prompt.lower() or "upsert" in prompt.lower()
            return "```cypher\nMERGE (p:Person {name: 'Bob'})\n```"
        pipe = LLMNLToCypher(llm_fn=llm, schema=schema)
        result = pipe("Bob exists.", mode="merge")
        assert "MERGE" in result

    def test_match_mode(self, schema):
        def llm(prompt):
            return "```cypher\nMATCH (p:Person {name: 'Bob'}) RETURN p\n```"
        pipe = LLMNLToCypher(llm_fn=llm, schema=schema)
        result = pipe("Find Bob.", mode="match")
        assert "MATCH" in result
        assert "RETURN" in result


# ---------------------------------------------------------------------------
# __call__ without schema (Mode B — inferred)
# ---------------------------------------------------------------------------

class TestModeB:
    def test_infer_schema(self, mock_llm_infer):
        pipe = LLMNLToCypher(llm_fn=mock_llm_infer)
        result = pipe("Alice lives in NYC.", mode="create")
        assert isinstance(result, str)
        assert "CREATE" in result
        # Schema should now be cached
        assert pipe._discovered_schema is not None
        assert "Person" in pipe._discovered_schema.node_labels()
        assert "City" in pipe._discovered_schema.node_labels()

    def test_inferred_schema_merges(self, mock_llm_infer):
        pipe = LLMNLToCypher(llm_fn=mock_llm_infer)
        pipe("Alice lives in NYC.", mode="create")
        # Call again — should merge, not replace
        pipe("Bob lives in LA.", mode="create")
        assert pipe._discovered_schema is not None


# ---------------------------------------------------------------------------
# #6 — reset_discovered_schema
# ---------------------------------------------------------------------------

class TestSchemaReset:
    def test_reset_clears_cache(self, mock_llm_infer):
        pipe = LLMNLToCypher(llm_fn=mock_llm_infer)
        pipe("Alice lives in NYC.", mode="create")
        assert pipe._discovered_schema is not None

        pipe.reset_discovered_schema()
        assert pipe._discovered_schema is None

    def test_reset_causes_reinference(self, mock_llm_infer):
        pipe = LLMNLToCypher(llm_fn=mock_llm_infer)
        pipe("Alice lives in NYC.", mode="create")
        pipe.reset_discovered_schema()
        ctx = pipe.ingest_with_context("Bob lives in LA.", mode="create")
        assert ctx["schema_source"] == "inferred"
        assert ctx["inferred_schema"] is not None


# ---------------------------------------------------------------------------
# ingest_with_context
# ---------------------------------------------------------------------------

class TestIngestWithContext:
    def test_returns_all_keys(self, schema, mock_llm_create):
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema)
        ctx = pipe.ingest_with_context("John works for Apple.", mode="create")
        assert "schema_source" in ctx
        assert "inferred_schema" in ctx
        assert "cypher" in ctx
        assert "is_valid" in ctx
        assert "validation_errors" in ctx
        assert "repair_attempts" in ctx
        assert "records" in ctx
        assert "execution_error" in ctx

    def test_schema_source_user(self, schema, mock_llm_create):
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema)
        ctx = pipe.ingest_with_context("text", mode="create")
        assert ctx["schema_source"] == "user"

    def test_schema_source_inferred(self, mock_llm_infer):
        pipe = LLMNLToCypher(llm_fn=mock_llm_infer)
        ctx = pipe.ingest_with_context("Alice lives in NYC.", mode="create")
        assert ctx["schema_source"] == "inferred"
        assert ctx["inferred_schema"] is not None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class TestExecution:
    def test_execute_returns_tuple(self, schema, mock_llm_create):
        mock_db = MagicMock()
        mock_db.execute.return_value = [{"name": "John"}]
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema, db=mock_db)
        cypher, records = pipe("John works for Apple.", mode="create", execute=True)
        assert isinstance(cypher, str)
        assert records == [{"name": "John"}]
        mock_db.execute.assert_called_once()

    def test_execute_no_db_raises(self, schema, mock_llm_create):
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema)
        with pytest.raises(RuntimeError, match="no 'db'"):
            pipe("text", mode="create", execute=True)

    def test_execute_error_captured(self, schema, mock_llm_create):
        mock_db = MagicMock()
        mock_db.execute.side_effect = RuntimeError("connection lost")
        pipe = LLMNLToCypher(llm_fn=mock_llm_create, schema=schema, db=mock_db)
        ctx = pipe.ingest_with_context("text", mode="create", execute=True)
        assert ctx["execution_error"] == "connection lost"
        assert ctx["records"] == []


# ---------------------------------------------------------------------------
# #2 — Repair prompt includes schema context
# ---------------------------------------------------------------------------

class TestRepair:
    def test_repair_prompt_includes_schema(self, schema):
        """Repair prompt must contain schema context so the LLM can fix labels."""
        repair_prompts_seen = []
        call_count = {"n": 0}

        def llm(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "```cypher\nCREATE (x:Alien {name: 'E.T.'})\n```"
            else:
                repair_prompts_seen.append(prompt)
                return "```cypher\nCREATE (p:Person {name: 'E.T.'})\n```"

        pipe = LLMNLToCypher(llm_fn=llm, schema=schema, max_repair_retries=2)
        ctx = pipe.ingest_with_context("E.T. is a person.", mode="create")
        assert ctx["repair_attempts"] >= 1
        assert "Person" in ctx["cypher"]
        # The repair prompt should include schema info
        assert len(repair_prompts_seen) >= 1
        assert "Schema" in repair_prompts_seen[0] or "Person" in repair_prompts_seen[0]

    def test_repair_called_on_invalid(self, schema):
        call_count = {"n": 0}

        def llm(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "```cypher\nCREATE (x:Alien {name: 'E.T.'})\n```"
            else:
                return "```cypher\nCREATE (p:Person {name: 'E.T.'})\n```"

        pipe = LLMNLToCypher(llm_fn=llm, schema=schema, max_repair_retries=2)
        ctx = pipe.ingest_with_context("E.T. is a person.", mode="create")
        assert ctx["repair_attempts"] >= 1
        assert "Person" in ctx["cypher"]


# ---------------------------------------------------------------------------
# #3 — CypherValidator(None) graceful handling
# ---------------------------------------------------------------------------

class TestNoSchemaValidation:
    def test_valid_syntax_no_schema(self):
        """Without a schema, syntax-valid Cypher should pass."""
        def llm(prompt):
            return "```cypher\nCREATE (n:Foo {name: 'bar'})\n```"

        pipe = LLMNLToCypher(llm_fn=llm)
        # Force no-schema path by not providing schema and ensuring Mode B
        # returns no inferred schema
        ctx = pipe.ingest_with_context.__wrapped__(pipe, "test", "create") \
            if hasattr(pipe.ingest_with_context, '__wrapped__') else None

        # Simpler: just directly test _validate_and_repair with None schema
        cypher, is_valid, errors, attempts = pipe._validate_and_repair(
            "CREATE (n:Foo {name: 'bar'})", None
        )
        assert is_valid
        assert errors == []
        assert attempts == 0

    def test_invalid_syntax_no_schema(self):
        """Syntax errors should be caught even without schema."""
        pipe = LLMNLToCypher(llm_fn=lambda p: "")
        cypher, is_valid, errors, attempts = pipe._validate_and_repair(
            "THIS IS NOT CYPHER AT ALL", None
        )
        assert not is_valid
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Schema resolution priority
# ---------------------------------------------------------------------------

class TestSchemaResolution:
    def test_user_schema_takes_priority(self, schema):
        mock_db = MagicMock()
        mock_db.introspect_schema.return_value = Schema(
            nodes={"Other": ["x"]}, relationships={}
        )
        fn = lambda p: "```cypher\nMATCH (n:Person) RETURN n\n```"
        pipe = LLMNLToCypher(llm_fn=fn, schema=schema, db=mock_db)
        ctx = pipe.ingest_with_context("text", mode="match")
        assert ctx["schema_source"] == "user"
        mock_db.introspect_schema.assert_not_called()

    def test_db_schema_used_when_no_user_schema(self):
        db_schema = Schema(
            nodes={"Person": ["name"]}, relationships={}
        )
        mock_db = MagicMock()
        mock_db.introspect_schema.return_value = db_schema

        fn = lambda p: "```cypher\nMATCH (n:Person) RETURN n\n```"
        pipe = LLMNLToCypher(llm_fn=fn, db=mock_db)
        ctx = pipe.ingest_with_context("text", mode="match")
        assert ctx["schema_source"] == "db"


# ---------------------------------------------------------------------------
# _parse_inferred_schema
# ---------------------------------------------------------------------------

class TestParseInferredSchema:
    def test_valid_json_and_cypher(self):
        response = (
            '```json\n'
            '{"inferred_schema": {"nodes": {"A": ["x"]}, "relationships": {}}}\n'
            '```\n\n'
            '```cypher\nCREATE (a:A {x: 1})\n```'
        )
        schema_dict, cypher = LLMNLToCypher._parse_inferred_schema(response)
        assert schema_dict is not None
        assert schema_dict["nodes"]["A"] == ["x"]
        assert "CREATE" in cypher

    def test_no_json_block(self):
        response = "```cypher\nCREATE (a:A {x: 1})\n```"
        schema_dict, cypher = LLMNLToCypher._parse_inferred_schema(response)
        assert schema_dict is None
        assert "CREATE" in cypher

    def test_malformed_json(self):
        response = '```json\n{bad json}\n```\n```cypher\nCREATE (a:A)\n```'
        schema_dict, cypher = LLMNLToCypher._parse_inferred_schema(response)
        assert schema_dict is None
        assert "CREATE" in cypher


# ---------------------------------------------------------------------------
# #7 — Context manager / close()
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_close_owned_db(self):
        mock_db = MagicMock()
        pipe = LLMNLToCypher(llm_fn=lambda p: "", db=mock_db)
        pipe._owns_db = True
        pipe.close()
        mock_db.close.assert_called_once()

    def test_close_does_not_close_user_db(self):
        mock_db = MagicMock()
        pipe = LLMNLToCypher(llm_fn=lambda p: "", db=mock_db)
        # _owns_db defaults to False
        pipe.close()
        mock_db.close.assert_not_called()

    def test_context_manager(self):
        mock_db = MagicMock()
        pipe = LLMNLToCypher(llm_fn=lambda p: "", db=mock_db)
        pipe._owns_db = True
        with pipe as p:
            assert p is pipe
        mock_db.close.assert_called_once()

    def test_from_env_sets_owns_db(self, schema):
        env = {"DEEPSEEK_API_KEY": "sk-test", "NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "pw"}
        with patch.dict(os.environ, env, clear=False), \
             patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build, \
             patch("cypher_validator.gliner2_integration.Neo4jDatabase") as mock_neo4j:
            mock_build.return_value = lambda p: ""
            mock_neo4j.return_value = MagicMock()
            pipe = LLMNLToCypher.from_env(schema=schema)
            assert pipe._owns_db is True


# ---------------------------------------------------------------------------
# #5 — temperature docs (behavioural: no effect on llm_fn)
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_temperature_stored(self, schema):
        pipe = LLMNLToCypher(llm_fn=lambda p: "", schema=schema, temperature=0.7)
        assert pipe.temperature == 0.7

    def test_temperature_passed_to_openai_adapter(self, schema):
        with patch("cypher_validator.llm_pipeline._build_openai_fn") as mock_build:
            mock_build.return_value = lambda p: ""
            LLMNLToCypher(model="gpt-4o", api_key="sk-test", schema=schema, temperature=0.5)
            _, kwargs = mock_build.call_args
            assert kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_with_schema(self, schema):
        pipe = LLMNLToCypher(llm_fn=lambda p: "", schema=schema)
        assert "schema" in repr(pipe)
        assert "no db" in repr(pipe)

    def test_repr_no_schema_with_db(self):
        mock_db = MagicMock()
        pipe = LLMNLToCypher(llm_fn=lambda p: "", db=mock_db)
        assert "no schema" in repr(pipe)
        assert "db" in repr(pipe)
        assert "no db" not in repr(pipe)


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------

class TestLangChain:
    def test_from_langchain_factory(self, schema):
        """from_langchain wraps a LangChain ChatModel correctly."""
        # Simulate a LangChain AIMessage
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = (
            "```cypher\n"
            "CREATE (p:Person {name: 'Alice'})-[:WORKS_FOR]->(c:Company {name: 'Acme'})\n"
            "```"
        )
        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_msg

        pipe = LLMNLToCypher.from_langchain(mock_chat_model, schema=schema)
        result = pipe("Alice works for Acme.", mode="create")

        assert "CREATE" in result
        assert "Person" in result
        mock_chat_model.invoke.assert_called_once()

    def test_langchain_receives_system_and_human_messages(self, schema):
        """The adapter should send SystemMessage + HumanMessage."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "```cypher\nMATCH (n:Person) RETURN n\n```"
        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_msg

        pipe = LLMNLToCypher.from_langchain(mock_chat_model, schema=schema)
        pipe("Find people.", mode="match")

        call_args = mock_chat_model.invoke.call_args[0][0]
        # Should be a list of LangChain message objects
        assert len(call_args) == 2
        # First should be SystemMessage, second HumanMessage
        assert call_args[0].__class__.__name__ == "SystemMessage"
        assert call_args[1].__class__.__name__ == "HumanMessage"
        assert "Schema" in call_args[0].content
        assert "Text:" in call_args[1].content

    def test_langchain_missing_package(self):
        """Should raise ImportError with helpful message."""
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.messages": None}):
            with pytest.raises(ImportError, match="langchain-core"):
                _build_langchain_fn(MagicMock())

    def test_langchain_end_to_end_with_execute(self, schema):
        """Full pipeline: LangChain model → Cypher → execute."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = (
            "```cypher\n"
            "CREATE (p:Person {name: 'Bob'})-[:WORKS_FOR]->(c:Company {name: 'X'})\n"
            "```"
        )
        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_msg

        mock_db = MagicMock()
        mock_db.execute.return_value = [{"p": "Bob"}]

        pipe = LLMNLToCypher.from_langchain(
            mock_chat_model, schema=schema, db=mock_db
        )
        cypher, records = pipe("Bob works for X.", mode="create", execute=True)

        assert "CREATE" in cypher
        assert records == [{"p": "Bob"}]
        mock_db.execute.assert_called_once()
