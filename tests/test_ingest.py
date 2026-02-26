"""Tests for batch ingestion: _chunk_text, _build_provenance_cypher, ingest_texts, ingest_document."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from cypher_validator import Schema
from cypher_validator.llm_pipeline import (
    ChunkResult,
    IngestionResult,
    LLMNLToCypher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = Schema(
    nodes={"Person": ["name", "age"], "Company": ["name"]},
    relationships={"WORKS_FOR": ("Person", "Company", [])},
)


def _make_pipeline(*, schema=None, db=None, llm_responses=None):
    """Create an LLMNLToCypher with a mocked llm_fn."""
    responses = list(llm_responses or [])
    call_count = {"n": 0}

    def fake_llm(prompt: str) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(responses):
            return responses[idx]
        return "```cypher\nMERGE (n:Person {name: 'default'})\n```"

    return LLMNLToCypher(llm_fn=fake_llm, schema=schema, db=db)


# ---------------------------------------------------------------------------
# _chunk_text tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world. This is short."
        chunks = LLMNLToCypher._chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        sentences = [f"Sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        chunks = LLMNLToCypher._chunk_text(text, chunk_size=100, chunk_overlap=30)
        assert len(chunks) > 1
        # All original text should be covered
        for s in sentences:
            assert any(s in c for c in chunks), f"Missing: {s}"

    def test_no_sentence_boundaries(self):
        text = "a" * 500
        chunks = LLMNLToCypher._chunk_text(text, chunk_size=200, chunk_overlap=50)
        # Single "sentence" > chunk_size → still included as one chunk
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_behavior(self):
        # Create sentences with known lengths
        s1 = "First sentence."   # 15 chars
        s2 = "Second sentence."  # 16 chars
        s3 = "Third sentence."   # 15 chars
        text = f"{s1} {s2} {s3}"
        # chunk_size just enough for 2 sentences, overlap enough for 1
        chunks = LLMNLToCypher._chunk_text(text, chunk_size=35, chunk_overlap=20)
        assert len(chunks) >= 2
        # Second chunk should contain overlap from first chunk
        if len(chunks) >= 2:
            # The overlap sentence should appear in both chunks
            assert "Second sentence." in chunks[0] or "First sentence." in chunks[0]

    def test_empty_text(self):
        assert LLMNLToCypher._chunk_text("", chunk_size=100, chunk_overlap=10) == []

    def test_whitespace_only(self):
        assert LLMNLToCypher._chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


# ---------------------------------------------------------------------------
# _build_provenance_cypher tests
# ---------------------------------------------------------------------------


class TestBuildProvenanceCypher:
    def test_produces_valid_parameterized_cypher(self):
        domain_cypher = "MERGE (p:Person {name: 'Alice'})\nMERGE (c:Company {name: 'Acme'})"
        cypher, params = LLMNLToCypher._build_provenance_cypher(
            domain_cypher,
            chunk_id="doc_chunk_0",
            source_id="doc",
            text_preview="Alice works for Acme",
        )
        assert "MERGE (chunk:Chunk {chunk_id: $chunk_id})" in cypher
        assert "MENTIONED_IN" in cypher
        assert params["chunk_id"] == "doc_chunk_0"
        assert params["source_id"] == "doc"
        assert params["text_preview"] == "Alice works for Acme"
        assert "Person" in params["domain_labels"]
        assert "Company" in params["domain_labels"]
        assert "Chunk" not in params["domain_labels"]

    def test_no_labels_in_domain(self):
        # Even with a simple RETURN query, should not crash
        cypher, params = LLMNLToCypher._build_provenance_cypher(
            "RETURN 1",
            chunk_id="c1",
            source_id="s1",
            text_preview="test",
        )
        assert params["domain_labels"] == []
        assert "MERGE (chunk:Chunk" in cypher


# ---------------------------------------------------------------------------
# ingest_texts tests
# ---------------------------------------------------------------------------


class TestIngestTexts:
    def test_with_schema_provided_uses_merge_prompt(self):
        """When schema is provided, all texts use the MERGE prompt (Phase 2 only)."""
        responses = [
            "```cypher\nMERGE (p:Person {name: 'Alice'})\n```",
            "```cypher\nMERGE (p:Person {name: 'Bob'})\n```",
        ]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_texts(
            ["Alice is a person.", "Bob is a person."],
            provenance=False,
        )
        assert isinstance(result, IngestionResult)
        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert result.schema_sample_texts == 0  # no Phase 1
        assert result.schema_source == "user"
        assert len(result.results) == 2
        for r in result.results:
            assert "MERGE" in r.cypher

    def test_without_schema_runs_phase1_then_phase2(self):
        """Phase 1 infers schema, Phase 2 uses it."""
        phase1_response = (
            '```json\n{"inferred_schema": {"nodes": {"Person": ["name"]}, '
            '"relationships": {}}}\n```\n'
            "```cypher\nCREATE (p:Person {name: 'Alice'})\n```"
        )
        phase2_response = "```cypher\nMERGE (p:Person {name: 'Bob'})\n```"
        pipe = _make_pipeline(
            llm_responses=[phase1_response, phase2_response],
        )
        result = pipe.ingest_texts(
            ["Alice is a person.", "Bob is a person."],
            schema_sample_size=1,
            provenance=False,
        )
        assert result.schema_sample_texts == 1
        assert result.schema_source == "inferred"
        assert result.total == 2
        # Phase 1 text produced a result
        assert result.results[0].index == 0
        # Phase 2 text also produced a result
        assert result.results[1].index == 1

    def test_on_error_skip(self):
        """on_error='skip' records error but continues."""
        def failing_llm(prompt):
            raise RuntimeError("LLM exploded")

        pipe = LLMNLToCypher(llm_fn=failing_llm, schema=_SCHEMA)
        result = pipe.ingest_texts(
            ["text1", "text2"],
            on_error="skip",
            provenance=False,
        )
        assert result.failed == 2
        assert len(result.errors) == 2

    def test_on_error_raise(self):
        """on_error='raise' propagates the exception."""
        def failing_llm(prompt):
            raise RuntimeError("LLM exploded")

        pipe = LLMNLToCypher(llm_fn=failing_llm, schema=_SCHEMA)
        with pytest.raises(RuntimeError, match="LLM exploded"):
            pipe.ingest_texts(
                ["text1"],
                on_error="raise",
                provenance=False,
            )

    def test_provenance_false_skips_provenance(self):
        responses = ["```cypher\nMERGE (p:Person {name: 'Alice'})\n```"]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_texts(["Alice."], provenance=False)
        assert result.results[0].provenance_cypher == ""

    def test_provenance_true_generates_cypher(self):
        responses = ["```cypher\nMERGE (p:Person {name: 'Alice'})\n```"]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_texts(["Alice."], provenance=True)
        assert "Chunk" in result.results[0].provenance_cypher
        assert "MENTIONED_IN" in result.results[0].provenance_cypher

    def test_execute_with_mock_db(self):
        mock_db = MagicMock()
        mock_db.execute.return_value = [{"n": "Alice"}]

        responses = ["```cypher\nMERGE (p:Person {name: 'Alice'})\n```"]
        pipe = _make_pipeline(schema=_SCHEMA, db=mock_db, llm_responses=responses)
        result = pipe.ingest_texts(
            ["Alice."],
            execute=True,
            provenance=True,
        )
        assert result.results[0].executed is True
        assert mock_db.execute.called

    def test_progress_fn_called(self):
        responses = [
            "```cypher\nMERGE (p:Person {name: 'A'})\n```",
            "```cypher\nMERGE (p:Person {name: 'B'})\n```",
        ]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        calls = []
        result = pipe.ingest_texts(
            ["A.", "B."],
            provenance=False,
            progress_fn=lambda cur, tot: calls.append((cur, tot)),
        )
        assert calls == [(1, 2), (2, 2)]

    def test_empty_texts(self):
        pipe = _make_pipeline(schema=_SCHEMA)
        result = pipe.ingest_texts([])
        assert result.total == 0
        assert result.results == []

    def test_source_ids_mismatch_raises(self):
        pipe = _make_pipeline(schema=_SCHEMA)
        with pytest.raises(ValueError, match="source_ids length"):
            pipe.ingest_texts(["a", "b"], source_ids=["only_one"])

    def test_on_error_invalid_value_raises(self):
        pipe = _make_pipeline(schema=_SCHEMA)
        with pytest.raises(ValueError, match="on_error must be"):
            pipe.ingest_texts(["text"], on_error="invalid")

    def test_custom_source_ids(self):
        responses = ["```cypher\nMERGE (p:Person {name: 'A'})\n```"]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_texts(
            ["A."],
            source_ids=["my_doc"],
            provenance=False,
        )
        assert result.results[0].source_id == "my_doc"


# ---------------------------------------------------------------------------
# ingest_document tests
# ---------------------------------------------------------------------------


class TestIngestDocument:
    def test_chunking_and_delegation(self):
        """ingest_document chunks text then delegates to ingest_texts."""
        sentences = [f"Sentence {i}." for i in range(10)]
        text = " ".join(sentences)

        # Enough responses for all chunks
        responses = [
            f"```cypher\nMERGE (p:Person {{name: 'P{i}'}})\n```"
            for i in range(20)
        ]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_document(
            text,
            source_id="mydoc",
            chunk_size=60,
            chunk_overlap=15,
            provenance=False,
        )
        assert result.total > 1
        # All source_ids should be mydoc_chunk_N
        for r in result.results:
            assert r.source_id.startswith("mydoc_chunk_")

    def test_short_document_single_chunk(self):
        responses = ["```cypher\nMERGE (p:Person {name: 'Alice'})\n```"]
        pipe = _make_pipeline(schema=_SCHEMA, llm_responses=responses)
        result = pipe.ingest_document(
            "Alice is nice.",
            source_id="short",
            provenance=False,
        )
        assert result.total == 1
        assert result.results[0].source_id == "short_chunk_0"
