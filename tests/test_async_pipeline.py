"""Tests for async LLMNLToCypher methods — all LLM calls are mocked."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from cypher_validator import Schema
from cypher_validator.llm_pipeline import (
    LLMNLToCypher,
    TokenBucketRateLimiter,
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


def _sync_llm_create(prompt: str) -> str:
    return (
        "Sure! Here's the query:\n\n"
        "```cypher\n"
        "CREATE (p:Person {name: 'John'})-[:WORKS_FOR]->(c:Company {name: 'Apple'})\n"
        "```"
    )


async def _async_llm_create(prompt: str) -> str:
    return (
        "Sure! Here's the query:\n\n"
        "```cypher\n"
        "CREATE (p:Person {name: 'John'})-[:WORKS_FOR]->(c:Company {name: 'Apple'})\n"
        "```"
    )


async def _async_llm_merge(prompt: str) -> str:
    return (
        "```cypher\n"
        "MERGE (p:Person {name: 'John'})-[:WORKS_FOR]->(c:Company {name: 'Apple'})\n"
        "```"
    )


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------

class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_budget(self):
        limiter = TokenBucketRateLimiter(tpm=10000)
        # Should return immediately — well within budget
        start = time.monotonic()
        await limiter.acquire(100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_acquire_waits_when_exhausted(self):
        # Very small bucket: 60 tokens/min = 1 token/sec
        limiter = TokenBucketRateLimiter(tpm=60)
        # Drain the bucket
        await limiter.acquire(60)
        # Next acquire should wait ~1 second for 1 token
        start = time.monotonic()
        await limiter.acquire(1)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.5  # at least half a second

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        limiter = TokenBucketRateLimiter(tpm=6000)
        # 10 concurrent tasks each requesting 100 tokens
        results = []

        async def _acquire():
            await limiter.acquire(100)
            results.append(True)

        await asyncio.gather(*[_acquire() for _ in range(10)])
        assert len(results) == 10

    def test_estimate_tokens(self):
        assert TokenBucketRateLimiter.estimate_tokens("hello") == 1
        assert TokenBucketRateLimiter.estimate_tokens("a" * 400) == 100


# ---------------------------------------------------------------------------
# acall
# ---------------------------------------------------------------------------

class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_acall_with_async_llm_fn(self, schema):
        pipe = LLMNLToCypher(
            llm_fn=_sync_llm_create,
            async_llm_fn=_async_llm_create,
            schema=schema,
        )
        result = await pipe.acall("John works for Apple.", mode="create")
        assert isinstance(result, str)
        assert "CREATE" in result
        assert "Person" in result

    @pytest.mark.asyncio
    async def test_acall_falls_back_to_sync(self, schema):
        """When only sync llm_fn is provided, async still works via to_thread."""
        pipe = LLMNLToCypher(llm_fn=_sync_llm_create, schema=schema)
        result = await pipe.acall("John works for Apple.", mode="create")
        assert isinstance(result, str)
        assert "CREATE" in result

    @pytest.mark.asyncio
    async def test_acall_with_execute(self, schema):
        mock_db = MagicMock()
        mock_db.execute.return_value = [{"name": "John"}]
        pipe = LLMNLToCypher(
            llm_fn=_sync_llm_create,
            schema=schema,
            db=mock_db,
        )
        cypher, records = await pipe.acall(
            "John works for Apple.", mode="create", execute=True
        )
        assert isinstance(cypher, str)
        assert records == [{"name": "John"}]
        mock_db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# aingest_texts
# ---------------------------------------------------------------------------

class TestAsyncIngestTexts:
    @pytest.mark.asyncio
    async def test_parallel_ingestion_with_schema(self, schema):
        """10 texts ingested concurrently, all results present."""
        texts = [f"Person_{i} works for Company_{i}." for i in range(10)]
        pipe = LLMNLToCypher(
            llm_fn=_sync_llm_create,
            async_llm_fn=_async_llm_merge,
            schema=schema,
        )
        result = await pipe.aingest_texts(texts, max_concurrency=5)
        assert result.total == 10
        assert len(result.results) == 10
        # All should have Phase 2 (no schema inference needed)
        assert result.schema_sample_texts == 0

    @pytest.mark.asyncio
    async def test_phase1_sequential_phase2_parallel(self):
        """Without schema, Phase 1 runs serially then Phase 2 in parallel."""
        import json

        schema_json = json.dumps({
            "inferred_schema": {
                "nodes": {"Person": ["name"], "City": ["name"]},
                "relationships": {
                    "LIVES_IN": ["Person", "City", []],
                },
            }
        }, indent=2)

        async def _llm(prompt: str) -> str:
            # Phase 1 prompts contain the schema-unknown system prompt
            # Phase 2 prompts contain the ingest system prompt with schema
            # Both should return valid Cypher
            return (
                f"```json\n{schema_json}\n```\n\n"
                "```cypher\n"
                "MERGE (p:Person {name: 'Alice'})-[:LIVES_IN]->(c:City {name: 'NYC'})\n"
                "```"
            )

        texts = [f"Person {i} lives in City {i}." for i in range(6)]
        pipe = LLMNLToCypher(
            llm_fn=lambda p: "",  # dummy sync
            async_llm_fn=_llm,
        )
        result = await pipe.aingest_texts(
            texts, schema_sample_size=3
        )
        assert result.total == 6
        assert result.schema_sample_texts == 3
        assert len(result.results) == 6

    @pytest.mark.asyncio
    async def test_on_error_skip_async(self, schema):
        """Errors are captured, not raised."""
        call_count = {"n": 0}

        async def _failing_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("LLM exploded")
            return (
                "```cypher\n"
                "MERGE (p:Person {name: 'Test'})\n"
                "```"
            )

        pipe = LLMNLToCypher(
            llm_fn=lambda p: "",
            async_llm_fn=_failing_llm,
            schema=schema,
        )
        result = await pipe.aingest_texts(
            ["text1", "text2", "text3"],
            on_error="skip",
        )
        assert result.total == 3
        assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_progress_fn_async(self, schema):
        """progress callback fires for each text."""
        progress_calls = []

        def _progress(current, total):
            progress_calls.append((current, total))

        pipe = LLMNLToCypher(
            llm_fn=_sync_llm_create,
            async_llm_fn=_async_llm_merge,
            schema=schema,
        )
        await pipe.aingest_texts(
            ["text1", "text2", "text3"],
            progress_fn=_progress,
        )
        assert len(progress_calls) == 3
        assert all(t == 3 for _, t in progress_calls)

    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self, schema):
        """Verify at most N concurrent LLM calls."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def _tracking_llm(prompt: str) -> str:
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.05)  # simulate work
            async with lock:
                current -= 1
            return (
                "```cypher\n"
                "MERGE (p:Person {name: 'Test'})\n"
                "```"
            )

        pipe = LLMNLToCypher(
            llm_fn=lambda p: "",
            async_llm_fn=_tracking_llm,
            schema=schema,
            max_concurrency=3,
        )
        texts = [f"text_{i}" for i in range(10)]
        await pipe.aingest_texts(texts, max_concurrency=3)
        assert max_concurrent <= 3


# ---------------------------------------------------------------------------
# Rate limiting integration
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_tpm_limit_throttles(self, schema):
        """With a low TPM limit, calls are spaced out."""
        call_times = []

        async def _timed_llm(prompt: str) -> str:
            call_times.append(time.monotonic())
            return "```cypher\nMERGE (p:Person {name: 'T'})\n```"

        # Each prompt is ~250 chars = ~62 tokens.
        # TPM=600 = 10 tokens/sec. First call consumes ~62 tokens,
        # leaving ~538. Three calls = ~186 tokens total, well within
        # initial budget for first burst, but second+ calls start
        # competing and rate limiter kicks in with very small waits.
        # Use a TPM that forces at least some wait but stays fast.
        # TPM=100 = ~1.67 tokens/sec. 62 tokens = ~37 sec wait.
        # That's too slow. Instead, use a bucket that fits 1.5 calls:
        # TPM=5580 = 93 tokens/sec. 62*3 = 186 tokens < 5580, no wait.
        # We want a bucket that's smaller than total tokens needed.
        # 3 calls * ~62 tokens = 186 tokens. If TPM=120, bucket starts
        # at 120, first call takes 62, leaves 58, second needs 62 but
        # only 58 available → wait ~0.24s. Total wait small enough.
        # But prompt is ~250 chars... let's compute actual token estimate.
        # _timed_llm receives the full system+user prompt which is large.
        # Let's use a simpler approach: pre-drain the bucket.
        limiter = TokenBucketRateLimiter(tpm=6000)  # 100 tokens/sec
        # Drain most of the bucket so next calls must wait
        await limiter.acquire(5950)  # leaves 50 tokens
        pipe = LLMNLToCypher(
            llm_fn=lambda p: "",
            async_llm_fn=_timed_llm,
            schema=schema,
            tpm_limit=6000,
            max_concurrency=3,
        )
        pipe._rate_limiter = limiter
        await pipe.aingest_texts(["a", "b", "c"], max_concurrency=3)
        assert len(call_times) == 3
        total_elapsed = call_times[-1] - call_times[0]
        # Should have some non-trivial delay from rate limiting
        assert total_elapsed > 0.05


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------

class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_with(self):
        mock_db = MagicMock()
        pipe = LLMNLToCypher(llm_fn=lambda p: "", db=mock_db)
        pipe._owns_db = True
        async with pipe as p:
            assert p is pipe
        mock_db.close.assert_called_once()
