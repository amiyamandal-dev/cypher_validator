"""
Graph RAG (Retrieval-Augmented Generation) pipeline.

Chains together:

1. **Schema context injection** — the graph schema is formatted and embedded
   in the LLM system prompt so the model knows which labels and types exist.
2. **Cypher generation** — an LLM generates a Cypher query from a natural
   language question.
3. **Extraction** — :func:`~cypher_validator.llm_utils.extract_cypher_from_text`
   pulls the query out of the raw LLM response.
4. **Validation + repair** — the query is validated against the schema; if
   invalid, the LLM is asked to fix it (up to *max_repair_retries* times).
5. **Execution** — the validated query is run against the Neo4j database.
6. **Result formatting** — records are formatted for LLM context.
7. **Answer generation** — the LLM synthesises a natural language answer from
   the question and the query results.

Usage
-----
::

    from cypher_validator import GraphRAGPipeline, Neo4jDatabase, Schema

    def call_llm(prompt: str) -> str:
        # wrap your Claude / OpenAI / etc. call here
        ...

    schema = Schema(
        nodes={"Person": ["name"], "Company": ["name"]},
        relationships={"WORKS_FOR": ("Person", "Company", [])},
    )
    db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")

    pipeline = GraphRAGPipeline(schema=schema, db=db, llm_fn=call_llm)
    answer = pipeline.query("Who works for Acme Corp?")

    # Or get all intermediate artefacts:
    ctx = pipeline.query_with_context("Who works for Acme Corp?")
    print(ctx["cypher"])
    print(ctx["formatted_results"])
    print(ctx["answer"])
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from cypher_validator.llm_utils import (
    extract_cypher_from_text,
    format_records,
)


class GraphRAGPipeline:
    """End-to-end Graph RAG pipeline: natural language → Cypher → execute → answer.

    Parameters
    ----------
    schema:
        :class:`~cypher_validator.Schema` that describes the graph model.
        Used for schema-context injection and Cypher validation.
    db:
        :class:`~cypher_validator.Neo4jDatabase` connection used to execute
        generated queries.
    llm_fn:
        A callable ``(prompt: str) -> str`` that wraps your LLM (Claude,
        GPT-4, Gemini, …).  It is called twice per :meth:`query` call:
        once to generate Cypher and once to produce the final answer.
    max_repair_retries:
        How many times to ask the LLM to fix an invalid query before giving
        up and executing it as-is (default: 2).
    result_format:
        Format used by :func:`~cypher_validator.llm_utils.format_records`
        when injecting results into the answer prompt — one of
        ``"markdown"`` (default), ``"csv"``, ``"json"``, or ``"text"``.
    cypher_system_prompt:
        Override the default system prompt used for Cypher generation.
        The schema context is appended automatically when ``None``.
    answer_system_prompt:
        Override the default system prompt used for answer synthesis.
    """

    def __init__(
        self,
        schema: Any,
        db: Any,
        llm_fn: Callable[[str], str],
        *,
        max_repair_retries: int = 2,
        result_format: str = "markdown",
        cypher_system_prompt: Optional[str] = None,
        answer_system_prompt: Optional[str] = None,
    ) -> None:
        from cypher_validator import CypherValidator

        self.schema = schema
        self.db = db
        self.llm_fn = llm_fn
        self.max_repair_retries = max_repair_retries
        self.result_format = result_format

        self._validator = CypherValidator(schema)
        self._cypher_system = cypher_system_prompt or self._default_cypher_system()
        self._answer_system = answer_system_prompt or self._default_answer_system()

    # ------------------------------------------------------------------
    # Default prompts
    # ------------------------------------------------------------------

    def _default_cypher_system(self) -> str:
        schema_block = self.schema.to_cypher_context()
        return (
            "You are a Neo4j Cypher expert. "
            "Generate a single Cypher query that answers the user's question.\n\n"
            f"Schema:\n{schema_block}\n"
            "Rules:\n"
            "- Return ONLY the Cypher query inside a ```cypher code fence, "
            "with no other text.\n"
            "- Use only the labels and relationship types defined in the schema above.\n"
            "- Prefer parameterised values ($name, $year, …) for user-supplied data.\n"
            "- Always include a RETURN clause.\n"
            "- Use LIMIT when the result could be very large."
        )

    def _default_answer_system(self) -> str:
        return (
            "You are a helpful assistant. "
            "Answer the user's question concisely based on the provided "
            "graph database query results. "
            "If the results are empty, say so clearly."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Run the full pipeline and return a natural language answer.

        Parameters
        ----------
        question:
            Natural language question about the graph data.

        Returns
        -------
        str
            LLM-generated answer based on the graph query results.
        """
        return self.query_with_context(question)["answer"]

    def query_with_context(self, question: str) -> Dict[str, Any]:
        """Run the full pipeline and return all intermediate artefacts.

        Parameters
        ----------
        question:
            Natural language question about the graph data.

        Returns
        -------
        dict
            Contains the following keys:

            ``question``
                The original question.
            ``cypher``
                The final Cypher query (possibly repaired).
            ``is_valid``
                Whether the final query passed schema validation.
            ``validation_errors``
                List of validation error strings (empty when valid).
            ``repair_attempts``
                Number of repair iterations performed (0 = first try was valid).
            ``records``
                Raw list of record dicts from Neo4j (empty on execution error).
            ``formatted_results``
                Records formatted as a string for LLM context.
            ``answer``
                Final natural language answer from the LLM.
            ``execution_error``
                Error message string if execution raised an exception, else ``None``.
        """
        # ── Step 1: Generate Cypher ──────────────────────────────────────
        cypher_prompt = f"{self._cypher_system}\n\nQuestion: {question}"
        raw_response = self.llm_fn(cypher_prompt)
        cypher = extract_cypher_from_text(raw_response)

        # ── Step 2: Validate + repair loop ───────────────────────────────
        repair_attempts = 0
        result = self._validator.validate(cypher)

        while not result.is_valid and repair_attempts < self.max_repair_retries:
            error_list = "\n".join(f"  - {e}" for e in result.errors)
            repair_prompt = (
                f"{self._cypher_system}\n\n"
                f"The Cypher query below has validation errors. "
                f"Fix it so it is valid and return ONLY the corrected query "
                f"inside a ```cypher code fence.\n\n"
                f"Faulty query:\n```cypher\n{cypher}\n```\n\n"
                f"Errors:\n{error_list}"
            )
            raw_response = self.llm_fn(repair_prompt)
            cypher = extract_cypher_from_text(raw_response)
            result = self._validator.validate(cypher)
            repair_attempts += 1

        # ── Step 3: Execute against Neo4j ────────────────────────────────
        records: List[Dict[str, Any]] = []
        execution_error: Optional[str] = None
        if cypher:
            try:
                records = self.db.execute(cypher)
            except Exception as exc:  # noqa: BLE001
                execution_error = str(exc)

        # ── Step 4: Format results ────────────────────────────────────────
        formatted = format_records(records, format=self.result_format)

        # ── Step 5: Generate answer ───────────────────────────────────────
        result_section = formatted if formatted else "No results found."
        if execution_error:
            result_section = f"Query execution failed: {execution_error}"

        answer_prompt = (
            f"{self._answer_system}\n\n"
            f"Question: {question}\n\n"
            f"Graph database results:\n{result_section}\n\n"
            f"Answer:"
        )
        answer = self.llm_fn(answer_prompt)

        return {
            "question": question,
            "cypher": cypher,
            "is_valid": result.is_valid,
            "validation_errors": list(result.errors),
            "repair_attempts": repair_attempts,
            "records": records,
            "formatted_results": formatted,
            "answer": answer,
            "execution_error": execution_error,
        }

    def __repr__(self) -> str:
        return (
            f"GraphRAGPipeline("
            f"schema={self.schema!r}, "
            f"db={self.db!r}, "
            f"max_repair_retries={self.max_repair_retries})"
        )
