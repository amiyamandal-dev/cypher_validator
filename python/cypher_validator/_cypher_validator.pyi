"""
Type stubs for the _cypher_validator Rust extension module.

These stubs provide IDE autocompletion and mypy/pyright type-checking support
for the compiled PyO3 classes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Schema:
    """Graph schema definition: node labels with properties, and relationship types.

    Example::

        schema = Schema(
            nodes={"Person": ["name", "age"], "Movie": ["title", "year"]},
            relationships={"ACTED_IN": ("Person", "Movie", ["role"])},
        )
    """

    def __init__(
        self,
        nodes: Dict[str, List[str]],
        relationships: Dict[str, Tuple[str, str, List[str]]],
    ) -> None: ...

    @staticmethod
    def from_dict(d: Dict[str, object]) -> Schema:
        """Create a Schema from a plain Python dict (e.g. from ``to_dict()``)."""
        ...

    @staticmethod
    def from_neo4j(
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        sample_limit: int = 1000,
    ) -> Schema:
        """Create a Schema by introspecting a live Neo4j database.

        Convenience wrapper that creates a temporary :class:`Neo4jDatabase`,
        calls :meth:`~Neo4jDatabase.introspect_schema`, closes the connection,
        and returns the resulting :class:`Schema`.
        """
        ...

    # ----- Label / type queries -----

    def node_labels(self) -> List[str]:
        """Return all node labels declared in this schema."""
        ...

    def rel_types(self) -> List[str]:
        """Return all relationship types declared in this schema."""
        ...

    def has_node_label(self, label: str) -> bool:
        """Return True if *label* is declared in the schema."""
        ...

    def has_rel_type(self, rel_type: str) -> bool:
        """Return True if *rel_type* is declared in the schema."""
        ...

    # ----- Property inspection -----

    def node_properties(self, label: str) -> List[str]:
        """Return the list of properties declared for *label*."""
        ...

    def rel_properties(self, rel_type: str) -> List[str]:
        """Return the list of properties declared for *rel_type*."""
        ...

    def rel_endpoints(self, rel_type: str) -> Optional[Tuple[str, str]]:
        """Return ``(src_label, tgt_label)`` for *rel_type*, or None if unknown."""
        ...

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, object]:
        """Export the schema as a plain Python dict compatible with the constructor."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class ValidationDiagnostic:
    """A single structured diagnostic with error code, suggestion, and position."""

    code: str
    """Short error code (e.g. ``"E201"``)."""

    code_name: str
    """Human-readable code name (e.g. ``"UnknownNodeLabel"``)."""

    severity: str
    """``"error"`` or ``"warning"``."""

    message: str
    """Full human-readable message."""

    suggestion_original: Optional[str]
    """Original text fragment to replace, or ``None``."""

    suggestion_replacement: Optional[str]
    """Suggested replacement text, or ``None``."""

    suggestion_description: Optional[str]
    """Human-readable description of the suggestion, or ``None``."""

    position_line: Optional[int]
    """1-based line number (parse errors only), or ``None``."""

    position_col: Optional[int]
    """1-based column number (parse errors only), or ``None``."""

    def to_dict(self) -> Dict[str, object]:
        """Return the diagnostic as a plain Python dict."""
        ...

    def __repr__(self) -> str: ...


class ValidationResult:
    """Result of validating a Cypher query against a :class:`Schema`."""

    is_valid: bool
    """True when the query has no errors."""

    errors: List[str]
    """All errors combined (syntax + semantic)."""

    syntax_errors: List[str]
    """Parse / syntax errors only."""

    semantic_errors: List[str]
    """Schema-level semantic errors only."""

    warnings: List[str]
    """Advisory warnings (not errors)."""

    diagnostics: List[ValidationDiagnostic]
    """Structured diagnostics with error codes and suggestions."""

    fixed_query: Optional[str]
    """Auto-corrected query when all errors have suggestions, or ``None``."""

    def to_dict(self) -> Dict[str, object]:
        """Return the result as a plain Python dict (includes diagnostics)."""
        ...

    def to_json(self) -> str:
        """Return the result as a compact JSON string."""
        ...

    def __bool__(self) -> bool:
        """``bool(result)`` — True when the query is valid."""
        ...

    def __len__(self) -> int:
        """``len(result)`` — number of errors."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# CypherValidator
# ---------------------------------------------------------------------------

class CypherValidator:
    """Validate Cypher queries against a :class:`Schema`.

    Example::

        v = CypherValidator(schema)
        result = v.validate("MATCH (p:Person) RETURN p")
        result.is_valid    # True
    """

    def __init__(self, schema: Schema) -> None: ...

    def validate(self, query: str) -> ValidationResult:
        """Validate *query* and return a :class:`ValidationResult`."""
        ...

    def validate_batch(self, queries: List[str]) -> List[ValidationResult]:
        """Validate multiple queries at once and return one result per query.

        Queries are validated in parallel (Rayon) and the Python GIL is released
        for the duration, so other Python threads can run concurrently.
        Result order matches input order.

        Example::

            results = v.validate_batch(["MATCH (p:Person) RETURN p", "bad query"])
            for r in results:
                print(r.is_valid, r.errors)
        """
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# QueryInfo
# ---------------------------------------------------------------------------

class QueryInfo:
    """Structural information extracted from a parsed Cypher query.

    Returned by :func:`parse_query`.  Does **not** require a schema.
    """

    is_valid: bool
    """True when the query is syntactically valid."""

    errors: List[str]
    """Syntax error messages (empty when valid)."""

    labels_used: List[str]
    """Node labels referenced in the query (sorted, deduplicated)."""

    rel_types_used: List[str]
    """Relationship types referenced in the query (sorted, deduplicated)."""

    properties_used: List[str]
    """Property keys accessed in the query, e.g. ``["age", "name"]``."""

    def __bool__(self) -> bool:
        """``bool(info)`` — True when the query is valid."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# parse_query
# ---------------------------------------------------------------------------

def parse_query(query: str) -> QueryInfo:
    """Parse a Cypher query and return structural information.

    Does **not** require a :class:`Schema`.  Use :class:`CypherValidator`
    for schema-aware validation.

    Example::

        info = parse_query("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name")
        info.is_valid        # True
        info.labels_used     # ["Movie", "Person"]
        info.rel_types_used  # ["ACTED_IN"]
        info.properties_used # ["name"]
    """
    ...


# ---------------------------------------------------------------------------
# CypherGenerator
# ---------------------------------------------------------------------------

class CypherGenerator:
    """Generate random, schema-valid Cypher queries.

    Useful for testing, fuzzing, and populating example datasets.

    Example::

        gen = CypherGenerator(schema, seed=42)
        gen.generate("match_return")   # "MATCH (n:Person) RETURN n"
        gen.generate("order_by")       # "MATCH (n:Movie) RETURN n ORDER BY n.year"
    """

    def __init__(self, schema: Schema, seed: Optional[int] = None) -> None: ...

    def generate(self, query_type: str) -> str:
        """Generate a single Cypher query of the given type.

        Raises :class:`ValueError` for unrecognised *query_type*.

        Supported types (see also :meth:`supported_types`):
        ``"match_return"``, ``"match_where_return"``, ``"create"``,
        ``"merge"``, ``"aggregation"``, ``"match_relationship"``,
        ``"create_relationship"``, ``"match_set"``, ``"match_delete"``,
        ``"with_chain"``, ``"distinct_return"``, ``"order_by"``,
        ``"unwind"``.
        """
        ...

    @staticmethod
    def supported_types() -> List[str]:
        """Return the list of all supported query type strings."""
        ...

    def __repr__(self) -> str: ...
