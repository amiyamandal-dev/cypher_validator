"""
Type stubs for the ``cypher_validator`` package.

Re-exports the Rust core and the pure-Python GLiNER2 / Neo4j integration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

# Re-export Rust core types
from cypher_validator._cypher_validator import (
    Schema as Schema,
    ValidationResult as ValidationResult,
    CypherValidator as CypherValidator,
    QueryInfo as QueryInfo,
    CypherGenerator as CypherGenerator,
    parse_query as parse_query,
)

# ---------------------------------------------------------------------------
# Schema (extended stubs for new methods)
# ---------------------------------------------------------------------------

class Schema:
    """Graph schema: node labels with their allowed property sets,
    and relationship types with their (source label, target label, property set).
    """

    def __init__(
        self,
        nodes: Dict[str, List[str]],
        relationships: Dict[str, Tuple[str, str, List[str]]],
    ) -> None: ...

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Schema":
        """Create a Schema from a plain Python dict."""
        ...

    @staticmethod
    def from_json(json_str: str) -> "Schema":
        """Deserialise a Schema from a JSON string produced by ``to_json()``."""
        ...

    def node_labels(self) -> List[str]: ...
    def rel_types(self) -> List[str]: ...
    def has_node_label(self, label: str) -> bool: ...
    def has_rel_type(self, rel_type: str) -> bool: ...
    def node_properties(self, label: str) -> List[str]: ...
    def rel_properties(self, rel_type: str) -> List[str]: ...
    def rel_endpoints(self, rel_type: str) -> Optional[Tuple[str, str]]: ...
    def to_dict(self) -> Dict[str, Any]: ...

    def to_json(self) -> str:
        """Serialise the schema to a JSON string (round-trips with ``from_json``)."""
        ...

    def merge(self, other: "Schema") -> "Schema":
        """Return a new Schema that is the union of *self* and *other*.

        Property sets for shared labels/types are merged (union).
        """
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# CypherGenerator (extended stubs)
# ---------------------------------------------------------------------------

class CypherGenerator:
    """Generates random but schema-valid Cypher queries."""

    def __init__(self, schema: Schema, seed: Optional[int] = None) -> None: ...

    def generate(self, query_type: str) -> str:
        """Generate a single random Cypher query of the given type."""
        ...

    def generate_batch(self, query_type: str, n: int) -> List[str]:
        """Generate *n* queries of the given type in a single call.

        Equivalent to calling :meth:`generate` *n* times but avoids per-call
        Python overhead.
        """
        ...

    @staticmethod
    def supported_types() -> List[str]: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Neo4jDatabase
# ---------------------------------------------------------------------------

class Neo4jDatabase:
    """Thin wrapper around the Neo4j Python driver for executing Cypher queries.

    Requires the ``neo4j`` package (``pip install neo4j``).

    Can be used as a context manager::

        with Neo4jDatabase("bolt://localhost:7687", "neo4j", "password") as db:
            results = db.execute("MATCH (n:Person) RETURN n LIMIT 5")

    Pass to :class:`NLToCypher` to enable database-aware execution::

        db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
        pipeline = NLToCypher.from_pretrained(..., db=db)
        cypher, results = pipeline("John works for Apple.", ["works_for"],
                                   mode="create", execute=True)
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        """
        Parameters
        ----------
        uri:
            Bolt/neo4j URI, e.g. ``"bolt://localhost:7687"``.
        username:
            Neo4j username.
        password:
            Neo4j password.
        database:
            Target database name (default: ``"neo4j"``).

        Raises
        ------
        ImportError
            If the ``neo4j`` package is not installed.
        """
        ...

    def execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as a list of record dicts.

        Parameters
        ----------
        cypher:
            Cypher query string.
        parameters:
            Optional query parameters.

        Returns
        -------
        list[dict]
            One dict per returned record.  Empty list for queries with no
            ``RETURN`` clause (e.g. a bare ``CREATE``).
        """
        ...

    def execute_many(
        self,
        queries: List[str],
        parameters_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Execute multiple Cypher queries in sequence and return all results.

        Parameters
        ----------
        queries:
            List of Cypher query strings to execute.
        parameters_list:
            Optional list of parameter dicts, one per query.

        Returns
        -------
        list[list[dict]]
            One inner list per query.
        """
        ...

    def close(self) -> None:
        """Close the underlying driver and release all connections."""
        ...

    def __enter__(self) -> "Neo4jDatabase": ...
    def __exit__(self, *args: Any) -> None: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# RelationToCypherConverter
# ---------------------------------------------------------------------------

class RelationToCypherConverter:
    """Convert GLiNER2 relation-extraction output to a Cypher query.

    Example::

        converter = RelationToCypherConverter(schema=schema)
        cypher = converter.convert(results, mode="merge")
    """

    schema: Any
    name_property: str

    def __init__(
        self,
        schema: Optional[Any] = None,
        name_property: str = "name",
    ) -> None: ...

    def to_match_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build MATCH clauses for the extracted relations.

        Returns an empty string when nothing was extracted.
        """
        ...

    def to_merge_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build MERGE clauses to upsert extracted entities/relationships.

        Node labels are added automatically when a schema is provided.
        Returns an empty string when nothing was extracted.
        """
        ...

    def to_create_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build CREATE clauses to insert extracted entities/relationships.

        Returns an empty string when nothing was extracted.
        """
        ...

    def convert(
        self,
        relations: Dict[str, Any],
        mode: str = "match",
        **kwargs: Any,
    ) -> str:
        """Convert extracted relations to a Cypher query.

        Raises
        ------
        ValueError
            For an unrecognised *mode*.
        """
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# GLiNER2RelationExtractor
# ---------------------------------------------------------------------------

class GLiNER2RelationExtractor:
    """Thin wrapper around the GLiNER2 model for zero-shot relation extraction.

    Requires the ``gliner2`` package (``pip install gliner2``).

    Example::

        extractor = GLiNER2RelationExtractor.from_pretrained("fastino/gliner2-large-v1")
        results = extractor.extract_relations(
            "John works for Apple Inc.",
            ["works_for"],
        )
        # {"relation_extraction": {"works_for": [("John", "Apple Inc.")]}}
    """

    DEFAULT_MODEL: str

    threshold: float

    def __init__(self, model: Any, threshold: float = 0.5) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ...,
        threshold: float = 0.5,
    ) -> "GLiNER2RelationExtractor":
        """Load a pretrained GLiNER2 model from the HuggingFace Hub or a local path.

        Raises
        ------
        ImportError
            If the ``gliner2`` package is not installed.
        """
        ...

    def extract_relations(
        self,
        text: str,
        relation_types: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract relations from *text* using the loaded GLiNER2 model.

        All requested relation types are present in the output dict
        (with an empty list when a relation was not found).

        Returns
        -------
        dict
            ``{"relation_extraction": {"works_for": [("John", "Apple Inc.")], ...}}``
        """
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# NLToCypher
# ---------------------------------------------------------------------------

class NLToCypher:
    """End-to-end pipeline: natural language text → Cypher query.

    Chains :class:`GLiNER2RelationExtractor` and :class:`RelationToCypherConverter`.
    Optionally executes the generated query against a Neo4j database when a
    :class:`Neo4jDatabase` is provided and ``execute=True`` is passed.

    Example — generation only::

        pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
        cypher = pipeline("John works for Apple Inc.", ["works_for"], mode="create")

    Example — db-aware execution::

        db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
        pipeline = NLToCypher.from_pretrained(..., db=db)
        cypher, results = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            mode="create",
            execute=True,
        )

    Example — credentials from environment variables::

        # export NEO4J_URI=bolt://localhost:7687
        # export NEO4J_PASSWORD=secret
        pipeline = NLToCypher.from_env("fastino/gliner2-large-v1", schema=schema)
        cypher, results = pipeline("...", ["works_for"], mode="create", execute=True)
    """

    extractor: GLiNER2RelationExtractor
    converter: RelationToCypherConverter
    db: Optional[Neo4jDatabase]

    def __init__(
        self,
        extractor: GLiNER2RelationExtractor,
        schema: Optional[Any] = None,
        name_property: str = "name",
        db: Optional[Neo4jDatabase] = None,
    ) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ...,
        schema: Optional[Any] = None,
        threshold: float = 0.5,
        name_property: str = "name",
        db: Optional[Neo4jDatabase] = None,
    ) -> "NLToCypher":
        """Create a pipeline by loading a pretrained GLiNER2 model.

        Raises
        ------
        ImportError
            If the ``gliner2`` package is not installed.
        """
        ...

    @classmethod
    def from_env(
        cls,
        model_name: str = ...,
        schema: Optional[Any] = None,
        threshold: float = 0.5,
        name_property: str = "name",
        database: str = "neo4j",
    ) -> "NLToCypher":
        """Create a pipeline reading Neo4j credentials from environment variables.

        Reads ``NEO4J_URI``, ``NEO4J_USERNAME`` (default ``"neo4j"``), and
        ``NEO4J_PASSWORD`` from the process environment.

        Raises
        ------
        KeyError
            If ``NEO4J_URI`` or ``NEO4J_PASSWORD`` are not set.
        ImportError
            If the ``gliner2`` or ``neo4j`` packages are not installed.
        """
        ...

    # execute=False → str
    @overload
    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = ...,
        threshold: Optional[float] = ...,
        execute: Literal[False] = ...,
        **kwargs: Any,
    ) -> str: ...

    # execute=True → (str, list[dict])
    @overload
    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = ...,
        threshold: Optional[float] = ...,
        execute: Literal[True] = ...,
        **kwargs: Any,
    ) -> Tuple[str, List[Dict[str, Any]]]: ...

    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        """Extract relations from *text* and return a Cypher query.

        Parameters
        ----------
        text:
            Input sentence or passage.
        relation_types:
            Relation labels to extract.
        mode:
            ``"match"``, ``"merge"``, or ``"create"``.
        threshold:
            Override extractor confidence threshold.
        execute:
            When ``True``, execute the query against the database and return
            ``(cypher, results)``.  Requires ``db`` to be set.
        **kwargs:
            Forwarded to the Cypher converter (e.g. ``return_clause="RETURN *"``).

        Returns
        -------
        str
            Cypher query when ``execute=False`` (default).
        tuple[str, list[dict]]
            ``(cypher, db_results)`` when ``execute=True``.

        Raises
        ------
        RuntimeError
            When ``execute=True`` but no ``db`` was provided.
        """
        ...

    # execute=False → (dict, str)
    @overload
    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = ...,
        threshold: Optional[float] = ...,
        execute: Literal[False] = ...,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]: ...

    # execute=True → (dict, str, list[dict])
    @overload
    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = ...,
        threshold: Optional[float] = ...,
        execute: Literal[True] = ...,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]: ...

    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[Dict[str, Any], str], Tuple[Dict[str, Any], str, List[Dict[str, Any]]]]:
        """Extract relations and return the raw dict and the Cypher string.

        Returns
        -------
        tuple[dict, str]
            ``(relations_dict, cypher_query)`` when ``execute=False``.
        tuple[dict, str, list[dict]]
            ``(relations_dict, cypher_query, db_results)`` when ``execute=True``.

        Raises
        ------
        RuntimeError
            When ``execute=True`` but no ``db`` was provided.
        """
        ...

    def __repr__(self) -> str: ...


__all__ = [
    "Schema",
    "CypherValidator",
    "ValidationResult",
    "CypherGenerator",
    "QueryInfo",
    "parse_query",
    "Neo4jDatabase",
    "GLiNER2RelationExtractor",
    "RelationToCypherConverter",
    "NLToCypher",
]
