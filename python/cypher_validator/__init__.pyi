"""
Type stubs for the ``cypher_validator`` package.

Re-exports the Rust core and the pure-Python GLiNER2 integration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

        Parameters
        ----------
        relations:
            Output from :meth:`GLiNER2RelationExtractor.extract_relations`.
        mode:
            ``"match"``, ``"merge"``, or ``"create"``.
        **kwargs:
            Forwarded to the chosen generator method
            (e.g. ``return_clause="RETURN *"``).

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

    Example::

        pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
        cypher = pipeline("John works for Apple Inc.", ["works_for"], mode="merge")
    """

    extractor: GLiNER2RelationExtractor
    converter: RelationToCypherConverter

    def __init__(
        self,
        extractor: GLiNER2RelationExtractor,
        schema: Optional[Any] = None,
        name_property: str = "name",
    ) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ...,
        schema: Optional[Any] = None,
        threshold: float = 0.5,
        name_property: str = "name",
    ) -> "NLToCypher":
        """Create a pipeline by loading a pretrained GLiNER2 model.

        Raises
        ------
        ImportError
            If the ``gliner2`` package is not installed.
        """
        ...

    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Extract relations from *text* and return a Cypher query.

        Returns an empty string when no relations were found.
        """
        ...

    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """Extract relations and return both the raw dict and the Cypher string.

        Returns
        -------
        tuple
            ``(relations_dict, cypher_query)``
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
    "GLiNER2RelationExtractor",
    "RelationToCypherConverter",
    "NLToCypher",
]
