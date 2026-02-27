/// Machine-readable error codes, severity levels, and fix suggestions for
/// Cypher validation diagnostics.
///
/// Each `ValidationDiagnostic` carries an `ErrorCode` (e.g. `E201`), a human-readable
/// message, an optional `Suggestion` for auto-fix, and an optional source position.

use std::fmt;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/// Structured error code for a validation diagnostic.
///
/// Naming convention:
/// - `E1xx` — parse / syntax errors
/// - `E2xx` — unknown label / relationship type
/// - `E3xx` — unknown property
/// - `E4xx` — wrong endpoint label
/// - `E5xx` — scope / binding errors
/// - `E6xx` — type / aggregate misuse
/// - `W1xx` — performance warnings
/// - `W2xx` — complexity warnings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCode {
    // Parse errors
    E101ParseError,

    // Unknown node label
    E201UnknownNodeLabel,
    E202UnknownNodeLabelInSet,
    E203UnknownNodeLabelInRemove,

    // Unknown relationship type
    E211UnknownRelType,

    // Unknown property
    E301UnknownNodeProperty,
    E302UnknownRelProperty,
    E303UnknownNodePropertyOnVar,
    E304UnknownRelPropertyOnVar,

    // Wrong endpoint
    E401WrongEndpointLabel,

    // Scope / binding
    E501UnboundVariable,
    E502DuplicateProjectionName,

    // Type / aggregate misuse
    E601AggregateStringArg,
    E602AggregateInForbiddenContext,
    E611PaginationTypeString,
    E612PaginationTypeBool,
    E613PaginationTypeNull,
    E614PaginationTypeFloat,

    // Function validation
    E603WrongArity,

    // Warnings
    W101CartesianProduct,
    W103UnknownFunction,
    W104DistinctOnNonAggregate,
    W201UnlabeledFullScan,
    W202UnboundedVarLength,
}

impl ErrorCode {
    /// Short numeric code string (e.g. `"E201"`).
    pub fn code(&self) -> &'static str {
        match self {
            Self::E101ParseError => "E101",
            Self::E201UnknownNodeLabel => "E201",
            Self::E202UnknownNodeLabelInSet => "E202",
            Self::E203UnknownNodeLabelInRemove => "E203",
            Self::E211UnknownRelType => "E211",
            Self::E301UnknownNodeProperty => "E301",
            Self::E302UnknownRelProperty => "E302",
            Self::E303UnknownNodePropertyOnVar => "E303",
            Self::E304UnknownRelPropertyOnVar => "E304",
            Self::E401WrongEndpointLabel => "E401",
            Self::E501UnboundVariable => "E501",
            Self::E502DuplicateProjectionName => "E502",
            Self::E601AggregateStringArg => "E601",
            Self::E602AggregateInForbiddenContext => "E602",
            Self::E611PaginationTypeString => "E611",
            Self::E612PaginationTypeBool => "E612",
            Self::E613PaginationTypeNull => "E613",
            Self::E614PaginationTypeFloat => "E614",
            Self::E603WrongArity => "E603",
            Self::W101CartesianProduct => "W101",
            Self::W103UnknownFunction => "W103",
            Self::W104DistinctOnNonAggregate => "W104",
            Self::W201UnlabeledFullScan => "W201",
            Self::W202UnboundedVarLength => "W202",
        }
    }

    /// Human-readable name (e.g. `"UnknownNodeLabel"`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::E101ParseError => "ParseError",
            Self::E201UnknownNodeLabel => "UnknownNodeLabel",
            Self::E202UnknownNodeLabelInSet => "UnknownNodeLabelInSet",
            Self::E203UnknownNodeLabelInRemove => "UnknownNodeLabelInRemove",
            Self::E211UnknownRelType => "UnknownRelType",
            Self::E301UnknownNodeProperty => "UnknownNodeProperty",
            Self::E302UnknownRelProperty => "UnknownRelProperty",
            Self::E303UnknownNodePropertyOnVar => "UnknownNodePropertyOnVar",
            Self::E304UnknownRelPropertyOnVar => "UnknownRelPropertyOnVar",
            Self::E401WrongEndpointLabel => "WrongEndpointLabel",
            Self::E501UnboundVariable => "UnboundVariable",
            Self::E502DuplicateProjectionName => "DuplicateProjectionName",
            Self::E601AggregateStringArg => "AggregateStringArg",
            Self::E602AggregateInForbiddenContext => "AggregateInForbiddenContext",
            Self::E611PaginationTypeString => "PaginationTypeString",
            Self::E612PaginationTypeBool => "PaginationTypeBool",
            Self::E613PaginationTypeNull => "PaginationTypeNull",
            Self::E614PaginationTypeFloat => "PaginationTypeFloat",
            Self::E603WrongArity => "WrongArity",
            Self::W101CartesianProduct => "CartesianProduct",
            Self::W103UnknownFunction => "UnknownFunction",
            Self::W104DistinctOnNonAggregate => "DistinctOnNonAggregate",
            Self::W201UnlabeledFullScan => "UnlabeledFullScan",
            Self::W202UnboundedVarLength => "UnboundedVarLength",
        }
    }

    pub fn severity(&self) -> Severity {
        match self {
            Self::W101CartesianProduct
            | Self::W103UnknownFunction
            | Self::W104DistinctOnNonAggregate
            | Self::W201UnlabeledFullScan
            | Self::W202UnboundedVarLength => Severity::Warning,
            _ => Severity::Error,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
        }
    }
}

// ---------------------------------------------------------------------------
// Suggestion
// ---------------------------------------------------------------------------

/// A concrete fix suggestion — "replace `original` with `replacement`".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    /// The text fragment to replace (e.g. `:Persn`).
    pub original: String,
    /// The replacement text (e.g. `:Person`).
    pub replacement: String,
    /// Human-readable description of the fix.
    pub description: String,
}

// ---------------------------------------------------------------------------
// ValidationDiagnostic
// ---------------------------------------------------------------------------

/// A single structured diagnostic produced during validation.
#[derive(Debug, Clone)]
pub struct ValidationDiagnostic {
    pub code: ErrorCode,
    pub severity: Severity,
    pub message: String,
    pub suggestion: Option<Suggestion>,
    /// `(line, col)` — 1-based, only populated for parse errors.
    pub position: Option<(u32, u32)>,
}

impl ValidationDiagnostic {
    pub fn error(code: ErrorCode, message: String) -> Self {
        Self {
            severity: Severity::Error,
            code,
            message,
            suggestion: None,
            position: None,
        }
    }

    pub fn warning(code: ErrorCode, message: String) -> Self {
        Self {
            severity: Severity::Warning,
            code,
            message,
            suggestion: None,
            position: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestion = Some(suggestion);
        self
    }

    pub fn with_position(mut self, line: u32, col: u32) -> Self {
        self.position = Some((line, col));
        self
    }
}
