use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use crate::bindings::py_schema::PySchema;
use crate::validator::{CypherValidator, ValidationResult};
use crate::diagnostics::ValidationDiagnostic;

// ---------------------------------------------------------------------------
// PyValidationDiagnostic
// ---------------------------------------------------------------------------

#[pyclass(name = "ValidationDiagnostic")]
pub struct PyValidationDiagnostic {
    /// Short error code (e.g. "E201").
    #[pyo3(get)]
    pub code: String,
    /// Human-readable code name (e.g. "UnknownNodeLabel").
    #[pyo3(get)]
    pub code_name: String,
    /// "error" or "warning".
    #[pyo3(get)]
    pub severity: String,
    /// Full human-readable message.
    #[pyo3(get)]
    pub message: String,
    /// Original text fragment to replace, or None.
    #[pyo3(get)]
    pub suggestion_original: Option<String>,
    /// Suggested replacement text, or None.
    #[pyo3(get)]
    pub suggestion_replacement: Option<String>,
    /// Human-readable description of the suggestion, or None.
    #[pyo3(get)]
    pub suggestion_description: Option<String>,
    /// 1-based line number (parse errors only), or None.
    #[pyo3(get)]
    pub position_line: Option<u32>,
    /// 1-based column number (parse errors only), or None.
    #[pyo3(get)]
    pub position_col: Option<u32>,
}

#[pymethods]
impl PyValidationDiagnostic {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("code", &self.code)?;
        d.set_item("code_name", &self.code_name)?;
        d.set_item("severity", &self.severity)?;
        d.set_item("message", &self.message)?;
        d.set_item("suggestion_original", &self.suggestion_original)?;
        d.set_item("suggestion_replacement", &self.suggestion_replacement)?;
        d.set_item("suggestion_description", &self.suggestion_description)?;
        d.set_item("position_line", self.position_line)?;
        d.set_item("position_col", self.position_col)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        if let Some(repl) = &self.suggestion_replacement {
            format!(
                "ValidationDiagnostic(code='{}', severity='{}', message='{}', suggestion='{}')",
                self.code, self.severity, self.message, repl
            )
        } else {
            format!(
                "ValidationDiagnostic(code='{}', severity='{}', message='{}')",
                self.code, self.severity, self.message
            )
        }
    }
}

impl PyValidationDiagnostic {
    fn from_rust(d: &ValidationDiagnostic) -> Self {
        let (suggestion_original, suggestion_replacement, suggestion_description) =
            if let Some(s) = &d.suggestion {
                (
                    Some(s.original.clone()),
                    Some(s.replacement.clone()),
                    Some(s.description.clone()),
                )
            } else {
                (None, None, None)
            };
        let (position_line, position_col) = if let Some((l, c)) = d.position {
            (Some(l), Some(c))
        } else {
            (None, None)
        };
        PyValidationDiagnostic {
            code: d.code.code().to_string(),
            code_name: d.code.name().to_string(),
            severity: d.severity.to_string(),
            message: d.message.clone(),
            suggestion_original,
            suggestion_replacement,
            suggestion_description,
            position_line,
            position_col,
        }
    }
}

// ---------------------------------------------------------------------------
// PyValidationResult
// ---------------------------------------------------------------------------

#[pyclass(name = "ValidationResult")]
pub struct PyValidationResult {
    #[pyo3(get)]
    pub is_valid: bool,
    /// All errors (syntax + semantic) combined.
    #[pyo3(get)]
    pub errors: Vec<String>,
    /// Parse/syntax errors only.
    #[pyo3(get)]
    pub syntax_errors: Vec<String>,
    /// Schema-level semantic errors only.
    #[pyo3(get)]
    pub semantic_errors: Vec<String>,
    /// Advisory warnings (not errors — query still runs, but may be slow or incorrect).
    #[pyo3(get)]
    pub warnings: Vec<String>,
    /// Auto-corrected query when all errors are fixable, or None.
    #[pyo3(get)]
    pub fixed_query: Option<String>,
    /// Internal storage for diagnostics (exposed via getter).
    diagnostics_inner: Vec<ValidationDiagnostic>,
}

impl From<ValidationResult> for PyValidationResult {
    fn from(r: ValidationResult) -> Self {
        PyValidationResult {
            is_valid: r.is_valid,
            errors: r.errors,
            syntax_errors: r.syntax_errors,
            semantic_errors: r.semantic_errors,
            warnings: r.warnings,
            fixed_query: r.fixed_query,
            diagnostics_inner: r.diagnostics,
        }
    }
}

#[pymethods]
impl PyValidationResult {
    /// Structured diagnostics with error codes, suggestions, and positions.
    #[getter]
    fn diagnostics(&self) -> Vec<PyValidationDiagnostic> {
        self.diagnostics_inner
            .iter()
            .map(PyValidationDiagnostic::from_rust)
            .collect()
    }

    /// `bool(result)` → True if valid.
    fn __bool__(&self) -> bool {
        self.is_valid
    }

    /// `len(result)` → number of errors.
    fn __len__(&self) -> usize {
        self.errors.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationResult(is_valid={}, errors={:?})",
            self.is_valid, self.errors
        )
    }

    /// Return the result as a plain Python dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("is_valid", self.is_valid)?;
        d.set_item("errors", self.errors.clone())?;
        d.set_item("syntax_errors", self.syntax_errors.clone())?;
        d.set_item("semantic_errors", self.semantic_errors.clone())?;
        d.set_item("warnings", self.warnings.clone())?;
        d.set_item("fixed_query", &self.fixed_query)?;
        // diagnostics as list of dicts
        let diag_dicts: Vec<Bound<'py, PyDict>> = self.diagnostics_inner
            .iter()
            .map(|diag| PyValidationDiagnostic::from_rust(diag).to_dict(py))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("diagnostics", diag_dicts)?;
        Ok(d)
    }

    /// Return the result as a compact JSON string.
    fn to_json(&self) -> String {
        let diags: Vec<serde_json::Value> = self.diagnostics_inner
            .iter()
            .map(|d| {
                let mut obj = serde_json::json!({
                    "code": d.code.code(),
                    "code_name": d.code.name(),
                    "severity": d.severity.to_string(),
                    "message": d.message,
                });
                if let Some(s) = &d.suggestion {
                    obj["suggestion_original"] = serde_json::json!(s.original);
                    obj["suggestion_replacement"] = serde_json::json!(s.replacement);
                    obj["suggestion_description"] = serde_json::json!(s.description);
                }
                if let Some((l, c)) = d.position {
                    obj["position_line"] = serde_json::json!(l);
                    obj["position_col"] = serde_json::json!(c);
                }
                obj
            })
            .collect();

        serde_json::json!({
            "is_valid": self.is_valid,
            "errors": self.errors,
            "syntax_errors": self.syntax_errors,
            "semantic_errors": self.semantic_errors,
            "warnings": self.warnings,
            "fixed_query": self.fixed_query,
            "diagnostics": diags,
        })
        .to_string()
    }
}

// ---------------------------------------------------------------------------
// PyCypherValidator
// ---------------------------------------------------------------------------

#[pyclass(name = "CypherValidator")]
pub struct PyCypherValidator {
    inner: CypherValidator,
}

#[pymethods]
impl PyCypherValidator {
    #[new]
    pub fn new(schema: &PySchema) -> Self {
        PyCypherValidator {
            inner: CypherValidator::new(schema.inner.clone()),
        }
    }

    /// The schema this validator was constructed with.
    #[getter]
    pub fn schema(&self) -> PySchema {
        PySchema { inner: self.inner.schema.clone() }
    }

    pub fn validate(&self, query: &str) -> PyValidationResult {
        self.inner.validate(query).into()
    }

    /// Validate multiple queries at once and return one ``ValidationResult`` per query.
    ///
    /// Queries are validated in parallel (via Rayon) and the Python GIL is released
    /// for the duration of the batch, so other Python threads (e.g. asyncio event loop)
    /// are not blocked.
    pub fn validate_batch(&self, py: Python<'_>, queries: Vec<String>) -> Vec<PyValidationResult> {
        py.detach(|| {
            queries
                .par_iter()
                .map(|q| self.inner.validate(q).into())
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        "CypherValidator()".to_string()
    }
}
