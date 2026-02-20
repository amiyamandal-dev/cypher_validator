use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use crate::bindings::py_schema::PySchema;
use crate::validator::CypherValidator;

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
}

#[pymethods]
impl PyValidationResult {
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
    ///
    /// Useful for serialising into an LLM retry prompt or logging pipeline::
    ///
    ///     result = validator.validate(query)
    ///     if not result.is_valid:
    ///         payload = result.to_dict()
    ///         # payload["errors"], payload["warnings"], etc.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("is_valid", self.is_valid)?;
        d.set_item("errors", self.errors.clone())?;
        d.set_item("syntax_errors", self.syntax_errors.clone())?;
        d.set_item("semantic_errors", self.semantic_errors.clone())?;
        d.set_item("warnings", self.warnings.clone())?;
        Ok(d)
    }

    /// Return the result as a compact JSON string.
    ///
    /// Example::
    ///
    ///     json_str = result.to_json()
    ///     # '{"is_valid":false,"errors":[...],"warnings":[...],...}'
    fn to_json(&self) -> String {
        serde_json::json!({
            "is_valid": self.is_valid,
            "errors": self.errors,
            "syntax_errors": self.syntax_errors,
            "semantic_errors": self.semantic_errors,
            "warnings": self.warnings,
        })
        .to_string()
    }
}

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
    ///
    /// Useful for building LLM retry prompts — pass ``validator.schema.to_cypher_context()``
    /// (or ``to_prompt()`` / ``to_markdown()``) directly into the system message so the
    /// model knows which labels, types, and properties are valid::
    ///
    ///     result = validator.validate(llm_query)
    ///     if not result.is_valid:
    ///         schema_hint = validator.schema.to_cypher_context()
    ///         retry_prompt = f"Fix this Cypher query.\\nErrors: {result.errors}\\nSchema:\\n{schema_hint}"
    #[getter]
    pub fn schema(&self) -> PySchema {
        PySchema { inner: self.inner.schema.clone() }
    }

    pub fn validate(&self, query: &str) -> PyValidationResult {
        let result = self.inner.validate(query);
        PyValidationResult {
            is_valid: result.is_valid,
            errors: result.errors,
            syntax_errors: result.syntax_errors,
            semantic_errors: result.semantic_errors,
            warnings: result.warnings,
        }
    }

    /// Validate multiple queries at once and return one ``ValidationResult`` per query.
    ///
    /// Queries are validated in parallel (via Rayon) and the Python GIL is released
    /// for the duration of the batch, so other Python threads (e.g. asyncio event loop)
    /// are not blocked.
    ///
    /// Example::
    ///
    ///     results = validator.validate_batch([
    ///         "MATCH (p:Person) RETURN p",
    ///         "MATCH (p:BadLabel) RETURN p",
    ///     ])
    ///     for r in results:
    ///         print(r.is_valid, r.errors)
    pub fn validate_batch(&self, py: Python<'_>, queries: Vec<String>) -> Vec<PyValidationResult> {
        // Release the GIL while doing the parallel Rust work so that other
        // Python threads (e.g. asyncio event loop) are not blocked.
        py.detach(|| {
            queries
                .par_iter()
                .map(|q| {
                    let result = self.inner.validate(q);
                    PyValidationResult {
                        is_valid: result.is_valid,
                        errors: result.errors,
                        syntax_errors: result.syntax_errors,
                        semantic_errors: result.semantic_errors,
                        warnings: result.warnings,
                    }
                })
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        "CypherValidator()".to_string()
    }
}
