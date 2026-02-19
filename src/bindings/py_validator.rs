use pyo3::prelude::*;
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

    pub fn validate(&self, query: &str) -> PyValidationResult {
        let result = self.inner.validate(query);
        PyValidationResult {
            is_valid: result.is_valid,
            errors: result.errors,
            syntax_errors: result.syntax_errors,
            semantic_errors: result.semantic_errors,
        }
    }

    /// Validate multiple queries at once and return one ``ValidationResult`` per query.
    ///
    /// Queries are validated in parallel (via Rayon) and the Python GIL is released
    /// for the duration of the batch, so other Python threads can run concurrently.
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
                    }
                })
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        "CypherValidator()".to_string()
    }
}
