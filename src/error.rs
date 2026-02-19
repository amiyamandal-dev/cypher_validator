#[derive(Debug, thiserror::Error)]
pub enum CypherError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Semantic error: {0}")]
    SemanticError(String),
    #[error("Generator error: {0}")]
    GeneratorError(String),
}

impl From<CypherError> for pyo3::PyErr {
    fn from(e: CypherError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}
