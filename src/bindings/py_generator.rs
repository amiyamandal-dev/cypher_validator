use pyo3::prelude::*;
use crate::bindings::py_schema::PySchema;
use crate::generator::CypherGenerator;

#[pyclass(name = "CypherGenerator")]
pub struct PyCypherGenerator {
    inner: CypherGenerator,
}

#[pymethods]
impl PyCypherGenerator {
    #[new]
    #[pyo3(signature = (schema, seed=None))]
    pub fn new(schema: &PySchema, seed: Option<u64>) -> Self {
        PyCypherGenerator {
            inner: CypherGenerator::new(schema.inner.clone(), seed),
        }
    }

    pub fn generate(&mut self, query_type: &str) -> PyResult<String> {
        self.inner.generate(query_type).map_err(|e| e.into())
    }

    /// Generate *n* queries of the given type in a single call.
    ///
    /// Equivalent to calling :meth:`generate` *n* times but avoids per-call
    /// Python overhead.
    ///
    /// Parameters
    /// ----------
    /// query_type:
    ///     One of the types returned by :meth:`supported_types`.
    /// n:
    ///     Number of queries to generate.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Generated Cypher query strings.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If *query_type* is not supported.
    pub fn generate_batch(&mut self, query_type: &str, n: usize) -> PyResult<Vec<String>> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.inner.generate(query_type).map_err(|e| PyErr::from(e))?);
        }
        Ok(results)
    }

    #[staticmethod]
    pub fn supported_types() -> Vec<&'static str> {
        CypherGenerator::supported_types()
    }

    fn __repr__(&self) -> String {
        "CypherGenerator()".to_string()
    }
}
