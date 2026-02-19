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

    #[staticmethod]
    pub fn supported_types() -> Vec<&'static str> {
        CypherGenerator::supported_types()
    }

    fn __repr__(&self) -> String {
        "CypherGenerator()".to_string()
    }
}
