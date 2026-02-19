pub mod error;
pub mod schema;
pub mod parser;
pub mod validator;
pub mod generator;
pub mod bindings;

use pyo3::prelude::*;
use bindings::py_schema::PySchema;
use bindings::py_validator::{PyCypherValidator, PyValidationResult};
use bindings::py_generator::PyCypherGenerator;
use bindings::py_parser::{PyQueryInfo, parse_query};

#[pymodule]
fn _cypher_validator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySchema>()?;
    m.add_class::<PyCypherValidator>()?;
    m.add_class::<PyValidationResult>()?;
    m.add_class::<PyCypherGenerator>()?;
    m.add_class::<PyQueryInfo>()?;
    m.add_function(wrap_pyfunction!(parse_query, m)?)?;
    Ok(())
}
