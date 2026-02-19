use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::schema::Schema;

#[pyclass(name = "Schema")]
pub struct PySchema {
    pub inner: Schema,
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (nodes, relationships))]
    pub fn new(nodes: &Bound<'_, PyDict>, relationships: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut node_map: HashMap<String, Vec<String>> = HashMap::new();
        for (k, v) in nodes.iter() {
            let label: String = k.extract()?;
            let props: Vec<String> = v.extract()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    format!("Expected list[str] for node properties of label '{}'", label)
                ))?;
            node_map.insert(label, props);
        }

        let mut rel_map: HashMap<String, (String, String, Vec<String>)> = HashMap::new();
        for (k, v) in relationships.iter() {
            let rel_type: String = k.extract()?;
            let (src, tgt, props): (String, String, Vec<String>) = v.extract()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    format!(
                        "Expected (str, str, list[str]) for relationship '{}', e.g. ('Person', 'Movie', ['role'])",
                        rel_type
                    )
                ))?;
            rel_map.insert(rel_type, (src, tgt, props));
        }

        Ok(PySchema { inner: Schema::new(node_map, rel_map) })
    }

    // ===== Construction =====

    /// Create a Schema from a plain Python dict (e.g. previously exported with ``to_dict()``).
    ///
    /// Example::
    ///
    ///     d = {
    ///         "nodes": {"Person": ["name", "age"], "Movie": ["title"]},
    ///         "relationships": {"ACTED_IN": ("Person", "Movie", ["role"])},
    ///     }
    ///     schema = Schema.from_dict(d)
    #[staticmethod]
    pub fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let nodes_obj = d
            .get_item("nodes")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("'nodes' key missing from dict"))?;
        let nodes = nodes_obj.cast::<PyDict>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'nodes' must be a dict"))?;

        let rels_obj = d
            .get_item("relationships")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("'relationships' key missing from dict"))?;
        let relationships = rels_obj.cast::<PyDict>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'relationships' must be a dict"))?;

        Self::new(nodes, relationships)
    }

    // ===== Label / type queries =====

    fn node_labels(&self) -> Vec<String> {
        self.inner.node_labels().into_iter().map(|s| s.to_string()).collect()
    }

    fn rel_types(&self) -> Vec<String> {
        self.inner.rel_types().into_iter().map(|s| s.to_string()).collect()
    }

    fn has_node_label(&self, label: &str) -> bool {
        self.inner.has_node_label(label)
    }

    fn has_rel_type(&self, rel_type: &str) -> bool {
        self.inner.has_rel_type(rel_type)
    }

    // ===== Property inspection =====

    /// Return the list of properties declared for a node label.
    fn node_properties(&self, label: &str) -> Vec<String> {
        self.inner.node_properties(label).into_iter().map(|s| s.to_string()).collect()
    }

    /// Return the list of properties declared for a relationship type.
    fn rel_properties(&self, rel_type: &str) -> Vec<String> {
        self.inner.relationships.get(rel_type)
            .map_or(vec![], |(_, _, props)| props.iter().map(|s| s.to_string()).collect())
    }

    /// Return `(src_label, tgt_label)` for a relationship type, or None if unknown.
    fn rel_endpoints(&self, rel_type: &str) -> Option<(String, String)> {
        self.inner.rel_src_tgt(rel_type).map(|(s, t)| (s.to_string(), t.to_string()))
    }

    // ===== Serialization =====

    /// Export the schema as a plain Python dict compatible with the Schema constructor.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);

        let nodes = PyDict::new(py);
        for (label, props) in &self.inner.nodes {
            let prop_list = PyList::new(py, props.iter().map(|s| s.as_str()))?;
            nodes.set_item(label, prop_list)?;
        }
        d.set_item("nodes", nodes)?;

        let rels = PyDict::new(py);
        for (rel_type, (src, tgt, props)) in &self.inner.relationships {
            let prop_list = PyList::new(py, props.iter().map(|s| s.as_str()))?;
            // Build (src, tgt, [props]) tuple
            let tup = pyo3::types::PyTuple::new(py, [
                src.as_str().into_pyobject(py)?.into_any().unbind(),
                tgt.as_str().into_pyobject(py)?.into_any().unbind(),
                prop_list.into_any().unbind(),
            ])?;
            rels.set_item(rel_type, tup)?;
        }
        d.set_item("relationships", rels)?;

        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Schema(nodes={:?}, relationships={:?})",
            self.inner.node_labels(),
            self.inner.rel_types()
        )
    }
}
