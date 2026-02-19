use std::collections::{HashMap, HashSet};
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
        let mut node_map: HashMap<String, HashSet<String>> = HashMap::new();
        for (k, v) in nodes.iter() {
            let label: String = k.extract()?;
            let props: Vec<String> = v.extract()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    format!("Expected list[str] for node properties of label '{}'", label)
                ))?;
            node_map.insert(label, props.into_iter().collect());
        }

        let mut rel_map: HashMap<String, (String, String, HashSet<String>)> = HashMap::new();
        for (k, v) in relationships.iter() {
            let rel_type: String = k.extract()?;
            let (src, tgt, props): (String, String, Vec<String>) = v.extract()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    format!(
                        "Expected (str, str, list[str]) for relationship '{}', e.g. ('Person', 'Movie', ['role'])",
                        rel_type
                    )
                ))?;
            rel_map.insert(rel_type, (src, tgt, props.into_iter().collect()));
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

    /// Deserialise a Schema from a JSON string produced by ``to_json()``.
    ///
    /// Example::
    ///
    ///     json_str = schema.to_json()
    ///     schema2 = Schema.from_json(json_str)
    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let inner: Schema = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Invalid Schema JSON: {}", e)
            ))?;
        Ok(PySchema { inner })
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
        self.inner.rel_properties(rel_type).into_iter().map(|s| s.to_string()).collect()
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
        for label in self.inner.node_labels() {
            let props = self.inner.node_properties(label);
            let prop_list = PyList::new(py, props.iter().copied())?;
            nodes.set_item(label, prop_list)?;
        }
        d.set_item("nodes", nodes)?;

        let rels = PyDict::new(py);
        for rel_type in self.inner.rel_types() {
            if let Some((src, tgt)) = self.inner.rel_src_tgt(rel_type) {
                let props = self.inner.rel_properties(rel_type);
                let prop_list = PyList::new(py, props.iter().copied())?;
                let tup = pyo3::types::PyTuple::new(py, [
                    src.into_pyobject(py)?.into_any().unbind(),
                    tgt.into_pyobject(py)?.into_any().unbind(),
                    prop_list.into_any().unbind(),
                ])?;
                rels.set_item(rel_type, tup)?;
            }
        }
        d.set_item("relationships", rels)?;

        Ok(d)
    }

    /// Serialise the schema to a JSON string.
    ///
    /// The result can be stored, transmitted, and later restored via
    /// ``Schema.from_json()``.
    ///
    /// Example::
    ///
    ///     json_str = schema.to_json()
    ///     schema2 = Schema.from_json(json_str)
    ///     assert schema2 == schema  # round-trip safe
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Schema serialisation failed: {}", e)
            ))
    }

    /// Return a new Schema that is the union of *self* and *other*.
    ///
    /// * Node labels and relationship types from both schemas are combined.
    /// * When a label/type exists in both, the property sets are merged
    ///   (union), so no declared property is lost.
    /// * The *other* schema's endpoint labels take precedence for
    ///   relationship types that appear in both schemas.
    ///
    /// Example::
    ///
    ///     s1 = Schema({"Person": ["name"]}, {"KNOWS": ("Person", "Person", [])})
    ///     s2 = Schema({"Movie": ["title"]}, {"ACTED_IN": ("Person", "Movie", ["role"])})
    ///     merged = s1.merge(s2)
    ///     # merged has Person, Movie, KNOWS, ACTED_IN
    fn merge(&self, other: &PySchema) -> PySchema {
        let mut nodes = self.inner.nodes.clone();
        for (label, props) in &other.inner.nodes {
            nodes.entry(label.clone())
                .and_modify(|existing: &mut HashSet<String>| existing.extend(props.iter().cloned()))
                .or_insert_with(|| props.clone());
        }

        let mut relationships = self.inner.relationships.clone();
        for (rel_type, (src, tgt, props)) in &other.inner.relationships {
            relationships.entry(rel_type.clone())
                .and_modify(|(_, _, existing_props)| existing_props.extend(props.iter().cloned()))
                .or_insert_with(|| (src.clone(), tgt.clone(), props.clone()));
        }

        PySchema { inner: Schema::new(nodes, relationships) }
    }

    // ===== LLM prompt helpers =====

    /// Return a human-readable, LLM-friendly representation of the schema.
    ///
    /// Suitable for injecting into an LLM system prompt to tell the model which
    /// labels, relationship types, and properties exist in the graph.
    ///
    /// Example::
    ///
    ///     print(schema.to_prompt())
    ///     # Graph Schema
    ///     # ============
    ///     #
    ///     # Nodes
    ///     # -----
    ///     #   :Person                   name, age, email
    ///     #   :Company                  name, founded
    ///     #
    ///     # Relationships
    ///     # -------------
    ///     #   :WORKS_FOR                (Person)-->(Company)   since, role
    ///     #   :LIVES_IN                 (Person)-->(City)
    fn to_prompt(&self) -> String {
        let mut out = String::from("Graph Schema\n============\n\nNodes\n-----\n");
        for label in self.inner.node_labels() {
            let props = self.inner.node_properties(label);
            if props.is_empty() {
                out.push_str(&format!("  :{}\n", label));
            } else {
                out.push_str(&format!("  :{:<26}  {}\n", label, props.join(", ")));
            }
        }
        out.push_str("\nRelationships\n-------------\n");
        for rel_type in self.inner.rel_types() {
            let props = self.inner.rel_properties(rel_type);
            if let Some((src, tgt)) = self.inner.rel_src_tgt(rel_type) {
                let endpoints = format!("({})-->({})", src, tgt);
                if props.is_empty() {
                    out.push_str(&format!("  :{:<26}  {}\n", rel_type, endpoints));
                } else {
                    out.push_str(&format!(
                        "  :{:<26}  {}   {}\n",
                        rel_type,
                        endpoints,
                        props.join(", ")
                    ));
                }
            }
        }
        out
    }

    /// Return the schema formatted as a Markdown table.
    ///
    /// Useful for LLM contexts that render Markdown, documentation, or
    /// README sections.
    ///
    /// Example::
    ///
    ///     print(schema.to_markdown())
    ///     # ### Nodes
    ///     # | Label | Properties |
    ///     # |---|---|
    ///     # | :Person | name, age, email |
    ///     # ...
    fn to_markdown(&self) -> String {
        let mut out = String::from("### Nodes\n\n| Label | Properties |\n|---|---|\n");
        for label in self.inner.node_labels() {
            let props = self.inner.node_properties(label);
            let props_str = if props.is_empty() {
                "\u{2014}".to_string() // em-dash
            } else {
                props.join(", ")
            };
            out.push_str(&format!("| :{} | {} |\n", label, props_str));
        }
        out.push_str("\n### Relationships\n\n| Type | Source \u{2192} Target | Properties |\n|---|---|---|\n");
        for rel_type in self.inner.rel_types() {
            if let Some((src, tgt)) = self.inner.rel_src_tgt(rel_type) {
                let props = self.inner.rel_properties(rel_type);
                let props_str = if props.is_empty() {
                    "\u{2014}".to_string()
                } else {
                    props.join(", ")
                };
                out.push_str(&format!(
                    "| :{} | :{} \u{2192} :{} | {} |\n",
                    rel_type, src, tgt, props_str
                ));
            }
        }
        out
    }

    /// Return the schema as inline Cypher patterns.
    ///
    /// This format is often the most natural for LLMs that already know Cypher,
    /// since it mirrors the query syntax they will generate.
    ///
    /// Example::
    ///
    ///     print(schema.to_cypher_context())
    ///     # // Node labels and their properties
    ///     # (:Person {name, age, email})
    ///     # (:Company {name, founded})
    ///     #
    ///     # // Relationship types
    ///     # (:Person)-[:WORKS_FOR {since, role}]->(:Company)
    ///     # (:Person)-[:LIVES_IN]->(:City)
    fn to_cypher_context(&self) -> String {
        let mut out = String::from("// Node labels and their properties\n");
        for label in self.inner.node_labels() {
            let props = self.inner.node_properties(label);
            if props.is_empty() {
                out.push_str(&format!("(:{label})\n"));
            } else {
                out.push_str(&format!("(:{label} {{{}}})\n", props.join(", ")));
            }
        }
        out.push_str("\n// Relationship types\n");
        for rel_type in self.inner.rel_types() {
            if let Some((src, tgt)) = self.inner.rel_src_tgt(rel_type) {
                let props = self.inner.rel_properties(rel_type);
                if props.is_empty() {
                    out.push_str(&format!("(:{src})-[:{rel_type}]->(:{tgt})\n"));
                } else {
                    out.push_str(&format!(
                        "(:{src})-[:{rel_type} {{{}}}]->(:{tgt})\n",
                        props.join(", ")
                    ));
                }
            }
        }
        out
    }

    fn __repr__(&self) -> String {
        format!(
            "Schema(nodes={:?}, relationships={:?})",
            self.inner.node_labels(),
            self.inner.rel_types()
        )
    }
}
