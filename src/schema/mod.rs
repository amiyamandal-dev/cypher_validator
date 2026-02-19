use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub nodes: HashMap<String, Vec<String>>,
    pub relationships: HashMap<String, (String, String, Vec<String>)>,
}

impl Schema {
    pub fn new(
        nodes: HashMap<String, Vec<String>>,
        relationships: HashMap<String, (String, String, Vec<String>)>,
    ) -> Self {
        Schema { nodes, relationships }
    }

    pub fn has_node_label(&self, label: &str) -> bool {
        self.nodes.contains_key(label)
    }

    pub fn has_rel_type(&self, rel_type: &str) -> bool {
        self.relationships.contains_key(rel_type)
    }

    pub fn node_has_property(&self, label: &str, prop: &str) -> bool {
        self.nodes.get(label).map_or(false, |props| props.iter().any(|p| p == prop))
    }

    pub fn rel_has_property(&self, rel_type: &str, prop: &str) -> bool {
        self.relationships
            .get(rel_type)
            .map_or(false, |(_, _, props)| props.iter().any(|p| p == prop))
    }

    pub fn node_labels(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    pub fn rel_types(&self) -> Vec<&str> {
        self.relationships.keys().map(|s| s.as_str()).collect()
    }

    pub fn node_properties(&self, label: &str) -> Vec<&str> {
        self.nodes.get(label).map_or(vec![], |props| props.iter().map(|s| s.as_str()).collect())
    }

    pub fn rel_src_tgt(&self, rel_type: &str) -> Option<(&str, &str)> {
        self.relationships.get(rel_type).map(|(src, tgt, _)| (src.as_str(), tgt.as_str()))
    }
}
