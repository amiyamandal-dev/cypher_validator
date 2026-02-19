use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Graph schema: node labels with their allowed property sets,
/// and relationship types with their (source label, target label, property set).
///
/// Properties are stored as `HashSet<String>` so that `node_has_property` and
/// `rel_has_property` are O(1) rather than the O(n) linear scan of a Vec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub nodes: HashMap<String, HashSet<String>>,
    pub relationships: HashMap<String, (String, String, HashSet<String>)>,
}

impl Schema {
    pub fn new(
        nodes: HashMap<String, HashSet<String>>,
        relationships: HashMap<String, (String, String, HashSet<String>)>,
    ) -> Self {
        Schema { nodes, relationships }
    }

    // ----- Label / type existence -----

    #[inline]
    pub fn has_node_label(&self, label: &str) -> bool {
        self.nodes.contains_key(label)
    }

    #[inline]
    pub fn has_rel_type(&self, rel_type: &str) -> bool {
        self.relationships.contains_key(rel_type)
    }

    // ----- Property existence (O(1) with HashSet) -----

    #[inline]
    pub fn node_has_property(&self, label: &str, prop: &str) -> bool {
        self.nodes.get(label).map_or(false, |props| props.contains(prop))
    }

    #[inline]
    pub fn rel_has_property(&self, rel_type: &str, prop: &str) -> bool {
        self.relationships
            .get(rel_type)
            .map_or(false, |(_, _, props)| props.contains(prop))
    }

    // ----- Enumeration (sorted for determinism) -----

    pub fn node_labels(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    pub fn rel_types(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.relationships.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Return sorted property list for a node label.
    pub fn node_properties(&self, label: &str) -> Vec<&str> {
        let mut v: Vec<&str> = self.nodes
            .get(label)
            .map_or(vec![], |props| props.iter().map(|s| s.as_str()).collect());
        v.sort_unstable();
        v
    }

    /// Return sorted property list for a relationship type.
    pub fn rel_properties(&self, rel_type: &str) -> Vec<&str> {
        let mut v: Vec<&str> = self.relationships
            .get(rel_type)
            .map_or(vec![], |(_, _, props)| props.iter().map(|s| s.as_str()).collect());
        v.sort_unstable();
        v
    }

    pub fn rel_src_tgt(&self, rel_type: &str) -> Option<(&str, &str)> {
        self.relationships.get(rel_type).map(|(src, tgt, _)| (src.as_str(), tgt.as_str()))
    }
}
