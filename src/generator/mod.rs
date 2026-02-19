use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::SmallRng;
use crate::error::CypherError;
use crate::schema::Schema;

pub struct CypherGenerator {
    schema: Schema,
    rng: SmallRng,
    /// All node labels, precomputed once so each `generate()` call avoids
    /// allocating a fresh `Vec` from the schema's HashMap.
    labels: Vec<String>,
    /// All relationship types, precomputed for the same reason.
    rel_types: Vec<String>,
    /// Node property lists keyed by label, precomputed for the same reason.
    props_by_label: HashMap<String, Vec<String>>,
}

impl CypherGenerator {
    pub fn new(schema: Schema, seed: Option<u64>) -> Self {
        // Precompute and own the label / rel-type / property lists so that
        // pick_node_label / pick_rel_type / pick_node_prop / gen_property_map
        // never need to re-allocate a Vec from the schema on every call.
        let labels: Vec<String> = schema.node_labels()
            .into_iter().map(|s| s.to_string()).collect();
        let rel_types: Vec<String> = schema.rel_types()
            .into_iter().map(|s| s.to_string()).collect();
        let props_by_label: HashMap<String, Vec<String>> = labels.iter()
            .map(|l| {
                let props = schema.node_properties(l)
                    .into_iter().map(|s| s.to_string()).collect();
                (l.clone(), props)
            })
            .collect();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_os_rng(),
        };
        CypherGenerator { schema, rng, labels, rel_types, props_by_label }
    }

    pub fn supported_types() -> Vec<&'static str> {
        vec![
            "match_return",
            "match_where_return",
            "create",
            "merge",
            "aggregation",
            "match_relationship",
            "create_relationship",
            "match_set",
            "match_delete",
            "with_chain",
            "distinct_return",
            "order_by",
            "unwind",
        ]
    }

    pub fn generate(&mut self, query_type: &str) -> Result<String, CypherError> {
        match query_type {
            "match_return"          => Ok(self.gen_match_return()),
            "match_where_return"    => Ok(self.gen_match_where_return()),
            "create"                => Ok(self.gen_create()),
            "merge"                 => Ok(self.gen_merge()),
            "aggregation"           => Ok(self.gen_aggregation()),
            "match_relationship"    => Ok(self.gen_match_relationship()),
            "create_relationship"   => Ok(self.gen_create_relationship()),
            "match_set"             => Ok(self.gen_match_set()),
            "match_delete"          => Ok(self.gen_match_delete()),
            "with_chain"            => Ok(self.gen_with_chain()),
            "distinct_return"       => Ok(self.gen_distinct_return()),
            "order_by"              => Ok(self.gen_order_by()),
            "unwind"                => Ok(self.gen_unwind()),
            other => Err(CypherError::GeneratorError(format!("Unknown query type: {}", other))),
        }
    }

    // ===== Value generation helpers =====

    fn pick_node_label(&mut self) -> Option<String> {
        self.labels.choose(&mut self.rng).cloned()
    }

    fn pick_rel_type(&mut self) -> Option<String> {
        self.rel_types.choose(&mut self.rng).cloned()
    }

    fn pick_node_prop(&mut self, label: &str) -> Option<String> {
        self.props_by_label
            .get(label)
            .and_then(|props| props.choose(&mut self.rng).cloned())
    }

    /// Generate a scalar value: string literal, integer, boolean, or $parameter.
    fn gen_scalar_value(&mut self) -> String {
        match self.rng.random_range(0u8..4) {
            0 => {
                let names = ["\"Alice\"", "\"Bob\"", "\"Carol\"", "\"Neo4j\"", "\"hello\""];
                names.choose(&mut self.rng).copied().unwrap_or("\"value\"").to_string()
            }
            1 => self.rng.random_range(1i64..=100).to_string(),
            2 => {
                if self.rng.random_range(0u8..2) == 0 { "true".to_string() } else { "false".to_string() }
            }
            _ => {
                let params = ["$name", "$id", "$value", "$limit"];
                params.choose(&mut self.rng).copied().unwrap_or("$param").to_string()
            }
        }
    }

    /// Generate a string-only value (for WHERE equality comparisons).
    fn gen_string_value(&mut self) -> String {
        if self.rng.random_range(0u8..3) == 0 {
            let params = ["$name", "$id", "$value"];
            return params.choose(&mut self.rng).copied().unwrap_or("$param").to_string();
        }
        let names = ["\"Alice\"", "\"Bob\"", "\"Carol\"", "\"Neo4j\""];
        names.choose(&mut self.rng).copied().unwrap_or("\"value\"").to_string()
    }

    /// Build `{prop: value, ...}` string for node properties using mixed value types.
    fn gen_property_map(&mut self, label: &str) -> String {
        let props = match self.props_by_label.get(label) {
            Some(p) if !p.is_empty() => p,
            _ => return String::new(),
        };
        let count = self.rng.random_range(1..=(props.len().min(3)));
        // Clone only the chosen property names to free the borrow on self before
        // calling gen_scalar_value (which also needs &mut self).
        let chosen: Vec<String> = props
            .choose_multiple(&mut self.rng, count)
            .cloned()
            .collect();
        let pairs: Vec<String> = chosen.iter()
            .map(|p| format!("{}: {}", p, self.gen_scalar_value()))
            .collect();
        format!(" {{{}}}", pairs.join(", "))
    }

    /// Optionally append LIMIT clause.
    fn maybe_limit(&mut self) -> String {
        if self.rng.random_range(0u8..3) == 0 {
            let n = self.rng.random_range(1i64..=50);
            format!(" LIMIT {}", n)
        } else {
            String::new()
        }
    }

    // ===== Query generators =====

    fn gen_match_return(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let limit = self.maybe_limit();
        format!("MATCH (n:{}) RETURN n{}", label, limit)
    }

    fn gen_match_where_return(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        if let Some(p) = prop {
            let val = self.gen_string_value();
            let limit = self.maybe_limit();
            format!("MATCH (n:{}) WHERE n.{} = {} RETURN n.{}{}", label, p, val, p, limit)
        } else {
            let limit = self.maybe_limit();
            format!("MATCH (n:{}) RETURN n{}", label, limit)
        }
    }

    fn gen_create(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let props = self.gen_property_map(&label);
        format!("CREATE (n:{}{}) RETURN n", label, props)
    }

    fn gen_merge(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let props = self.gen_property_map(&label);
        format!("MERGE (n:{}{}) RETURN n", label, props)
    }

    fn gen_aggregation(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        if let Some(p) = prop {
            format!("MATCH (n:{}) RETURN count(n.{}) AS result", label, p)
        } else {
            format!("MATCH (n:{}) RETURN count(n) AS result", label)
        }
    }

    fn gen_match_relationship(&mut self) -> String {
        if let Some(rel_type) = self.pick_rel_type() {
            let (src, tgt) = self.schema.rel_src_tgt(&rel_type)
                .map(|(s, t)| (s.to_string(), t.to_string()))
                .unwrap_or_else(|| ("Node".to_string(), "Node".to_string()));
            let optional = if self.rng.random_range(0u8..4) == 0 { "OPTIONAL " } else { "" };
            format!("{}MATCH (a:{})-[r:{}]->(b:{}) RETURN a, r, b", optional, src, rel_type, tgt)
        } else {
            let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
            format!("MATCH (n:{}) RETURN n", label)
        }
    }

    fn gen_create_relationship(&mut self) -> String {
        if let Some(rel_type) = self.pick_rel_type() {
            let (src, tgt) = self.schema.rel_src_tgt(&rel_type)
                .map(|(s, t)| (s.to_string(), t.to_string()))
                .unwrap_or_else(|| ("Node".to_string(), "Node".to_string()));
            format!(
                "MATCH (a:{}),(b:{}) CREATE (a)-[r:{}]->(b) RETURN r",
                src, tgt, rel_type
            )
        } else {
            let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
            format!("CREATE (n:{}) RETURN n", label)
        }
    }

    fn gen_match_set(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        if let Some(p) = prop {
            let val = self.gen_string_value();
            format!("MATCH (n:{}) SET n.{} = {} RETURN n", label, p, val)
        } else {
            format!("MATCH (n:{}) RETURN n", label)
        }
    }

    fn gen_match_delete(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        format!("MATCH (n:{}) DETACH DELETE n", label)
    }

    fn gen_with_chain(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        if let Some(p) = prop {
            format!("MATCH (n:{}) WITH n.{} AS val RETURN count(*)", label, p)
        } else {
            format!("MATCH (n:{}) WITH n AS val RETURN count(*)", label)
        }
    }

    fn gen_distinct_return(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        let limit = self.maybe_limit();
        if let Some(p) = prop {
            format!("MATCH (n:{}) RETURN DISTINCT n.{}{}", label, p, limit)
        } else {
            format!("MATCH (n:{}) RETURN DISTINCT n{}", label, limit)
        }
    }

    fn gen_order_by(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        let direction = if self.rng.random_range(0u8..2) == 0 { "" } else { " DESC" };
        let limit = self.maybe_limit();
        if let Some(p) = prop {
            format!(
                "MATCH (n:{}) RETURN n ORDER BY n.{}{}{}",
                label, p, direction, limit
            )
        } else {
            format!("MATCH (n:{}) RETURN n ORDER BY n{}", label, limit)
        }
    }

    fn gen_unwind(&mut self) -> String {
        let label = self.pick_node_label().unwrap_or_else(|| "Node".to_string());
        let prop = self.pick_node_prop(&label);
        // Generate a small list of scalar values to unwind
        let count = self.rng.random_range(2u8..=4);
        let values: Vec<String> = (0..count).map(|_| self.gen_scalar_value()).collect();
        let list = values.join(", ");
        if let Some(p) = prop {
            format!(
                "MATCH (n:{}) UNWIND n.{} AS item RETURN item",
                label, p
            )
        } else {
            format!("UNWIND [{}] AS item RETURN item", list)
        }
    }
}
