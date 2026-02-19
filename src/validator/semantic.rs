use std::collections::HashMap;
use crate::schema::Schema;
use crate::parser::ast::*;

/// TypeEnv maps variable names to a list of labels/rel-types they are bound to.
pub type TypeEnv = HashMap<String, Vec<String>>;

// ---------------------------------------------------------------------------
// Levenshtein-based "did you mean" helper
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance (case-insensitive) between two strings,
/// stopping early and returning `cap + 1` as soon as the distance provably exceeds `cap`.
///
/// Uses a 1-D rolling array (O(n) space) rather than a full m×n matrix.
fn levenshtein_capped(a: &str, b: &str, cap: usize) -> usize {
    // Schema identifiers are ASCII, so byte comparison is correct after lowercasing.
    let a_lo = a.to_lowercase();
    let b_lo = b.to_lowercase();
    let a_bytes = a_lo.as_bytes();
    let b_bytes = b_lo.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    // If the lengths differ by more than cap, the distance must exceed cap.
    if m.abs_diff(n) > cap {
        return cap + 1;
    }

    // prev[j] = edit distance between a[0..i] and b[0..j] from the previous row.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        let mut row_min = i; // track minimum in current row for early-exit
        for j in 1..=n {
            curr[j] = if a_bytes[i - 1] == b_bytes[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
            if curr[j] < row_min { row_min = curr[j]; }
        }
        // Every remaining row can only increase the minimum by at most 1 per row,
        // so if the current row minimum already exceeds cap we can stop early.
        if row_min > cap {
            return cap + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Return the closest candidate from `candidates` within `max_dist`, or None.
/// Uses the capped Levenshtein so expensive full-matrix computation is avoided
/// for strings that are clearly too far apart.
fn closest_match<'a>(name: &str, candidates: &'a [String], max_dist: usize) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for c in candidates {
        let d = levenshtein_capped(name, c, max_dist);
        if d <= max_dist {
            match best {
                None => best = Some((c.as_str(), d)),
                Some((_, bd)) if d < bd => best = Some((c.as_str(), d)),
                _ => {}
            }
        }
    }
    best.map(|(c, _)| c)
}

/// Build a "did you mean :X?" hint string, or empty string if no close match.
fn did_you_mean_label(name: &str, candidates: &[String]) -> String {
    match closest_match(name, candidates, 3) {
        Some(s) => format!(", did you mean :{s}?"),
        None => String::new(),
    }
}

fn did_you_mean_rel(name: &str, candidates: &[String]) -> String {
    match closest_match(name, candidates, 3) {
        Some(s) => format!(", did you mean :{s}?"),
        None => String::new(),
    }
}

pub struct SemanticValidator<'a> {
    pub schema: &'a Schema,
    pub errors: Vec<String>,
    pub env: TypeEnv,
    /// Cached label list for "did you mean" suggestions — computed once per validation.
    known_labels: Vec<String>,
    /// Cached rel-type list for "did you mean" suggestions — computed once per validation.
    known_rel_types: Vec<String>,
}

impl<'a> SemanticValidator<'a> {
    pub fn new(schema: &'a Schema) -> Self {
        let known_labels = schema.node_labels().into_iter().map(|s| s.to_string()).collect();
        let known_rel_types = schema.rel_types().into_iter().map(|s| s.to_string()).collect();
        SemanticValidator {
            schema,
            errors: vec![],
            env: HashMap::new(),
            known_labels,
            known_rel_types,
        }
    }

    pub fn validate_query(&mut self, query: &CypherQuery) {
        match &query.statement {
            Statement::RegularQuery(rq) => self.validate_regular_query(rq),
            Statement::StandaloneCall(_) => {} // procedure calls are open-world
        }
    }

    fn validate_regular_query(&mut self, rq: &RegularQuery) {
        self.validate_single_query(&rq.single_query);
        for (_, sq) in &rq.unions {
            // Each UNION branch is validated with an independent env
            let saved_env = self.env.clone();
            self.env.clear();
            self.validate_single_query(sq);
            self.env = saved_env;
        }
    }

    fn validate_single_query(&mut self, sq: &SingleQuery) {
        match sq {
            SingleQuery::SinglePart(spq) => self.validate_single_part(spq),
            SingleQuery::MultiPart(mpq) => self.validate_multi_part(mpq),
        }
    }

    fn validate_single_part(&mut self, spq: &SinglePartQuery) {
        // First pass: collect all bindings so expressions can reference any bound var
        for rc in &spq.reading {
            self.collect_reading_bindings(rc);
        }
        for uc in &spq.updating {
            self.collect_updating_bindings(uc);
        }
        // Second pass: validate labels, properties, directions, and variable references
        for rc in &spq.reading {
            self.validate_reading_clause(rc);
        }
        for uc in &spq.updating {
            self.validate_updating_clause(uc);
        }
        if let Some(ret) = &spq.ret {
            self.validate_return_clause(ret);
        }
    }

    fn validate_multi_part(&mut self, mpq: &MultiPartQuery) {
        for (reading, updating, with) in &mpq.parts {
            // Collect bindings for this part
            for rc in reading { self.collect_reading_bindings(rc); }
            for uc in updating { self.collect_updating_bindings(uc); }
            // Validate this part against current env
            for rc in reading { self.validate_reading_clause(rc); }
            for uc in updating { self.validate_updating_clause(uc); }
            // Process WITH: validate its expressions, then reset env to projected vars only
            self.process_with_clause(with);
        }
        // Final part: collect then validate
        for rc in &mpq.reading {
            self.collect_reading_bindings(rc);
        }
        for uc in &mpq.updating {
            self.collect_updating_bindings(uc);
        }
        for rc in &mpq.reading {
            self.validate_reading_clause(rc);
        }
        for uc in &mpq.updating {
            self.validate_updating_clause(uc);
        }
        if let Some(ret) = &mpq.ret {
            self.validate_return_clause(ret);
        }
    }

    // ===== WITH scope management =====

    fn process_with_clause(&mut self, with: &WithClause) {
        // Validate WITH expressions against the current (pre-reset) env
        for item in &with.items {
            self.validate_expr(&item.expr);
        }
        if let Some(order) = &with.order {
            for si in order { self.validate_expr(&si.expr); }
        }
        if let Some(s) = &with.skip { self.validate_expr(s); }
        if let Some(l) = &with.limit { self.validate_expr(l); }
        // WHERE is also evaluated against the old env (pre-WITH scope)
        if let Some(where_expr) = &with.where_clause {
            self.validate_expr(where_expr);
        }

        // Reset env: only projected variables survive the WITH boundary
        let mut new_env: TypeEnv = HashMap::new();
        for item in &with.items {
            let name = item.alias.clone().or_else(|| {
                if let Expr::Variable(v) = &item.expr { Some(v.clone()) } else { None }
            });
            if let Some(name) = name {
                // Carry labels forward when projecting a bare variable
                let labels = if let Expr::Variable(v) = &item.expr {
                    self.env.get(v.as_str()).cloned().unwrap_or_default()
                } else {
                    vec![]
                };
                new_env.insert(name, labels);
            }
        }
        self.env = new_env;
    }

    // ===== Binding collection =====

    fn collect_reading_bindings(&mut self, rc: &ReadingClause) {
        match rc {
            ReadingClause::Match(m) => self.collect_pattern_bindings(&m.pattern),
            ReadingClause::Unwind(u) => {
                self.env.insert(u.variable.clone(), vec![]);
            }
            ReadingClause::Call(c) => {
                // CALL YIELD variables are bound in scope
                for yi in &c.yield_items {
                    self.env.insert(yi.variable.clone(), vec![]);
                }
            }
        }
    }

    fn collect_updating_bindings(&mut self, uc: &UpdatingClause) {
        match uc {
            UpdatingClause::Create(pat) => self.collect_pattern_bindings(pat),
            UpdatingClause::Merge(m) => {
                let pattern = Pattern { parts: vec![m.pattern_part.clone()] };
                self.collect_pattern_bindings(&pattern);
            }
            _ => {}
        }
    }

    fn collect_pattern_bindings(&mut self, pattern: &Pattern) {
        for part in &pattern.parts {
            if let Some(v) = &part.variable {
                self.env.insert(v.clone(), vec![]);
            }
            let elem = &part.element;
            self.collect_node_bindings(&elem.start);
            for (rel, node) in &elem.chain {
                self.collect_rel_bindings(rel);
                self.collect_node_bindings(node);
            }
        }
    }

    fn collect_node_bindings(&mut self, node: &NodePattern) {
        if let Some(var) = &node.variable {
            let labels = node.labels.clone();
            self.env.entry(var.clone())
                .and_modify(|existing| {
                    for l in &labels {
                        if !existing.contains(l) {
                            existing.push(l.clone());
                        }
                    }
                })
                .or_insert(labels);
        }
    }

    fn collect_rel_bindings(&mut self, rel: &RelationshipPattern) {
        if let Some(var) = &rel.variable {
            let types = rel.rel_types.clone();
            self.env.entry(var.clone())
                .and_modify(|existing| {
                    for t in &types {
                        if !existing.contains(t) {
                            existing.push(t.clone());
                        }
                    }
                })
                .or_insert(types);
        }
    }

    // ===== Validation =====

    fn validate_reading_clause(&mut self, rc: &ReadingClause) {
        match rc {
            ReadingClause::Match(m) => {
                self.validate_pattern(&m.pattern);
                if let Some(w) = &m.where_clause {
                    self.validate_expr(w);
                }
            }
            ReadingClause::Unwind(u) => self.validate_expr(&u.expr),
            ReadingClause::Call(_) => {}
        }
    }

    fn validate_updating_clause(&mut self, uc: &UpdatingClause) {
        match uc {
            UpdatingClause::Create(pat) => self.validate_pattern(pat),
            UpdatingClause::Merge(m) => {
                let pattern = Pattern { parts: vec![m.pattern_part.clone()] };
                self.validate_pattern(&pattern);
            }
            UpdatingClause::Set(items) => {
                for item in items { self.validate_set_item(item); }
            }
            UpdatingClause::Remove(items) => {
                for item in items { self.validate_remove_item(item); }
            }
            UpdatingClause::Delete { exprs, .. } => {
                for e in exprs { self.validate_expr(e); }
            }
        }
    }

    fn validate_return_clause(&mut self, ret: &ReturnClause) {
        match &ret.items {
            ReturnItems::Items(items) => {
                for item in items { self.validate_expr(&item.expr); }
            }
            ReturnItems::Wildcard => {}
        }
        if let Some(order) = &ret.order {
            for si in order { self.validate_expr(&si.expr); }
        }
        if let Some(s) = &ret.skip { self.validate_expr(s); }
        if let Some(l) = &ret.limit { self.validate_expr(l); }
    }

    // ===== Pattern validation (direction + endpoint aware) =====

    fn validate_pattern(&mut self, pattern: &Pattern) {
        for part in &pattern.parts {
            let elem = &part.element;
            self.validate_node_pattern(&elem.start);
            let mut prev = &elem.start;
            for (rel, node) in &elem.chain {
                self.validate_rel_pattern_with_endpoints(rel, prev, node);
                self.validate_node_pattern(node);
                prev = node;
            }
        }
    }

    fn validate_node_pattern(&mut self, node: &NodePattern) {
        for label in &node.labels {
            if !self.schema.has_node_label(label) {
                let hint = did_you_mean_label(label, &self.known_labels);
                self.errors.push(format!("Unknown node label: :{}{}", label, hint));
            }
        }
        if let Some(MapOrParam::Map(map)) = &node.properties {
            for key in map.keys() {
                for label in &node.labels {
                    if self.schema.has_node_label(label) && !self.schema.node_has_property(label, key) {
                        self.errors.push(format!(
                            "Unknown property '{}' for node label :{}", key, label
                        ));
                    }
                }
            }
        }
    }

    /// Validates relationship type, properties, direction, and endpoint labels.
    fn validate_rel_pattern_with_endpoints(
        &mut self,
        rel: &RelationshipPattern,
        from_node: &NodePattern,
        to_node: &NodePattern,
    ) {
        for rel_type in &rel.rel_types {
            if !self.schema.has_rel_type(rel_type) {
                let hint = did_you_mean_rel(rel_type, &self.known_rel_types);
                self.errors.push(format!("Unknown relationship type: :{}{}", rel_type, hint));
                continue; // skip endpoint check for unknown types
            }

            // Validate relationship properties
            if let Some(MapOrParam::Map(map)) = &rel.properties {
                for key in map.keys() {
                    if !self.schema.rel_has_property(rel_type, key) {
                        self.errors.push(format!(
                            "Unknown property '{}' for relationship type :{}", key, rel_type
                        ));
                    }
                }
            }

            // Validate direction and endpoint labels against schema
            if let Some((schema_src, schema_tgt)) = self.schema.rel_src_tgt(rel_type) {
                let schema_src = schema_src.to_string();
                let schema_tgt = schema_tgt.to_string();
                match &rel.direction {
                    Direction::Outgoing => {
                        // Pattern: (from_node)-[r]->(to_node)
                        // from_node is the source, to_node is the target
                        self.check_endpoint_label(from_node, &schema_src, "source", rel_type);
                        self.check_endpoint_label(to_node, &schema_tgt, "target", rel_type);
                    }
                    Direction::Incoming => {
                        // Pattern: (from_node)<-[r]-(to_node)
                        // to_node is the source, from_node is the target
                        self.check_endpoint_label(to_node, &schema_src, "source", rel_type);
                        self.check_endpoint_label(from_node, &schema_tgt, "target", rel_type);
                    }
                    Direction::Undirected => {
                        // Undirected — either orientation is valid, skip endpoint check
                    }
                }
            }
        }
    }

    /// Reports an error if `node` has known labels that don't include `expected`.
    /// Nodes without labels are skipped (open-world assumption).
    fn check_endpoint_label(
        &mut self,
        node: &NodePattern,
        expected: &str,
        role: &str,
        rel_type: &str,
    ) {
        if node.labels.is_empty() {
            return; // no labels declared: open-world, skip
        }
        if node.labels.iter().any(|l| l == expected) {
            return; // at least one label matches: valid
        }
        // Only report if at least one label is a known schema label
        // (avoids redundant errors on top of "Unknown node label" errors)
        if node.labels.iter().any(|l| self.schema.has_node_label(l)) {
            self.errors.push(format!(
                "Relationship :{} expects {} label :{}, but node has label(s): {}",
                rel_type,
                role,
                expected,
                node.labels.iter().map(|l| format!(":{}", l)).collect::<Vec<_>>().join(", ")
            ));
        }
    }

    fn validate_set_item(&mut self, item: &SetItem) {
        match item {
            SetItem::PropertySet { prop, value } => {
                self.validate_property_access(&prop.variable, &prop.properties);
                self.validate_expr(value);
            }
            SetItem::VariableSet { value, .. } => self.validate_expr(value),
            SetItem::VariableAdd { value, .. } => self.validate_expr(value),
            SetItem::LabelSet { var: _, labels } => {
                for label in labels {
                    if !self.schema.has_node_label(label) {
                        let hint = did_you_mean_label(label, &self.known_labels);
                        self.errors.push(format!("Unknown node label in SET: :{}{}", label, hint));
                    }
                }
            }
        }
    }

    fn validate_remove_item(&mut self, item: &RemoveItem) {
        match item {
            RemoveItem::LabelRemove { labels, .. } => {
                for label in labels {
                    if !self.schema.has_node_label(label) {
                        let hint = did_you_mean_label(label, &self.known_labels);
                        self.errors.push(format!("Unknown node label in REMOVE: :{}{}", label, hint));
                    }
                }
            }
            RemoveItem::PropertyRemove(prop) => {
                self.validate_property_access(&prop.variable, &prop.properties);
            }
        }
    }

    fn validate_property_access(&mut self, var: &str, props: &[String]) {
        if let Some(labels) = self.env.get(var).cloned() {
            for label in &labels {
                if self.schema.has_node_label(label) {
                    for prop in props {
                        if !self.schema.node_has_property(label, prop) {
                            self.errors.push(format!(
                                "Unknown property '{}' on variable '{}' with label :{}", prop, var, label
                            ));
                        }
                    }
                } else if self.schema.has_rel_type(label) {
                    for prop in props {
                        if !self.schema.rel_has_property(label, prop) {
                            self.errors.push(format!(
                                "Unknown property '{}' on variable '{}' with relationship type :{}", prop, var, label
                            ));
                        }
                    }
                }
            }
            // If labels is empty: open-world assumption — no property check
        }
        // Variable not in env: open-world — no error here (unbound check is in validate_expr)
    }

    fn validate_expr(&mut self, expr: &Expr) {
        match expr {
            // ===== Unbound variable detection =====
            Expr::Variable(name) => {
                if !self.env.contains_key(name.as_str()) {
                    self.errors.push(format!("Variable '{}' is not bound in this scope", name));
                }
            }

            // ===== Property access =====
            Expr::Property { expr: inner, key } => {
                if let Expr::Variable(var) = inner.as_ref() {
                    self.validate_property_access(var, &[key.clone()]);
                }
                self.validate_expr(inner);
            }

            // ===== Binary operators =====
            Expr::Or(a, b) | Expr::Xor(a, b) | Expr::And(a, b)
            | Expr::Eq(a, b) | Expr::Ne(a, b) | Expr::Lt(a, b) | Expr::Lte(a, b)
            | Expr::Gt(a, b) | Expr::Gte(a, b) | Expr::Regex(a, b)
            | Expr::In(a, b) | Expr::StartsWith(a, b) | Expr::EndsWith(a, b)
            | Expr::Contains(a, b)
            | Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Mod(a, b) | Expr::Pow(a, b) => {
                self.validate_expr(a);
                self.validate_expr(b);
            }

            // ===== Unary operators =====
            Expr::Not(e) | Expr::Neg(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => {
                self.validate_expr(e);
            }

            // ===== Function calls =====
            Expr::FunctionCall { args, .. } => {
                for a in args { self.validate_expr(a); }
            }

            // ===== Collections =====
            Expr::List(items) => {
                for i in items { self.validate_expr(i); }
            }
            Expr::Map(map) => {
                for v in map.values() { self.validate_expr(v); }
            }

            // ===== Access operators =====
            Expr::Subscript { expr: e, index } => {
                self.validate_expr(e);
                self.validate_expr(index);
            }
            Expr::Slice { expr: e, from, to } => {
                self.validate_expr(e);
                if let Some(f) = from { self.validate_expr(f); }
                if let Some(t) = to { self.validate_expr(t); }
            }

            // ===== CASE expression =====
            Expr::Case { subject, alternatives, default } => {
                if let Some(s) = subject { self.validate_expr(s); }
                for (w, t) in alternatives {
                    self.validate_expr(w);
                    self.validate_expr(t);
                }
                if let Some(d) = default { self.validate_expr(d); }
            }

            // ===== List comprehension (locally-scoped variable) =====
            Expr::ListComprehension { variable, source, filter, projection } => {
                self.validate_expr(source);
                let was_present = self.env.contains_key(variable.as_str());
                if !was_present { self.env.insert(variable.clone(), vec![]); }
                if let Some(f) = filter { self.validate_expr(f); }
                if let Some(p) = projection { self.validate_expr(p); }
                if !was_present { self.env.remove(variable.as_str()); }
            }

            // ===== Quantifiers (locally-scoped variable) =====
            Expr::All { variable, source, filter }
            | Expr::Any { variable, source, filter }
            | Expr::None { variable, source, filter }
            | Expr::Single { variable, source, filter } => {
                self.validate_expr(source);
                let was_present = self.env.contains_key(variable.as_str());
                if !was_present { self.env.insert(variable.clone(), vec![]); }
                if let Some(f) = filter { self.validate_expr(f); }
                if !was_present { self.env.remove(variable.as_str()); }
            }

            // ===== Leaf nodes — no further validation =====
            Expr::Parameter(_) | Expr::Integer(_) | Expr::Float(_)
            | Expr::Str(_) | Expr::Bool(_) | Expr::Null | Expr::CountStar => {}

            Expr::Exists(_) => {}
        }
    }
}
