pub mod semantic;

use crate::error::CypherError;
use crate::schema::Schema;
use crate::parser;
use crate::validator::semantic::SemanticValidator;

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    /// All errors combined (backward-compatible).
    pub errors: Vec<String>,
    /// Parse/syntax errors only.
    pub syntax_errors: Vec<String>,
    /// Schema-level semantic errors only.
    pub semantic_errors: Vec<String>,
}

pub struct CypherValidator {
    pub schema: Schema,
}

impl CypherValidator {
    pub fn new(schema: Schema) -> Self {
        CypherValidator { schema }
    }

    pub fn validate(&self, query: &str) -> ValidationResult {
        // Phase 1: syntax
        let ast = match parser::parse(query) {
            Ok(ast) => ast,
            Err(CypherError::ParseError(msg)) => {
                return ValidationResult {
                    is_valid: false,
                    errors: vec![msg.clone()],
                    syntax_errors: vec![msg],
                    semantic_errors: vec![],
                };
            }
            Err(e) => {
                let msg = e.to_string();
                return ValidationResult {
                    is_valid: false,
                    errors: vec![msg.clone()],
                    syntax_errors: vec![msg],
                    semantic_errors: vec![],
                };
            }
        };

        // Phase 2: semantic
        let mut sem = SemanticValidator::new(&self.schema);
        sem.validate_query(&ast);

        let semantic_errors = sem.errors;
        let errors = semantic_errors.clone();
        let is_valid = errors.is_empty();
        ValidationResult { is_valid, errors, syntax_errors: vec![], semantic_errors }
    }
}
