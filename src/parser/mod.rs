pub mod ast;
pub mod builder;

use pest::Parser as PestParser;
use pest_derive::Parser;
use crate::error::CypherError;
use crate::parser::ast::CypherQuery;

#[derive(Parser)]
#[grammar = "grammar/cypher.pest"]
pub struct CypherParser;

pub fn parse(input: &str) -> Result<CypherQuery, CypherError> {
    let pairs = <CypherParser as PestParser<Rule>>::parse(Rule::cypher, input)
        .map_err(|e| CypherError::ParseError(e.to_string()))?;
    let cypher_pair = pairs.into_iter().next()
        .ok_or_else(|| CypherError::ParseError("No parse result".into()))?;
    builder::build_query(cypher_pair)
}

/// Parse result that preserves error position info before stringifying.
pub struct ParseDetail {
    pub error_message: String,
    /// 1-based `(line, col)` extracted from pest error, if available.
    pub position: Option<(u32, u32)>,
}

/// Like `parse()` but on failure returns structured error detail with position info.
pub fn parse_with_detail(input: &str) -> Result<CypherQuery, ParseDetail> {
    let pairs = <CypherParser as PestParser<Rule>>::parse(Rule::cypher, input);
    match pairs {
        Err(e) => {
            let position = match e.line_col {
                pest::error::LineColLocation::Pos((line, col)) => {
                    Some((line as u32, col as u32))
                }
                pest::error::LineColLocation::Span((line, col), _) => {
                    Some((line as u32, col as u32))
                }
            };
            Err(ParseDetail {
                error_message: e.to_string(),
                position,
            })
        }
        Ok(pairs) => {
            let cypher_pair = pairs.into_iter().next()
                .ok_or_else(|| ParseDetail {
                    error_message: "No parse result".into(),
                    position: None,
                })?;
            builder::build_query(cypher_pair).map_err(|e| ParseDetail {
                error_message: e.to_string(),
                position: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pest::Parser as PestParser;

    fn try_parse(rule: Rule, input: &str) -> bool {
        <CypherParser as PestParser<Rule>>::parse(rule, input).is_ok()
    }

    // --- null_check_op unit tests ---

    #[test]
    fn test_null_check_op_is_null() {
        assert!(try_parse(Rule::null_check_op, "IS NULL"),    "IS NULL should parse");
        assert!(try_parse(Rule::null_check_op, "is null"),    "is null (lowercase) should parse");
        assert!(try_parse(Rule::null_check_op, "IS NOT NULL"),"IS NOT NULL should parse");
        assert!(try_parse(Rule::null_check_op, "is not null"),"is not null (lowercase) should parse");
    }

    #[test]
    fn test_null_check_op_rejects_partial() {
        // "ISNULL" (no space) must not match — the rule requires at least one whitespace
        assert!(!try_parse(Rule::null_check_op, "ISNULL"),    "ISNULL without space must not match");
        assert!(!try_parse(Rule::null_check_op, "ISNOTNULL"), "ISNOTNULL without spaces must not match");
    }

    // --- postfix_expr tests ---

    #[test]
    fn test_postfix_expr_is_null_on_variable() {
        assert!(try_parse(Rule::postfix_expr, "n IS NULL"),    "n IS NULL");
        assert!(try_parse(Rule::postfix_expr, "n IS NOT NULL"),"n IS NOT NULL");
    }

    #[test]
    fn test_postfix_expr_is_null_on_property() {
        assert!(try_parse(Rule::postfix_expr, "n.age IS NULL"),    "n.age IS NULL");
        assert!(try_parse(Rule::postfix_expr, "n.age IS NOT NULL"),"n.age IS NOT NULL");
    }

    // --- where_clause tests ---

    #[test]
    fn test_where_clause_is_null() {
        assert!(try_parse(Rule::where_clause, "WHERE n.age IS NULL"));
        assert!(try_parse(Rule::where_clause, "WHERE n.age IS NOT NULL"));
        assert!(try_parse(Rule::where_clause, "WHERE n IS NULL"));
    }

    // --- full cypher query tests ---

    #[test]
    fn test_full_query_is_null() {
        assert!(try_parse(Rule::cypher, "MATCH (n:Person) WHERE n.age IS NULL RETURN n"));
    }

    #[test]
    fn test_full_query_is_not_null() {
        assert!(try_parse(Rule::cypher, "MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n"));
    }

    #[test]
    fn test_full_query_is_null_lowercase_keywords() {
        assert!(try_parse(Rule::cypher, "MATCH (n:Person) WHERE n.age is null RETURN n"));
        assert!(try_parse(Rule::cypher, "MATCH (n:Person) WHERE n.age is not null RETURN n"));
    }

    #[test]
    fn test_full_query_is_null_with_and() {
        assert!(try_parse(Rule::cypher,
            "MATCH (n:Person) WHERE n.age IS NULL AND n.name IS NOT NULL RETURN n"));
    }

    #[test]
    fn test_full_query_is_null_with_or() {
        assert!(try_parse(Rule::cypher,
            "MATCH (n:Person) WHERE n.age IS NULL OR n.name IS NOT NULL RETURN n"));
    }

    #[test]
    fn test_full_query_is_null_in_relationship_pattern() {
        assert!(try_parse(Rule::cypher,
            "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) WHERE p.age IS NOT NULL RETURN p, c"));
    }

    #[test]
    fn test_full_query_is_null_in_return() {
        // IS NULL used as a projected expression, not just in WHERE
        assert!(try_parse(Rule::cypher,
            "MATCH (n:Person) RETURN n.age IS NULL"));
    }

    #[test]
    fn test_full_query_not_is_null() {
        assert!(try_parse(Rule::cypher,
            "MATCH (n:Person) WHERE NOT n.age IS NULL RETURN n"));
    }
}
