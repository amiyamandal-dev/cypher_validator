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
