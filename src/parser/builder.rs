use std::collections::HashMap;
use pest::iterators::Pair;
use crate::error::CypherError;
use super::Rule;
use crate::parser::ast::*;

pub fn build_query(pair: Pair<Rule>) -> Result<CypherQuery, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty query".into()))?;
    let statement = match inner.as_rule() {
        Rule::statement => build_statement(inner)?,
        _ => return Err(CypherError::ParseError(format!("Expected statement, got {:?}", inner.as_rule()))),
    };
    Ok(CypherQuery { statement })
}

fn build_statement(pair: Pair<Rule>) -> Result<Statement, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty statement".into()))?;
    match inner.as_rule() {
        Rule::regular_query => Ok(Statement::RegularQuery(build_regular_query(inner)?)),
        Rule::standalone_call => Ok(Statement::StandaloneCall(build_call_clause(inner)?)),
        _ => Err(CypherError::ParseError(format!("Unknown statement type: {:?}", inner.as_rule()))),
    }
}

fn build_regular_query(pair: Pair<Rule>) -> Result<RegularQuery, CypherError> {
    let mut inner = pair.into_inner();
    let single_query = build_single_query(inner.next().ok_or_else(|| CypherError::ParseError("Missing single_query".into()))?)?;
    let mut unions = vec![];
    while let Some(union_pair) = inner.next() {
        if union_pair.as_rule() == Rule::union_clause {
            let text = union_pair.as_str().to_uppercase();
            let is_all = text.contains("ALL");
            let sub = union_pair.into_inner()
                .find(|p| p.as_rule() == Rule::single_query)
                .ok_or_else(|| CypherError::ParseError("Missing union single_query".into()))?;
            unions.push((is_all, build_single_query(sub)?));
        }
    }
    Ok(RegularQuery { single_query, unions })
}

fn build_single_query(pair: Pair<Rule>) -> Result<SingleQuery, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty single_query".into()))?;
    match inner.as_rule() {
        Rule::single_part_query => Ok(SingleQuery::SinglePart(build_single_part_query(inner)?)),
        Rule::multi_part_query => Ok(SingleQuery::MultiPart(build_multi_part_query(inner)?)),
        _ => Err(CypherError::ParseError(format!("Unknown single_query: {:?}", inner.as_rule()))),
    }
}

fn build_single_part_query(pair: Pair<Rule>) -> Result<SinglePartQuery, CypherError> {
    let mut reading = vec![];
    let mut updating = vec![];
    let mut ret = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::reading_clause => reading.push(build_reading_clause(child)?),
            Rule::updating_clause => updating.push(build_updating_clause(child)?),
            Rule::return_clause => ret = Some(build_return_clause(child)?),
            _ => {}
        }
    }
    Ok(SinglePartQuery { reading, updating, ret })
}

fn build_multi_part_query(pair: Pair<Rule>) -> Result<MultiPartQuery, CypherError> {
    let mut parts = vec![];
    let mut reading = vec![];
    let mut updating = vec![];
    let mut ret = None;
    let mut part_reading = vec![];
    let mut part_updating = vec![];
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::reading_clause => {
                reading.push(build_reading_clause(child)?);
                part_reading.push(reading.last().unwrap().clone());
            }
            Rule::updating_clause => {
                updating.push(build_updating_clause(child)?);
                part_updating.push(updating.last().unwrap().clone());
            }
            Rule::with_clause => {
                let with = build_with_clause(child)?;
                parts.push((part_reading.clone(), part_updating.clone(), with));
                part_reading.clear();
                part_updating.clear();
                reading.clear();
                updating.clear();
            }
            Rule::return_clause => ret = Some(build_return_clause(child)?),
            _ => {}
        }
    }
    Ok(MultiPartQuery { parts, reading, updating, ret })
}

fn build_reading_clause(pair: Pair<Rule>) -> Result<ReadingClause, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty reading_clause".into()))?;
    match inner.as_rule() {
        Rule::match_clause => Ok(ReadingClause::Match(build_match_clause(inner)?)),
        Rule::unwind_clause => Ok(ReadingClause::Unwind(build_unwind_clause(inner)?)),
        Rule::in_query_call => Ok(ReadingClause::Call(build_call_clause(inner)?)),
        _ => Err(CypherError::ParseError(format!("Unknown reading clause: {:?}", inner.as_rule()))),
    }
}

fn build_match_clause(pair: Pair<Rule>) -> Result<MatchClause, CypherError> {
    let text = pair.as_str().to_uppercase();
    let optional = text.starts_with("OPTIONAL");
    let mut pattern = None;
    let mut where_clause = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::pattern => pattern = Some(build_pattern(child)?),
            Rule::where_clause => {
                let expr_pair = child.into_inner().next()
                    .ok_or_else(|| CypherError::ParseError("Empty WHERE".into()))?;
                where_clause = Some(build_expression(expr_pair)?);
            }
            _ => {}
        }
    }
    Ok(MatchClause {
        optional,
        pattern: pattern.ok_or_else(|| CypherError::ParseError("MATCH missing pattern".into()))?,
        where_clause,
    })
}

fn build_unwind_clause(pair: Pair<Rule>) -> Result<UnwindClause, CypherError> {
    let mut children = pair.into_inner();
    let expr = build_expression(children.next().ok_or_else(|| CypherError::ParseError("UNWIND missing expr".into()))?)?;
    let variable = children.next().ok_or_else(|| CypherError::ParseError("UNWIND missing variable".into()))?.as_str().to_string();
    Ok(UnwindClause { expr, variable })
}

fn build_call_clause(pair: Pair<Rule>) -> Result<CallClause, CypherError> {
    let mut procedure_name = String::new();
    let mut args = vec![];
    let mut yield_items = vec![];
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::explicit_procedure_invocation => {
                for sub in child.into_inner() {
                    match sub.as_rule() {
                        Rule::procedure_name => procedure_name = sub.as_str().to_string(),
                        Rule::expression => args.push(build_expression(sub)?),
                        _ => {}
                    }
                }
            }
            Rule::yield_items => {
                for yi in child.into_inner() {
                    if yi.as_rule() == Rule::yield_item {
                        let mut field = None;
                        let mut var = String::new();
                        for sub in yi.into_inner() {
                            match sub.as_rule() {
                                Rule::procedure_result_field => field = Some(sub.as_str().to_string()),
                                Rule::variable => var = sub.as_str().to_string(),
                                _ => {}
                            }
                        }
                        yield_items.push(YieldItem { field, variable: var });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(CallClause { procedure_name, args, yield_items })
}

fn build_updating_clause(pair: Pair<Rule>) -> Result<UpdatingClause, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty updating_clause".into()))?;
    match inner.as_rule() {
        Rule::create_clause => {
            let pat = inner.into_inner().next()
                .ok_or_else(|| CypherError::ParseError("CREATE missing pattern".into()))?;
            Ok(UpdatingClause::Create(build_pattern(pat)?))
        }
        Rule::merge_clause => Ok(UpdatingClause::Merge(build_merge_clause(inner)?)),
        Rule::set_clause => Ok(UpdatingClause::Set(build_set_clause(inner)?)),
        Rule::remove_clause => Ok(UpdatingClause::Remove(build_remove_clause(inner)?)),
        Rule::delete_clause => {
            let text = inner.as_str().to_uppercase();
            let detach = text.starts_with("DETACH");
            let exprs = inner.into_inner()
                .filter(|p| p.as_rule() == Rule::expression)
                .map(|p| build_expression(p))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(UpdatingClause::Delete { detach, exprs })
        }
        Rule::foreach_clause => build_foreach_clause(inner),
        _ => Err(CypherError::ParseError(format!("Unknown updating clause: {:?}", inner.as_rule()))),
    }
}

fn build_merge_clause(pair: Pair<Rule>) -> Result<MergeClause, CypherError> {
    let mut on_match = vec![];
    let mut on_create = vec![];
    let mut pattern_part = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::pattern_part => pattern_part = Some(build_pattern_part(child)?),
            Rule::on_match_or_create => {
                let text = child.as_str().to_uppercase();
                for sub in child.into_inner() {
                    if sub.as_rule() == Rule::set_clause {
                        let items = build_set_clause(sub)?;
                        if text.contains("MATCH") {
                            on_match.push(items);
                        } else {
                            on_create.push(items);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(MergeClause {
        pattern_part: pattern_part.ok_or_else(|| CypherError::ParseError("MERGE missing pattern_part".into()))?,
        on_match,
        on_create,
    })
}

fn build_set_clause(pair: Pair<Rule>) -> Result<Vec<SetItem>, CypherError> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::set_item)
        .map(|p| {
            let text = p.as_str();
            let children: Vec<Pair<Rule>> = p.into_inner().collect();
            if text.contains("+=") {
                let var = children[0].as_str().to_string();
                let value = build_expression(children[1].clone())?;
                Ok(SetItem::VariableAdd { var, value })
            } else if children.len() >= 2 {
                match children[0].as_rule() {
                    Rule::property_expression => {
                        let prop = build_property_expression(children[0].clone())?;
                        let value = build_expression(children[1].clone())?;
                        Ok(SetItem::PropertySet { prop, value })
                    }
                    Rule::variable => {
                        let var = children[0].as_str().to_string();
                        if children[1].as_rule() == Rule::node_labels {
                            let labels = children[1].clone().into_inner()
                                .filter(|p| p.as_rule() == Rule::label_name)
                                .map(|p| p.as_str().to_string())
                                .collect();
                            Ok(SetItem::LabelSet { var, labels })
                        } else {
                            let value = build_expression(children[1].clone())?;
                            Ok(SetItem::VariableSet { var, value })
                        }
                    }
                    _ => Err(CypherError::ParseError("Unknown SET item".into())),
                }
            } else {
                Err(CypherError::ParseError("Invalid SET item".into()))
            }
        })
        .collect()
}

fn build_remove_clause(pair: Pair<Rule>) -> Result<Vec<RemoveItem>, CypherError> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::remove_item)
        .map(|p| {
            let children: Vec<Pair<Rule>> = p.into_inner().collect();
            if children.is_empty() {
                return Err(CypherError::ParseError("Empty remove_item".into()));
            }
            match children[0].as_rule() {
                Rule::variable => {
                    let var = children[0].as_str().to_string();
                    let labels = if children.len() > 1 && children[1].as_rule() == Rule::node_labels {
                        children[1].clone().into_inner()
                            .filter(|p| p.as_rule() == Rule::label_name)
                            .map(|p| p.as_str().to_string())
                            .collect()
                    } else {
                        vec![]
                    };
                    Ok(RemoveItem::LabelRemove { var, labels })
                }
                Rule::property_expression => {
                    Ok(RemoveItem::PropertyRemove(build_property_expression(children[0].clone())?))
                }
                _ => Err(CypherError::ParseError("Unknown REMOVE item".into())),
            }
        })
        .collect()
}

fn build_property_expression(pair: Pair<Rule>) -> Result<PropertyExpr, CypherError> {
    let mut variable = String::new();
    let mut properties = vec![];
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::variable => variable = child.as_str().to_string(),
            Rule::property_lookup => {
                let key = child.into_inner().next()
                    .ok_or_else(|| CypherError::ParseError("Property lookup missing key".into()))?
                    .as_str().to_string();
                properties.push(key);
            }
            _ => {}
        }
    }
    Ok(PropertyExpr { variable, properties })
}

fn build_with_clause(pair: Pair<Rule>) -> Result<WithClause, CypherError> {
    let text = pair.as_str().to_uppercase();
    let distinct = text.contains("DISTINCT");
    let mut items = vec![];
    let mut order = None;
    let mut skip = None;
    let mut limit = None;
    let mut where_clause = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::return_body => {
                let (i, o, s, l) = build_return_body(child)?;
                items = i;
                order = o;
                skip = s;
                limit = l;
            }
            Rule::where_clause => {
                let e = child.into_inner().next().ok_or_else(|| CypherError::ParseError("WHERE missing expr".into()))?;
                where_clause = Some(build_expression(e)?);
            }
            _ => {}
        }
    }
    Ok(WithClause { distinct, items, order, skip, limit, where_clause })
}

fn build_return_clause(pair: Pair<Rule>) -> Result<ReturnClause, CypherError> {
    let text = pair.as_str().to_uppercase();
    let distinct = text.contains("DISTINCT");
    let mut items = ReturnItems::Wildcard;
    let mut order = None;
    let mut skip = None;
    let mut limit = None;
    for child in pair.into_inner() {
        if child.as_rule() == Rule::return_body {
            let (i, o, s, l) = build_return_body(child)?;
            items = ReturnItems::Items(i);
            order = o;
            skip = s;
            limit = l;
        }
    }
    Ok(ReturnClause { distinct, items, order, skip, limit })
}

fn build_return_body(pair: Pair<Rule>) -> Result<(Vec<ReturnItem>, Option<Vec<SortItem>>, Option<Expr>, Option<Expr>), CypherError> {
    let mut items = vec![];
    let mut order = None;
    let mut skip = None;
    let mut limit = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::return_items => {
                let s = child.as_str().trim();
                if s == "*" {
                    // wildcard - leave items empty
                } else {
                    for ri in child.into_inner().filter(|p| p.as_rule() == Rule::return_item) {
                        items.push(build_return_item(ri)?);
                    }
                }
            }
            Rule::order_clause => {
                order = Some(
                    child.into_inner()
                        .filter(|p| p.as_rule() == Rule::sort_item)
                        .map(|p| build_sort_item(p))
                        .collect::<Result<Vec<_>, _>>()?
                );
            }
            Rule::skip_clause => {
                let e = child.into_inner().next().ok_or_else(|| CypherError::ParseError("SKIP missing expr".into()))?;
                skip = Some(build_expression(e)?);
            }
            Rule::limit_clause => {
                let e = child.into_inner().next().ok_or_else(|| CypherError::ParseError("LIMIT missing expr".into()))?;
                limit = Some(build_expression(e)?);
            }
            _ => {}
        }
    }
    Ok((items, order, skip, limit))
}

fn build_return_item(pair: Pair<Rule>) -> Result<ReturnItem, CypherError> {
    let mut children = pair.into_inner();
    let expr = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Return item missing expr".into()))?)?;
    let alias = children.find(|p| p.as_rule() == Rule::variable).map(|p| p.as_str().to_string());
    Ok(ReturnItem { expr, alias })
}

fn build_sort_item(pair: Pair<Rule>) -> Result<SortItem, CypherError> {
    let text = pair.as_str().to_uppercase();
    let descending = text.contains("DESC");
    let expr = build_expression(pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Sort item missing expr".into()))?)?;
    Ok(SortItem { expr, descending })
}

// ===== Pattern building =====
pub fn build_pattern(pair: Pair<Rule>) -> Result<Pattern, CypherError> {
    let parts = pair.into_inner()
        .filter(|p| p.as_rule() == Rule::pattern_part)
        .map(|p| build_pattern_part(p))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Pattern { parts })
}

fn build_pattern_part(pair: Pair<Rule>) -> Result<PatternPart, CypherError> {
    let mut variable = None;
    let mut element = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::variable => variable = Some(child.as_str().to_string()),
            Rule::pattern_element => element = Some(build_pattern_element(child)?),
            _ => {}
        }
    }
    Ok(PatternPart {
        variable,
        element: element.ok_or_else(|| CypherError::ParseError("Pattern part missing element".into()))?,
    })
}

fn build_pattern_element(pair: Pair<Rule>) -> Result<PatternElement, CypherError> {
    let mut children = pair.into_inner();
    let first = children.next()
        .ok_or_else(|| CypherError::ParseError("Pattern element missing node".into()))?;

    // pattern_element may start with shortest_path_call; unwrap it to get the inner chain.
    let (start, rest): (NodePattern, Box<dyn Iterator<Item = Pair<Rule>>>) =
        if first.as_rule() == Rule::shortest_path_call {
            let mut sp = first.into_inner();
            let sp_start = build_node_pattern(
                sp.next().ok_or_else(|| CypherError::ParseError("shortestPath missing node".into()))?,
            )?;
            // chain from inside shortest_path_call, then nothing left in outer children
            (sp_start, Box::new(sp))
        } else {
            // Regular path: first child is node_pattern, rest is the outer iterator
            (build_node_pattern(first)?, Box::new(children))
        };

    let mut chain = vec![];
    let mut rel_opt: Option<RelationshipPattern> = None;
    for child in rest {
        match child.as_rule() {
            Rule::relationship_pattern => rel_opt = Some(build_relationship_pattern(child)?),
            Rule::node_pattern => {
                let node = build_node_pattern(child)?;
                if let Some(rel) = rel_opt.take() {
                    chain.push((rel, node));
                }
            }
            _ => {}
        }
    }
    Ok(PatternElement { start, chain })
}

pub fn build_node_pattern(pair: Pair<Rule>) -> Result<NodePattern, CypherError> {
    let mut variable = None;
    let mut labels = vec![];
    let mut properties = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::variable => variable = Some(child.as_str().to_string()),
            Rule::node_labels => {
                labels = child.into_inner()
                    .filter(|p| p.as_rule() == Rule::label_name)
                    .map(|p| p.as_str().to_string())
                    .collect();
            }
            Rule::properties => properties = Some(build_properties(child)?),
            _ => {}
        }
    }
    Ok(NodePattern { variable, labels, properties })
}

fn build_relationship_pattern(pair: Pair<Rule>) -> Result<RelationshipPattern, CypherError> {
    let full = pair.as_str();
    let direction = if full.starts_with('<') {
        Direction::Incoming
    } else if full.ends_with('>') {
        Direction::Outgoing
    } else {
        Direction::Undirected
    };
    let mut variable = None;
    let mut rel_types = vec![];
    let mut range = None;
    let mut properties = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::variable => variable = Some(child.as_str().to_string()),
            Rule::rel_types => {
                rel_types = child.into_inner()
                    .filter(|p| p.as_rule() == Rule::rel_type_name)
                    .map(|p| p.as_str().to_string())
                    .collect();
            }
            Rule::range_literal => range = Some(build_range_literal(child)?),
            Rule::properties => properties = Some(build_properties(child)?),
            _ => {}
        }
    }
    Ok(RelationshipPattern { direction, variable, rel_types, range, properties })
}

fn build_range_literal(pair: Pair<Rule>) -> Result<RangeLiteral, CypherError> {
    let text = pair.as_str().trim_start_matches('*').trim();
    if text.is_empty() {
        return Ok(RangeLiteral { min: None, max: None });
    }
    if text.contains("..") {
        let parts: Vec<&str> = text.split("..").collect();
        let min = parts[0].trim().parse::<i64>().ok();
        let max = parts.get(1).and_then(|s| s.trim().parse::<i64>().ok());
        Ok(RangeLiteral { min, max })
    } else {
        let n = text.parse::<i64>().ok();
        Ok(RangeLiteral { min: n, max: n })
    }
}

fn build_properties(pair: Pair<Rule>) -> Result<MapOrParam, CypherError> {
    let child = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty properties".into()))?;
    match child.as_rule() {
        Rule::map_literal => Ok(MapOrParam::Map(build_map_literal(child)?)),
        Rule::parameter => Ok(MapOrParam::Param(child.as_str().trim_start_matches('$').to_string())),
        _ => Err(CypherError::ParseError("Unknown properties type".into())),
    }
}

fn build_map_literal(pair: Pair<Rule>) -> Result<HashMap<String, Expr>, CypherError> {
    let mut map = HashMap::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::property_key_value_pair {
            let mut kv = child.into_inner();
            let key = kv.next().ok_or_else(|| CypherError::ParseError("Map missing key".into()))?.as_str().to_string();
            let val = build_expression(kv.next().ok_or_else(|| CypherError::ParseError("Map missing value".into()))?)?;
            map.insert(key, val);
        }
    }
    Ok(map)
}

// ===== Expression building =====
pub fn build_expression(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    match pair.as_rule() {
        Rule::expression => {
            let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty expression".into()))?;
            build_expression(inner)
        }
        Rule::or_expr => build_or_expr(pair),
        Rule::xor_expr => build_xor_expr(pair),
        Rule::and_expr => build_and_expr(pair),
        Rule::not_expr => build_not_expr(pair),
        Rule::comparison_expr => build_comparison_expr(pair),
        Rule::add_sub_expr => build_add_sub_expr(pair),
        Rule::mul_div_expr => build_mul_div_expr(pair),
        Rule::mod_expr => build_mod_expr(pair),
        Rule::power_expr => build_power_expr(pair),
        Rule::unary_expr => build_unary_expr(pair),
        Rule::postfix_expr => build_postfix_expr(pair),
        Rule::atom => build_atom(pair),
        _ => Err(CypherError::ParseError(format!("Unexpected rule in expression: {:?}", pair.as_rule()))),
    }
}

fn build_or_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    // Filter out or_kw tokens — they now appear in the parse tree because
    // or_kw uses @{ } (atomic) to fix the word-boundary lookahead bug.
    let mut children = pair.into_inner().filter(|p| p.as_rule() != Rule::or_kw);
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty or_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::Or(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_xor_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    // Filter out xor_kw tokens — same reason as or_kw above.
    let mut children = pair.into_inner().filter(|p| p.as_rule() != Rule::xor_kw);
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty xor_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::Xor(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_and_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    // Filter out and_kw tokens — same reason as or_kw above.
    let mut children = pair.into_inner().filter(|p| p.as_rule() != Rule::and_kw);
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty and_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::And(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_not_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let text = pair.as_str().to_uppercase();
    let nots = text.matches("NOT").count();
    let inner = pair.into_inner()
        .find(|p| p.as_rule() == Rule::comparison_expr)
        .ok_or_else(|| CypherError::ParseError("NOT missing comparison_expr".into()))?;
    let mut expr = build_comparison_expr(inner)?;
    for _ in 0..nots {
        expr = Expr::Not(Box::new(expr));
    }
    Ok(expr)
}

fn build_comparison_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty comparison_expr".into()))?)?;
    let mut result = first;
    loop {
        let op_pair = match children.next() {
            Some(p) => p,
            None => break,
        };
        let right_pair = match children.next() {
            Some(p) => p,
            None => break,
        };
        let right = build_expression(right_pair)?;
        let op = op_pair.as_str().to_uppercase();
        result = match op.as_str() {
            "=" => Expr::Eq(Box::new(result), Box::new(right)),
            "<>" => Expr::Ne(Box::new(result), Box::new(right)),
            "<" => Expr::Lt(Box::new(result), Box::new(right)),
            "<=" => Expr::Lte(Box::new(result), Box::new(right)),
            ">" => Expr::Gt(Box::new(result), Box::new(right)),
            ">=" => Expr::Gte(Box::new(result), Box::new(right)),
            "=~" => Expr::Regex(Box::new(result), Box::new(right)),
            "IN" => Expr::In(Box::new(result), Box::new(right)),
            s if s.starts_with("STARTS") => Expr::StartsWith(Box::new(result), Box::new(right)),
            s if s.starts_with("ENDS") => Expr::EndsWith(Box::new(result), Box::new(right)),
            "CONTAINS" => Expr::Contains(Box::new(result), Box::new(right)),
            _ => return Err(CypherError::ParseError(format!("Unknown comparison op: {}", op))),
        };
    }
    Ok(result)
}

fn build_add_sub_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner().peekable();
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty add_sub_expr".into()))?)?;
    let mut result = first;
    while children.peek().is_some() {
        // In the pest grammar, the operator appears as a separate token in the string
        // We need to check the string between pairs
        let next = children.next().unwrap();
        // Determine operator from current string context - look at next pair's rule
        // The operators appear inline in the parsed text
        match next.as_rule() {
            Rule::mul_div_expr | Rule::mod_expr | Rule::power_expr | Rule::unary_expr | Rule::postfix_expr | Rule::atom => {
                // No operator pair — check if we need to figure out the operator from text
                // For left-associative, operator was embedded in pair structure
                result = Expr::Add(Box::new(result), Box::new(build_expression(next)?));
            }
            _ => {
                result = Expr::Add(Box::new(result), Box::new(build_expression(next)?));
            }
        }
    }
    Ok(result)
}

fn build_mul_div_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty mul_div_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::Mul(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_mod_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty mod_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::Mod(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_power_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let first = build_expression(children.next().ok_or_else(|| CypherError::ParseError("Empty power_expr".into()))?)?;
    children.try_fold(first, |acc, next| {
        Ok(Expr::Pow(Box::new(acc), Box::new(build_expression(next)?)))
    })
}

fn build_unary_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let text = pair.as_str();
    let negated = text.starts_with('-');
    let inner = pair.into_inner()
        .find(|p| p.as_rule() == Rule::postfix_expr)
        .ok_or_else(|| CypherError::ParseError("Unary missing postfix".into()))?;
    let expr = build_postfix_expr(inner)?;
    if negated {
        Ok(Expr::Neg(Box::new(expr)))
    } else {
        Ok(expr)
    }
}

fn build_postfix_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let base = build_atom(children.next().ok_or_else(|| CypherError::ParseError("Postfix missing atom".into()))?)?;
    let mut result = base;
    for child in children {
        match child.as_rule() {
            Rule::postfix_op => {
                let op = child.into_inner().next()
                    .ok_or_else(|| CypherError::ParseError("Empty postfix_op".into()))?;
                match op.as_rule() {
                    Rule::null_check_op => {
                        let text = op.as_str().to_uppercase();
                        result = if text.contains("NOT") {
                            Expr::IsNotNull(Box::new(result))
                        } else {
                            Expr::IsNull(Box::new(result))
                        };
                    }
                    Rule::property_lookup => {
                        let key = op.into_inner().next()
                            .ok_or_else(|| CypherError::ParseError("Property lookup missing key".into()))?
                            .as_str().to_string();
                        result = Expr::Property { expr: Box::new(result), key };
                    }
                    Rule::subscript_op => {
                        let idx = op.into_inner().next()
                            .ok_or_else(|| CypherError::ParseError("Subscript missing index".into()))?;
                        let index = build_expression(idx)?;
                        result = Expr::Subscript { expr: Box::new(result), index: Box::new(index) };
                    }
                    Rule::list_slice_op => {
                        let mut sub = op.into_inner();
                        let from = sub.next().map(|p| build_expression(p)).transpose()?;
                        let to = sub.next().map(|p| build_expression(p)).transpose()?;
                        result = Expr::Slice {
                            expr: Box::new(result),
                            from: from.map(Box::new),
                            to: to.map(Box::new),
                        };
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(result)
}

fn build_atom(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty atom".into()))?;
    match inner.as_rule() {
        Rule::reduce_expression => build_reduce_expression(inner),
        Rule::shortest_path_call => build_shortest_path_expr(inner),
        Rule::literal => build_literal(inner),
        Rule::parameter => Ok(Expr::Parameter(inner.as_str().trim_start_matches('$').to_string())),
        Rule::case_expression => build_case_expression(inner),
        Rule::list_comprehension => build_list_comprehension(inner),
        Rule::quantifier_expression => build_quantifier_expression(inner),
        Rule::exists_subquery => build_exists_subquery(inner),
        Rule::count_star => Ok(Expr::CountStar),
        Rule::function_invocation => build_function_invocation(inner),
        Rule::variable => Ok(Expr::Variable(inner.as_str().to_string())),
        Rule::parenthesized_expr => {
            let e = inner.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty paren expr".into()))?;
            build_expression(e)
        }
        _ => Err(CypherError::ParseError(format!("Unknown atom: {:?}", inner.as_rule()))),
    }
}

fn build_literal(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty literal".into()))?;
    match inner.as_rule() {
        Rule::integer_literal => {
            let s = inner.as_str();
            let n = if s.starts_with("0x") || s.starts_with("0X") {
                i64::from_str_radix(&s[2..], 16).unwrap_or(0)
            } else if s.starts_with("0o") {
                i64::from_str_radix(&s[2..], 8).unwrap_or(0)
            } else {
                s.parse::<i64>().unwrap_or(0)
            };
            Ok(Expr::Integer(n))
        }
        Rule::float_literal => Ok(Expr::Float(inner.as_str().parse::<f64>().unwrap_or(0.0))),
        Rule::string_literal => {
            let s = inner.as_str();
            let unquoted = &s[1..s.len()-1];
            Ok(Expr::Str(unquoted.to_string()))
        }
        Rule::boolean_literal => Ok(Expr::Bool(inner.as_str().to_uppercase() == "TRUE")),
        Rule::null_literal => Ok(Expr::Null),
        Rule::list_literal => {
            let items = inner.into_inner()
                .filter(|p| p.as_rule() == Rule::expression)
                .map(|p| build_expression(p))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::List(items))
        }
        Rule::map_literal => Ok(Expr::Map(build_map_literal(inner)?)),
        _ => Err(CypherError::ParseError(format!("Unknown literal: {:?}", inner.as_rule()))),
    }
}

fn build_case_expression(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner().peekable();
    let subject = if let Some(p) = children.peek() {
        if p.as_rule() == Rule::expression {
            Some(Box::new(build_expression(children.next().unwrap())?))
        } else {
            None
        }
    } else {
        None
    };
    let mut alternatives = vec![];
    let mut default = None;
    let pairs: Vec<Pair<Rule>> = children.collect();
    let mut i = 0;
    while i < pairs.len() {
        let text = pairs[i].as_str().to_uppercase();
        if text == "ELSE" {
            if i + 1 < pairs.len() {
                default = Some(Box::new(build_expression(pairs[i + 1].clone())?));
            }
            break;
        } else if i + 1 < pairs.len() {
            let when = build_expression(pairs[i].clone())?;
            let then = build_expression(pairs[i + 1].clone())?;
            alternatives.push((when, then));
            i += 2;
            continue;
        }
        i += 1;
    }
    Ok(Expr::Case { subject, alternatives, default })
}

/// Parse a `filter_expression` pair into (variable, source, optional filter predicate).
fn parse_filter_expression(pair: Pair<Rule>) -> Result<(String, Expr, Option<Box<Expr>>), CypherError> {
    let mut variable = String::new();
    let mut source = Expr::Null;
    let mut filter = None;
    for sub in pair.into_inner() {
        match sub.as_rule() {
            Rule::id_in_coll => {
                let mut ic = sub.into_inner();
                variable = ic.next().map(|p| p.as_str().to_string()).unwrap_or_default();
                if let Some(src_pair) = ic.next() {
                    source = build_expression(src_pair)?;
                }
            }
            Rule::expression => filter = Some(Box::new(build_expression(sub)?)),
            _ => {}
        }
    }
    Ok((variable, source, filter))
}

fn build_list_comprehension(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut variable = String::new();
    let mut source = Expr::Null;
    let mut filter = None;
    let mut projection = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::filter_expression => {
                let parsed = parse_filter_expression(child)?;
                variable = parsed.0;
                source = parsed.1;
                filter = parsed.2;
            }
            Rule::expression => projection = Some(Box::new(build_expression(child)?)),
            _ => {}
        }
    }
    Ok(Expr::ListComprehension { variable, source: Box::new(source), filter, projection })
}

fn build_quantifier_expression(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let text = pair.as_str().to_uppercase();
    let kind = if text.starts_with("ALL") { "ALL" }
               else if text.starts_with("ANY") { "ANY" }
               else if text.starts_with("NONE") { "NONE" }
               else { "SINGLE" };
    let mut variable = String::new();
    let mut source = Expr::Null;
    let mut filter = None;
    for child in pair.into_inner() {
        if child.as_rule() == Rule::filter_expression {
            let parsed = parse_filter_expression(child)?;
            variable = parsed.0;
            source = parsed.1;
            filter = parsed.2;
        }
    }
    match kind {
        "ALL"  => Ok(Expr::All  { variable, source: Box::new(source), filter }),
        "ANY"  => Ok(Expr::Any  { variable, source: Box::new(source), filter }),
        "NONE" => Ok(Expr::None { variable, source: Box::new(source), filter }),
        _      => Ok(Expr::Single { variable, source: Box::new(source), filter }),
    }
}

fn build_exists_subquery(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let inner = pair.into_inner().next().ok_or_else(|| CypherError::ParseError("Empty EXISTS".into()))?;
    match inner.as_rule() {
        Rule::regular_query => Ok(Expr::Exists(Box::new(ExistsSubquery::Query(build_regular_query(inner)?)))),
        Rule::pattern => Ok(Expr::Exists(Box::new(ExistsSubquery::Pattern(build_pattern(inner)?)))),
        _ => Err(CypherError::ParseError("Unknown EXISTS subquery type".into())),
    }
}

/// Build an `Expr::ShortestPath` from a `shortest_path_call` pair.
/// The pair's children are: node_pattern, (relationship_pattern node_pattern)*.
fn build_shortest_path_expr(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let text = pair.as_str();
    let all = text.len() >= 3 && text[..3].eq_ignore_ascii_case("all");
    let mut children = pair.into_inner();
    let start = build_node_pattern(
        children.next().ok_or_else(|| CypherError::ParseError("shortestPath missing start node".into()))?,
    )?;
    let mut chain = vec![];
    let mut rel_opt: Option<RelationshipPattern> = None;
    for child in children {
        match child.as_rule() {
            Rule::relationship_pattern => rel_opt = Some(build_relationship_pattern(child)?),
            Rule::node_pattern => {
                if let Some(rel) = rel_opt.take() {
                    chain.push((rel, build_node_pattern(child)?));
                }
            }
            _ => {}
        }
    }
    Ok(Expr::ShortestPath {
        all,
        element: Box::new(PatternElement { start, chain }),
    })
}

fn build_foreach_clause(pair: Pair<Rule>) -> Result<UpdatingClause, CypherError> {
    // Children (silent tokens stripped): variable, expression, (updating_clause)+
    let mut children = pair.into_inner();
    let variable = children
        .next()
        .ok_or_else(|| CypherError::ParseError("FOREACH missing variable".into()))?
        .as_str()
        .to_string();
    let list_expr = build_expression(
        children
            .next()
            .ok_or_else(|| CypherError::ParseError("FOREACH missing list expression".into()))?,
    )?;
    let mut body = vec![];
    for child in children {
        if child.as_rule() == Rule::updating_clause {
            body.push(build_updating_clause(child)?);
        }
    }
    if body.is_empty() {
        return Err(CypherError::ParseError("FOREACH body is empty".into()));
    }
    Ok(UpdatingClause::Foreach { variable, list_expr, body })
}

fn build_reduce_expression(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    // Children (in order, silent tokens filtered by pest):
    //   variable (accumulator), expression (init),
    //   variable (loop var), expression (source), expression (projection)
    let mut children = pair.into_inner();
    let accumulator = children
        .next()
        .ok_or_else(|| CypherError::ParseError("REDUCE missing accumulator".into()))?
        .as_str()
        .to_string();
    let init = build_expression(
        children
            .next()
            .ok_or_else(|| CypherError::ParseError("REDUCE missing init expression".into()))?,
    )?;
    let variable = children
        .next()
        .ok_or_else(|| CypherError::ParseError("REDUCE missing loop variable".into()))?
        .as_str()
        .to_string();
    let source = build_expression(
        children
            .next()
            .ok_or_else(|| CypherError::ParseError("REDUCE missing source expression".into()))?,
    )?;
    let projection = build_expression(
        children
            .next()
            .ok_or_else(|| CypherError::ParseError("REDUCE missing projection expression".into()))?,
    )?;
    Ok(Expr::Reduce {
        accumulator,
        init: Box::new(init),
        variable,
        source: Box::new(source),
        projection: Box::new(projection),
    })
}

fn build_function_invocation(pair: Pair<Rule>) -> Result<Expr, CypherError> {
    let mut children = pair.into_inner();
    let name = children.next().ok_or_else(|| CypherError::ParseError("Function missing name".into()))?.as_str().to_string();
    let rest: Vec<Pair<Rule>> = children.collect();
    let distinct = rest.iter().any(|p| p.as_str().to_uppercase() == "DISTINCT");
    let args = rest.into_iter()
        .filter(|p| p.as_rule() == Rule::expression)
        .map(|p| build_expression(p))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::FunctionCall { name, distinct, args })
}
