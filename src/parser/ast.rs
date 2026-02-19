use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub statement: Statement,
}

#[derive(Debug, Clone)]
pub enum Statement {
    RegularQuery(RegularQuery),
    StandaloneCall(CallClause),
}

#[derive(Debug, Clone)]
pub struct RegularQuery {
    pub single_query: SingleQuery,
    pub unions: Vec<(bool, SingleQuery)>, // (is_all, query)
}

#[derive(Debug, Clone)]
pub enum SingleQuery {
    SinglePart(SinglePartQuery),
    MultiPart(MultiPartQuery),
}

#[derive(Debug, Clone)]
pub struct SinglePartQuery {
    pub reading: Vec<ReadingClause>,
    pub updating: Vec<UpdatingClause>,
    pub ret: Option<ReturnClause>,
}

#[derive(Debug, Clone)]
pub struct MultiPartQuery {
    pub parts: Vec<(Vec<ReadingClause>, Vec<UpdatingClause>, WithClause)>,
    pub reading: Vec<ReadingClause>,
    pub updating: Vec<UpdatingClause>,
    pub ret: Option<ReturnClause>,
}

#[derive(Debug, Clone)]
pub enum ReadingClause {
    Match(MatchClause),
    Unwind(UnwindClause),
    Call(CallClause),
}

#[derive(Debug, Clone)]
pub struct MatchClause {
    pub optional: bool,
    pub pattern: Pattern,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct UnwindClause {
    pub expr: Expr,
    pub variable: String,
}

#[derive(Debug, Clone)]
pub struct CallClause {
    pub procedure_name: String,
    pub args: Vec<Expr>,
    pub yield_items: Vec<YieldItem>,
}

#[derive(Debug, Clone)]
pub struct YieldItem {
    pub field: Option<String>,
    pub variable: String,
}

#[derive(Debug, Clone)]
pub enum UpdatingClause {
    Create(Pattern),
    Merge(MergeClause),
    Set(Vec<SetItem>),
    Remove(Vec<RemoveItem>),
    Delete { detach: bool, exprs: Vec<Expr> },
}

#[derive(Debug, Clone)]
pub struct MergeClause {
    pub pattern_part: PatternPart,
    pub on_match: Vec<Vec<SetItem>>,
    pub on_create: Vec<Vec<SetItem>>,
}

#[derive(Debug, Clone)]
pub enum SetItem {
    PropertySet { prop: PropertyExpr, value: Expr },
    VariableSet { var: String, value: Expr },
    VariableAdd { var: String, value: Expr },
    LabelSet { var: String, labels: Vec<String> },
}

#[derive(Debug, Clone)]
pub enum RemoveItem {
    LabelRemove { var: String, labels: Vec<String> },
    PropertyRemove(PropertyExpr),
}

#[derive(Debug, Clone)]
pub struct PropertyExpr {
    pub variable: String,
    pub properties: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct WithClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
    pub order: Option<Vec<SortItem>>,
    pub skip: Option<Expr>,
    pub limit: Option<Expr>,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: ReturnItems,
    pub order: Option<Vec<SortItem>>,
    pub skip: Option<Expr>,
    pub limit: Option<Expr>,
}

#[derive(Debug, Clone)]
pub enum ReturnItems {
    Wildcard,
    Items(Vec<ReturnItem>),
}

#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SortItem {
    pub expr: Expr,
    pub descending: bool,
}

// ===== Pattern types =====
#[derive(Debug, Clone)]
pub struct Pattern {
    pub parts: Vec<PatternPart>,
}

#[derive(Debug, Clone)]
pub struct PatternPart {
    pub variable: Option<String>,
    pub element: PatternElement,
}

#[derive(Debug, Clone)]
pub struct PatternElement {
    pub start: NodePattern,
    pub chain: Vec<(RelationshipPattern, NodePattern)>,
}

#[derive(Debug, Clone)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: Option<MapOrParam>,
}

#[derive(Debug, Clone)]
pub struct RelationshipPattern {
    pub direction: Direction,
    pub variable: Option<String>,
    pub rel_types: Vec<String>,
    pub range: Option<RangeLiteral>,
    pub properties: Option<MapOrParam>,
}

#[derive(Debug, Clone)]
pub enum Direction {
    Outgoing,    // -->
    Incoming,    // <--
    Undirected,  // --
}

#[derive(Debug, Clone)]
pub struct RangeLiteral {
    pub min: Option<i64>,
    pub max: Option<i64>,
}

#[derive(Debug, Clone)]
pub enum MapOrParam {
    Map(HashMap<String, Expr>),
    Param(String),
}

// ===== Expressions =====
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Integer(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    List(Vec<Expr>),
    Map(HashMap<String, Expr>),

    // Variables and properties
    Variable(String),
    Property { expr: Box<Expr>, key: String },
    Subscript { expr: Box<Expr>, index: Box<Expr> },
    Slice { expr: Box<Expr>, from: Option<Box<Expr>>, to: Option<Box<Expr>> },

    // Parameter
    Parameter(String),

    // Operators
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),

    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Lte(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Gte(Box<Expr>, Box<Expr>),
    Regex(Box<Expr>, Box<Expr>),

    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    In(Box<Expr>, Box<Expr>),
    StartsWith(Box<Expr>, Box<Expr>),
    EndsWith(Box<Expr>, Box<Expr>),
    Contains(Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),

    // Function calls
    FunctionCall { name: String, distinct: bool, args: Vec<Expr> },
    CountStar,

    // Case
    Case {
        subject: Option<Box<Expr>>,
        alternatives: Vec<(Expr, Expr)>,
        default: Option<Box<Expr>>,
    },

    // List comprehension
    ListComprehension {
        variable: String,
        source: Box<Expr>,
        filter: Option<Box<Expr>>,
        projection: Option<Box<Expr>>,
    },

    // Quantifiers
    All { variable: String, source: Box<Expr>, filter: Option<Box<Expr>> },
    Any { variable: String, source: Box<Expr>, filter: Option<Box<Expr>> },
    None { variable: String, source: Box<Expr>, filter: Option<Box<Expr>> },
    Single { variable: String, source: Box<Expr>, filter: Option<Box<Expr>> },

    // Exists subquery
    Exists(Box<ExistsSubquery>),
}

#[derive(Debug, Clone)]
pub enum ExistsSubquery {
    Query(RegularQuery),
    Pattern(Pattern),
}
