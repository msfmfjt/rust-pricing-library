//! Strict source-only graph documents; no executable state or callbacks.
use super::*;
use crate::core::{FiniteF64, NodeId};
use crate::product::{CompactC2Smoothing, GraphProduct, SourceGraph, SourceNode, SourceOpcode};
const KIND: &str = "product_graph";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GraphProductV1 {
    currency_id: u16,
    underlying_ids: Vec<u32>,
    nodes: Vec<NodeV1>,
    cashflows: Vec<CashflowV1>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeV1 {
    id: u32,
    operation: OperationV1,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CashflowV1 {
    output_node: u32,
    payment_date: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OperationV1 {
    Literal {
        value: f64,
    },
    TerminalSpot {
        underlying_id: u32,
        observation_date: String,
    },
    PreDividendSpot {
        underlying_id: u32,
        observation_date: String,
    },
    Add {
        left: u32,
        right: u32,
    },
    Subtract {
        left: u32,
        right: u32,
    },
    Multiply {
        left: u32,
        right: u32,
    },
    Divide {
        numerator: u32,
        denominator: u32,
    },
    Minimum {
        left: u32,
        right: u32,
    },
    Maximum {
        left: u32,
        right: u32,
    },
    Indicator {
        input: u32,
    },
    Negate {
        input: u32,
    },
    BarrierHitState {
        previous: u32,
        hit_weight: u32,
    },
    SmoothMinimum {
        left: u32,
        right: u32,
        half_width: f64,
    },
    SmoothMaximum {
        left: u32,
        right: u32,
        half_width: f64,
    },
    SmoothIndicator {
        input: u32,
        half_width: f64,
    },
}
impl From<SourceOpcode> for OperationV1 {
    fn from(op: SourceOpcode) -> Self {
        match op {
            SourceOpcode::Literal(value) => Self::Literal { value: value.get() },
            SourceOpcode::TerminalSpot {
                underlying,
                observation_date,
            } => Self::TerminalSpot {
                underlying_id: underlying.get(),
                observation_date: observation_date.to_string(),
            },
            SourceOpcode::PreDividendSpot {
                underlying,
                observation_date,
            } => Self::PreDividendSpot {
                underlying_id: underlying.get(),
                observation_date: observation_date.to_string(),
            },
            SourceOpcode::Add { left, right } => Self::Add {
                left: left.get(),
                right: right.get(),
            },
            SourceOpcode::Subtract { left, right } => Self::Subtract {
                left: left.get(),
                right: right.get(),
            },
            SourceOpcode::Multiply { left, right } => Self::Multiply {
                left: left.get(),
                right: right.get(),
            },
            SourceOpcode::Divide {
                numerator,
                denominator,
            } => Self::Divide {
                numerator: numerator.get(),
                denominator: denominator.get(),
            },
            SourceOpcode::Minimum { left, right } => Self::Minimum {
                left: left.get(),
                right: right.get(),
            },
            SourceOpcode::Maximum { left, right } => Self::Maximum {
                left: left.get(),
                right: right.get(),
            },
            SourceOpcode::Indicator { input } => Self::Indicator { input: input.get() },
            SourceOpcode::Negate { input } => Self::Negate { input: input.get() },
            SourceOpcode::BarrierHitState {
                previous,
                hit_weight,
            } => Self::BarrierHitState {
                previous: previous.get(),
                hit_weight: hit_weight.get(),
            },
            SourceOpcode::SmoothMinimum {
                left,
                right,
                smoothing,
            } => Self::SmoothMinimum {
                left: left.get(),
                right: right.get(),
                half_width: smoothing.half_width().get(),
            },
            SourceOpcode::SmoothMaximum {
                left,
                right,
                smoothing,
            } => Self::SmoothMaximum {
                left: left.get(),
                right: right.get(),
                half_width: smoothing.half_width().get(),
            },
            SourceOpcode::SmoothIndicator { input, smoothing } => Self::SmoothIndicator {
                input: input.get(),
                half_width: smoothing.half_width().get(),
            },
        }
    }
}
impl OperationV1 {
    fn into_domain(self, pointer: &str) -> Result<SourceOpcode, WireError> {
        Ok(match self {
            Self::Literal { value } => SourceOpcode::Literal(
                FiniteF64::new(value, "graph literal").map_err(|e| domain_at(pointer, e))?,
            ),
            Self::TerminalSpot {
                underlying_id,
                observation_date,
            } => SourceOpcode::TerminalSpot {
                underlying: UnderlyingId::new(underlying_id),
                observation_date: parse_date_owned_at(
                    &observation_date,
                    format!("{pointer}/observation_date"),
                )?,
            },
            Self::PreDividendSpot {
                underlying_id,
                observation_date,
            } => SourceOpcode::PreDividendSpot {
                underlying: UnderlyingId::new(underlying_id),
                observation_date: parse_date_owned_at(
                    &observation_date,
                    format!("{pointer}/observation_date"),
                )?,
            },
            Self::Add { left, right } => SourceOpcode::Add {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Self::Subtract { left, right } => SourceOpcode::Subtract {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Self::Multiply { left, right } => SourceOpcode::Multiply {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Self::Divide {
                numerator,
                denominator,
            } => SourceOpcode::Divide {
                numerator: NodeId::new(numerator),
                denominator: NodeId::new(denominator),
            },
            Self::Minimum { left, right } => SourceOpcode::Minimum {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Self::Maximum { left, right } => SourceOpcode::Maximum {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Self::Indicator { input } => SourceOpcode::Indicator {
                input: NodeId::new(input),
            },
            Self::Negate { input } => SourceOpcode::Negate {
                input: NodeId::new(input),
            },
            Self::BarrierHitState {
                previous,
                hit_weight,
            } => SourceOpcode::BarrierHitState {
                previous: NodeId::new(previous),
                hit_weight: NodeId::new(hit_weight),
            },
            Self::SmoothMinimum {
                left,
                right,
                half_width,
            } => SourceOpcode::SmoothMinimum {
                left: NodeId::new(left),
                right: NodeId::new(right),
                smoothing: CompactC2Smoothing::new(half_width)
                    .map_err(|e| domain_at(pointer, e))?,
            },
            Self::SmoothMaximum {
                left,
                right,
                half_width,
            } => SourceOpcode::SmoothMaximum {
                left: NodeId::new(left),
                right: NodeId::new(right),
                smoothing: CompactC2Smoothing::new(half_width)
                    .map_err(|e| domain_at(pointer, e))?,
            },
            Self::SmoothIndicator { input, half_width } => SourceOpcode::SmoothIndicator {
                input: NodeId::new(input),
                smoothing: CompactC2Smoothing::new(half_width)
                    .map_err(|e| domain_at(pointer, e))?,
            },
        })
    }
}
impl From<&GraphProduct> for GraphProductV1 {
    fn from(product: &GraphProduct) -> Self {
        let mut nodes: Vec<_> = product
            .source_graph()
            .nodes()
            .iter()
            .map(|node| NodeV1 {
                id: node.id().get(),
                operation: node.opcode().into(),
            })
            .collect();
        nodes.sort_unstable_by_key(|node| node.id);
        Self {
            currency_id: product.currency().get(),
            underlying_ids: product.underlyings().iter().map(|id| id.get()).collect(),
            nodes,
            cashflows: product
                .source_graph()
                .outputs()
                .iter()
                .zip(product.payment_dates())
                .map(|(id, date)| CashflowV1 {
                    output_node: id.get(),
                    payment_date: date.to_string(),
                })
                .collect(),
        }
    }
}
impl GraphProductV1 {
    pub(super) fn into_domain(self, pointer: &str) -> Result<GraphProduct, WireError> {
        let nodes = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, node)| {
                Ok(SourceNode::new(
                    NodeId::new(node.id),
                    node.operation
                        .into_domain(&format!("{pointer}/nodes/{i}/operation"))?,
                ))
            })
            .collect::<Result<Vec<_>, WireError>>()?;
        let outputs = self
            .cashflows
            .iter()
            .map(|flow| NodeId::new(flow.output_node))
            .collect();
        let dates = self
            .cashflows
            .into_iter()
            .enumerate()
            .map(|(i, flow)| {
                parse_date_owned_at(
                    &flow.payment_date,
                    format!("{pointer}/cashflows/{i}/payment_date"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        GraphProduct::new(
            CurrencyId::new(self.currency_id),
            self.underlying_ids
                .into_iter()
                .map(UnderlyingId::new)
                .collect(),
            SourceGraph::new(nodes, outputs),
            dates,
        )
        .map_err(|e| domain_at(pointer, e))
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentV1 {
    document_kind: String,
    schema_version: u32,
    product: GraphProductV1,
}
impl From<&GraphProduct> for DocumentV1 {
    fn from(product: &GraphProduct) -> Self {
        Self {
            document_kind: KIND.into(),
            schema_version: 1,
            product: product.into(),
        }
    }
}
pub fn parse_product_graph_json(
    input: &[u8],
    limits: JsonLimits,
) -> Result<GraphProduct, WireError> {
    let text = validate_and_decode(input, limits)?;
    let header: Envelope = serde_json::from_str(text).map_err(json)?;
    if header.document_kind != KIND {
        return Err(WireError::WrongDocumentKind {
            expected: KIND,
            actual: header.document_kind,
        });
    }
    if header.schema_version != 1 {
        return Err(WireError::UnsupportedSchemaVersion(header.schema_version));
    }
    let document: DocumentV1 = serde_json::from_str(text).map_err(json)?;
    document.product.into_domain("/product")
}
pub fn product_graph_to_json(product: &GraphProduct) -> Result<String, WireError> {
    serialize(&DocumentV1::from(product), false)
}
pub fn product_graph_to_pretty_json(product: &GraphProduct) -> Result<String, WireError> {
    serialize(&DocumentV1::from(product), true)
}
pub fn fingerprint_product_graph(product: &GraphProduct) -> Result<Fingerprint, WireError> {
    let value = serde_json::to_value(DocumentV1::from(product)).map_err(json)?;
    let mut bytes = b"pricing/product_graph\0".to_vec();
    bytes.extend_from_slice(&1_u32.to_be_bytes());
    encode_value(&value, &mut bytes)?;
    Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
}
#[must_use]
pub fn current_product_graph_schema() -> &'static str {
    include_str!("../../../../schemas/product-graph-v1/product_graph.schema.json")
}
