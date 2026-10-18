//! Compare a restricted paper-chain diagnostic with a saved pricing request.
use pricing::mc::ExecutionPolicy;
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::{Value, json};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args()
        .nth(1)
        .ok_or("usage: vega_kt_chain_probe REQUEST.json")?;
    let bytes = std::fs::read(file)?;
    let wire: Value = serde_json::from_slice(&bytes)?;
    let request = parse_request_json(&bytes, JsonLimits::DEFAULT)?;
    let plan = PricingPlan::compile(&request, ExecutionPolicy::new(1, Some(256))?)?;
    let spot = wire["market"]["spot"]
        .as_f64()
        .ok_or("equity spot required")?;
    let strikes = wire["model"]["local_variance_grid"]["log_forward_moneyness_nodes"]
        .as_array()
        .ok_or("LV spatial nodes required")?
        .iter()
        .map(|x| Ok(spot * x.as_f64().ok_or("finite spatial node required")?.exp()))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let old = plan.evaluate()?;
    let r = plan.evaluate_vega_kt_chain_diagnostic(strikes, 0.04, 2)?;
    let e = r.estimate;
    let width = e.reporting_basis.log_moneyness_nodes().len();
    let local_vega = old
        .pricing_result
        .risks
        .vega
        .as_ref()
        .ok_or("probe requires a scalar Vega risk request")?
        .raw()
        .value()
        .get();
    let legacy_sum = old
        .pricing_result
        .risks
        .vega_kt
        .as_ref()
        .ok_or("probe requires a VegaKT risk request")?
        .raw_buckets()
        .iter()
        .map(|v| v.get())
        .sum::<f64>();
    println!(
        "{}",
        json!({
            "diagnostic_only": true,
            "policy": e.policy_label,
            "request_fingerprint": r.request_fingerprint.to_string(),
            "local_vega": local_vega,
            "legacy_sum": legacy_sum,
            "chain": e.total_position_iv_vega.mean(),
            "chain_se": e.total_position_iv_vega.standard_error(),
            "initial_difference": e.initial_reconstruction_difference.mean(),
            "quadrature_difference": e.total_quadrature_difference.mean(),
            "requested_initial_strikes": r.requested_initial_strikes.to_vec(),
            "initial_strikes": r.initial_strikes.to_vec(),
            "recovery_log_moneyness_nodes": r.recovery_log_moneyness_nodes.to_vec(),
            "maturity_totals": e.position_iv_buckets.chunks_exact(width)
                .map(|row| row.iter().map(|v| v.mean()).sum::<f64>()).collect::<Vec<_>>()
        })
    );
    Ok(())
}
