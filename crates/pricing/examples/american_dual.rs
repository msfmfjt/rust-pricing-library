//! cargo run --locked --release -p pricing --example american_dual
use std::sync::Arc;

use pricing::PricingRequest;
use pricing::core::{CurrencyId, CurveId, Date, PositiveF64, UnderlyingId};
use pricing::dual::{AndersenBroadieConfig, AndersenBroadiePlan};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{
    CpqrConfig, EngineConfig, ExecutionPolicy, LsmConfig, LsmStateVariable, PolynomialBasisSpec,
    PseudoMcConfig, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{RiskRequest, SmileDynamics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let dates = ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
        .into_iter()
        .map(str::parse)
        .collect::<Result<Vec<Date>, _>>()?;
    let curve = |id, rate: f64| -> Result<_, pricing::market::MarketError> {
        Ok(Arc::new(LogLinearDiscountCurve::new(
            CurveId::new(id),
            vec![0., 1.],
            vec![1., (-rate).exp()],
        )?))
    };
    let engine = |seed, units| {
        PseudoMcConfig::new(seed, units, VarianceReduction::new(true, false))
            .map(EngineConfig::PseudoMonteCarlo)
    };
    let request = PricingRequest::new_with_lsm(
        "2026-09-04".parse()?,
        ProductSpec::AmericanVanilla(AmericanVanillaSpec::new(
            underlying,
            currency,
            *dates.last().expect("dates"),
            100.,
            1.,
            OptionSide::Put,
            dates,
        )?),
        MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100., "spot")?,
                curve(1, 0.05)?,
                curve(2, 0.)?,
            ),
        )),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2)?),
        engine(71, 4096)?,
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        Some(LsmConfig::new(
            engine(17, 8192)?,
            vec![LsmStateVariable::Spot],
            PolynomialBasisSpec::new(1, 3, 8, 8)?,
            0.,
            CpqrConfig::new(1e-14, 1e-12)?,
            4_000_000,
        )?),
    )?;
    let plan = AndersenBroadiePlan::compile(
        &request,
        ExecutionPolicy::new(4, Some(64))?,
        AndersenBroadieConfig::new(256, 128, 0x1234)?,
    )?;
    let result = plan.evaluate()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "method": pricing::dual::ANDERSEN_BROADIE_ABI,
            "scope": "declared four-date Bermudan grid; conditional on trained policy",
            "lower_bound": result.lower_bound.value().get(),
            "lower_standard_error": result.lower_bound.standard_error().get(),
            "upper_bound": result.upper_bound.value().get(),
            "upper_standard_error": result.upper_bound.standard_error().get(),
            "duality_gap": result.duality_gap.value().get(),
            "gap_standard_error": result.duality_gap.standard_error().get(),
            "price_confidence_interval_95": result.price_confidence_interval_95,
            "outer_trajectories": result.outer_trajectories,
            "policy_fingerprint": result.policy_fingerprint.to_string(),
            "plan_fingerprint": result.plan_fingerprint.to_string(),
        }))?
    );
    Ok(())
}
