use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, Date, DayCountConvention, PositiveF64, UnderlyingId};
use pricing::market::{
    DiscountCurve, EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext,
};
use pricing::mc::{
    CpqrConfig, EngineConfig, ExecutionPolicy, LsmConfig, LsmStateVariable, PolynomialBasisSpec,
    PseudoMcConfig, RqmcConfig, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{RiskRequest, SmileDynamics};
use pricing::{Estimate, PricingRequest, price_monte_carlo};

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const DIVIDEND_YIELD: f64 = 0.0;
const VOLATILITY: f64 = 0.20;
const REFERENCE_STEPS_PER_DAY: usize = 10;
const REFERENCE_DISCRETIZATION_TOLERANCE: f64 = 0.03;
// Out-of-sample LSM is a lower-bound estimator. Keep its finite cubic-basis
// shortfall separate from the stochastic confidence bound.
const POLICY_SHORTFALL_TOLERANCE: f64 = 0.005 * SPOT;

fn date(value: &str) -> Date {
    value.parse().expect("valid date")
}

fn exercise_dates() -> Vec<Date> {
    ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
        .map(date)
        .to_vec()
}

#[test]
fn zero_variance_american_call_put_grid_uses_optimal_declared_date() {
    for side in [OptionSide::Call, OptionSide::Put] {
        for strike in [80.0, 100.0, 120.0] {
            let request = american_request(
                side,
                strike,
                0.0,
                pseudo_engine(0x1111, 2),
                pseudo_engine(0x2222, 2),
            );
            let result = price_monte_carlo(&request, execution_policy()).expect("American price");
            let expected = deterministic_american_value(&request, side, strike);
            assert_eq!(
                result.pricing_result.value.value().get().to_bits(),
                expected.to_bits(),
                "side={side:?} strike={strike}"
            );
            assert_eq!(result.pricing_result.value.standard_error().get(), 0.0);
            let diagnostics = result
                .early_exercise_diagnostics
                .expect("early-exercise diagnostics");
            assert_eq!(diagnostics.exercise_counts.iter().sum::<usize>(), 4);
            assert_eq!(diagnostics.stopping_indices.len(), 4);
        }
    }
}

#[test]
#[ignore = "American statistical acceptance is run explicitly by CI"]
fn pseudo_and_rqmc_call_put_grids_match_independent_binomial_reference() {
    for side in [OptionSide::Call, OptionSide::Put] {
        let strikes = match side {
            OptionSide::Call => [90.0, 100.0, 110.0],
            OptionSide::Put => [100.0, 110.0, 120.0],
        };
        for strike in strikes {
            let pseudo = evaluate(american_request(
                side,
                strike,
                VOLATILITY,
                pseudo_engine(0x1000 + strike as u64, 8_192),
                pseudo_engine(0x2000 + strike as u64, 16_384),
            ));
            let rqmc = evaluate(american_request(
                side,
                strike,
                VOLATILITY,
                rqmc_engine(0x3000 + strike as u64, 8_192, 16),
                rqmc_engine(0x4000 + strike as u64, 4_096, 16),
            ));
            let reference = bermudan_binomial_reference(side, strike, VOLATILITY);

            assert_matches_reference("Pseudo-MC", side, strike, pseudo, reference);
            assert_matches_reference("RQMC", side, strike, rqmc, reference);
            assert_estimates_agree(side, strike, pseudo, rqmc);
        }
    }
}

#[test]
#[ignore = "American in/out-of-sample bias evidence is run explicitly by CI"]
fn independent_training_replicates_bound_in_sample_optimism() {
    const REPLICATES: usize = 8;
    let reference = bermudan_binomial_reference(OptionSide::Put, 100.0, VOLATILITY);
    let mut out_of_sample = Vec::with_capacity(REPLICATES);
    let mut paired_gaps = Vec::with_capacity(REPLICATES);
    let mut conditional_variances = Vec::with_capacity(REPLICATES);

    for replicate in 0..REPLICATES {
        let result = price_monte_carlo(
            &american_request(
                OptionSide::Put,
                100.0,
                VOLATILITY,
                pseudo_engine(0x5100 + replicate as u64, 4_096),
                pseudo_engine(0x6100 + replicate as u64, 8_192),
            ),
            execution_policy(),
        )
        .expect("American replicate");
        let out_value = result.pricing_result.value.value().get();
        let diagnostics = result
            .early_exercise_diagnostics
            .expect("early-exercise diagnostics");
        out_of_sample.push(out_value);
        paired_gaps.push(diagnostics.in_sample_value - out_value);
        conditional_variances.push(result.estimator_variance);
    }

    let out_mean = mean(&out_of_sample);
    let gap_mean = mean(&paired_gaps);
    let out_mean_se = sample_standard_error(&out_of_sample);
    let gap_mean_se = sample_standard_error(&paired_gaps);
    let conditional_mean_se = (mean(&conditional_variances) / REPLICATES as f64).sqrt();
    let reference_bound = 6.0 * out_mean_se.hypot(conditional_mean_se) + 0.03;
    let optimism_lower_bound = gap_mean - 3.0 * gap_mean_se;

    println!(
        "American Put K=100: reference={reference:.12} out_mean={out_mean:.12} \
         reference_bound={reference_bound:.6e} in_minus_out={gap_mean:.6e} \
         gap_se={gap_mean_se:.6e}"
    );
    assert!(
        (out_mean - reference).abs() <= reference_bound,
        "out-of-sample mean={out_mean}, reference={reference}, bound={reference_bound}"
    );
    assert!(
        optimism_lower_bound >= -0.03,
        "in-sample optimism is materially inverted: mean={gap_mean}, se={gap_mean_se}"
    );
}

fn evaluate(request: PricingRequest) -> Estimate {
    price_monte_carlo(&request, execution_policy())
        .expect("American price")
        .pricing_result
        .value
}

fn execution_policy() -> ExecutionPolicy {
    ExecutionPolicy::new(2, Some(256)).expect("execution policy")
}

fn american_request(
    side: OptionSide,
    strike: f64,
    volatility: f64,
    training_engine: EngineConfig,
    valuation_engine: EngineConfig,
) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let dates = exercise_dates();
    let product = ProductSpec::AmericanVanilla(
        AmericanVanillaSpec::new(
            underlying,
            currency,
            *dates.last().expect("exercise dates"),
            strike,
            1.0,
            side,
            dates,
        )
        .expect("American product"),
    );
    let market = MarketContext::Equity(EquityMarket::new(
        currency,
        EquityForward::new(
            underlying,
            PositiveF64::new(SPOT, "spot").expect("spot"),
            curve(1, RATE),
            curve(2, DIVIDEND_YIELD),
        ),
    ));
    let lsm = LsmConfig::new(
        training_engine,
        vec![LsmStateVariable::Spot],
        PolynomialBasisSpec::new(1, 3, 8, 8).expect("basis"),
        0.0,
        CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
        4_000_000,
    )
    .expect("LSM config");
    PricingRequest::new_with_lsm(
        date("2026-09-04"),
        product,
        market,
        ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
        valuation_engine,
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        Some(lsm),
    )
    .expect("American request")
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
    )
}

fn pseudo_engine(seed: u64, units: u64) -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(seed, units, VarianceReduction::new(true, false))
            .expect("Pseudo-MC engine"),
    )
}

fn rqmc_engine(seed: u64, points: u64, scrambles: u32) -> EngineConfig {
    EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(points, scrambles, seed, VarianceReduction::new(true, true))
            .expect("RQMC engine"),
    )
}

fn deterministic_american_value(request: &PricingRequest, side: OptionSide, strike: f64) -> f64 {
    let forward = request.market().equity().forward();
    exercise_dates()
        .into_iter()
        .map(|exercise_date| {
            let time =
                DayCountConvention::Act365F.year_fraction(request.valuation_date(), exercise_date);
            let evaluated = forward.evaluate(time).expect("forward evaluation");
            let discount = forward
                .discount_curve()
                .evaluate(time)
                .expect("discount evaluation")
                .discount;
            discount * payoff(side, evaluated.forward, strike)
        })
        .fold(0.0, f64::max)
}

fn bermudan_binomial_reference(side: OptionSide, strike: f64, volatility: f64) -> f64 {
    let valuation_date = date("2026-09-04");
    let expiry = *exercise_dates().last().expect("exercise dates");
    let day_count = usize::try_from(valuation_date.days_until(expiry)).expect("positive term");
    let steps = day_count * REFERENCE_STEPS_PER_DAY;
    let dt = 1.0 / (365.0 * REFERENCE_STEPS_PER_DAY as f64);
    let up = (volatility * dt.sqrt()).exp();
    let down = up.recip();
    let probability = (((RATE - DIVIDEND_YIELD) * dt).exp() - down) / (up - down);
    let discount = (-RATE * dt).exp();
    let exercise_steps: Vec<_> = exercise_dates()
        .into_iter()
        .map(|exercise_date| {
            usize::try_from(valuation_date.days_until(exercise_date)).expect("exercise term")
                * REFERENCE_STEPS_PER_DAY
        })
        .collect();
    let mut values = (0..=steps)
        .map(|up_count| {
            let spot = SPOT * up.powi(up_count as i32) * down.powi((steps - up_count) as i32);
            payoff(side, spot, strike)
        })
        .collect::<Vec<_>>();

    for step in (0..steps).rev() {
        let may_exercise = exercise_steps.binary_search(&step).is_ok();
        for up_count in 0..=step {
            let continuation = discount
                * (probability * values[up_count + 1] + (1.0 - probability) * values[up_count]);
            values[up_count] = if may_exercise {
                let spot = SPOT * up.powi(up_count as i32) * down.powi((step - up_count) as i32);
                continuation.max(payoff(side, spot, strike))
            } else {
                continuation
            };
        }
    }
    values[0]
}

fn payoff(side: OptionSide, spot: f64, strike: f64) -> f64 {
    match side {
        OptionSide::Call => (spot - strike).max(0.0),
        OptionSide::Put => (strike - spot).max(0.0),
    }
}

fn assert_matches_reference(
    engine: &str,
    side: OptionSide,
    strike: f64,
    estimate: Estimate,
    reference: f64,
) {
    let signed_difference = estimate.value().get() - reference;
    let difference = signed_difference.abs();
    let statistical_bound = 6.0 * estimate.standard_error().get();
    let upper_bound = statistical_bound + REFERENCE_DISCRETIZATION_TOLERANCE;
    let lower_bound = statistical_bound + POLICY_SHORTFALL_TOLERANCE;
    println!(
        "{engine} {side:?} K={strike}: value={:.12} reference={reference:.12} \
         difference={difference:.6e} lower_bound={lower_bound:.6e} upper_bound={upper_bound:.6e}",
        estimate.value().get()
    );
    assert!(
        signed_difference <= upper_bound,
        "estimate exceeds reference: difference={signed_difference}, bound={upper_bound}"
    );
    assert!(
        -signed_difference <= lower_bound,
        "policy shortfall={}; bound={lower_bound}",
        -signed_difference
    );
}

fn assert_estimates_agree(side: OptionSide, strike: f64, pseudo: Estimate, rqmc: Estimate) {
    let difference = (pseudo.value().get() - rqmc.value().get()).abs();
    let combined_error = pseudo
        .standard_error()
        .get()
        .hypot(rqmc.standard_error().get());
    let bound = 6.0 * combined_error + POLICY_SHORTFALL_TOLERANCE;
    assert!(
        difference <= bound,
        "{side:?} K={strike}: difference={difference}, bound={bound}"
    );
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_standard_error(values: &[f64]) -> f64 {
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (variance / values.len() as f64).sqrt()
}
