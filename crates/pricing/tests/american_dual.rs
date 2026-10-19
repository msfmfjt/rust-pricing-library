//! Public adapter conformance and independent Bermudan lattice checks.
use std::sync::Arc;

use pricing::core::{
    CurrencyId, CurveId, Date, DayCountConvention, EventId, PositiveF64, UnderlyingId,
};
use pricing::dual::{AndersenBroadieConfig, AndersenBroadieError, AndersenBroadiePlan};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
    MarketContext,
};
use pricing::mc::{
    CpqrConfig, EngineConfig, ExecutionPolicy, LsmConfig, LsmStateVariable, PolynomialBasisSpec,
    PseudoMcConfig, RqmcConfig, VarianceReduction,
};
use pricing::models::{Black76Spec, BlackScholesSpec, ModelSpec};
use pricing::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{RiskRequest, SmileDynamics};
use pricing::{PricingPlan, PricingRequest};

fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn dates() -> Vec<Date> {
    ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
        .map(date)
        .to_vec()
}
fn engine(seed: u64, units: u64, antithetic: bool) -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(seed, units, VarianceReduction::new(antithetic, false)).unwrap(),
    )
}
fn execution(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(64)).unwrap()
}
fn config(paths: u32) -> AndersenBroadieConfig {
    AndersenBroadieConfig::new(paths, paths / 2 + 1, 0x1234).unwrap()
}

struct Case {
    side: OptionSide,
    strike: f64,
    vol: f64,
    dividend: f64,
    dates: Vec<Date>,
    units: u64,
    training_units: u64,
    antithetic: bool,
    seed: u64,
    notional: f64,
    discrete: bool,
    black76: bool,
}
impl Default for Case {
    fn default() -> Self {
        Self {
            side: OptionSide::Put,
            strike: 100.,
            vol: 0.2,
            dividend: 0.,
            dates: dates(),
            units: 256,
            training_units: 2048,
            antithetic: true,
            seed: 91,
            notional: 1.,
            discrete: false,
            black76: false,
        }
    }
}
impl Case {
    fn request(&self) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let curve = |id, rate: f64| {
            Arc::new(
                LogLinearDiscountCurve::new(
                    CurveId::new(id),
                    vec![0., 1.],
                    vec![1., (-rate).exp()],
                )
                .unwrap(),
            )
        };
        let mut forward = EquityForward::new(
            underlying,
            PositiveF64::new(100., "spot").unwrap(),
            curve(1, 0.05),
            curve(2, self.dividend),
        );
        if self.discrete {
            let time = DayCountConvention::Act365F.year_fraction(date("2026-09-04"), self.dates[1]);
            forward = EquityForward::with_discrete_dividends(
                underlying,
                PositiveF64::new(100., "spot").unwrap(),
                curve(1, 0.05),
                curve(2, self.dividend),
                vec![
                    DividendEvent::new(
                        EventId::new(1),
                        time,
                        DividendQuote::FixedCashAndProportional {
                            fixed_cash: 2.,
                            beta: 0.03,
                        },
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
        }
        let model = if self.black76 {
            ModelSpec::Black76(Black76Spec::new(self.vol).unwrap())
        } else {
            ModelSpec::BlackScholes(BlackScholesSpec::new(self.vol).unwrap())
        };
        PricingRequest::new_with_lsm(
            date("2026-09-04"),
            ProductSpec::AmericanVanilla(
                AmericanVanillaSpec::new(
                    underlying,
                    currency,
                    *self.dates.last().unwrap(),
                    self.strike,
                    self.notional,
                    self.side,
                    self.dates.clone(),
                )
                .unwrap(),
            ),
            MarketContext::Equity(EquityMarket::new(currency, forward)),
            model,
            engine(self.seed, self.units, self.antithetic),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            Some(
                LsmConfig::new(
                    engine(17, self.training_units, true),
                    vec![LsmStateVariable::Spot],
                    PolynomialBasisSpec::new(1, 3, 8, 8).unwrap(),
                    0.,
                    CpqrConfig::new(1e-14, 1e-12).unwrap(),
                    4_000_000,
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }
}

#[test]
fn single_exercise_matches_existing_mc_and_has_zero_gap() {
    for antithetic in [false, true] {
        let request = Case {
            dates: vec![date("2027-09-04")],
            antithetic,
            ..Case::default()
        }
        .request();
        let result = AndersenBroadiePlan::compile(&request, execution(2), config(8))
            .unwrap()
            .evaluate()
            .unwrap();
        let ordinary = PricingPlan::compile(&request, execution(2))
            .unwrap()
            .evaluate()
            .unwrap();
        assert!(
            (result.lower_bound.value().get() - ordinary.pricing_result.value.value().get()).abs()
                < 1e-13
        );
        assert_eq!(result.lower_bound, result.upper_bound);
        assert_eq!(result.duality_gap.value().get(), 0.);
        assert_eq!(result.duality_gap.standard_error().get(), 0.);
    }
}

#[test]
fn deterministic_grid_matches_max_payoff_including_time_zero_and_dividends() {
    for side in [OptionSide::Put, OptionSide::Call] {
        for discrete in [false, true] {
            let mut grid = dates();
            grid.insert(0, date("2026-09-04"));
            let case = Case {
                side,
                vol: 0.,
                strike: 105.,
                discrete,
                dates: grid.clone(),
                notional: 2.5,
                units: 2,
                training_units: 2,
                ..Case::default()
            };
            let request = case.request();
            let expected = grid
                .iter()
                .map(|d| {
                    let t = DayCountConvention::Act365F.year_fraction(request.valuation_date(), *d);
                    // Independent deterministic dividend jump and carry.
                    let ex = DayCountConvention::Act365F
                        .year_fraction(request.valuation_date(), grid[1]);
                    let spot = if discrete && t >= ex {
                        0.97 * 100. * (0.05 * t).exp() - 2. * (0.05 * (t - ex)).exp()
                    } else {
                        100. * (0.05 * t).exp()
                    };
                    payoff(side, spot, case.strike) * case.notional * (-0.05 * t).exp()
                })
                .fold(0., f64::max);
            let result = AndersenBroadiePlan::compile(&request, execution(2), config(8))
                .unwrap()
                .evaluate()
                .unwrap();
            assert!(
                (result.upper_bound.value().get() - expected).abs() < 1e-11,
                "{result:?} expected={expected}"
            );
            assert!((result.lower_bound.value().get() - expected).abs() < 1e-11);
            assert_eq!(result.upper_bound.standard_error().get(), 0.);
        }
    }
}

#[test]
fn frozen_policy_lower_bound_and_worker_replay_match_existing_engine() {
    let request = Case {
        discrete: true,
        notional: 2.5,
        ..Case::default()
    }
    .request();
    let one = AndersenBroadiePlan::compile(&request, execution(1), config(12))
        .unwrap()
        .evaluate()
        .unwrap();
    let two = AndersenBroadiePlan::compile(&request, execution(2), config(12))
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(one.lower_bound, two.lower_bound);
    assert_eq!(one.upper_bound, two.upper_bound);
    assert_eq!(one.duality_gap, two.duality_gap);
    assert_eq!(one.policy_fingerprint, two.policy_fingerprint);
    let ordinary = PricingPlan::compile(&request, execution(2))
        .unwrap()
        .evaluate()
        .unwrap();
    assert!(
        (one.lower_bound.value().get() - ordinary.pricing_result.value.value().get()).abs() < 1e-12
    );
    assert_eq!(
        one.policy_fingerprint,
        ordinary
            .early_exercise_diagnostics
            .unwrap()
            .policy_fingerprint
    );
    assert!(one.upper_bound.value().get() >= one.lower_bound.value().get());
    assert!(
        (one.upper_bound.value().get()
            - one.lower_bound.value().get()
            - one.duality_gap.value().get())
        .abs()
            < 1e-12
    );
    assert_eq!(one.outer_trajectories, 512);
    let replay = AndersenBroadiePlan::compile(&request, execution(1), config(12))
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(one, replay);
}

#[test]
fn black76_constant_vol_adapter_matches_black_scholes() {
    let bs = Case::default().request();
    let b76 = Case {
        black76: true,
        ..Case::default()
    }
    .request();
    let a = AndersenBroadiePlan::compile(&bs, execution(1), config(4))
        .unwrap()
        .evaluate()
        .unwrap();
    let b = AndersenBroadiePlan::compile(&b76, execution(1), config(4))
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(a.lower_bound, b.lower_bound);
    assert_eq!(a.upper_bound, b.upper_bound);
}

#[test]
fn explicit_configuration_and_engine_rejections() {
    assert!(matches!(
        AndersenBroadieConfig::new(0, 1, 1),
        Err(AndersenBroadieError::ZeroInnerPaths { .. })
    ));
    assert!(AndersenBroadieConfig::new(1, 0, 1).is_err());
    let request = Case::default().request();
    let a = AndersenBroadiePlan::compile(&request, execution(1), config(8)).unwrap();
    let b = AndersenBroadiePlan::compile(&request, execution(1), config(9)).unwrap();
    assert_ne!(a.fingerprint(), b.fingerprint());
    // Build RQMC through the public request constructor; no silent downgrade.
    let rqmc = EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(16, 4, 29, VarianceReduction::new(false, false)).unwrap(),
    );
    let changed = PricingRequest::new_with_lsm(
        request.valuation_date(),
        request.product().clone(),
        request.market().clone(),
        request.model().clone(),
        rqmc,
        request.risk().clone(),
        request.lsm().cloned(),
    )
    .unwrap();
    assert!(matches!(
        AndersenBroadiePlan::compile(&changed, execution(1), config(8)),
        Err(AndersenBroadieError::Unsupported { .. })
    ));
    let risk = RiskRequest::new(
        true,
        None,
        false,
        None,
        SmileDynamics::StickyLogMoneyness,
        None,
        None,
    )
    .unwrap();
    let risk_request = request.clone().with_risk(risk).unwrap();
    assert!(matches!(
        AndersenBroadiePlan::compile(&risk_request, execution(1), config(8)),
        Err(AndersenBroadieError::Unsupported { .. })
    ));
    let enormous = Case {
        units: u64::MAX,
        vol: 0.,
        ..Case::default()
    }
    .request();
    assert!(matches!(
        AndersenBroadiePlan::compile(&enormous, execution(1), config(8)),
        Err(AndersenBroadieError::RandomCoordinateOverflow)
    ));
}

#[test]
#[ignore = "Release-mode independent lattice and inner-refinement acceptance"]
fn primal_dual_bracket_and_inner_refinement_against_independent_tree() {
    for (side, dividend, strike, discrete) in [
        (OptionSide::Put, 0., 100., false),
        (OptionSide::Call, 0.1, 100., false),
        (OptionSide::Put, 0., 100., true),
    ] {
        let reference = tree(side, strike, dividend, discrete, 4);
        let coarser = tree(side, strike, dividend, discrete, 2);
        assert!((reference - coarser).abs() < 0.03);
        let mut gaps = [0.; 2];
        let mut gap_variances = [0.; 2];
        for seed in [71, 97, 113] {
            let request = Case {
                side,
                strike,
                dividend,
                discrete,
                units: 2048,
                training_units: 8192,
                seed,
                ..Case::default()
            }
            .request();
            for (level, inner) in [16, 256].into_iter().enumerate() {
                let result = AndersenBroadiePlan::compile(&request, execution(4), config(inner))
                    .unwrap()
                    .evaluate()
                    .unwrap();
                let lower = result.lower_bound.value().get();
                let upper = result.upper_bound.value().get();
                println!(
                    "{side:?} q={dividend} discrete={discrete} seed={seed} inner={inner} tree={reference:.8} lower={lower:.8} se_l={:.8} upper={upper:.8} se_u={:.8} gap={:.8}",
                    result.lower_bound.standard_error().get(),
                    result.upper_bound.standard_error().get(),
                    result.duality_gap.value().get()
                );
                assert!(lower - 5. * result.lower_bound.standard_error().get() <= reference + 0.03);
                assert!(upper + 5. * result.upper_bound.standard_error().get() >= reference - 0.03);
                assert!(result.duality_gap.value().get() < 2.0);
                gaps[level] += result.duality_gap.value().get() / 3.;
                gap_variances[level] += result.duality_gap.standard_error().get().powi(2) / 9.;
            }
        }
        assert!(gaps[1] <= gaps[0] + 5. * (gap_variances[0] + gap_variances[1]).sqrt());
    }
}

fn payoff(side: OptionSide, spot: f64, strike: f64) -> f64 {
    match side {
        OptionSide::Call => (spot - strike).max(0.),
        OptionSide::Put => (strike - spot).max(0.),
    }
}

fn tree(side: OptionSide, strike: f64, dividend: f64, discrete: bool, steps_per_day: usize) -> f64 {
    let steps = 365 * steps_per_day;
    let dt = 1. / steps as f64;
    let up = (0.2 * dt.sqrt()).exp();
    let drift = if discrete { 0. } else { 0.05 - dividend };
    let p = ((drift * dt).exp() - 1. / up) / (up - 1. / up);
    let discount = (-0.05 * dt).exp();
    let exercise: Vec<usize> = dates()
        .iter()
        .map(|d| date("2026-09-04").days_until(*d) as usize * steps_per_day)
        .collect();
    // Independent escrowed residual-martingale tree for the dividend case.
    let ex_step = exercise[1];
    let ex = ex_step as f64 * dt;
    let reserve = 2. / (0.97 * ((0.05 - dividend) * ex).exp());
    let node_spot = |i: usize, j: usize| {
        let factor = up.powi(2 * j as i32 - i as i32);
        if !discrete {
            return 100. * factor;
        }
        let t = i as f64 * dt;
        let carry = ((0.05 - dividend) * t).exp();
        if i < ex_step {
            carry * ((100. - reserve) * factor + reserve)
        } else {
            0.97 * carry * (100. - reserve) * factor
        }
    };
    let mut values: Vec<f64> = (0..=steps)
        .map(|j| payoff(side, node_spot(steps, j), strike))
        .collect();
    for i in (0..steps).rev() {
        for j in 0..=i {
            let continuation = discount * ((1. - p) * values[j] + p * values[j + 1]);
            values[j] = if exercise.contains(&i) {
                continuation.max(payoff(side, node_spot(i, j), strike))
            } else {
                continuation
            };
        }
    }
    values[0]
}
