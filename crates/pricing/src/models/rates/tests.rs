use super::*;
use crate::mc::LocalVolTimeGrid;
use crate::mc::hull_white::{HullWhiteEquityPlan, HybridEquityVolatility, HybridState};
use crate::models::{HullWhite1Factor, HybridCorrelation};

fn bits(a: f64, b: f64) {
    assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
}

#[test]
fn extracted_rate_step_replays_legacy_hybrid_states() {
    let times = vec![0.0, 0.13, 0.4, 0.9];
    let grid = LocalVolTimeGrid::compile(times.clone(), 1.0).unwrap();
    let corr = HybridCorrelation::new(-0.3, 0.2, -0.1).unwrap();
    let noise = [
        [0.03, -0.02, 0.001, 0.0004, 0.0],
        [-0.08, 0.04, -0.002, -0.0001, 0.0],
        [0.02, -0.01, 0.0005, 0.0008, 0.0],
    ];
    for a in [0.0, 0.07, 2.0] {
        let rates =
            HullWhite1Factor::new(a, vec![0.0, 0.2, 0.6], vec![0.01, 0.025, 0.015]).unwrap();
        let plan = HullWhiteEquityPlan::new(
            rates.clone(),
            HybridEquityVolatility::BlackScholes(0.25),
            corr,
            &grid,
        )
        .unwrap();
        let actual = plan.evolve_with_innovations(100.0, &noise).unwrap();
        let mut s = HybridState::initial(100.0).unwrap();
        for (i, pair) in times.windows(2).enumerate() {
            let tr = rates.transition(pair[0], pair[1], 0.0, corr).unwrap();
            let shift = rates.integrated_shift(pair[0], pair[1]).unwrap();
            let variance = 0.25 * 0.25 * (0.0 * s.volatility_factor).exp();
            let integral = tr.integral_loading * s.rate_factor + noise[i][3];
            // Frozen pre-extraction recurrence, including its multiplication
            // and addition order, supplies an independent bitwise witness.
            s = HybridState {
                normalized_equity: s.normalized_equity
                    * (integral + shift - 0.5 * variance * (pair[1] - pair[0])
                        + variance.sqrt() * noise[i][0])
                        .exp(),
                volatility_factor: tr.vol_decay * s.volatility_factor + noise[i][1],
                rate_factor: tr.rate_decay * s.rate_factor + noise[i][2],
                integrated_rate_factor: s.integrated_rate_factor + integral,
            };
            bits(actual[i + 1].normalized_equity, s.normalized_equity);
            bits(actual[i + 1].volatility_factor, s.volatility_factor);
            bits(actual[i + 1].rate_factor, s.rate_factor);
            bits(
                actual[i + 1].integrated_rate_factor,
                s.integrated_rate_factor,
            );
        }
    }
}

#[test]
fn rate_step_pullback_matches_state_and_innovation_finite_differences() {
    let step = GaussianRateStep::new(0.94, 0.21, 0.0003);
    let input = [0.02, 0.06, -0.003, 0.001];
    let seed = CenteredRateState {
        factor: 0.7,
        integral: -1.3,
    };
    let integral_seed = 0.4;
    let adj = step.pullback(seed, integral_seed);
    let gradient = [
        adj.state.factor,
        adj.state.integral,
        adj.innovations.factor,
        adj.innovations.integral,
    ];
    let objective = |x: [f64; 4]| {
        let out = step.advance(
            CenteredRateState {
                factor: x[0],
                integral: x[1],
            },
            RateInnovations {
                factor: x[2],
                integral: x[3],
            },
        );
        seed.factor * out.state.factor
            + seed.integral * out.state.integral
            + integral_seed * out.step_integral
    };
    for (i, &g) in gradient.iter().enumerate() {
        let (mut up, mut down) = (input, input);
        up[i] += 1e-6;
        down[i] -= 1e-6;
        let fd = (objective(up) - objective(down)) / 2e-6;
        assert!((g - fd).abs() < 2e-10, "coordinate {i}: {g} vs {fd}");
    }
}

#[test]
fn gaussian_moments_and_discounts_keep_piecewise_hw_conventions() {
    let rates = HullWhite1Factor::new(0.07, vec![0.0, 0.2, 0.6], vec![0.01, 0.025, 0.015]).unwrap();
    let (t, payment, x, integrated) = (0.13, 1.1, -0.007, 0.004);
    let zero = HybridCorrelation::new(0.0, 0.0, 0.0).unwrap();
    let tr = rates.transition(t, payment, 0.0, zero).unwrap();
    let moments = rates.rate_covariance(t, payment).unwrap();
    bits(moments.factor_variance, tr.covariance[2][2]);
    bits(moments.integral_variance, tr.covariance[3][3]);
    bits(moments.factor_integral, tr.covariance[2][3]);
    for k in [0.0, 0.5, 3.0] {
        let old = rates
            .transition(
                t,
                payment,
                k,
                HybridCorrelation::new(0.0, 0.0, 1.0).unwrap(),
            )
            .unwrap();
        let cross = rates.ou_rate_covariance(t, payment, k).unwrap();
        bits(cross.factor, old.covariance[1][2]);
        bits(cross.integral, old.covariance[1][3]);
    }
    let bond = rates.bond_exposure(t, payment).unwrap();
    let legacy_bond = (-tr.integral_loading * x - rates.integrated_shift(t, payment).unwrap()
        + 0.5 * tr.covariance[3][3])
        .exp();
    bits(bond.relative_price(x), legacy_bond);
    bits(rates.relative_bond(t, payment, x).unwrap(), legacy_bond);
    bits(
        rates.relative_discount(t, integrated).unwrap(),
        (-integrated - 0.5 * rates.integrated_variance(t).unwrap()).exp(),
    );
    let discount = rates
        .bond_transition(t, payment)
        .unwrap()
        .conditional_discount(rates.integrated_variance(payment).unwrap());
    let state = CenteredRateState {
        factor: x,
        integral: integrated,
    };
    let old = (-0.5 * rates.integrated_variance(payment).unwrap() + 0.5 * tr.covariance[3][3]
        - integrated
        - tr.integral_loading * x)
        .exp();
    bits(discount.relative_discount(state), old);
    let h = 1e-6;
    let fd = (discount.relative_discount(CenteredRateState {
        factor: x + h,
        ..state
    }) - discount.relative_discount(CenteredRateState {
        factor: x - h,
        ..state
    })) / (2.0 * h);
    let aad = -discount.loading.duration() * old;
    assert!((aad - fd).abs() < 2e-10);
}

#[test]
fn bond_reserve_exposure_derivatives_and_deterministic_adapter_are_consistent() {
    let (amount, x, h) = (4.2, -0.01, 1e-6);
    for duration in [0.0, 0.7, 4.5] {
        let b = BondStateLoading::new(duration);
        let first = amount * (b.multiplier(x + h) - b.multiplier(x - h)) / (2.0 * h);
        let second =
            (b.rate_derivative(amount, x + h) - b.rate_derivative(amount, x - h)) / (2.0 * h);
        assert!((b.rate_derivative(amount, x) - first).abs() < 2e-8);
        assert!((b.rate_second_derivative(amount, x) - second).abs() < 2e-8);
    }
    let rate = DeterministicRates.advance((), ());
    bits(rate.step_integral, 0.0);
    bits(rate.integrated_shift, 0.0);
    bits(DeterministicRates.relative_discount(()), 1.0);
    let hw = HullWhite1Factor::new(0.1, vec![0.0], vec![0.0]).unwrap();
    bits(hw.relative_discount(0.4, 0.0).unwrap(), 1.0);
    bits(hw.relative_bond(0.4, 1.2, 0.0).unwrap(), 1.0);
    // Zero volatility still permits a nonzero conditional HW state; it must
    // not be replaced by the stateless deterministic adapter.
    assert_ne!(hw.relative_bond(0.4, 1.2, 0.02).unwrap(), 1.0);
}
