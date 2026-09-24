use super::*;
use crate::models::Bergomi2Factor;

fn plan<F: BergomiDynamics>(factor: F) -> BergomiLsvPlan<F> {
    let times = vec![0.0, 0.08, 0.3, 0.7];
    let surface = LsvLeverageSurface::new(
        times.clone(),
        vec![-0.8, 0.0, 0.8],
        (0..12).map(|i| 0.04 + 0.002 * (i % 3) as f64).collect(),
        100.0,
    )
    .unwrap();
    let grid = LocalVolTimeGrid::compile(times, 1.0).unwrap();
    BergomiLsvPlan::new(factor, surface, &grid).unwrap()
}

fn legacy_path<F: BergomiDynamics>(p: &BergomiLsvPlan<F>) {
    let n = p.transitions.len();
    let z = p
        .pseudo_shocks(812, 17, RandomDomain::LsvCalibration)
        .unwrap();
    let mut f = 103.0;
    let mut x = F::State::default();
    let mut states = vec![f];
    let mut factors = vec![x];
    for i in 0..n {
        let variance =
            p.surface.lookup(p.times[i], f).unwrap().value * p.factor.multiplier_squared(x);
        f = advance(f, variance, p.times[i + 1] - p.times[i], z[i], i, 0).unwrap();
        // Original public array adapter, assembled independently of the new
        // strided view. NaN padding must remain unused for the one-factor model.
        x = p.factor.evolve(
            p.transitions[i],
            x,
            z[i],
            [
                z[n + i],
                if F::FACTOR_COUNT == 2 {
                    z[2 * n + i]
                } else {
                    f64::NAN
                },
            ],
        );
        states.push(f);
        factors.push(x);
    }
    let path = p.evolve_path(103.0, &z).unwrap();
    assert_eq!(
        path.states()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        states.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(path.factors(), factors);
    let mut price_only = Vec::with_capacity(states.len());
    let capacity = price_only.capacity();
    p.evolve_states(103.0, &z, &mut price_only).unwrap();
    assert_eq!(price_only, states);
    assert_eq!(price_only.capacity(), capacity);
}

#[test]
fn typed_normals_preserve_legacy_paths_and_unused_one_factor_padding() {
    legacy_path(&plan(Bergomi1Factor::new(1.2, 0.6, -0.5).unwrap()));
    legacy_path(&plan(
        Bergomi2Factor::new([3.0, 0.2], 0.6, 0.4, [-0.6, -0.2], 0.3).unwrap(),
    ));
}

fn external_pullback<F: BergomiDynamics>(p: &BergomiLsvPlan<F>) {
    let n = p.transitions.len();
    let spot: Vec<_> = (0..n).map(|i| 0.2 + 0.07 * i as f64).collect();
    let increments: Vec<_> = (0..F::FACTOR_COUNT * n)
        .map(|i| -0.08 + 0.03 * i as f64)
        .collect();
    let seeds: Vec<_> = (0..=n).map(|i| 0.3 + 0.1 * i as f64).collect();
    let path = p
        .evolve_with_ou_innovations(103.0, &spot, &increments)
        .unwrap();
    let adj = path.reverse(&seeds).unwrap();
    assert_eq!(adj.orthogonal_shocks.len(), increments.len());
    let mut x = F::State::default();
    for i in 0..n {
        x = p.factor.evolve_ou(
            p.transitions[i],
            x,
            [
                increments[i],
                if F::FACTOR_COUNT == 2 {
                    increments[n + i]
                } else {
                    f64::NAN
                },
            ],
        );
        assert_eq!(x, path.factors()[i + 1]);
    }
    let value = |z: &[f64], v: &[f64]| {
        p.evolve_with_ou_innovations(103.0, z, v)
            .unwrap()
            .states()
            .iter()
            .zip(&seeds)
            .map(|(s, a)| s * a)
            .sum::<f64>()
    };
    for (input, gradient, is_spot) in [
        (&spot, &adj.spot_shocks, true),
        (&increments, &adj.orthogonal_shocks, false),
    ] {
        for i in 0..input.len() {
            let mut plus = input.to_vec();
            let mut minus = input.to_vec();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let fd = if is_spot {
                (value(&plus, &increments) - value(&minus, &increments)) / 2e-6
            } else {
                (value(&spot, &plus) - value(&spot, &minus)) / 2e-6
            };
            assert!(
                (gradient[i] - fd).abs() < 2e-7 * (1.0 + fd.abs()),
                "coordinate {i}, spot={is_spot}: {} vs {fd}",
                gradient[i]
            );
        }
    }
}

#[test]
fn joint_ou_increment_pullback_covers_each_spot_and_volatility_coordinate() {
    external_pullback(&plan(Bergomi1Factor::new(1.2, 0.6, -0.5).unwrap()));
    external_pullback(&plan(
        Bergomi2Factor::new([3.0, 0.2], 0.6, 0.4, [-0.6, -0.2], 0.3).unwrap(),
    ));
}
