use pricing::core::{Date, UnderlyingId};
use pricing::product::{CompiledPayoff, GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op};
use pricing::risk::{
    ReportingIvBasis, VegaKtGammaAtoms, gaussian_log_hat_probabilities,
    terminal_payoff_gamma_measure,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a:.16e} != {b:.16e}, tolerance {tol}"
    );
}
fn date() -> Date {
    "2027-09-04".parse().unwrap()
}
fn graph(kind: usize) -> CompiledPayoff {
    let mut b = SourceGraphBuilder::new();
    let s = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: date(),
        })
        .unwrap();
    let k = b.literal(100.).unwrap();
    let z = b.literal(0.).unwrap();
    let x = b.push(Op::Subtract { left: s, right: k }).unwrap();
    let neg = b.push(Op::Negate { input: x }).unwrap();
    let call = b.push(Op::Maximum { left: x, right: z }).unwrap();
    let put = b
        .push(Op::Maximum {
            left: neg,
            right: z,
        })
        .unwrap();
    let out = match kind {
        0 => call,
        1 => put,
        2 => {
            let cap = b.literal(5.).unwrap();
            b.push(Op::Minimum {
                left: call,
                right: cap,
            })
            .unwrap()
        }
        3 => b
            .push(Op::Add {
                left: call,
                right: put,
            })
            .unwrap(),
        4 => {
            let c = b.literal(-2.).unwrap();
            b.push(Op::Multiply {
                left: call,
                right: c,
            })
            .unwrap()
        }
        5 => b
            .push(Op::Subtract {
                left: call,
                right: put,
            })
            .unwrap(),
        6 => b.push(Op::Multiply { left: s, right: s }).unwrap(),
        7 => b
            .push(Op::Divide {
                numerator: call,
                denominator: s,
            })
            .unwrap(),
        8 => b.push(Op::Indicator { input: x }).unwrap(),
        9 => {
            let c = b.literal(2.).unwrap();
            b.push(Op::Divide {
                numerator: call,
                denominator: c,
            })
            .unwrap()
        }
        10 => {
            // The crossing lies INSIDE existing pieces, not at an input knot.
            let c = b.literal(3.).unwrap();
            let shifted = b
                .push(Op::Add {
                    left: put,
                    right: c,
                })
                .unwrap();
            b.push(Op::Maximum {
                left: call,
                right: shifted,
            })
            .unwrap()
        }
        11 => b
            .push(Op::Maximum {
                left: call,
                right: call,
            })
            .unwrap(),
        _ => panic!(),
    };
    b.finish(vec![out])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap()
}

#[test]
fn compiled_payoffs_preserve_signed_kinks_affine_terms_and_values() {
    let expected = [
        (0, vec![(100., 1.)], 0., 0.),
        (1, vec![(100., 1.)], 100., -1.),
        (2, vec![(100., 1.), (105., -1.)], 0., 0.),
        (3, vec![(100., 2.)], 100., -1.),
        (4, vec![(100., -2.)], 0., 0.),
        (5, vec![], -100., 1.),
        (9, vec![(100., 0.5)], 0., 0.),
        (10, vec![(100., 1.), (103., 1.)], 103., -1.),
        (11, vec![(100., 1.)], 0., 0.),
    ];
    for (kind, atoms, cash, slope) in expected {
        let compiled = graph(kind);
        let p = terminal_payoff_gamma_measure(&compiled, UnderlyingId::new(1), date()).unwrap();
        assert_eq!(
            p.gamma()
                .atoms()
                .iter()
                .map(|a| (a.strike(), a.mass()))
                .collect::<Vec<_>>(),
            atoms
        );
        assert_eq!((p.cash(), p.slope()), (cash, slope));
        for spot in [
            0.001, 55., 99.999999, 100., 100.000001, 102.9, 103., 103.1, 104.999999, 105.,
            105.000001, 250., 10000.,
        ] {
            let actual = compiled.evaluate(|_, _| Some(spot)).unwrap()[0];
            close(p.value(spot).unwrap(), actual, 1e-11);
        }
    }
}

#[test]
fn unsupported_expressions_and_observations_fail_explicitly() {
    for kind in [6, 7, 8] {
        assert!(terminal_payoff_gamma_measure(&graph(kind), UnderlyingId::new(1), date()).is_err());
    }
    assert!(terminal_payoff_gamma_measure(&graph(0), UnderlyingId::new(2), date()).is_err());
    assert!(
        terminal_payoff_gamma_measure(
            &graph(0),
            UnderlyingId::new(1),
            "2028-09-04".parse().unwrap()
        )
        .is_err()
    );
}

#[test]
fn canonical_measure_accepts_signed_cancellation_and_rejects_invalid_atoms() {
    let a = VegaKtGammaAtoms::new(vec![
        (105., -1.),
        (100., 1e16),
        (100., 1.),
        (100., -1e16),
        (90., 1.),
        (90., -1.),
    ])
    .unwrap();
    assert_eq!(
        a.atoms()
            .iter()
            .map(|a| (a.strike(), a.mass()))
            .collect::<Vec<_>>(),
        vec![(100., 1.), (105., -1.)]
    );
    for atoms in [
        vec![(0., 1.)],
        vec![(-1., 1.)],
        vec![(100., f64::NAN)],
        vec![(f64::INFINITY, 1.)],
        vec![(100., f64::MAX); 2],
        vec![(100., 1.); 4097],
    ] {
        assert!(VegaKtGammaAtoms::new(atoms).is_err());
    }
    assert!(a.gamma_density(100., 0.).is_err());
    assert!(
        a.expected_source_hat_vegas(100., -1., 0.1, 0.02, &[-1., 0., 1.])
            .is_err()
    );
    let empty = VegaKtGammaAtoms::new(vec![]).unwrap();
    assert_eq!(empty.gamma_density(100., 0.04).unwrap(), 0.);
    assert_eq!(
        empty
            .expected_source_hat_vegas(100., 0., 0.04, 0.2, &[-1., 0., 1.])
            .unwrap(),
        vec![0.; 3]
    );
}

fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut total = f(a) + f(b);
    for i in 1..n {
        total += (if i % 2 == 0 { 2. } else { 4. }) * f(a + i as f64 * h);
    }
    total * h / 3.
}
fn pdf(x: f64) -> f64 {
    (-x * x / 2.).exp() / (2. * std::f64::consts::PI).sqrt()
}

#[test]
fn signed_gamma_propagation_obeys_semigroup_and_price_curvature() {
    let atoms = VegaKtGammaAtoms::new(vec![(95., 1.7), (108., -0.8)]).unwrap();
    for spot in [65_f64, 100., 145.] {
        let v1: f64 = 0.013;
        let v2: f64 = 0.027;
        let composed = simpson(
            |y| {
                let k = y.exp();
                let d1 = (spot.ln() - y) / v1.sqrt() + 0.5 * v1.sqrt();
                pdf(d1) / (spot * v1.sqrt()) * atoms.gamma_density(k, v2).unwrap() * k
            },
            spot.ln() - 3.,
            spot.ln() + 3.,
            12000,
        );
        close(composed, atoms.gamma_density(spot, v1 + v2).unwrap(), 2e-13);
    }
    // Independent expectation of compiled call payoff, split at its kink, then
    // spot finite differences. This checks density units against native payoff.
    let payoff = graph(0);
    let v: f64 = 0.04;
    let price = |s: f64| {
        let kink = ((100_f64 / s).ln() + 0.5 * v) / v.sqrt();
        simpson(
            |z| {
                payoff
                    .evaluate(|_, _| Some(s * (-0.5 * v + v.sqrt() * z).exp()))
                    .unwrap()[0]
                    * pdf(z)
            },
            kink,
            10.,
            4000,
        )
    };
    let h = 0.05;
    let fd = (price(100. + h) - 2. * price(100.) + price(100. - h)) / (h * h);
    close(
        fd,
        VegaKtGammaAtoms::new(vec![(100., 1.)])
            .unwrap()
            .gamma_density(100., v)
            .unwrap(),
        2e-8,
    );
}

#[test]
fn forward_hats_match_direct_source_integral_including_time_zero() {
    let atoms = VegaKtGammaAtoms::new(vec![(95., 1.7), (108., -0.8)]).unwrap();
    let nodes = [-0.4, -0.1, 0., 0.17, 0.5];
    let u: f64 = 0.03;
    let w: f64 = 0.01;
    let sdt = 0.05;
    let exact = atoms
        .expected_source_hat_vegas(100., u, w, sdt, &nodes)
        .unwrap();
    for (j, &expected) in exact.iter().enumerate() {
        // Unconditional source distribution times propagated terminal Gamma;
        // independent of the bridge conditional-hat expression under test.
        let mut bounds = vec![-2.];
        bounds.extend(nodes);
        bounds.push(2.);
        let numeric: f64 = bounds
            .windows(2)
            .map(|p| {
                simpson(
                    |x| {
                        let spot = 100. * x.exp();
                        let z = (x + 0.5 * u) / u.sqrt();
                        let h = if x <= nodes[0] {
                            if j == 0 { 1. } else { 0. }
                        } else if x >= nodes[4] {
                            if j == 4 { 1. } else { 0. }
                        } else {
                            let i = nodes.partition_point(|&n| n <= x) - 1;
                            let r = (x - nodes[i]) / (nodes[i + 1] - nodes[i]);
                            if j == i {
                                1. - r
                            } else if j == i + 1 {
                                r
                            } else {
                                0.
                            }
                        };
                        h * sdt * spot * spot * atoms.gamma_density(spot, w).unwrap() * pdf(z)
                            / u.sqrt()
                    },
                    p[0],
                    p[1],
                    1000,
                )
            })
            .sum();
        close(expected, numeric, 3e-10);
    }
    let initial = atoms
        .expected_source_hat_vegas(100., 0., u + w, sdt, &nodes)
        .unwrap();
    let total = 10000. * sdt * atoms.gamma_density(100., u + w).unwrap();
    for (j, &v) in initial.iter().enumerate() {
        close(v, if j == 2 { total } else { 0. }, 1e-12);
    }
    for (mean, expected) in [
        (-1., vec![1., 0., 0.]),
        (0., vec![0., 1., 0.]),
        (0.5, vec![0., 0.5, 0.5]),
        (2., vec![0., 0., 1.]),
    ] {
        assert_eq!(
            gaussian_log_hat_probabilities(mean, 0., &[-1., 0., 1.]).unwrap(),
            expected
        );
    }
}

#[test]
fn terminal_atoms_stay_on_strike_and_offgrid_projection_multiplies_vega_first() {
    let nodes = vec![(90_f64 / 100.).ln(), 0., (120_f64 / 100.).ln()];
    let basis = ReportingIvBasis::new(vec![0.5, 1.], nodes.clone(), vec![0.2; 6]).unwrap();
    let vega = |k: f64| 100. * pdf((100_f64 / k).ln() / 0.2 + 0.1);
    let a = VegaKtGammaAtoms::new(vec![(100., 1.)])
        .unwrap()
        .project_terminal(&basis, 1., 100.)
        .unwrap();
    assert_eq!(a.hedge_quantities(), &[0., 0., 0., 0., -1., 0.]);
    close(a.position_risk().raw_buckets()[4], vega(100.), 1e-12);
    let b = VegaKtGammaAtoms::new(vec![(105., -2.)])
        .unwrap()
        .project_terminal(&basis, 1., 100.)
        .unwrap();
    let r = (105_f64 / 100.).ln() / nodes[2];
    close(
        b.position_risk().raw_buckets()[4],
        -2. * vega(105.) * (1. - r),
        1e-12,
    );
    close(
        b.position_risk().raw_buckets()[5],
        -2. * vega(105.) * r,
        1e-12,
    );
    close(
        b.hedge_quantities()[5],
        2. * vega(105.) * r / vega(120.),
        1e-12,
    );
    assert!(b.position_risk().reconciles());
}
