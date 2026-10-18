use pricing::risk::{ReportingIvBasis, VegaKtGammaAtoms};
use pricing_numerics::standard_normal_pdf as pdf;

fn hat(x: f64, nodes: &[f64], i: usize) -> f64 {
    if x <= nodes[0] {
        return f64::from(i == 0);
    }
    if x >= nodes[nodes.len() - 1] {
        return f64::from(i == nodes.len() - 1);
    }
    let j = nodes.partition_point(|&v| v <= x) - 1;
    if i == j {
        (nodes[j + 1] - x) / (nodes[j + 1] - nodes[j])
    } else if i == j + 1 {
        (x - nodes[j]) / (nodes[j + 1] - nodes[j])
    } else {
        0.
    }
}
fn quadrature(f: impl Fn(f64) -> f64, nodes: &[f64]) -> f64 {
    let mut boundaries = vec![-6.];
    boundaries.extend(nodes);
    boundaries.push(6.);
    boundaries
        .windows(2)
        .map(|p| {
            let n = 16384;
            let h = (p[1] - p[0]) / n as f64;
            (0..=n)
                .map(|j| {
                    let w = if j == 0 || j == n {
                        1.
                    } else if j % 2 == 0 {
                        2.
                    } else {
                        4.
                    };
                    w * f(p[0] + h * j as f64) * h / 3.
                })
                .sum::<f64>()
        })
        .sum()
}
#[test]
fn propagated_signed_atoms_match_direct_gamma_times_continuum_vega_integral() {
    let spot = 103.;
    let t: f64 = 0.7;
    let v: f64 = 0.035;
    let next: f64 = 0.081;
    let nodes = vec![-0.09, -0.02, 0.015, 0.12];
    let basis = ReportingIvBasis::new(
        vec![t, 1.5],
        nodes.clone(),
        [vec![(v / t).sqrt(); 4], vec![0.3; 4]].concat(),
    )
    .unwrap();
    let atoms = VegaKtGammaAtoms::new(vec![(71., 0.7), (108., -1.3), (150., 2.1)]).unwrap();
    let result = atoms.project_propagated(&basis, t, spot, v, next).unwrap();
    for (i, &actual) in result[..4].iter().enumerate() {
        let expected = quadrature(
            |x| {
                let s = spot * x.exp();
                let gamma = atoms
                    .atoms()
                    .iter()
                    .map(|a| {
                        a.mass() * pdf(((s / a.strike()).ln() + 0.5 * next) / next.sqrt())
                            / (s * next.sqrt())
                    })
                    .sum::<f64>();
                let iv_vega = spot * pdf(-x / v.sqrt() + 0.5 * v.sqrt()) * t.sqrt();
                gamma * iv_vega * s * hat(x, &nodes, i)
            },
            &nodes,
        );
        assert!(
            (actual - expected).abs() < 2e-10,
            "{i}: {actual} vs {expected}"
        );
    }
    assert_eq!(&result[4..], &[0.; 4]);
    // Summing hats covers both tails, not just the visible strike interval.
    let exact = atoms
        .atoms()
        .iter()
        .map(|a| {
            a.mass()
                * spot
                * pdf(((spot / a.strike()).ln() + 0.5 * (v + next)) / (v + next).sqrt())
                * (t * v / (v + next)).sqrt()
        })
        .sum::<f64>();
    assert!((result.iter().sum::<f64>() - exact).abs() < 2e-12);
}
#[test]
fn propagation_is_linear_keeps_tails_and_validates_matching_flat_iv() {
    let basis = ReportingIvBasis::new(vec![0.5, 1.], vec![-0.01, 0.01], vec![0.2; 4]).unwrap();
    let a = VegaKtGammaAtoms::new(vec![(100., 1.)]).unwrap();
    let b = VegaKtGammaAtoms::new(vec![(100., -3.)]).unwrap();
    let va = a.project_propagated(&basis, 0.5, 100., 0.02, 0.03).unwrap();
    let vb = b.project_propagated(&basis, 0.5, 100., 0.02, 0.03).unwrap();
    for (a, b) in va.iter().zip(vb) {
        assert!((b + 3. * a).abs() < 1e-12);
    }
    assert!(va[..2].iter().all(|&v| v > 5.));
    assert!(a.project_propagated(&basis, 0.5, 100., 0., 0.03).is_err());
    assert!(a.project_propagated(&basis, 0.5, 100., 0.03, 0.03).is_err());
    assert!(a.project_propagated(&basis, 0.4, 100., 0.02, 0.03).is_err());
    let smile = ReportingIvBasis::new(vec![0.5, 1.], vec![-0.01, 0.01], vec![0.19, 0.21, 0.2, 0.2])
        .unwrap();
    assert!(a.project_propagated(&smile, 0.5, 100., 0.02, 0.03).is_err());
    let empty = VegaKtGammaAtoms::new(vec![]).unwrap();
    assert_eq!(
        empty
            .project_propagated(&basis, 0.5, 100., 0.02, 0.03)
            .unwrap(),
        vec![0.; 4]
    );
}
