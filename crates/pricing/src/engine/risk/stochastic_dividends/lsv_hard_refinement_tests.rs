//! Cross-language primal-path checks for the independent hard refinement panel.
//! Leverage knots/values and contractual checkpoint times remain fixed.
use super::lsv_path_refinement_tests::{Contract, compile, request_value};
use super::*;
use crate::mc::lsv::LsvLeverageSurface;
use serde_json::Value;

fn values(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn fixed_surface_fine_paths_match_independent_checkpoints() {
    let source: Value = serde_json::from_str(include_str!(
        "../../../../../../fixtures/stochastic-dividends/rough-survival-barrier-reference.json"
    ))
    .unwrap();
    let reference: Value = serde_json::from_str(include_str!(
        "../../../../../../fixtures/stochastic-dividends/rough-hard-barrier-paths.json"
    ))
    .unwrap();
    for case in source["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["id"] == "h01_n8_four" || c["id"] == "h03_n8_four")
    {
        let h = case["hurst"].as_f64().unwrap();
        let base = compile(&request_value(Contract::Barrier), h, 0.6, 0.7);
        let knots = values(&case["times"]);
        let surface = LsvLeverageSurface::new(
            knots.clone(),
            values(&case["log_nodes"]),
            values(&case["squared_leverage"]),
            base.risky_spot(),
        )
        .unwrap();
        for steps in [16, 32, 64, 128] {
            let ratio = steps / (knots.len() - 1);
            let times = knots
                .windows(2)
                .flat_map(|w| {
                    (1..=ratio).map(move |j| {
                        if j == ratio {
                            w[1]
                        } else {
                            w[0] + (w[1] - w[0]) * (j as f64 / ratio as f64)
                        }
                    })
                })
                .collect();
            let grid = LocalVolTimeGrid::compile(times, 1.0).unwrap();
            let path = StochasticDividendPathPlan::compile(
                &base.market,
                BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
                0.0,
                &grid,
            )
            .unwrap()
            .with_rough_bergomi_lsv(
                RoughBergomi::new(h, 0.6, -0.4).unwrap(),
                0.15,
                surface.clone(),
            )
            .unwrap();
            assert_eq!(path.times().len(), steps + 1);
            assert_eq!(
                path.times()
                    .iter()
                    .step_by(ratio)
                    .copied()
                    .collect::<Vec<_>>(),
                knots
            );
            assert_eq!(
                path.lsv_surface().unwrap().squared_leverage(),
                surface.squared_leverage()
            );
            assert_eq!(path.lsv_surface().unwrap().times(), surface.times());
            for pattern in 0..3 {
                let record = reference["records"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| {
                        r["case"] == case["id"] && r["steps"] == steps && r["pattern"] == pattern
                    })
                    .unwrap();
                let sd: f64 = -0.25;
                let sv: f64 = -0.4;
                let dv: f64 = 0.15;
                let orth = (sv - sd * dv) / (1.0 - dv * dv).sqrt();
                let vol_second = (dv - sd * sv) / (1.0 - sd * sd).sqrt();
                let vol_third = (1.0 - sv * sv - vol_second * vol_second).sqrt();
                let mut normals = Vec::with_capacity(4 * steps);
                for i in 0..steps {
                    let z: [f64; 4] = std::array::from_fn(|factor| {
                        (((37 * (i + 1) * (factor + 3) + 17 * (i + 1) * (i + 1) + 13 * pattern)
                            % 101) as f64
                            - 50.0)
                            / 25.0
                    });
                    // Map dividend-first conditional coordinates into the
                    // production equity-first Cholesky ordering.
                    let equity =
                        sd * z[0] + orth * z[1] + (1.0 - sd * sd - orth * orth).sqrt() * z[3];
                    let dividend = (z[0] - sd * equity) / (1.0 - sd * sd).sqrt();
                    let vol = dv * z[0] + (1.0 - dv * dv).sqrt() * z[1];
                    normals.extend([
                        equity,
                        dividend,
                        (vol - sv * equity - vol_second * dividend) / vol_third,
                        z[2],
                    ]);
                }
                let states = path.evolve_path(&normals).unwrap();
                for checkpoint in record["checkpoints"].as_array().unwrap() {
                    let expected = values(checkpoint);
                    let index = path
                        .times()
                        .binary_search_by(|t| t.total_cmp(&expected[0]))
                        .unwrap();
                    for (actual, expected) in [states[index].equity(), states[index].dividend()]
                        .into_iter()
                        .zip(&expected[1..])
                    {
                        assert!(
                            (actual - expected).abs() < 2e-11,
                            "H={h}, steps={steps}, pattern={pattern}, node={index}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}
