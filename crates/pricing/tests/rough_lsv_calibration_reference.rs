//! Independent NumPy finite-particle calibration with shared Gaussian inputs.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::{LsvError, LsvParticleConfig, calibrate_rough_bergomi_lsv};
use pricing::mc::{Philox4x32, RandomCoordinate, RandomDomain};
use pricing::models::RoughBergomi;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/stochastic-dividends/rough-lsv-calibration-reference.json"
        ))
        .unwrap(),
    )
    .unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-12 * (1.0 + expected.abs()),
        "{actual:.16e} != {expected:.16e}"
    );
}

#[test]
fn calibration_random_coordinates_match_independent_reference_inputs() {
    let f = fixture();
    let rng = Philox4x32::from_seed(f["seed"].as_u64().unwrap());
    for (p, row) in f["normals"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            row.as_array().unwrap().len(),
            3 * f["steps"].as_u64().unwrap() as usize
        );
        for (d, expected) in vector(row).into_iter().enumerate() {
            let actual = rng.standard_normal(RandomCoordinate::new(
                p as u64,
                d as u32,
                RandomDomain::LsvCalibration,
            ));
            assert!((actual - expected).abs() < 2e-14);
        }
    }
}

#[test]
fn rough_particle_calibration_matches_independent_numpy_moments_and_fallbacks() {
    let f = fixture();
    for case in f["cases"].as_array().unwrap() {
        let number = |name: &str| case[name].as_f64().unwrap();
        let target = LocalVarianceGrid::new(
            vector(&case["times"]),
            vector(&case["log_nodes"]),
            vector(&case["target_variances"]),
            1e-8,
            4.0,
        )
        .unwrap();
        let model =
            RoughBergomi::new(number("hurst"), number("eta"), number("correlation")).unwrap();
        let config = LsvParticleConfig::new(
            f["normals"].as_array().unwrap().len(),
            f["seed"].as_u64().unwrap(),
            number("bandwidth"),
            number("minimum_effective_samples"),
            false,
        )
        .unwrap();
        let result = calibrate_rough_bergomi_lsv(&target, model, number("initial_f"), config);
        if let Some(row) = case["unsupported_time_index"].as_u64() {
            assert!(
                matches!(result, Err(LsvError::NoSupportedCalibrationNode { time_index }) if time_index == row as usize)
            );
            continue;
        }
        let calibrated = result.unwrap();
        let expected = &case["expected"];
        assert_eq!(
            calibrated.surface().squared_leverage().len(),
            expected["squared_leverage"].as_array().unwrap().len()
        );
        assert_eq!(
            calibrated.conditional_moments().len(),
            expected["moments"].as_array().unwrap().len()
        );
        assert_eq!(
            calibrated.diagnostics().len(),
            expected["diagnostics"].as_array().unwrap().len()
        );
        if case["id"] == "interior_donor_tie" {
            let middle = calibrated.conditional_moments()[4];
            assert!(middle.extrapolated);
            assert_eq!(middle.source_node, 0);
            assert!(middle.effective_samples < number("minimum_effective_samples"));
        }

        for (actual, expected) in calibrated
            .surface()
            .squared_leverage()
            .iter()
            .zip(vector(&expected["squared_leverage"]))
        {
            close(*actual, expected);
        }
        for (i, actual) in calibrated.conditional_moments().iter().enumerate() {
            let row = &expected["moments"][i];
            for (actual, name) in [
                (actual.second, "second"),
                (actual.third, "third"),
                (actual.fourth, "fourth"),
                (actual.effective_samples, "effective_samples"),
            ] {
                close(actual, row[name].as_f64().unwrap());
            }
            assert_eq!(
                actual.source_node,
                row["source_node"].as_u64().unwrap() as usize
            );
            assert_eq!(actual.extrapolated, row["extrapolated"].as_bool().unwrap());
        }
        for (i, actual) in calibrated.diagnostics().iter().enumerate() {
            let row = &expected["diagnostics"][i];
            close(actual.time, row["time"].as_f64().unwrap());
            close(
                actual.particle_mean_f,
                row["particle_mean_f"].as_f64().unwrap(),
            );
            close(
                actual.minimum_effective_samples,
                row["minimum_effective_samples"].as_f64().unwrap(),
            );
            assert_eq!(
                actual.extrapolated_nodes,
                row["extrapolated_nodes"].as_u64().unwrap() as usize
            );
        }
    }
}

/// Manual input capture only: never runs in ordinary test/CI discovery.
#[test]
#[ignore = "manual Gaussian input export for independent calibration reference"]
fn export_reference_calibration_normals() {
    let (seed, particles, steps) = (42, 64, 8);
    let rng = Philox4x32::from_seed(seed);
    let normals = (0..particles)
        .map(|p| {
            (0..3 * steps)
                .map(|d| {
                    rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::LsvCalibration))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!(
        "CALIBRATION_NORMALS {}",
        json!({"seed":seed,"steps":steps,"normals":normals})
    );
}
