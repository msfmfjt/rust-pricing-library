//! The operation matrix is one classification read by capability queries, task
//! construction and the published guide. These tests keep the three aligned.
use pricing::risk::SpotBump;
use pricing::{
    CapabilityReason, CapabilityStatus, JsonLimits, Measure, MeasureKind, MeasureMethodKind,
    MeasureRequest, PricingOperation, PricingSource, SourceFamily, measure_matrix_markdown,
    operation_matrix_markdown, parse_pricing_task_json, parse_specification_json,
};
use std::path::{Path, PathBuf};

const BEGIN: &str =
    "<!-- operation-matrix:begin (generated; see crates/pricing/tests/operation_matrix.rs) -->\n";
const END: &str = "<!-- operation-matrix:end -->\n";

const EVALUATE: PricingOperation = PricingOperation::Evaluate;
const AAD: PricingOperation = PricingOperation::Aad {
    gamma_relative_bump: None,
};
const GAMMA: PricingOperation = PricingOperation::Aad {
    gamma_relative_bump: Some(0.01),
};
const LOCAL_VARIANCE: PricingOperation = PricingOperation::LocalVarianceRisk;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn specification(name: &str) -> PricingSource {
    let bytes = std::fs::read(repository().join("fixtures/specification-v1").join(name)).unwrap();
    parse_specification_json(&bytes, JsonLimits::DEFAULT)
        .unwrap()
        .into()
}

fn task(directory: &str, name: &str) -> PricingSource {
    let bytes = std::fs::read(repository().join("fixtures").join(directory).join(name)).unwrap();
    parse_pricing_task_json(&bytes, JsonLimits::DEFAULT)
        .unwrap()
        .source()
        .clone()
}

fn fixtures() -> Vec<(&'static str, PricingSource, SourceFamily)> {
    use SourceFamily as F;
    vec![
        (
            "black_scholes",
            specification("black_scholes.golden.json"),
            F::StandardSingleAsset,
        ),
        (
            "bergomi_lsv",
            specification("bergomi_lsv.golden.json"),
            F::LocalStochasticVolatility,
        ),
        (
            "bergomi_hull_white",
            specification("bergomi_hull_white.golden.json"),
            F::SingleAssetHullWhite,
        ),
        (
            "rough_heston",
            specification("rough_heston.golden.json"),
            F::DirectRoughVolatility,
        ),
        (
            "standard_risk",
            task("task-v1", "standard_risk.golden.json"),
            F::StandardSingleAsset,
        ),
        (
            "multi_asset_bs",
            task("task-v1", "multi_asset_bs.golden.json"),
            F::MultiAsset,
        ),
        (
            "multi_asset_joint",
            task("task-v1", "multi_asset_joint.golden.json"),
            F::MultiAsset,
        ),
        (
            "stochastic_dividend_bs",
            task("task-v1", "stochastic_dividend_bs.golden.json"),
            F::StochasticDividend,
        ),
        (
            "stochastic_dividend_hw",
            task("task-v1", "stochastic_dividend_hw.golden.json"),
            F::StochasticDividend,
        ),
        (
            "stochastic_dividend_lsv",
            task("task-v1", "stochastic_dividend_lsv.golden.json"),
            F::StochasticDividendLsv,
        ),
        (
            "contract_digital",
            task("task-v2", "contract_digital.json"),
            F::StandardSingleAsset,
        ),
    ]
}

#[test]
fn exposed_operations_are_an_explicit_reviewed_table() {
    use SourceFamily as F;
    // Columns: evaluate, first-order aad, aad with Gamma, local_variance_risk.
    let expected = [
        (F::StandardSingleAsset, [true, false, false, false]),
        (F::StochasticVolatility, [true, true, false, false]),
        (F::LocalStochasticVolatility, [true, false, false, true]),
        (F::RoughFamilyLsv, [true, false, false, true]),
        (F::SingleAssetHullWhite, [true, true, false, false]),
        (F::DirectRoughVolatility, [true, false, false, false]),
        (F::MultiAsset, [true, true, true, false]),
        (F::StochasticDividend, [true, true, false, false]),
        (F::StochasticDividendLsv, [true, false, false, true]),
    ];
    assert_eq!(expected.map(|(family, _)| family), SourceFamily::ALL);
    for (family, row) in expected {
        let actual = [EVALUATE, AAD, GAMMA, LOCAL_VARIANCE].map(|o| family.exposes(&o));
        assert_eq!(actual, row, "{}", family.as_str());
    }
    let mut labels = SourceFamily::ALL.map(SourceFamily::as_str).to_vec();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels.len(), SourceFamily::ALL.len());
}

#[test]
fn saved_sources_classify_into_their_documented_family() {
    for (name, source, family) in fixtures() {
        assert_eq!(source.family(), family, "{name}");
    }
}

#[test]
fn capability_denies_exactly_the_operations_a_family_does_not_expose() {
    for (name, source, _) in fixtures() {
        for operation in [EVALUATE, AAD, GAMMA, LOCAL_VARIANCE] {
            let capability = source.capability(operation.clone());
            let denied_as_unexposed = capability.status == CapabilityStatus::Unsupported
                && capability.reason == Some(CapabilityReason::DedicatedRiskApi);
            assert_eq!(
                denied_as_unexposed,
                !source.family().exposes(&operation),
                "{name}: {operation:?} -> {capability:?}"
            );
        }
    }
}

#[test]
fn discovery_lists_the_common_operations_in_a_fixed_order() {
    for (name, source, _) in fixtures() {
        let listed = source.available_operations();
        assert_eq!(
            listed
                .iter()
                .map(|c| c.operation.clone())
                .collect::<Vec<_>>(),
            [EVALUATE, AAD, LOCAL_VARIANCE],
            "{name}"
        );
        for capability in listed {
            assert_eq!(
                capability,
                source.capability(capability.operation.clone()),
                "{name}"
            );
        }
    }
}

fn embedded(begin_marker: &str, end_marker: &str, generated: &str) {
    // Both generated blocks live in one file; rewrite them one at a time.
    static GUIDE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = GUIDE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = repository().join("docs/library/pricing-tasks.md");
    let guide = std::fs::read_to_string(&path).unwrap();
    let begin = guide.find(begin_marker).expect("begin marker") + begin_marker.len();
    let end = guide.find(end_marker).expect("end marker");
    if std::env::var_os("UPDATE_OPERATION_MATRIX").is_some() {
        let updated = format!("{}{generated}{}", &guide[..begin], &guide[end..]);
        std::fs::write(&path, updated).unwrap();
        return;
    }
    assert_eq!(
        &guide[begin..end],
        generated,
        "docs/library/pricing-tasks.md is stale; rerun this test with UPDATE_OPERATION_MATRIX=1"
    );
}

/// Run with `UPDATE_OPERATION_MATRIX=1` to rewrite the block in the guide.
#[test]
fn task_guide_embeds_the_generated_operation_matrix() {
    embedded(BEGIN, END, &operation_matrix_markdown());
}

/// Run with `UPDATE_OPERATION_MATRIX=1` to rewrite the block in the guide.
#[test]
fn task_guide_embeds_the_generated_measure_matrix() {
    embedded(
        "<!-- measure-matrix:begin (generated; see crates/pricing/tests/operation_matrix.rs) -->\n",
        "<!-- measure-matrix:end -->\n",
        &measure_matrix_markdown(),
    );
}

#[test]
fn measure_methods_are_an_explicit_reviewed_table() {
    use MeasureKind as K;
    use MeasureMethodKind as M;
    for family in SourceFamily::ALL {
        for measure in K::ALL {
            use SourceFamily as F;
            let expected: &[M] = match (family, measure) {
                (_, K::Price) => &[M::Native],
                (F::StandardSingleAsset, K::SpotGamma) => &[M::NativeSpotBump, M::BumpAndRevalue],
                (F::StandardSingleAsset, K::VegaKt) => &[M::Native],
                (F::DirectRoughVolatility, K::SpotDelta) => &[M::Native, M::BumpAndRevalue],
                (F::DirectRoughVolatility, K::SpotGamma) => &[M::NativeSpotBump, M::BumpAndRevalue],
                (F::DirectRoughVolatility, K::ModelParameters) => &[M::Native],
                (
                    F::StochasticDividend,
                    K::Correlations | K::ModelParameters | K::DividendCash | K::DiscountCurve,
                ) => &[M::Native],
                (F::StochasticDividendLsv, K::DividendCash | K::DiscountCurve) => {
                    &[M::NativeCalibrated]
                }
                (F::MultiAsset, K::SpotDelta) => {
                    &[M::Native, M::NativeCalibrated, M::BumpAndRevalue]
                }
                (F::MultiAsset, K::CrossGamma) => &[M::NativeSpotBump, M::NativeSpotBumpCalibrated],
                (F::MultiAsset, K::DiscountCurve) => &[M::Native, M::NativeCalibrated],
                (F::MultiAsset, K::Correlations) => &[M::BumpAndRevalue],
                (_, K::Correlations | K::DividendCash | K::CrossGamma | K::DiscountCurve) => &[],
                (F::StochasticDividend, K::SpotDelta) => &[M::Native, M::BumpAndRevalue],
                (F::StochasticDividend, K::SpotGamma) => &[M::NativeSpotBump, M::BumpAndRevalue],
                (F::StochasticDividendLsv, K::SpotDelta) => {
                    &[M::NativeCalibrated, M::BumpAndRevalue]
                }
                (F::StochasticDividendLsv, K::SpotGamma) => {
                    &[M::NativeSpotBumpCalibrated, M::BumpAndRevalue]
                }
                (_, K::ModelParameters) if family != F::RoughFamilyLsv => &[],
                (F::StandardSingleAsset, _) => &[M::Native, M::BumpAndRevalue],
                (F::RoughFamilyLsv, K::SpotDelta) => &[M::NativeCalibrated, M::BumpAndRevalue],
                (F::RoughFamilyLsv, K::SpotGamma) => {
                    &[M::NativeSpotBumpCalibrated, M::BumpAndRevalue]
                }
                (F::RoughFamilyLsv, K::ParallelMarketIvVega) => {
                    &[M::NativeCalibrated, M::BumpAndRevalue]
                }
                (F::LocalStochasticVolatility, K::ParallelMarketIvVega) => &[M::BumpAndRevalue],
                (F::RoughFamilyLsv, K::ModelParameters) => &[M::NativeCalibrated],
                // Every source reprices under a moved Spot.
                (_, K::SpotDelta | K::SpotGamma) => &[M::BumpAndRevalue],
                (F::SingleAssetHullWhite, K::ParallelMarketIvVega) => &[M::BumpAndRevalue],
                _ => &[],
            };
            assert_eq!(
                family.measure_methods(measure),
                expected,
                "{family:?} {measure:?}"
            );
        }
    }
}

#[test]
fn measure_capability_follows_the_table_for_every_fixture() {
    let bump = SpotBump::relative(0.01).unwrap();
    for (name, source, family) in fixtures() {
        let available = source.available_measures();
        let requests = [
            MeasureRequest::native(Measure::Price),
            MeasureRequest::native(Measure::SpotDelta),
            MeasureRequest::native(Measure::SpotGamma),
            MeasureRequest::spot_gamma(bump),
            MeasureRequest::native(Measure::ParallelMarketIvVega),
        ];
        for capability in source.measure_capabilities(&requests) {
            let kind = capability.request.measure.kind();
            let methods = family.measure_methods(kind);
            assert_eq!(
                available
                    .iter()
                    .find(|a| a.measure == kind)
                    .map(|a| &a.methods[..]),
                (!methods.is_empty()).then_some(methods),
                "{name} {kind:?}"
            );
            let expected = if methods.is_empty() {
                Some(CapabilityReason::DedicatedRiskApi)
            } else if !methods.contains(&capability.request.method.kind()) {
                Some(CapabilityReason::NoNativeEstimator)
            } else if name == "standard_risk" && kind != MeasureKind::Price {
                // This fixture selects Delta and Vega in its nested request.
                Some(CapabilityReason::AmbiguousRiskSelection)
            } else if name == "multi_asset_joint" && kind == MeasureKind::SpotDelta {
                // A calibrated basket states recalibrated for its native Delta.
                Some(CapabilityReason::BumpCalibrationMismatch)
            } else if name == "contract_digital" && kind != MeasureKind::Price {
                // An unsmoothed digital payoff has no pathwise Greeks.
                Some(CapabilityReason::DiscontinuousPayoff)
            } else {
                None
            };
            match expected {
                Some(reason) => {
                    assert_eq!(
                        capability.status,
                        CapabilityStatus::Unsupported,
                        "{name} {kind:?}"
                    );
                    assert_eq!(capability.reason, Some(reason), "{name} {kind:?}");
                }
                None => assert_eq!(
                    (capability.status, capability.reason),
                    (
                        CapabilityStatus::Conditional,
                        Some(CapabilityReason::CompilationRequired)
                    ),
                    "{name} {kind:?}: {}",
                    capability.detail
                ),
            }
        }
    }
}
