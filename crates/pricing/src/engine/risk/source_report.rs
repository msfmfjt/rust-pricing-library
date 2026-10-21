//! Reporting only: copy native estimates, without recomputing sampling statistics.
use crate::mc::EngineConfig;
use crate::multi_asset::{MultiAssetBergomiLsvConfig, MultiAssetHullWhiteLsvRisk, MultiAssetPrice};
use crate::*;

pub(super) fn build(
    task: &PricingTask,
    fingerprint: Fingerprint,
    plan_fingerprint: String,
    out: &PricingTaskOutput,
) -> Result<ValuationReport, WireError> {
    let (engine, currency, ids, seeds) = source_metadata(task.source());
    let (value, error, units, paths) = match out {
        PricingTaskOutput::SingleAsset(r) => match &r.output {
            SpecificationOutput::Standard(p) => (
                p.pricing_result.value.value().get(),
                p.pricing_result.value.standard_error().get(),
                p.independent_sampling_units,
                p.evaluated_paths,
            ),
            SpecificationOutput::Hybrid(p) | SpecificationOutput::RoughVolatility(p) => (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            ),
            SpecificationOutput::LocalStochasticVolatility(p) => (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            ),
        },
        PricingTaskOutput::MultiAsset(p) => (
            p.price.value().get(),
            p.price.standard_error().get(),
            p.price.effective_sampling_units().get(),
            p.evaluated_paths,
        ),
        PricingTaskOutput::StochasticDividend(p) => (
            p.value,
            p.standard_error,
            p.independent_sampling_units,
            p.evaluated_paths,
        ),
        PricingTaskOutput::HullWhiteRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
        PricingTaskOutput::LocalVarianceRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
        PricingTaskOutput::StochasticDividendRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
        PricingTaskOutput::StochasticDividendLocalVarianceRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
    };
    let mut report = ValuationReport {
        task_fingerprint: fingerprint,
        plan_fingerprint,
        library_version: crate::version().into(),
        platform: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        currency_id: currency,
        value,
        standard_error: error,
        estimator: match engine {
            EngineConfig::PseudoMonteCarlo(_) => EstimatorKind::PseudoMonteCarlo,
            EngineConfig::RandomizedQuasiMonteCarlo(_) => EstimatorKind::RandomizedQuasiMonteCarlo,
        },
        independent_sampling_units: units,
        evaluated_paths: paths,
        uncertainty_scope: if seeds.is_empty() {
            UncertaintyScope::PricingSampling
        } else {
            UncertaintyScope::ConditionalOnCalibration
        },
        calibration_seeds: seeds,
        risks: Vec::new(),
        vega_kt: None,
    };
    let calibrated = if report.calibration_seeds.is_empty() {
        CalibrationTreatment::NotApplicable
    } else {
        CalibrationTreatment::Recalibrated
    };
    match out {
        PricingTaskOutput::SingleAsset(r) => {
            if let SpecificationOutput::Standard(p) = &r.output {
                if p.early_exercise_diagnostics.is_some() {
                    report.uncertainty_scope = UncertaintyScope::ConditionalOnExercisePolicy;
                }
                let d = &p.risk_diagnostics.methods;
                let treatment = if p.early_exercise_diagnostics.is_some() {
                    CalibrationTreatment::FixedExercisePolicy
                } else {
                    CalibrationTreatment::NotApplicable
                };
                let convention = format!(
                    "smile={:?}; exercise_strategy={:?}; stopping_indices={:?}; gamma_spot_bump={:?}; bump_policy_version={}",
                    d.smile_dynamics,
                    d.exercise_strategy,
                    d.stopping_indices,
                    d.gamma_spot_bump,
                    d.bump_policy_version
                );
                for (name, risk, method) in [
                    ("delta", p.pricing_result.risks.delta, d.delta),
                    ("gamma", p.pricing_result.risks.gamma, d.gamma),
                    ("vega", p.pricing_result.risks.vega, d.vega),
                ] {
                    if let Some(risk) = risk {
                        for (suffix, e, u) in [
                            ("raw", risk.raw(), risk.raw_unit()),
                            (
                                "market_scaled",
                                risk.market_scaled(),
                                risk.market_scaled_unit(),
                            ),
                        ] {
                            scalar(
                                &mut report,
                                &format!("{name}.{suffix}"),
                                &ids,
                                e,
                                unit(u),
                                method.map(method_name).unwrap_or("native"),
                                treatment,
                                &convention,
                            );
                        }
                    }
                }
                report.vega_kt = p.pricing_result.risks.vega_kt.clone();
            }
        }
        PricingTaskOutput::HullWhiteRisk(r) => labeled(
            &mut report,
            &ids,
            &r.parameter_labels,
            &r.derivatives,
            r.standard_errors.as_deref(),
            r.method,
            calibrated,
            "native escrowed equity coordinates; fixed model parameters, correlations, dates and grid; target derivatives include retained calibration",
        ),
        PricingTaskOutput::StochasticDividendRisk(r) => labeled(
            &mut report,
            &ids,
            &r.parameter_labels,
            &r.derivatives,
            Some(&r.standard_errors),
            r.method,
            CalibrationTreatment::NotApplicable,
            "physical Spot; initial volatility of residual equity; Q-mean cash dividends; fixed correlations, rates parameters, dates and grid",
        ),
        PricingTaskOutput::LocalVarianceRisk(r) => block(
            &mut report,
            "local_variance",
            &ids,
            grid(&r.time_nodes, &r.log_moneyness_nodes),
            &r.node_adjoints,
            r.standard_errors.as_deref(),
            SensitivityUnit::CurrencyPerVariance,
            r.method,
            CalibrationTreatment::Recalibrated,
            "effective target variance; finite-particle recalibration with fixed calibration seed, model and grid",
        ),
        PricingTaskOutput::StochasticDividendLocalVarianceRisk(r) => block(
            &mut report,
            "residual_local_variance",
            &ids,
            grid(&r.time_nodes, &r.log_moneyness_nodes),
            &r.node_adjoints,
            r.standard_errors.as_deref(),
            SensitivityUnit::CurrencyPerVariance,
            r.method,
            CalibrationTreatment::Recalibrated,
            "funded residual-equity target variance; finite-particle recalibration with fixed calibration seed, model and grid",
        ),
        PricingTaskOutput::MultiAsset(p) => multi_asset(&mut report, p, calibrated),
        PricingTaskOutput::StochasticDividend(_) => {}
    }
    report.validate()?;
    Ok(report)
}

fn source_metadata(s: &PricingSource) -> (EngineConfig, u16, Vec<u32>, Vec<u64>) {
    match s {
        PricingSource::SingleAsset(s) => {
            let seeds = match &s.model().equity {
                EquityModelSpec::LocalStochasticVolatility { particles, .. }
                | EquityModelSpec::RoughLocalStochasticVolatility { particles, .. } => {
                    vec![particles.seed()]
                }
                _ => vec![],
            };
            (
                s.engine(),
                s.product().currency().get(),
                vec![s.product().underlying().get()],
                seeds,
            )
        }
        PricingSource::StochasticDividend(s) => {
            source_metadata(&PricingSource::SingleAsset(s.equity().clone()))
        }
        PricingSource::MultiAsset(s) => {
            let mut seeds: Vec<_> = s
                .model()
                .lsv
                .iter()
                .flatten()
                .map(|c| match c {
                    MultiAssetBergomiLsvConfig::OneFactor(c) => c.particles.seed(),
                    MultiAssetBergomiLsvConfig::TwoFactor(c) => c.particles.seed(),
                    MultiAssetBergomiLsvConfig::Rough(c) => c.particles.seed(),
                })
                .collect();
            if let Some((c, _)) = &s.model().local_correlation {
                seeds.push(c.particles.seed());
            }
            (
                s.engine(),
                s.product().currency().get(),
                s.product().underlyings().iter().map(|u| u.get()).collect(),
                seeds,
            )
        }
    }
}
fn method_name(m: RiskMethod) -> &'static str {
    match m {
        RiskMethod::AadReverse => "aad_reverse",
        RiskMethod::CentralBump => "central_bump",
        RiskMethod::CentralBumpOfAadDelta => "central_bump_of_aad_delta",
    }
}
fn unit(u: RiskUnit) -> SensitivityUnit {
    match u {
        RiskUnit::DeltaRaw => SensitivityUnit::CurrencyPerSpot,
        RiskUnit::GammaRaw => SensitivityUnit::CurrencyPerSpotSquared,
        RiskUnit::VegaRaw => SensitivityUnit::CurrencyPerUnitVolatility,
        RiskUnit::DeltaOnePercentSpot => SensitivityUnit::CurrencyPerOnePercentSpot,
        RiskUnit::GammaOnePercentSpotSquared => SensitivityUnit::CurrencyPerOnePercentSpotSquared,
        RiskUnit::VegaOneVolPoint => SensitivityUnit::CurrencyPerVolatilityPoint,
    }
}
fn grid(t: &[f64], x: &[f64]) -> Vec<ReportAxis> {
    vec![
        ReportAxis {
            name: "time".into(),
            values: t.to_vec(),
        },
        ReportAxis {
            name: "log_moneyness".into(),
            values: x.to_vec(),
        },
    ]
}
#[allow(clippy::too_many_arguments)]
fn block(
    r: &mut ValuationReport,
    name: &str,
    ids: &[u32],
    mut axes: Vec<ReportAxis>,
    values: &[f64],
    errors: Option<&[f64]>,
    unit: SensitivityUnit,
    method: &str,
    calibration: CalibrationTreatment,
    convention: &str,
) {
    if values.is_empty() {
        return;
    }
    if axes.is_empty() && values.len() > 1 {
        axes.push(ReportAxis {
            name: "native_parameter_index".into(),
            values: (0..values.len()).map(|i| i as f64).collect(),
        });
    }
    r.risks.push(ReportedRisk {
        name: name.into(),
        underlying_ids: ids.to_vec(),
        axes,
        values: values.to_vec(),
        standard_errors: errors.map(<[f64]>::to_vec),
        unit,
        method: method.into(),
        calibration,
        convention: convention.into(),
    });
}
#[allow(clippy::too_many_arguments)]
fn scalar(
    r: &mut ValuationReport,
    name: &str,
    ids: &[u32],
    estimate: Estimate,
    unit: SensitivityUnit,
    method: &str,
    calibration: CalibrationTreatment,
    convention: &str,
) {
    block(
        r,
        name,
        ids,
        vec![],
        &[estimate.value().get()],
        Some(&[estimate.standard_error().get()]),
        unit,
        method,
        calibration,
        convention,
    );
}
#[allow(clippy::too_many_arguments)]
fn labeled(
    r: &mut ValuationReport,
    ids: &[u32],
    labels: &[String],
    values: &[f64],
    errors: Option<&[f64]>,
    method: &str,
    calibration: CalibrationTreatment,
    convention: &str,
) {
    for (i, label) in labels.iter().enumerate() {
        let unit = if label == "spot" {
            SensitivityUnit::CurrencyPerSpot
        } else if label.contains("volatility")
            || label.starts_with("market_iv[")
            || label == "parallel_market_iv"
        {
            SensitivityUnit::CurrencyPerUnitVolatility
        } else if label.starts_with("local_variance[") {
            SensitivityUnit::CurrencyPerVariance
        } else if label.contains("log_df[") {
            SensitivityUnit::CurrencyPerLogDiscountFactor
        } else if label.starts_with("forward_log_density[") {
            SensitivityUnit::CurrencyPerLogDensity
        } else {
            SensitivityUnit::CurrencyPerUnitParameter
        };
        block(
            r,
            label,
            ids,
            vec![],
            &values[i..i + 1],
            errors.map(|e| &e[i..i + 1]),
            unit,
            method,
            calibration,
            convention,
        );
    }
}
fn multi_asset(r: &mut ValuationReport, p: &MultiAssetPrice, calibration: CalibrationTreatment) {
    use SensitivityUnit as U;
    let convention = "physical Spot; fixed payout references, dates, model parameters, correlations and grid axes; native marginal/joint recalibration policy";
    for a in &p.risks {
        let ids = [a.underlying.get()];
        scalar(
            r,
            "delta.raw",
            &ids,
            a.delta,
            U::CurrencyPerSpot,
            "aad_reverse",
            calibration,
            convention,
        );
        scalar(
            r,
            "delta.market_scaled",
            &ids,
            a.delta_per_one_percent_spot,
            U::CurrencyPerOnePercentSpot,
            "aad_reverse",
            calibration,
            convention,
        );
        if let Some(v) = a.bs_vega {
            scalar(
                r,
                "bs_vega.raw",
                &ids,
                v,
                U::CurrencyPerUnitVolatility,
                "aad_reverse",
                calibration,
                convention,
            );
        }
        if let Some(v) = a.bs_vega_per_vol_point {
            scalar(
                r,
                "bs_vega.market_scaled",
                &ids,
                v,
                U::CurrencyPerVolatilityPoint,
                "aad_reverse",
                calibration,
                convention,
            );
        }
        if !a.local_variance.is_empty() {
            let values: Vec<_> = a.local_variance.iter().map(|v| v.value().get()).collect();
            let errors: Vec<_> = a
                .local_variance
                .iter()
                .map(|v| v.standard_error().get())
                .collect();
            block(
                r,
                "local_variance",
                &ids,
                grid(
                    &a.local_variance_time_nodes,
                    &a.local_variance_log_moneyness_nodes,
                ),
                &values,
                Some(&errors),
                U::CurrencyPerVariance,
                "aad_reverse",
                calibration,
                convention,
            );
        }
        if let Some(v) = &a.lsv_local_variance {
            block(
                r,
                "lsv_local_variance",
                &ids,
                grid(&v.time_nodes, &v.log_moneyness_nodes),
                &v.node_adjoints,
                v.standard_errors.as_deref(),
                U::CurrencyPerVariance,
                v.method,
                CalibrationTreatment::Recalibrated,
                convention,
            );
        }
        if let Some(v) = &a.hull_white_lsv {
            hw_lsv(r, "hw", &ids, v, convention);
        }
    }
    for (i, row) in p.gamma.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            scalar(
                r,
                "gamma",
                &[p.underlyings[i].get(), p.underlyings[j].get()],
                v,
                U::CurrencyPerSpotSquared,
                "central_bump_of_aad_delta",
                calibration,
                &format!(
                    "{convention}; unsymmetrized dDelta_i/dSpot_j; relative_bump={:?}",
                    p.gamma_relative_bump
                ),
            );
        }
    }
    if let Some(v) = &p.hull_white_curve_risk {
        estimates(
            r,
            "discount_log_df",
            &[],
            &v.discount_time_nodes,
            &v.discount_log_df_adjoints,
            U::CurrencyPerLogDiscountFactor,
            calibration,
            convention,
        );
        estimates(
            r,
            "discount_dv01",
            &[],
            &v.discount_time_nodes,
            &v.discount_node_dv01,
            U::CurrencyPerBasisPoint,
            calibration,
            "signed price change for +1bp continuously compounded zero-rate pillar",
        );
        for (i, values) in v.dividend_log_df_adjoints.iter().enumerate() {
            estimates(
                r,
                "repo_spread_log_df",
                &[p.underlyings[i].get()],
                &v.dividend_time_nodes[i],
                values,
                U::CurrencyPerLogDiscountFactor,
                calibration,
                convention,
            );
        }
    }
    if let Some(v) = &p.local_correlation_risk {
        block(
            r,
            "basket_local_variance",
            &[],
            grid(&v.basket_time_nodes, &v.basket_log_nodes),
            &v.basket_variance_adjoints,
            v.basket_standard_errors.as_deref(),
            U::CurrencyPerVariance,
            v.method,
            CalibrationTreatment::Recalibrated,
            convention,
        );
        for (i, values) in v.asset_adjoints.iter().enumerate() {
            let is_grid = !v.asset_time_nodes[i].is_empty();
            block(
                r,
                "joint_asset_target",
                &[p.underlyings[i].get()],
                if is_grid {
                    grid(&v.asset_time_nodes[i], &v.asset_log_nodes[i])
                } else {
                    vec![]
                },
                values,
                v.asset_standard_errors.as_ref().map(|s| s[i].as_slice()),
                if is_grid {
                    U::CurrencyPerVariance
                } else {
                    U::CurrencyPerUnitVolatility
                },
                v.method,
                CalibrationTreatment::Recalibrated,
                convention,
            );
        }
        if let Some(hw) = &v.basket_hull_white {
            hw_lsv(r, "basket_hw", &[], hw, convention);
        }
        for (i, hw) in v.asset_hull_white.iter().enumerate() {
            if let Some(hw) = hw {
                hw_lsv(
                    r,
                    "joint_asset_hw",
                    &[p.underlyings[i].get()],
                    hw,
                    convention,
                );
            }
        }
    }
}
fn hw_lsv(
    r: &mut ValuationReport,
    prefix: &str,
    ids: &[u32],
    v: &MultiAssetHullWhiteLsvRisk,
    convention: &str,
) {
    use SensitivityUnit as U;
    block(
        r,
        &format!("{prefix}.forward_log_density"),
        ids,
        vec![],
        &v.forward_log_density_adjoints,
        v.density_standard_errors.as_deref(),
        U::CurrencyPerLogDensity,
        v.method,
        CalibrationTreatment::Recalibrated,
        convention,
    );
    let axes = grid(&v.vega_kt_maturity_nodes, &v.vega_kt_log_moneyness_nodes);
    if let Some(values) = &v.vega_kt_raw {
        block(
            r,
            &format!("{prefix}.market_iv.raw"),
            ids,
            axes.clone(),
            values,
            v.vega_kt_standard_errors.as_deref(),
            U::CurrencyPerUnitVolatility,
            v.method,
            CalibrationTreatment::Recalibrated,
            convention,
        );
    }
    if let Some(values) = &v.vega_kt_market_scaled {
        let errors = v
            .vega_kt_standard_errors
            .as_ref()
            .map(|e| e.iter().map(|v| v * 0.01).collect::<Vec<_>>());
        block(
            r,
            &format!("{prefix}.market_iv.market_scaled"),
            ids,
            axes,
            values,
            errors.as_deref(),
            U::CurrencyPerVolatilityPoint,
            v.method,
            CalibrationTreatment::Recalibrated,
            convention,
        );
    }
    if let Some(vv) = v.parallel_vega {
        block(
            r,
            &format!("{prefix}.parallel_market_iv"),
            ids,
            vec![],
            &[vv],
            v.parallel_vega_standard_error
                .as_ref()
                .map(std::slice::from_ref),
            U::CurrencyPerUnitVolatility,
            v.method,
            CalibrationTreatment::Recalibrated,
            convention,
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn estimates(
    r: &mut ValuationReport,
    name: &str,
    ids: &[u32],
    times: &[f64],
    e: &[Estimate],
    unit: SensitivityUnit,
    calibration: CalibrationTreatment,
    convention: &str,
) {
    let values: Vec<_> = e.iter().map(|v| v.value().get()).collect();
    let errors: Vec<_> = e.iter().map(|v| v.standard_error().get()).collect();
    block(
        r,
        name,
        ids,
        vec![ReportAxis {
            name: "time".into(),
            values: times.to_vec(),
        }],
        &values,
        Some(&errors),
        unit,
        "aad_reverse",
        calibration,
        convention,
    );
}
