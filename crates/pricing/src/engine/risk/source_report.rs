//! Reporting only: copy native estimates, without recomputing sampling statistics.
use crate::mc::EngineConfig;
use crate::multi_asset::{MultiAssetBergomiLsvConfig, MultiAssetHullWhiteLsvRisk, MultiAssetPrice};
use crate::*;

pub(super) fn build(
    task: &PricingTask,
    fingerprint: Fingerprint,
    plan_fingerprint: String,
    out: &PricingTaskOutput,
    calibration_diagnostics: Vec<ReportedCalibration>,
) -> Result<ValuationReport, WireError> {
    let (engine, currency, ids, seeds) = source_metadata(task.source());
    let (value, error, units, paths) = headline(out);
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
        calibration_diagnostics: Some(calibration_diagnostics),
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
                for (name, kind, risk, method) in [
                    (
                        "delta",
                        MeasureKind::SpotDelta,
                        p.pricing_result.risks.delta,
                        d.delta,
                    ),
                    (
                        "gamma",
                        MeasureKind::SpotGamma,
                        p.pricing_result.risks.gamma,
                        d.gamma,
                    ),
                    (
                        "vega",
                        MeasureKind::ParallelMarketIvVega,
                        p.pricing_result.risks.vega,
                        d.vega,
                    ),
                ] {
                    if let Some(risk) = risk {
                        // A measures task types its rows; legacy rows stay
                        // named only, so their documents are unchanged.
                        let requested = task.measures().and_then(|set| set.get(kind));
                        let first = report.risks.len();
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
                        if let Some(requested) = requested {
                            for row in &mut report.risks[first..] {
                                row.measure = Some(kind);
                                row.method_detail = Some(requested.method);
                            }
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
            &r.parameter_kinds,
            &r.derivatives,
            r.standard_errors.as_deref(),
            r.method,
            calibrated,
            "native escrowed equity coordinates; fixed model parameters, correlations, dates and grid; target derivatives include retained calibration",
        )?,
        PricingTaskOutput::StochasticDividendRisk(r) => labeled(
            &mut report,
            &ids,
            &r.parameter_labels,
            &r.parameter_kinds,
            &r.derivatives,
            Some(&r.standard_errors),
            r.method,
            CalibrationTreatment::NotApplicable,
            "physical Spot; initial volatility of residual equity; Q-mean cash dividends; fixed correlations, rates parameters, dates and grid",
        )?,
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
        PricingTaskOutput::MultiAsset(p) if task.measures().is_some() => {
            basket_rows(&mut report, task, p, calibrated);
        }
        PricingTaskOutput::MultiAsset(p) => multi_asset(&mut report, p, calibrated),
        PricingTaskOutput::RoughFamilyLsvMeasureRisk(r) => {
            rough_family_lsv_rows(&mut report, task, &ids, r)
        }
        PricingTaskOutput::RoughVolatilityMeasureRisk(r) => {
            rough_volatility_rows(&mut report, task, &ids, r)
        }
        PricingTaskOutput::StochasticDividendMeasureRisk(r) => {
            stochastic_dividend_rows(&mut report, task, &ids, r, calibrated)
        }
        PricingTaskOutput::StochasticDividend(_) => {}
    }
    report.validate()?;
    Ok(report)
}

/// Price, its standard error, independent sampling units and evaluated paths
/// of any native output.
pub(super) fn headline(out: &PricingTaskOutput) -> (f64, f64, u64, u128) {
    match out {
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
        PricingTaskOutput::RoughFamilyLsvMeasureRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
        PricingTaskOutput::StochasticDividendMeasureRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
        PricingTaskOutput::RoughVolatilityMeasureRisk(r) => {
            let p = &r.price;
            (
                p.value,
                p.standard_error,
                p.independent_sampling_units,
                p.evaluated_paths,
            )
        }
    }
}

/// Native rows of rough-family LSV, copied from the dedicated results.
fn rough_family_lsv_rows(
    report: &mut ValuationReport,
    task: &PricingTask,
    ids: &[u32],
    r: &crate::lsv::RoughFamilyLsvMeasureRisk,
) {
    use crate::lsv::RoughFamilyLsvDeltaConvention as C;
    let treatment = |c: C| match c {
        C::FrozenLeverage => CalibrationTreatment::FixedCalibration,
        C::StickyRelativeLocalVariance => CalibrationTreatment::Recalibrated,
    };
    let held = |c: C| match c {
        C::FrozenLeverage => {
            "calibrated leverage and its reference funded-forward anchor held; initial residual equity moves with Spot"
        }
        C::StickyRelativeLocalVariance => {
            "relative local-variance target held in log-moneyness and recalibrated (scale equivariant: leverage values unchanged, anchor follows Spot); not sticky-strike market-IV risk"
        }
    };
    let mut push = |kind: MeasureKind,
                    value,
                    error: Option<f64>,
                    unit,
                    method: &str,
                    calibration,
                    convention| {
        let requested = task
            .measures()
            .and_then(|set| set.get(kind))
            .expect("native rows answer listed measures");
        report.risks.push(ReportedRisk {
            name: format!("{}.{}", kind.as_str(), requested.method.kind().as_str()),
            underlying_ids: ids.to_vec(),
            axes: Vec::new(),
            values: vec![value],
            standard_errors: error.map(|e| vec![e]),
            unit,
            method: method.into(),
            calibration,
            convention,
            measure: Some(kind),
            method_detail: Some(requested.method),
            model_parameter: None,
            correlation: None,
        });
    };
    if let Some(d) = &r.delta {
        push(
            MeasureKind::SpotDelta,
            d.delta,
            Some(d.delta_standard_error),
            SensitivityUnit::CurrencyPerSpot,
            d.method,
            treatment(d.convention),
            format!(
                "physical Spot; {}: {}; error conditional on one calibration",
                d.convention.as_str(),
                held(d.convention)
            ),
        );
    }
    if let Some(g) = &r.gamma {
        let c = if g.convention == C::FrozenLeverage.as_str() {
            C::FrozenLeverage
        } else {
            C::StickyRelativeLocalVariance
        };
        push(
            MeasureKind::SpotGamma,
            g.gamma,
            Some(g.gamma_standard_error),
            SensitivityUnit::CurrencyPerSpotSquared,
            g.method,
            treatment(c),
            format!(
                "physical Spot; {}: {}; central second difference with absolute bump {}; the half-bump Gamma and its paired difference are in the native result; error conditional on one calibration",
                g.convention,
                held(c),
                g.spot_bump
            ),
        );
    }
    if let Some(m) = &r.market_iv {
        push(
            MeasureKind::ParallelMarketIvVega,
            m.parallel_vega,
            m.parallel_standard_error,
            SensitivityUnit::CurrencyPerUnitVolatility,
            m.method,
            CalibrationTreatment::Recalibrated,
            format!(
                "per one absolute annualized Black IV unit added to every node of the target's reporting basis ({} maturities x {} relative log-moneyness nodes); the Dupire target is rebuilt from the moved surface and the particle calibration recalibrated through its adjoint; model parameters, Spot, curves and dividends held; node risks in the native result; {}",
                m.maturity_nodes.len(),
                m.log_moneyness_nodes.len(),
                if m.parallel_standard_error.is_some() {
                    "RQMC error across scrambles, conditional on one calibration"
                } else {
                    "pseudo-random sampling error not available for this estimator"
                }
            ),
        );
    }
    if let Some(p) = &r.model_parameters {
        let requested = task
            .measures()
            .and_then(|set| set.get(MeasureKind::ModelParameters))
            .expect("native rows answer listed measures");
        let Measure::ModelParameters(selection) = &requested.measure else {
            unreachable!("the kind selects the variant")
        };
        let values = p.risk.parameter_adjoints();
        let errors = p.risk.standard_errors();
        for &parameter in selection.parameters() {
            let index = p
                .parameters
                .iter()
                .position(|&q| q == parameter)
                .expect("preflight admitted only the model's parameters");
            report.risks.push(ReportedRisk {
                name: format!(
                    "{}.{}.{}",
                    MeasureKind::ModelParameters.as_str(),
                    parameter.label(),
                    requested.method.kind().as_str()
                ),
                underlying_ids: ids.to_vec(),
                axes: Vec::new(),
                values: vec![values[index]],
                standard_errors: errors.map(|e| vec![e[index]]),
                unit: SensitivityUnit::CurrencyPerUnitParameter,
                method: p.risk.method().into(),
                calibration: CalibrationTreatment::Recalibrated,
                convention: format!(
                    "per one absolute unit of the model's {} at the fixed relative local-variance target; total derivative: direct valuation plus the particle recalibration's response (both parts in the native result); other parameters, Spot, curves and dividends held; not a stochastic-parameter refit and not market-IV Vega; {}",
                    parameter.label(),
                    if errors.is_some() {
                        "RQMC error across scrambles, conditional on one calibration"
                    } else {
                        "pseudo-random sampling error not available for this estimator"
                    }
                ),
                measure: Some(MeasureKind::ModelParameters),
                method_detail: Some(requested.method),
                model_parameter: Some(parameter),
                correlation: None,
            });
        }
    }
}

/// Native rows of a direct rough-volatility source, copied from the
/// dedicated results. Nothing is calibrated.
fn rough_volatility_rows(
    report: &mut ValuationReport,
    task: &PricingTask,
    ids: &[u32],
    r: &crate::rough_volatility::RoughVolatilityMeasureRisk,
) {
    let requested = |kind: MeasureKind| {
        task.measures()
            .and_then(|set| set.get(kind))
            .expect("native rows answer listed measures")
    };
    let row = |name: String,
               kind: MeasureKind,
               value: f64,
               error: f64,
               unit,
               method: &str,
               convention: String,
               model_parameter| ReportedRisk {
        name,
        underlying_ids: ids.to_vec(),
        axes: Vec::new(),
        values: vec![value],
        standard_errors: Some(vec![error]),
        unit,
        method: method.into(),
        calibration: CalibrationTreatment::NotApplicable,
        convention,
        measure: Some(kind),
        method_detail: Some(requested(kind).method),
        model_parameter,
        correlation: None,
    };
    let name = |kind: MeasureKind| {
        format!(
            "{}.{}",
            kind.as_str(),
            requested(kind).method.kind().as_str()
        )
    };
    const HELD: &str = "rough-model parameters, rate and repo curves and cash and proportional dividend amounts held";
    if let Some(d) = &r.delta {
        report.risks.push(row(
            name(MeasureKind::SpotDelta),
            MeasureKind::SpotDelta,
            d.delta,
            d.delta_standard_error,
            SensitivityUnit::CurrencyPerSpot,
            d.method,
            format!("physical Spot; pathwise through payoff, path and dividend escrow; {HELD}"),
            None,
        ));
    }
    if let Some(g) = &r.gamma {
        report.risks.push(row(
            name(MeasureKind::SpotGamma),
            MeasureKind::SpotGamma,
            g.gamma,
            g.gamma_standard_error,
            SensitivityUnit::CurrencyPerSpotSquared,
            g.method,
            format!(
                "physical Spot; central second difference with absolute bump {} on common random numbers; the half-bump Gamma and its paired difference are in the native result; {HELD}; error excludes bump bias",
                g.spot_bump
            ),
            None,
        ));
    }
    if let Some(p) = &r.model_parameters {
        let Measure::ModelParameters(selection) = &requested(MeasureKind::ModelParameters).measure
        else {
            unreachable!("the kind selects the variant")
        };
        for &parameter in selection.parameters() {
            let (value, error, method) = p
                .coordinate(parameter)
                .expect("preflight admitted only the model's parameters");
            report.risks.push(row(
                format!(
                    "{}.{}.{}",
                    MeasureKind::ModelParameters.as_str(),
                    parameter.label(),
                    requested(MeasureKind::ModelParameters).method.kind().as_str()
                ),
                MeasureKind::ModelParameters,
                value,
                error,
                SensitivityUnit::CurrencyPerUnitParameter,
                method,
                format!(
                    "per one absolute unit of the model's {}; fixed model: pathwise through the discrete variance history{}, with every other parameter, Spot, curves, dividends and Gaussian draws held; not a recalibrated or market-IV sensitivity",
                    parameter.label(),
                    if parameter == ModelParameterId::Hurst {
                        " and the H-dependent kernel loadings"
                    } else {
                        ""
                    }
                ),
                Some(parameter),
            ));
        }
    }
}

/// Native Spot rows of a stochastic-dividend source, copied from the
/// dedicated results. `treatment` is `Recalibrated` on a residual-equity LSV.
/// The two `discount_curve` rows: log-DF adjoints and their DV01
/// projection, on the curve's pillar times.
#[allow(clippy::too_many_arguments)]
fn discount_curve_rows(
    report: &mut ValuationReport,
    task: &PricingTask,
    ids: &[u32],
    times: &[f64],
    log_df: (Vec<f64>, Vec<f64>),
    dv01: (Vec<f64>, Vec<f64>),
    method: &str,
    treatment: CalibrationTreatment,
    held: &str,
) {
    let requested = task
        .measures()
        .and_then(|set| set.get(MeasureKind::DiscountCurve))
        .expect("native rows answer listed measures");
    let rows = [
        (
            "log_df",
            log_df,
            SensitivityUnit::CurrencyPerLogDiscountFactor,
            "per unit log discount factor of each discount-curve pillar, the curve interpolated between pillars as the source builds it",
        ),
        (
            "dv01",
            dv01,
            SensitivityUnit::CurrencyPerBasisPoint,
            "signed price change for +1bp in each continuously compounded zero-rate pillar (-1e-4 t times the log-DF row)",
        ),
    ];
    for (part, (values, errors), unit, what) in rows {
        report.risks.push(ReportedRisk {
            name: format!(
                "{}.{part}.{}",
                MeasureKind::DiscountCurve.as_str(),
                requested.method.kind().as_str()
            ),
            underlying_ids: ids.to_vec(),
            axes: vec![ReportAxis {
                name: "discount_time".into(),
                values: times.to_vec(),
            }],
            values,
            standard_errors: Some(errors),
            unit,
            method: method.into(),
            calibration: treatment,
            convention: format!("{what}; {held}"),
            measure: Some(MeasureKind::DiscountCurve),
            method_detail: Some(requested.method),
            model_parameter: None,
            correlation: None,
        });
    }
}

/// `-1e-4 t` times each log-DF adjoint and its error.
fn dv01_projection(times: &[f64], (values, errors): &(Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>) {
    times
        .iter()
        .zip(values.iter().zip(errors))
        .map(|(t, (v, e))| (-1e-4 * t * v, 1e-4 * t * e))
        .unzip()
}

fn stochastic_dividend_rows(
    report: &mut ValuationReport,
    task: &PricingTask,
    ids: &[u32],
    r: &crate::stochastic_dividends::StochasticDividendMeasureRisk,
    treatment: CalibrationTreatment,
) {
    let held = if treatment == CalibrationTreatment::Recalibrated {
        "residual-equity LSV surface re-anchored to the moved funded residual equity, its squared-leverage values unchanged (recalibrated by scale invariance); Q-mean cash dividends, dividend model, curves, dates and grid held; error conditional on one calibration"
    } else {
        "Q-mean cash dividends, volatility and dividend model parameters, correlations, curves, dates and grid held"
    };
    let mut push = |kind: MeasureKind, value: f64, error: f64, unit, method: &str, convention| {
        let requested = task
            .measures()
            .and_then(|set| set.get(kind))
            .expect("native rows answer listed measures");
        report.risks.push(ReportedRisk {
            name: format!("{}.{}", kind.as_str(), requested.method.kind().as_str()),
            underlying_ids: ids.to_vec(),
            axes: Vec::new(),
            values: vec![value],
            standard_errors: Some(vec![error]),
            unit,
            method: method.into(),
            calibration: treatment,
            convention,
            measure: Some(kind),
            method_detail: Some(requested.method),
            model_parameter: None,
            correlation: None,
        });
    };
    if let Some(d) = &r.delta {
        push(
            MeasureKind::SpotDelta,
            d.delta(),
            d.standard_error(),
            SensitivityUnit::CurrencyPerSpot,
            d.method(),
            format!("physical Spot; pathwise through the Buehler escrow split; {held}"),
        );
    }
    if let Some(g) = &r.gamma {
        push(
            MeasureKind::SpotGamma,
            g.gamma(),
            g.standard_error(),
            SensitivityUnit::CurrencyPerSpotSquared,
            g.method,
            format!(
                "physical Spot; central difference of native Deltas with absolute bump {} on common random numbers; the half- and double-bump Gammas and their paired differences are in the native result; {held}; error excludes bump bias",
                g.spot_bumps[1]
            ),
        );
    }
    if let Some(c) = &r.correlations {
        let requested = task
            .measures()
            .and_then(|set| set.get(MeasureKind::Correlations))
            .expect("native rows answer listed measures");
        let Measure::Correlations(selection) = &requested.measure else {
            unreachable!("the kind selects the variant")
        };
        for &correlation in selection.correlations() {
            let (value, error) = c
                .coordinate(correlation)
                .expect("preflight admitted only the source's correlations");
            report.risks.push(ReportedRisk {
                name: format!(
                    "{}.{}.{}",
                    MeasureKind::Correlations.as_str(),
                    correlation.label(),
                    requested.method.kind().as_str()
                ),
                underlying_ids: ids.to_vec(),
                axes: Vec::new(),
                values: vec![value],
                standard_errors: Some(vec![error]),
                unit: SensitivityUnit::CurrencyPerUnitParameter,
                method: c.risk.method.into(),
                calibration: treatment,
                convention: format!(
                    "per one absolute unit of the {} Brownian correlation, varying that symmetric pair of the driver correlation matrix with every other entry held; no projection to a valid matrix and no recalibration; Spot, cash means, model parameters, curves, dates and grid held; the other coordinates of the joint reverse are in the native result",
                    correlation.label()
                ),
                measure: Some(MeasureKind::Correlations),
                method_detail: Some(requested.method),
                model_parameter: None,
                correlation: Some(correlation),
            });
        }
    }
    if let Some(c) = &r.dividend_cash {
        let requested = task
            .measures()
            .and_then(|set| set.get(MeasureKind::DividendCash))
            .expect("native rows answer listed measures");
        let (values, errors) = c.adjoints();
        let reverse = if treatment == CalibrationTreatment::Recalibrated {
            "the residual-equity LSV surface re-anchored to the moved funded residual equity with its squared-leverage values unchanged (recalibrated by scale invariance)"
        } else {
            "pathwise through the Buehler escrow split, the funded reserve and the payment discount"
        };
        report.risks.push(ReportedRisk {
            name: format!(
                "{}.{}",
                MeasureKind::DividendCash.as_str(),
                requested.method.kind().as_str()
            ),
            underlying_ids: ids.to_vec(),
            axes: vec![ReportAxis {
                name: "ex_time".into(),
                values: c.cash_times().to_vec(),
            }],
            values,
            standard_errors: Some(errors),
            unit: SensitivityUnit::CurrencyPerUnitParameter,
            method: c.method().into(),
            calibration: treatment,
            convention: format!(
                "per one currency unit of each cash dividend's risk-neutral mean, in the market's schedule order on the ex-dividend-time axis; {reverse}; Spot, the dividend and volatility models, correlations, curves, dates and grid held"
            ),
            measure: Some(MeasureKind::DividendCash),
            method_detail: Some(requested.method),
            model_parameter: None,
            correlation: None,
        });
    }
    if let Some(c) = &r.discount_curve {
        let log_df = c.discount_adjoints();
        let dv01 = dv01_projection(c.discount_times(), &log_df);
        let reverse = if treatment == CalibrationTreatment::Recalibrated {
            "the residual-equity LSV surface re-anchored to the moved funded residual equity with its squared-leverage values unchanged (recalibrated by scale invariance)"
        } else {
            "pathwise through the escrow, the funded reserve and the payment discount"
        };
        discount_curve_rows(
            report,
            task,
            ids,
            c.discount_times(),
            log_df,
            dv01,
            c.method(),
            treatment,
            &format!(
                "{reverse}; the repo-spread curve, Spot, cash means, models, correlations, dates and grid held; repo-spread pillars are in the native result"
            ),
        );
    }
    if let Some(p) = &r.model_parameters {
        let requested = task
            .measures()
            .and_then(|set| set.get(MeasureKind::ModelParameters))
            .expect("native rows answer listed measures");
        let Measure::ModelParameters(selection) = &requested.measure else {
            unreachable!("the kind selects the variant")
        };
        for &parameter in selection.parameters() {
            let (value, error) = p
                .coordinate(parameter)
                .expect("preflight admitted only the source's parameters");
            report.risks.push(ReportedRisk {
                name: format!(
                    "{}.{}.{}",
                    MeasureKind::ModelParameters.as_str(),
                    parameter.label(),
                    requested.method.kind().as_str()
                ),
                underlying_ids: ids.to_vec(),
                axes: Vec::new(),
                values: vec![value],
                standard_errors: Some(vec![error]),
                unit: SensitivityUnit::CurrencyPerUnitParameter,
                method: p.risk.method.into(),
                calibration: treatment,
                convention: format!(
                    "per one absolute unit of the model's {}; fixed model: pathwise through the Buehler escrow split and the discrete path, with every other parameter, Spot, correlations, cash means, curves, dates, grid and Gaussian draws held; not a recalibrated or market-IV sensitivity",
                    parameter.label()
                ),
                measure: Some(MeasureKind::ModelParameters),
                method_detail: Some(requested.method),
                model_parameter: Some(parameter),
                correlation: None,
            });
        }
    }
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
        measure: None,
        method_detail: None,
        model_parameter: None,
        correlation: None,
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
    kinds: &[RiskParameterKind],
    values: &[f64],
    errors: Option<&[f64]>,
    method: &str,
    calibration: CalibrationTreatment,
    convention: &str,
) -> Result<(), WireError> {
    if labels.len() != kinds.len()
        || labels.len() != values.len()
        || errors.is_some_and(|e| e.len() != values.len())
    {
        return Err(WireError::DomainAt {
            pointer: "/report/risks".into(),
            message: "native risk labels, kinds, values and errors must have matching lengths"
                .into(),
        });
    }
    for (i, label) in labels.iter().enumerate() {
        block(
            r,
            label,
            ids,
            vec![],
            &values[i..i + 1],
            errors.map(|e| &e[i..i + 1]),
            kinds[i].unit(),
            method,
            calibration,
            convention,
        );
    }
    Ok(())
}
/// Native rows of a basket `Measures` task, copied from `evaluate_aad`: one
/// Spot Delta per asset and one cross Gamma per ordered pair, each with the
/// underlyings it concerns.
fn basket_rows(
    report: &mut ValuationReport,
    task: &PricingTask,
    p: &MultiAssetPrice,
    treatment: CalibrationTreatment,
) {
    let held = if treatment == CalibrationTreatment::Recalibrated {
        "LSV marginals and Local Correlation recalibrated through the native reverse (forward-normalized, sticky moneyness)"
    } else {
        "no calibration step"
    };
    let mut push = |kind: MeasureKind,
                    ids: Vec<u32>,
                    value: Estimate,
                    unit,
                    method: &str,
                    convention: String| {
        let requested = task
            .measures()
            .and_then(|set| set.get(kind))
            .expect("native rows answer listed measures");
        report.risks.push(ReportedRisk {
            name: format!("{}.{}", kind.as_str(), requested.method.kind().as_str()),
            underlying_ids: ids,
            axes: Vec::new(),
            values: vec![value.value().get()],
            standard_errors: Some(vec![value.standard_error().get()]),
            unit,
            method: method.into(),
            calibration: treatment,
            convention,
            measure: Some(kind),
            method_detail: Some(requested.method),
            model_parameter: None,
            correlation: None,
        });
    };
    let listed = |kind| {
        task.measures()
            .and_then(|set| set.get(kind))
            .is_some_and(|r| !matches!(r.method, MeasureMethod::BumpAndRevalue { .. }))
    };
    if listed(MeasureKind::SpotDelta) {
        for a in &p.risks {
            push(
                MeasureKind::SpotDelta,
                vec![a.underlying.get()],
                a.delta,
                SensitivityUnit::CurrencyPerSpot,
                "aad_reverse",
                format!(
                    "physical Spot of this underlying, the others held; pathwise through payoff and paths; {held}; payout references, dates, model parameters, correlations and grid axes fixed"
                ),
            );
        }
    }
    if listed(MeasureKind::CrossGamma) {
        for (i, row) in p.gamma.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                push(
                    MeasureKind::CrossGamma,
                    vec![p.underlyings[i].get(), p.underlyings[j].get()],
                    value,
                    SensitivityUnit::CurrencyPerSpotSquared,
                    "central_bump_of_aad_delta",
                    format!(
                        "unsymmetrized dDelta_i/dSpot_j for underlyings [i, j]: central difference of native Deltas with relative Spot bump {:?} of underlying j on common random numbers; {held}; error excludes bump bias",
                        p.gamma_relative_bump.unwrap_or_default()
                    ),
                );
            }
        }
    }
    if listed(MeasureKind::DiscountCurve)
        && let Some(v) = &p.hull_white_curve_risk
    {
        let split = |values: &[Estimate]| -> (Vec<f64>, Vec<f64>) {
            values
                .iter()
                .map(|e| (e.value().get(), e.standard_error().get()))
                .unzip()
        };
        discount_curve_rows(
            report,
            task,
            &[],
            &v.discount_time_nodes,
            split(&v.discount_log_df_adjoints),
            split(&v.discount_node_dv01),
            "aad_reverse",
            treatment,
            &format!(
                "the shared Hull-White initial curve, through the rate paths and every cashflow discount; {held}; repo-spread curves, Spot, model parameters, correlations, dates and grid held; repo-spread pillars are in the native result"
            ),
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

/// Rows of bump-and-revalue estimates. They carry no sampling error: the
/// paired error of a revaluation difference is not available yet.
pub(super) fn bump_rows(
    report: &mut ValuationReport,
    estimates: &[super::bump::BumpEstimate],
) -> Result<(), WireError> {
    for e in estimates {
        let MeasureMethod::BumpAndRevalue { calibration, .. } = e.request.method else {
            unreachable!("bump rows come from bump-and-revalue requests")
        };
        let kind = e.request.measure.kind();
        let unit = match kind {
            MeasureKind::SpotGamma => SensitivityUnit::CurrencyPerSpotSquared,
            MeasureKind::ParallelMarketIvVega => SensitivityUnit::CurrencyPerUnitVolatility,
            _ => SensitivityUnit::CurrencyPerSpot,
        };
        report.risks.push(ReportedRisk {
            name: format!("{}.bump_and_revalue", kind.as_str()),
            underlying_ids: vec![e.underlying],
            axes: Vec::new(),
            values: vec![e.value],
            standard_errors: e.standard_error.map(|se| vec![se]),
            unit,
            method: "bump_and_revalue".into(),
            calibration: match calibration {
                BumpCalibration::NotApplicable => CalibrationTreatment::NotApplicable,
                BumpCalibration::Fixed => CalibrationTreatment::FixedCalibration,
                BumpCalibration::Recalibrated => CalibrationTreatment::Recalibrated,
            },
            convention: format!(
                "central difference of complete revaluations on the task's random coordinates; half-width {} in input units; calibration {}{}; {}",
                e.amount,
                calibration.as_str(),
                if calibration == BumpCalibration::Fixed {
                    " (unbumped leverage reused as a function of the absolute level of its calibrated state; paths start at the bumped Spot)"
                } else if kind == MeasureKind::ParallelMarketIvVega
                    && calibration == BumpCalibration::Recalibrated
                {
                    " (every IV of the target's reporting basis moved together; the Dupire target rebuilt on its axes, floor and cap, then recalibrated)"
                } else {
                    ""
                },
                if e.standard_error.is_some() {
                    "standard error of the paired per-sampling-unit differences; excludes bump bias"
                } else {
                    "sampling error not reported: this engine has no per-sampling-unit readout"
                }
            ),
            measure: Some(kind),
            method_detail: Some(e.request.method),
            model_parameter: None,
            correlation: None,
        });
    }
    report.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dividend() -> (PricingTask, PricingTaskResult) {
        let task = crate::parse_pricing_task_json(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/task-v1/stochastic_dividend_bs.golden.json"
            )),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        let result = task.compile().unwrap().evaluate().unwrap();
        (task, result)
    }

    #[test]
    fn display_labels_cannot_change_units_or_estimates() {
        let (task, mut result) = dividend();
        let original = result.report.clone();
        let PricingTaskOutput::StochasticDividendRisk(risk) = &mut result.native_result else {
            panic!("AAD expected")
        };
        assert_eq!(
            &risk.parameter_kinds[..5],
            &[
                RiskParameterKind::Spot,
                RiskParameterKind::Volatility,
                RiskParameterKind::ModelParameter,
                RiskParameterKind::ModelParameter,
                RiskParameterKind::Volatility,
            ]
        );
        // Deliberately misleading display labels: the old classifier swapped units.
        risk.parameter_labels[0] = "volatility".into();
        risk.parameter_labels[1] = "spot".into();
        for (i, label) in risk.parameter_labels.iter_mut().enumerate().skip(2) {
            *label = format!("display_{i}");
        }
        let renamed = build(
            &task,
            task.fingerprint().unwrap(),
            original.plan_fingerprint().into(),
            &result.native_result,
            original.calibration_diagnostics().unwrap().to_vec(),
        )
        .unwrap();
        for (before, after) in original.risks().iter().zip(renamed.risks()) {
            assert_eq!(before.unit, after.unit);
            assert_eq!(before.values, after.values);
            assert_eq!(before.standard_errors, after.standard_errors);
        }
        assert_eq!(renamed.risks()[0].unit, SensitivityUnit::CurrencyPerSpot);
        assert_eq!(
            renamed.risks()[1].unit,
            SensitivityUnit::CurrencyPerUnitVolatility
        );
    }

    #[test]
    fn inconsistent_native_metadata_is_rejected_before_indexing() {
        let (task, result) = dividend();
        for field in 0..4 {
            let mut output = result.native_result.clone();
            let PricingTaskOutput::StochasticDividendRisk(risk) = &mut output else {
                unreachable!()
            };
            match field {
                0 => risk.parameter_kinds = risk.parameter_kinds[1..].into(),
                1 => risk.parameter_labels = risk.parameter_labels[1..].into(),
                2 => risk.derivatives = risk.derivatives[1..].into(),
                _ => risk.standard_errors = risk.standard_errors[1..].into(),
            }
            let error = build(
                &task,
                task.fingerprint().unwrap(),
                result.report.plan_fingerprint().into(),
                &output,
                result.report.calibration_diagnostics().unwrap().to_vec(),
            )
            .unwrap_err();
            assert!(
                matches!(error, WireError::DomainAt { pointer, .. } if pointer == "/report/risks")
            );
        }
    }
}
