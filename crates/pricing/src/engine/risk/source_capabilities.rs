//! The same operation selection is used by discovery and task compilation.
use crate::multi_asset::MultiAssetBergomiLsvConfig;
use crate::*;

impl PricingSpecification {
    #[must_use]
    pub fn capability(&self, operation: PricingOperation) -> PricingCapability {
        PricingSource::SingleAsset(self.clone()).capability(operation)
    }
}

impl PricingSource {
    /// Capabilities of the three common operations, in a fixed order:
    /// `evaluate`, first-order `aad` and `local_variance_risk`. This is the
    /// discovery call; it performs the same pure inspection as `capability`.
    /// Basket cross Gamma depends on a bump and is queried with `capability`.
    #[must_use]
    pub fn available_operations(&self) -> Vec<PricingCapability> {
        [
            PricingOperation::Evaluate,
            PricingOperation::Aad {
                gamma_relative_bump: None,
            },
            PricingOperation::LocalVarianceRisk,
        ]
        .into_iter()
        .map(|operation| self.capability(operation))
        .collect()
    }

    /// The measures the `Measures` operation does not statically refuse for
    /// this source, each with the methods that apply. Compilation and native
    /// preflight conditions remain, as `measure_capabilities` reports.
    #[must_use]
    pub fn available_measures(&self) -> Vec<AvailableMeasure> {
        let family = self.family();
        MeasureKind::ALL
            .into_iter()
            .filter_map(|measure| {
                let methods = family.measure_methods(measure);
                (!methods.is_empty()).then(|| AvailableMeasure {
                    measure,
                    methods: methods.to_vec(),
                })
            })
            .collect()
    }

    /// One answer per request, in request order. Pure source inspection.
    #[must_use]
    pub fn measure_capabilities(&self, requests: &[MeasureRequest]) -> Vec<MeasureCapability> {
        requests
            .iter()
            .map(|request| self.measure_capability(request))
            .collect()
    }

    /// Whether the payoff admits native pathwise estimators: differentiable,
    /// or smoothed explicitly. A standard source reports its own reason
    /// through `measured_request`.
    fn native_pathwise(&self) -> bool {
        match self {
            Self::SingleAsset(_) if self.family() == SourceFamily::StandardSingleAsset => true,
            Self::SingleAsset(s) => {
                s.product().supports_pathwise_risk() || s.risk().payoff_smoothing().is_some()
            }
            Self::StochasticDividend(s) => {
                let equity = s.equity();
                equity.product().supports_pathwise_risk()
                    || equity.risk().payoff_smoothing().is_some()
            }
            Self::MultiAsset(s) => s.product().supports_pathwise_risk(),
        }
    }

    fn selects_nested_risk(&self) -> bool {
        match self {
            Self::SingleAsset(s) => s.selects_nested_risk(),
            Self::StochasticDividend(s) => s.equity().selects_nested_risk(),
            Self::MultiAsset(_) => false,
        }
    }

    fn measure_capability(&self, request: &MeasureRequest) -> MeasureCapability {
        use CapabilityReason as R;
        use CapabilityStatus as S;
        let denied = |reason, detail: String| {
            MeasureCapability::new(request, S::Unsupported, Some(reason), detail)
        };
        let family = self.family();
        let kind = request.measure.kind();
        let methods = family.measure_methods(kind);
        let names = || {
            methods
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        if methods.is_empty() {
            return denied(
                R::DedicatedRiskApi,
                format!(
                    "{} has no measure adapter for {} sources yet; use the aad or local_variance_risk operation where exposed, or a dedicated native risk API",
                    kind.as_str(),
                    family.as_str()
                ),
            );
        }
        if !methods.contains(&request.method.kind()) {
            return denied(
                R::NoNativeEstimator,
                format!(
                    "{} is not estimated by {} for {} sources; available methods: {}",
                    kind.as_str(),
                    request.method.kind().as_str(),
                    family.as_str(),
                    names()
                ),
            );
        }
        if kind != MeasureKind::Price {
            if self.selects_nested_risk() {
                return denied(
                    R::AmbiguousRiskSelection,
                    "the nested risk request already selects Greeks; a measures operation requires it to select none".into(),
                );
            }
            if let MeasureMethod::BumpAndRevalue { calibration, .. } = request.method {
                return self.revaluation_capability(request, calibration);
            }
            if !self.native_pathwise() {
                return denied(
                    R::DiscontinuousPayoff,
                    "native risk requires a differentiable payoff or explicit smoothing".into(),
                );
            }
            if let Some(calibration) = request.method.calibration() {
                // A calibrated native estimator: the table only offers it on
                // calibrated single-asset sources.
                if family == SourceFamily::StochasticDividendLsv
                    && matches!(kind, MeasureKind::SpotDelta | MeasureKind::SpotGamma)
                {
                    match calibration {
                        BumpCalibration::Recalibrated => {}
                        BumpCalibration::Fixed => {
                            return denied(
                                R::NoNativeEstimator,
                                super::measure_backend::DIVIDEND_LSV_FIXED.into(),
                            );
                        }
                        BumpCalibration::NotApplicable => {
                            return denied(
                                R::BumpCalibrationMismatch,
                                super::measure_backend::DIVIDEND_LSV_CALIBRATION.into(),
                            );
                        }
                    }
                }
                if calibration == BumpCalibration::NotApplicable
                    && matches!(kind, MeasureKind::SpotDelta | MeasureKind::SpotGamma)
                {
                    return denied(
                        R::BumpCalibrationMismatch,
                        "this source is calibrated; state fixed (frozen leverage) or recalibrated (sticky relative local variance)".into(),
                    );
                }
                if kind == MeasureKind::ParallelMarketIvVega
                    && calibration != BumpCalibration::Recalibrated
                {
                    return denied(
                        R::BumpCalibrationMismatch,
                        "a market-IV move reaches LSV prices only through the calibration; state recalibrated".into(),
                    );
                }
                if let Measure::ModelParameters(selection) = &request.measure
                    && let Self::SingleAsset(s) = self
                    && let EquityModelSpec::RoughLocalStochasticVolatility {
                        model, particles, ..
                    } = &s.model().equity
                {
                    if calibration != BumpCalibration::Recalibrated {
                        return denied(
                            R::BumpCalibrationMismatch,
                            super::measure_backend::PARAMETER_CALIBRATION.into(),
                        );
                    }
                    let Some((available, hurst)) = model.native_parameters() else {
                        return denied(
                            R::NoNativeEstimator,
                            format!("{} LSV has no native parameter risk", model.name()),
                        );
                    };
                    if let Some(detail) =
                        super::measure_backend::missing_parameter(selection, &available, hurst)
                    {
                        return denied(R::ModelCombination, detail);
                    }
                    if !particles.retain_reverse_trace() {
                        return denied(
                            R::ReverseTraceRequired,
                            super::measure_backend::PARAMETER_TRACE.into(),
                        );
                    }
                }
                if kind == MeasureKind::ParallelMarketIvVega {
                    if let Err(detail) = self.market_iv_source() {
                        return denied(R::ModelCombination, detail);
                    }
                    if let Self::SingleAsset(s) = self
                        && let EquityModelSpec::RoughLocalStochasticVolatility { particles, .. } =
                            &s.model().equity
                        && !particles.retain_reverse_trace()
                    {
                        return denied(
                            R::ReverseTraceRequired,
                            "native market-IV Vega differentiates through the calibration and requires retain_reverse_trace=true".into(),
                        );
                    }
                }
            } else if let Measure::Correlations(selection) = &request.measure
                && let Self::StochasticDividend(s) = self
            {
                let factors = match &s.equity().model().equity {
                    EquityModelSpec::StochasticVolatility {
                        factor: VolatilityFactorSpec::BergomiTwoFactor(_),
                        ..
                    } => 2,
                    EquityModelSpec::StochasticVolatility { .. } => 1,
                    _ => 0,
                };
                let hull_white =
                    matches!(s.equity().model().rates, RateModelSpec::HullWhite { .. });
                let available =
                    super::stochastic_dividends::dividend_correlations(factors, hull_white);
                if let Some(detail) =
                    super::measure_backend::missing_correlation(selection, &available)
                {
                    return denied(R::ModelCombination, detail);
                }
            } else if let Measure::ModelParameters(selection) = &request.measure
                && let Self::StochasticDividend(s) = self
            {
                use super::stochastic_dividends::DividendVolatility as V;
                let model = s.equity().model();
                let volatility = match (&model.equity, &model.rates) {
                    (_, RateModelSpec::HullWhite { model, .. }) => V::HullWhite {
                        knots: u16::try_from(model.volatilities().len()).unwrap_or(u16::MAX),
                    },
                    (
                        EquityModelSpec::StochasticVolatility {
                            factor: VolatilityFactorSpec::RoughBergomi(_),
                            ..
                        },
                        _,
                    ) => V::Rough,
                    (
                        EquityModelSpec::StochasticVolatility {
                            factor: VolatilityFactorSpec::BergomiTwoFactor(_),
                            ..
                        },
                        _,
                    ) => V::Bergomi { factors: 2 },
                    (EquityModelSpec::StochasticVolatility { .. }, _) => V::Bergomi { factors: 1 },
                    _ => V::Flat,
                };
                let available = super::stochastic_dividends::dividend_parameter_ids(volatility);
                if let Some(detail) =
                    super::measure_backend::missing_parameter(selection, &available, false)
                {
                    return denied(R::ModelCombination, detail);
                }
            } else if let Self::SingleAsset(s) = self
                && family == SourceFamily::DirectRoughVolatility
                && let EquityModelSpec::RoughVolatility(model) = &s.model().equity
            {
                // Direct rough volatility: fixed-model estimators on the
                // compiled plan; no standard risk request is built.
                if let Measure::ModelParameters(selection) = &request.measure {
                    let Some((available, hurst)) = model.native_parameters() else {
                        return denied(
                            R::NoNativeEstimator,
                            format!(
                                "{} {}",
                                model.name(),
                                super::measure_backend::DIRECT_PARAMETERS_MISSING
                            ),
                        );
                    };
                    if let Some(detail) =
                        super::measure_backend::missing_parameter(selection, &available, hurst)
                    {
                        return denied(R::ModelCombination, detail);
                    }
                }
            } else if let Self::SingleAsset(s) = self
                && let Err(error) = s.measured_request(std::slice::from_ref(request))
            {
                let reason = match error {
                    SpecificationValidationError::Request(
                        RequestValidationError::RiskUnsupportedForDiscontinuousProduct,
                    ) => R::DiscontinuousPayoff,
                    _ => R::ModelCombination,
                };
                return denied(reason, error.to_string());
            }
        }
        MeasureCapability::new(
            request,
            S::Conditional,
            Some(R::CompilationRequired),
            "native product, grid, covariance and resource validation is required",
        )
    }

    /// The calibration treatment must be stated and must fit the source, the
    /// bumped sides must be valid sources, and the cost is stated up front.
    fn revaluation_capability(
        &self,
        request: &MeasureRequest,
        calibration: BumpCalibration,
    ) -> MeasureCapability {
        use CapabilityReason as R;
        use CapabilityStatus as S;
        let denied = |reason, detail: String| {
            MeasureCapability::new(request, S::Unsupported, Some(reason), detail)
        };
        let calibrated = self.calibration_count() > 0;
        if calibrated
            && calibration == BumpCalibration::Fixed
            && request.measure.kind() == MeasureKind::ParallelMarketIvVega
        {
            return denied(
                R::BumpCalibrationMismatch,
                "a market-IV move reaches LSV prices only through the calibration; state recalibrated".into(),
            );
        }
        match (calibrated, calibration) {
            (false, BumpCalibration::NotApplicable) | (true, BumpCalibration::Recalibrated) => {}
            (true, BumpCalibration::Fixed) if self.supports_fixed_calibration() => {}
            (true, BumpCalibration::Fixed) => {
                return denied(
                    R::FixedCalibrationUnavailable,
                    "holding the calibration fixed across a bump is available only for single-asset LSV; state recalibrated to recalibrate both sides with the same seeds".into(),
                );
            }
            (true, _) => {
                return denied(
                    R::BumpCalibrationMismatch,
                    "this source is calibrated; state fixed or recalibrated".into(),
                );
            }
            (false, _) => {
                return denied(
                    R::BumpCalibrationMismatch,
                    "this source has no calibration step; state not_applicable".into(),
                );
            }
        }
        let checked = self.request_scenarios(request).and_then(|scenarios| {
            scenarios
                .iter()
                .try_for_each(|scenario| self.scenario_sides(scenario).map(drop))
        });
        if let Err(error) = checked {
            return denied(R::ModelCombination, error.to_string());
        }
        let cost = self
            .bump_cost(std::slice::from_ref(request))
            .expect("scenarios were validated above");
        MeasureCapability::new(
            request,
            S::Conditional,
            Some(R::CompilationRequired),
            format!(
                "bumped sides need {} calibrations and {} evaluations beyond the base price; native compilation of every side is still required",
                cost.calibrations, cost.evaluations
            ),
        )
    }

    fn measures_capability(
        &self,
        operation: PricingOperation,
        set: &MeasureSet,
    ) -> PricingCapability {
        use CapabilityReason as R;
        use CapabilityStatus as S;
        if self.selects_nested_risk() {
            return PricingCapability::new(
                operation,
                S::Unsupported,
                Some(R::AmbiguousRiskSelection),
                "the nested risk request already selects Greeks; a measures operation requires it to select none",
            );
        }
        if let Some(refused) = self
            .measure_capabilities(set.requests())
            .into_iter()
            .find(|c| c.status == S::Unsupported)
        {
            return PricingCapability::new(
                operation,
                S::Unsupported,
                refused.reason,
                refused.detail,
            );
        }
        // Measures are validated one at a time above; the combination can
        // still be refused, for example by a shared smoothing requirement.
        if let Self::SingleAsset(s) = self
            && self.family() == SourceFamily::StandardSingleAsset
            && let Err(error) = s.measured_request(set.requests())
        {
            return PricingCapability::new(
                operation,
                S::Unsupported,
                Some(R::ModelCombination),
                error.to_string(),
            );
        }
        PricingCapability::new(
            operation,
            S::Conditional,
            Some(R::CompilationRequired),
            "native product, grid, covariance and resource validation is required",
        )
    }

    /// Pure source inspection: never calibrates, generates paths or allocates an
    /// AAD workspace. Conditional means native compilation/preflight is pending.
    #[must_use]
    pub fn capability(&self, operation: PricingOperation) -> PricingCapability {
        use CapabilityReason as R;
        use CapabilityStatus as S;
        use PricingOperation as O;
        if let O::Measures(set) = &operation {
            let set = set.clone();
            return self.measures_capability(operation, &set);
        }
        let denied = |reason, detail| {
            PricingCapability::new(operation.clone(), S::Unsupported, Some(reason), detail)
        };
        if let O::Aad {
            gamma_relative_bump: Some(b),
        } = operation
        {
            if !b.is_finite() || b <= 0.0 || b >= 1.0 {
                return denied(R::InvalidGammaBump, "Gamma relative bump must lie in (0,1)");
            }
            if !self.family().exposes(&operation) {
                return denied(
                    R::DedicatedRiskApi,
                    "the common Gamma operation is basket AAD; single-asset Gamma uses its explicit risk request/API",
                );
            }
        }
        if operation == O::Evaluate {
            return PricingCapability::new(
                operation.clone(),
                S::Conditional,
                Some(R::CompilationRequired),
                "native product, grid, covariance and resource validation is required",
            );
        }
        let available = self.family().exposes(&operation);
        let (pathwise, trace) = match self {
            Self::SingleAsset(s) => {
                let pathwise =
                    s.product().supports_pathwise_risk() || s.risk().payoff_smoothing().is_some();
                let trace = match &s.model().equity {
                    EquityModelSpec::LocalStochasticVolatility { particles, .. }
                    | EquityModelSpec::RoughLocalStochasticVolatility { particles, .. } => {
                        particles.retain_reverse_trace()
                    }
                    _ => true,
                };
                (pathwise, trace)
            }
            Self::MultiAsset(s) => {
                let trace = s.model().lsv.iter().flatten().all(|c| match c {
                    MultiAssetBergomiLsvConfig::OneFactor(c) => c.particles.retain_reverse_trace(),
                    MultiAssetBergomiLsvConfig::TwoFactor(c) => c.particles.retain_reverse_trace(),
                    MultiAssetBergomiLsvConfig::Rough(c) => c.particles.retain_reverse_trace(),
                }) && s
                    .model()
                    .local_correlation
                    .as_ref()
                    .is_none_or(|(c, _)| c.particles.retain_reverse_trace());
                (s.product().supports_pathwise_risk(), trace)
            }
            Self::StochasticDividend(s) => {
                let equity = s.equity();
                let pathwise = equity.product().supports_pathwise_risk()
                    || equity.risk().payoff_smoothing().is_some();
                let trace = match &equity.model().equity {
                    EquityModelSpec::LocalStochasticVolatility { particles, .. } => {
                        particles.retain_reverse_trace()
                    }
                    _ => true,
                };
                // HW's native risk has a separate result type and is adapted too.
                (pathwise, trace)
            }
        };
        if !available {
            return denied(
                R::DedicatedRiskApi,
                "this operation is not exposed for this source; use its configured standard risk request or a dedicated native risk API",
            );
        }
        if !pathwise {
            return denied(
                R::DiscontinuousPayoff,
                "pathwise risk requires a differentiable payoff or explicit smoothing",
            );
        }
        if !trace {
            return denied(
                R::ReverseTraceRequired,
                "recalibrated risk requires retain_reverse_trace=true for every calibration",
            );
        }
        PricingCapability::new(
            operation.clone(),
            S::Conditional,
            Some(R::NativePreflightRequired),
            "native compilation and realized-calibration/risk preflight are required",
        )
    }
}
