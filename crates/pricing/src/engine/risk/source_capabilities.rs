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
    /// Pure source inspection: never calibrates, generates paths or allocates an
    /// AAD workspace. Conditional means native compilation/preflight is pending.
    #[must_use]
    pub fn capability(&self, operation: PricingOperation) -> PricingCapability {
        use CapabilityReason as R;
        use CapabilityStatus as S;
        use PricingOperation as O;
        let denied = |reason, detail| {
            PricingCapability::new(operation, S::Unsupported, Some(reason), detail)
        };
        if let O::Aad {
            gamma_relative_bump: Some(b),
        } = operation
        {
            if !b.is_finite() || b <= 0.0 || b >= 1.0 {
                return denied(R::InvalidGammaBump, "Gamma relative bump must lie in (0,1)");
            }
            if !matches!(self, Self::MultiAsset(_)) {
                return denied(
                    R::DedicatedRiskApi,
                    "the common Gamma operation is basket AAD; single-asset Gamma uses its explicit risk request/API",
                );
            }
        }
        if operation == O::Evaluate {
            return PricingCapability::new(
                operation,
                S::Conditional,
                Some(R::CompilationRequired),
                "native product, grid, covariance and resource validation is required",
            );
        }
        let (pathwise, trace, available) = match self {
            Self::SingleAsset(s) => {
                let pathwise =
                    s.product().supports_pathwise_risk() || s.risk().payoff_smoothing().is_some();
                let lsv = matches!(
                    s.model().equity,
                    EquityModelSpec::LocalStochasticVolatility { .. }
                        | EquityModelSpec::RoughLocalStochasticVolatility { .. }
                );
                let hybrid = matches!(s.model().rates, RateModelSpec::HullWhite { .. });
                let trace = match &s.model().equity {
                    EquityModelSpec::LocalStochasticVolatility { particles, .. }
                    | EquityModelSpec::RoughLocalStochasticVolatility { particles, .. } => {
                        particles.retain_reverse_trace()
                    }
                    _ => true,
                };
                let available = match operation {
                    O::Aad { .. } => {
                        hybrid
                            || matches!(
                                s.model().equity,
                                EquityModelSpec::StochasticVolatility { .. }
                            )
                    }
                    O::LocalVarianceRisk => lsv && !hybrid,
                    O::Evaluate => true,
                };
                (pathwise, trace, available)
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
                (
                    s.product().supports_pathwise_risk(),
                    trace,
                    matches!(operation, O::Aad { .. }),
                )
            }
            Self::StochasticDividend(s) => {
                let equity = s.equity();
                let pathwise = equity.product().supports_pathwise_risk()
                    || equity.risk().payoff_smoothing().is_some();
                let (lsv, trace) = match &equity.model().equity {
                    EquityModelSpec::LocalStochasticVolatility { particles, .. } => {
                        (true, particles.retain_reverse_trace())
                    }
                    _ => (false, true),
                };
                // HW's native risk has a separate result type and is adapted too.
                let available = match operation {
                    O::Aad { .. } => !lsv,
                    O::LocalVarianceRisk => lsv,
                    O::Evaluate => true,
                };
                (pathwise, trace, available)
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
            operation,
            S::Conditional,
            Some(R::NativePreflightRequired),
            "native compilation and realized-calibration/risk preflight are required",
        )
    }
}
