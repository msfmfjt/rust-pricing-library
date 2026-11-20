//! Bump-and-revalue above the measure backends (ADR 0043, phase 2).
//!
//! A side is a complete price task on a source rebuilt with one input moved.
//! Both sides keep the task's engine configuration and seeds, so they run on
//! the same random coordinates as the base price. Nothing here touches a
//! native estimator.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::PositiveF64;
use crate::market::MarketContext;
use crate::models::{Black76Spec, BlackScholesSpec, ModelSpec};
use crate::*;

/// The input a scenario moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BumpTarget {
    /// Spot of the asset at this position in the source's asset order.
    Spot { asset: usize },
    /// The single volatility of a constant-volatility model.
    Volatility,
    /// Every market IV of an LSV target's reproducing reporting basis, moved
    /// together; the target is rebuilt from the moved surface.
    MarketIv,
    /// One symmetric pair of a basket's Spot correlations, at the assets at
    /// these positions, moved by the same amount at every date. Explicit
    /// driver matrices move with it, since their Spot block must match.
    Correlation {
        first: usize,
        second: usize,
        id: CorrelationId,
    },
}

/// One up/down pair. Requests that move the same input by the same amount
/// share a pair, which is why Delta and Gamma of one size cost two evaluations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scenario {
    pub target: BumpTarget,
    pub underlying: u32,
    /// Half-width in units of the input.
    pub amount: f64,
    pub calibration: BumpCalibration,
}

fn invalid(message: impl ToString) -> SpecificationValidationError {
    SpecificationValidationError::Invalid {
        field: "operation.measures",
        message: message.to_string(),
    }
}

impl PricingSource {
    /// Underlying identifiers and Spots in the source's asset order.
    pub(crate) fn spots(&self) -> Vec<(u32, f64)> {
        let one = |s: &PricingSpecification| {
            let forward = s.market().equity().forward();
            vec![(forward.underlying().get(), forward.spot().get())]
        };
        match self {
            Self::SingleAsset(s) => one(s),
            Self::StochasticDividend(s) => one(s.equity()),
            Self::MultiAsset(s) => s
                .markets()
                .iter()
                .map(|m| (m.forward().underlying().get(), m.forward().spot().get()))
                .collect(),
        }
    }

    /// The volatility a parallel market-IV bump moves, when the source has
    /// exactly one: a Black-Scholes or Black-76 equity.
    pub(crate) fn constant_volatility(&self) -> Option<f64> {
        let Self::SingleAsset(s) = self else {
            return None;
        };
        match &s.model().equity {
            EquityModelSpec::Standard(ModelSpec::BlackScholes(m)) => Some(m.volatility().get()),
            EquityModelSpec::Standard(ModelSpec::Black76(m)) => Some(m.volatility().get()),
            _ => None,
        }
    }

    /// The local-variance target of a deterministic-rate single-asset LSV
    /// source (Bergomi or rough family).
    fn lsv_target(&self) -> Option<&crate::models::LocalVolatilitySpec> {
        let Self::SingleAsset(s) = self else {
            return None;
        };
        if !matches!(s.model().rates, RateModelSpec::Deterministic) {
            return None;
        }
        match &s.model().equity {
            EquityModelSpec::LocalStochasticVolatility {
                target,
                forward_density: None,
                ..
            }
            | EquityModelSpec::RoughLocalStochasticVolatility { target, .. } => Some(target),
            _ => None,
        }
    }

    /// The market-IV surface a parallel market-IV measure moves on an LSV
    /// source: its target's reporting basis, when that basis reproduces the
    /// target exactly. Pure; rebuilds the target grid once to check.
    pub(crate) fn market_iv_source(&self) -> Result<crate::market::MarketIvSurface, String> {
        self.lsv_target()
            .ok_or_else(|| {
                "only deterministic-rate single-asset LSV has a market-IV basis".to_owned()
            })?
            .market_iv_source()
    }

    fn with_parallel_iv_shift(&self, shift: f64) -> Result<Self, SpecificationValidationError> {
        let Self::SingleAsset(s) = self else {
            return Err(invalid("the source has no market-IV basis"));
        };
        let mut equity = s.model().equity.clone();
        match &mut equity {
            EquityModelSpec::LocalStochasticVolatility { target, .. }
            | EquityModelSpec::RoughLocalStochasticVolatility { target, .. } => {
                *target = target.with_parallel_iv_shift(shift).map_err(invalid)?;
            }
            _ => return Err(invalid("the source has no market-IV basis")),
        }
        s.clone()
            .with_model(
                PricingModelSpec::new(equity, s.model().rates.clone()),
                s.numerics().maximum_step,
            )
            .map(Self::SingleAsset)
    }

    /// Whether a bump can reuse this source's calibration instead of
    /// recalibrating: single-asset LSV, with deterministic or Hull-White
    /// rates and with or without stochastic dividends. The leverage is kept
    /// as a function of the absolute level of the state it was calibrated on
    /// (forward, escrowed forward or funded residual equity).
    ///
    /// Not offered for Bass local volatility, whose paths are fixed by the
    /// calibrated marginals alone (a reused calibration ignores Spot), nor
    /// for baskets, whose deterministic-rate marginals are calibrated in
    /// forward-normalized coordinates (reuse would be sticky moneyness).
    #[must_use]
    pub fn supports_fixed_calibration(&self) -> bool {
        let equity = match self {
            Self::SingleAsset(s) => s,
            Self::StochasticDividend(s) => s.equity(),
            Self::MultiAsset(_) => return false,
        };
        matches!(
            equity.model().equity,
            EquityModelSpec::LocalStochasticVolatility { .. }
                | EquityModelSpec::RoughLocalStochasticVolatility { .. }
        )
    }

    /// Calibrations one compilation of this source runs.
    pub(crate) fn calibration_count(&self) -> u64 {
        let single = |s: &PricingSpecification| match &s.model().equity {
            EquityModelSpec::LocalStochasticVolatility { .. }
            | EquityModelSpec::RoughLocalStochasticVolatility { .. }
            | EquityModelSpec::Standard(ModelSpec::BassLocalVolatility(_)) => 1,
            _ => 0,
        };
        match self {
            Self::SingleAsset(s) => single(s),
            Self::StochasticDividend(s) => single(s.equity()),
            Self::MultiAsset(s) => {
                let model = s.model();
                // One calibration per LSV marginal, Hull-White or not, and
                // one for Local Correlation.
                model.lsv.iter().flatten().count() as u64
                    + u64::from(model.local_correlation.is_some())
            }
        }
    }

    fn with_spot(&self, asset: usize, spot: f64) -> Result<Self, SpecificationValidationError> {
        let spot = PositiveF64::new(spot, "bumped_spot").map_err(invalid)?;
        let single = |s: &PricingSpecification| {
            let market = s.market().equity();
            let forward = market.forward().with_spot(spot).map_err(invalid)?;
            PricingSpecification::new(
                s.valuation_date(),
                s.product().clone(),
                MarketContext::Equity(crate::market::EquityMarket::new(market.currency(), forward)),
                s.model().clone(),
                s.engine(),
                s.risk().clone(),
                s.numerics().clone(),
            )
        };
        Ok(match self {
            Self::SingleAsset(s) => Self::SingleAsset(single(s)?),
            Self::StochasticDividend(s) => {
                Self::StochasticDividend(StochasticDividendSpecification::new(
                    single(s.equity())?,
                    s.dividends(),
                    s.volatility_correlations().to_vec(),
                    s.rate_correlation(),
                    s.maximum_step(),
                )?)
            }
            Self::MultiAsset(s) => {
                let mut markets = s.markets().to_vec();
                let market = &markets[asset];
                markets[asset] = crate::market::EquityMarket::new(
                    market.currency(),
                    market.forward().with_spot(spot).map_err(invalid)?,
                );
                Self::MultiAsset(MultiAssetSpecification::new(
                    s.valuation_date(),
                    s.product().clone(),
                    markets,
                    s.model().clone(),
                    s.engine(),
                    s.execution(),
                    s.maximum_step(),
                )?)
            }
        })
    }

    /// The basket with one Spot correlation pair moved by `shift` at every
    /// date of the term structure and of any explicit driver matrices.
    fn with_correlation_shift(
        &self,
        first: usize,
        second: usize,
        shift: f64,
    ) -> Result<Self, SpecificationValidationError> {
        let Self::MultiAsset(s) = self else {
            return Err(invalid("only a basket has asset-pair correlations"));
        };
        let moved = |matrix: &mut [Vec<f64>]| {
            matrix[first][second] += shift;
            matrix[second][first] += shift;
        };
        let mut model = s.model().clone();
        let correlation = &model.correlation;
        let entries = correlation
            .entries()
            .iter()
            .map(|(date, factor)| {
                let n = factor.dimension();
                let mut matrix = factor
                    .raw()
                    .chunks(n)
                    .map(<[f64]>::to_vec)
                    .collect::<Vec<_>>();
                moved(&mut matrix);
                (*date, matrix)
            })
            .collect();
        model.correlation = crate::market::CorrelationTermStructure::new(
            correlation.underlyings().to_vec(),
            entries,
            correlation.tolerances(),
        )
        .map_err(invalid)?;
        if let Some(drivers) = &mut model.driver_correlations {
            drivers.iter_mut().for_each(|m| moved(m));
        }
        MultiAssetSpecification::new(
            s.valuation_date(),
            s.product().clone(),
            s.markets().to_vec(),
            model,
            s.engine(),
            s.execution(),
            s.maximum_step(),
        )
        .map(Self::MultiAsset)
    }

    /// The scenarios of a basket correlation request: one per listed pair.
    fn correlation_scenarios(
        &self,
        selection: &CorrelationSelection,
        size: BumpSize,
        calibration: BumpCalibration,
    ) -> Result<Vec<Scenario>, SpecificationValidationError> {
        let Self::MultiAsset(s) = self else {
            return Err(invalid(
                "correlations are estimated by bump_and_revalue only on baskets",
            ));
        };
        if s.model().local_correlation.is_some() {
            return Err(invalid(
                "a Local Correlation basket mixes two correlation endpoints calibrated to the basket target; it has no single asset-pair correlation to move",
            ));
        }
        let BumpSize::Absolute(amount) = size else {
            return Err(invalid(
                "a correlation can be zero and has no level to scale; state an absolute size",
            ));
        };
        let ids = s
            .model()
            .correlation
            .underlyings()
            .iter()
            .map(|u| u.get())
            .collect::<Vec<_>>();
        selection
            .correlations()
            .iter()
            .map(|&id| {
                let position = |u: u32| ids.iter().position(|&v| v == u);
                let pair = match id {
                    CorrelationId::AssetPair { first, second } => {
                        position(first).zip(position(second))
                    }
                    _ => None,
                };
                let (first, second) = pair.ok_or_else(|| {
                    invalid(format!(
                        "this basket has no correlation {}; it has the asset pairs of underlyings {ids:?}",
                        id.label()
                    ))
                })?;
                Ok(Scenario {
                    target: BumpTarget::Correlation { first, second, id },
                    underlying: ids[first],
                    amount: amount.get(),
                    calibration,
                })
            })
            .collect()
    }

    fn with_constant_volatility(
        &self,
        volatility: f64,
    ) -> Result<Self, SpecificationValidationError> {
        let Self::SingleAsset(s) = self else {
            return Err(invalid("the source has no single volatility"));
        };
        if !volatility.is_finite() || volatility <= 0.0 {
            return Err(invalid("the bumped volatility must stay positive"));
        }
        let equity = match &s.model().equity {
            EquityModelSpec::Standard(ModelSpec::BlackScholes(_)) => {
                ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).map_err(invalid)?)
            }
            EquityModelSpec::Standard(ModelSpec::Black76(_)) => {
                ModelSpec::Black76(Black76Spec::new(volatility).map_err(invalid)?)
            }
            _ => return Err(invalid("the source has no single volatility")),
        };
        s.clone()
            .with_model(
                PricingModelSpec::new(EquityModelSpec::Standard(equity), s.model().rates.clone()),
                s.numerics().maximum_step,
            )
            .map(Self::SingleAsset)
    }

    /// The up and down sources of a scenario. Pure: validates, never compiles.
    pub(crate) fn scenario_sides(
        &self,
        scenario: &Scenario,
    ) -> Result<[Self; 2], SpecificationValidationError> {
        let side = |sign: f64| match scenario.target {
            BumpTarget::Spot { asset } => {
                self.with_spot(asset, self.spots()[asset].1 + sign * scenario.amount)
            }
            BumpTarget::Volatility => self.with_constant_volatility(
                self.constant_volatility()
                    .ok_or_else(|| invalid("the source has no single volatility"))?
                    + sign * scenario.amount,
            ),
            BumpTarget::MarketIv => self.with_parallel_iv_shift(sign * scenario.amount),
            BumpTarget::Correlation { first, second, .. } => {
                self.with_correlation_shift(first, second, sign * scenario.amount)
            }
        };
        Ok([side(1.0)?, side(-1.0)?])
    }

    /// The scenarios one revaluation request needs, in asset order.
    pub(crate) fn request_scenarios(
        &self,
        request: &MeasureRequest,
    ) -> Result<Vec<Scenario>, SpecificationValidationError> {
        let MeasureMethod::BumpAndRevalue { size, calibration } = request.method else {
            return Ok(Vec::new());
        };
        if let Measure::Correlations(selection) = &request.measure {
            return self.correlation_scenarios(selection, size, calibration);
        }
        match request.measure.kind() {
            MeasureKind::SpotDelta | MeasureKind::SpotGamma => Ok(self
                .spots()
                .into_iter()
                .enumerate()
                .map(|(asset, (underlying, spot))| Scenario {
                    target: BumpTarget::Spot { asset },
                    underlying,
                    amount: size.amount(spot),
                    calibration,
                })
                .collect()),
            MeasureKind::ParallelMarketIvVega => {
                if let Some(volatility) = self.constant_volatility() {
                    return Ok(vec![Scenario {
                        target: BumpTarget::Volatility,
                        underlying: self.spots()[0].0,
                        amount: size.amount(volatility),
                        calibration,
                    }]);
                }
                if self.lsv_target().is_none() {
                    return Err(invalid(
                        "a parallel market-IV bump needs a constant-volatility model or an LSV target with a reporting IV basis",
                    ));
                }
                self.market_iv_source().map_err(invalid)?;
                let BumpSize::Absolute(amount) = size else {
                    return Err(invalid(
                        "a parallel market-IV surface bump has no single level to scale; state an absolute size in IV units",
                    ));
                };
                if calibration != BumpCalibration::Recalibrated {
                    return Err(invalid(
                        "a market-IV move reaches LSV prices only through the calibration; state recalibrated",
                    ));
                }
                Ok(vec![Scenario {
                    target: BumpTarget::MarketIv,
                    underlying: self.spots()[0].0,
                    amount: amount.get(),
                    calibration,
                }])
            }
            other => Err(invalid(format!(
                "{} cannot be estimated by bump_and_revalue",
                other.as_str()
            ))),
        }
    }

    /// The distinct scenarios of a request list, in first-use order.
    pub(crate) fn scenarios(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<Vec<Scenario>, SpecificationValidationError> {
        let mut distinct: Vec<Scenario> = Vec::new();
        for request in requests {
            for scenario in self.request_scenarios(request)? {
                if !distinct.contains(&scenario) {
                    distinct.push(scenario);
                }
            }
        }
        Ok(distinct)
    }

    /// What the bump-and-revalue requests in `requests` cost beyond the base
    /// evaluation. Pure source inspection; nothing is compiled or run. Each
    /// scenario is two side evaluations and, for a calibrated source, two
    /// compilations with their calibrations. A Gamma request adds one
    /// recorded price evaluation of the task's own plan for its paired error.
    pub fn bump_cost(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<BumpCost, SpecificationValidationError> {
        let scenarios = self.scenarios(requests)?;
        let recalibrated = scenarios
            .iter()
            .filter(|s| s.calibration == BumpCalibration::Recalibrated)
            .count() as u64;
        let scenarios = scenarios.len() as u64;
        let base_readout = requests
            .iter()
            .any(|r| r.method.is_revaluation() && r.measure.kind() == MeasureKind::SpotGamma);
        Ok(BumpCost {
            calibrations: 2 * recalibrated * self.calibration_count(),
            evaluations: 2 * scenarios + u64::from(base_readout),
        })
    }
}

/// The engine's sampling units from the reductions recorded while it priced:
/// one reduction over every pseudo-random unit, or one reduction per RQMC
/// scramble whose point average is that scramble's unit. Any other shape
/// means the engine's price is not a plain unit average, and gives `None`.
pub(crate) fn units_from_recorded(
    recorded: &[crate::engine::mc::execution::RecordedReduction],
    engine: crate::mc::EngineConfig,
) -> Option<Vec<f64>> {
    use crate::mc::EngineConfig as E;
    let expected = match engine {
        E::PseudoMonteCarlo(e) => e.independent_sampling_units().get(),
        E::RandomizedQuasiMonteCarlo(e) => u64::from(e.scramble_count().get()),
    };
    // Units an engine recorded itself take precedence over the reductions it
    // also ran, for example for risk under a fixed exercise policy.
    let direct: Vec<_> = recorded.iter().filter(|r| r.units).collect();
    if !direct.is_empty() {
        return match direct[..] {
            [only] if only.values.len() as u64 == expected => Some(only.values.clone()),
            _ => None,
        };
    }
    match engine {
        E::PseudoMonteCarlo(e) => match recorded {
            [only] if only.values.len() as u64 == e.independent_sampling_units().get() => {
                Some(only.values.clone())
            }
            _ => None,
        },
        E::RandomizedQuasiMonteCarlo(e) => {
            let points = e.points_per_scramble().get();
            (recorded.len() as u64 == u64::from(e.scramble_count().get())
                && recorded.iter().all(|r| r.values.len() as u64 == points))
            .then(|| recorded.iter().map(|r| r.sum / points as f64).collect())
        }
    }
}

impl PricingSource {
    /// The engine configuration that sets the sampling units.
    pub(crate) fn engine(&self) -> crate::mc::EngineConfig {
        match self {
            Self::SingleAsset(s) => s.engine(),
            Self::StochasticDividend(s) => s.equity().engine(),
            Self::MultiAsset(s) => s.engine(),
        }
    }
}

/// Compiled up and down sides of every scenario of one task.
#[derive(Debug)]
pub(crate) struct CompiledScenarios {
    /// The requests these sides were compiled for.
    pub requests: Vec<MeasureRequest>,
    sides: Vec<(Scenario, [CompiledPricingTask; 2])>,
    evaluations: AtomicU64,
}

/// Central-difference rows of one evaluation, before they enter a report.
pub(crate) struct BumpEstimate {
    pub request: MeasureRequest,
    pub underlying: u32,
    /// The correlation a correlation scenario moved.
    pub correlation: Option<CorrelationId>,
    pub value: f64,
    /// Standard error of the mean of the paired per-unit differences.
    pub standard_error: Option<f64>,
    pub amount: f64,
}

fn paired(values: &[f64]) -> Result<(f64, f64), PricingTaskError> {
    let statistics = crate::mc::DeterministicStatistics::from_ordered_values_two_pass(values);
    let n = values.len() as f64;
    let variance = statistics.moments().sample_variance().ok_or_else(|| {
        PricingTaskError::Validation(invalid(
            "a paired standard error needs at least two sampling units",
        ))
    })?;
    Ok((statistics.sum().total() / n, (variance / n).sqrt()))
}

impl CompiledScenarios {
    pub fn compile(
        base: &CompiledPricingTask,
        requests: Vec<MeasureRequest>,
    ) -> Result<Self, PricingTaskError> {
        let source = base.task().source();
        let mut sides = Vec::new();
        for scenario in source
            .scenarios(&requests)
            .map_err(PricingTaskError::Validation)?
        {
            let [up, down] = source
                .scenario_sides(&scenario)
                .map_err(PricingTaskError::Validation)?;
            let compile = |side: PricingSource| {
                if scenario.calibration == BumpCalibration::Fixed {
                    return base.with_fixed_calibration(side);
                }
                PricingTask::new(side, PricingOperation::Evaluate)
                    .map_err(PricingTaskError::Validation)?
                    .compile()
            };
            sides.push((scenario, [compile(up)?, compile(down)?]));
        }
        Ok(Self {
            requests,
            sides,
            evaluations: AtomicU64::new(0),
        })
    }

    /// Calibrations realized while compiling the sides, counted from their
    /// calibration diagnostics, and side evaluations run so far.
    pub fn realized_cost(&self) -> BumpCost {
        // A fixed side reuses the base calibration and runs none of its own.
        let calibrations = self
            .sides
            .iter()
            .filter(|(scenario, _)| scenario.calibration == BumpCalibration::Recalibrated)
            .flat_map(|(_, sides)| sides)
            .map(|side| {
                side.calibration_blocks()
                    .iter()
                    .filter(|b| b.kind != CalibrationKind::BassSurfaceProjection)
                    .count() as u64
            })
            .sum();
        BumpCost {
            calibrations,
            evaluations: self.evaluations.load(Ordering::Relaxed),
        }
    }

    /// Price both sides of every scenario once, recording their sampling
    /// units, and form the requested differences around `base`, the task's
    /// own price. Where every side's engine shows its units, each estimate is
    /// the mean of the paired per-unit differences with its standard error;
    /// otherwise it is the difference of the side prices with no error.
    /// `base_units` records the task's own plan and is called only for Gamma.
    pub fn evaluate(
        &self,
        source: &PricingSource,
        base: f64,
        base_units: impl FnOnce() -> Result<Option<Vec<f64>>, PricingTaskError>,
    ) -> Result<Vec<BumpEstimate>, PricingTaskError> {
        let mut sides = Vec::with_capacity(self.sides.len());
        for (scenario, [up, down]) in &self.sides {
            let pair = (up.price_and_units()?, down.price_and_units()?);
            self.evaluations.fetch_add(2, Ordering::Relaxed);
            sides.push((*scenario, pair));
        }
        let gamma = self
            .requests
            .iter()
            .any(|r| r.measure.kind() == MeasureKind::SpotGamma);
        let base_units = if gamma {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            base_units()?
        } else {
            None
        };
        let mut estimates = Vec::new();
        for request in &self.requests {
            for scenario in source
                .request_scenarios(request)
                .map_err(PricingTaskError::Validation)?
            {
                let (_, pair) = sides
                    .iter()
                    .find(|(s, _)| *s == scenario)
                    .expect("every scenario of a compiled request has sides");
                let h = scenario.amount;
                let second = request.measure.kind() == MeasureKind::SpotGamma;
                let ((up_price, up_units), (down_price, down_units)) = pair;
                let paired_units = match (up_units, down_units, second) {
                    (Some(u), Some(d), false) if u.len() == d.len() => Some((u, d, None)),
                    (Some(u), Some(d), true) => base_units
                        .as_ref()
                        .filter(|b| b.len() == u.len() && d.len() == u.len())
                        .map(|b| (u, d, Some(b))),
                    _ => None,
                };
                let (value, standard_error) = match paired_units {
                    None => (
                        if second {
                            ((up_price - base) + (down_price - base)) / (h * h)
                        } else {
                            (up_price - down_price) / (2.0 * h)
                        },
                        None,
                    ),
                    Some((up, down, base_units)) => {
                        let differences: Vec<f64> = match base_units {
                            Some(b) => (0..up.len())
                                .map(|i| ((up[i] - b[i]) + (down[i] - b[i])) / (h * h))
                                .collect(),
                            None => (0..up.len())
                                .map(|i| (up[i] - down[i]) / (2.0 * h))
                                .collect(),
                        };
                        let (mean, se) = paired(&differences)?;
                        (mean, Some(se))
                    }
                };
                if !value.is_finite() || standard_error.is_some_and(|se| !se.is_finite()) {
                    return Err(PricingTaskError::Validation(invalid(
                        "the bump is too small or too large for a finite difference",
                    )));
                }
                estimates.push(BumpEstimate {
                    request: request.clone(),
                    underlying: scenario.underlying,
                    correlation: match scenario.target {
                        BumpTarget::Correlation { id, .. } => Some(id),
                        _ => None,
                    },
                    value,
                    standard_error,
                    amount: h,
                });
            }
        }
        Ok(estimates)
    }
}

pub(crate) type SharedScenarios = Arc<CompiledScenarios>;

#[cfg(test)]
mod readout_tests {
    use serde_json::{Value, json};

    use crate::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }
    /// Every saved fixture as an `Evaluate` task, under its own engine and
    /// under RQMC.
    pub(super) fn fixtures() -> Vec<(String, PricingTask)> {
        let rqmc = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":16,
            "scramble_count":8,"master_scramble_seed":5,
            "variance_reduction":{"antithetic":true,"brownian_bridge":false}});
        let mut out = Vec::new();
        for name in [
            "black_scholes",
            "bergomi_lsv",
            "bergomi_hull_white",
            "rough_heston",
        ] {
            let path = root().join(format!("specification-v1/{name}.golden.json"));
            let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for (label, engine) in [("mc", None), ("rqmc", Some(&rqmc))] {
                if let Some(engine) = engine {
                    v["engine"] = engine.clone();
                }
                let s =
                    parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
                        .unwrap();
                out.push((
                    format!("{name} {label}"),
                    PricingTask::new(s, PricingOperation::Evaluate).unwrap(),
                ));
            }
        }
        for name in [
            "multi_asset_bs",
            "multi_asset_joint",
            "stochastic_dividend_bs",
            "stochastic_dividend_hw",
            "stochastic_dividend_lsv",
        ] {
            let path = root().join(format!("task-v1/{name}.golden.json"));
            let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            v["operation"] = json!({"type":"evaluate"});
            for (label, engine) in [("mc", None), ("rqmc", Some(&rqmc))] {
                if let Some(engine) = engine {
                    match v["source"]["type"].as_str().unwrap() {
                        "multi_asset" => v["source"]["engine"] = engine.clone(),
                        _ => v["source"]["equity"]["engine"] = engine.clone(),
                    }
                }
                let t =
                    parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
                        .unwrap();
                out.push((format!("{name} {label}"), t));
            }
        }
        // Branches of the standard engine with their own execution paths.
        let request = |edit: &dyn Fn(&mut Value)| {
            let mut v: Value = serde_json::from_str(include_str!(
                "../../../../../fixtures/v3/pricing_request.golden.json"
            ))
            .unwrap();
            v["engine"]["independent_sampling_units"] = json!(64);
            v["risk"] = json!({"delta":false,"vega":false,
                "smile_dynamics":{"type":"sticky_log_moneyness"}});
            edit(&mut v);
            let r =
                parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
            PricingTask::new(
                PricingSpecification::from_request(
                    r,
                    crate::mc::ExecutionPolicy::new(2, Some(16)).unwrap(),
                ),
                PricingOperation::Evaluate,
            )
            .unwrap()
        };
        out.push((
            "continuous_barrier mc".into(),
            request(&|v| {
                v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,
                    "expiry":"2027-09-04","strike":80.0,"barrier":130.0,"notional":1.0,
                    "side":{"type":"call"},"direction":{"type":"up"},
                    "style":{"type":"knock_out"},"monitoring":{"type":"continuous"},"monitoring_dates":["2027-09-04"],
                    "payment_date":"2027-09-04"});
            }),
        ));
        out.push((
            "local_volatility mc".into(),
            request(&|v| {
                v["model"] = json!({"type":"local_volatility","local_variance_grid":{
                    "time_nodes":[0.0,0.25,0.5,1.0],"log_forward_moneyness_nodes":[-1.0,0.0,1.0],
                    "shape":[4,3],"values":[0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045],
                    "floor":1e-8,"cap":4.0}});
            }),
        ));
        out.push((
            "local_volatility rqmc".into(),
            request(&|v| {
                v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":16,
                    "scramble_count":8,"master_scramble_seed":5,
                    "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
                v["model"] = json!({"type":"local_volatility","local_variance_grid":{
                    "time_nodes":[0.0,0.25,0.5,1.0],"log_forward_moneyness_nodes":[-1.0,0.0,1.0],
                    "shape":[4,3],"values":[0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045],
                    "floor":1e-8,"cap":4.0}});
            }),
        ));
        out.push((
            "black_scholes with greeks mc".into(),
            request(&|v| {
                v["risk"]["delta"] = json!(true);
                v["risk"]["vega"] = json!(true);
            }),
        ));
        out.push((
            "bass mc".into(),
            request(&|v| {
                v["model"] = json!({"type":"bass_local_volatility","parameters":{
                    "maturity_nodes":[0.5,1.0],"log_forward_moneyness_nodes":[-2.0,0.0,2.0],
                    "implied_volatilities":vec![0.2;6],
                    "projection_nodes":(0..=200).map(|i|-2.0+4.0*i as f64/200.0).collect::<Vec<_>>(),
                    "grid_points":201,"grid_width":8.0,"max_iterations":2000,"cdf_tolerance":1e-7,
                    "tail_tolerance":1e-7,"projection_tail_tolerance":1e-7,
                    "projection_mean_tolerance":1e-3,"iv_bump":1e-4}});
            }),
        ));
        let american: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/v3/pricing_request_american.golden.json"
        ))
        .unwrap();
        out.push((
            "american mc".into(),
            PricingTask::new(
                PricingSpecification::from_request(
                    parse_request_json(
                        &serde_json::to_vec(&american).unwrap(),
                        JsonLimits::DEFAULT,
                    )
                    .unwrap(),
                    crate::mc::ExecutionPolicy::new(2, Some(16)).unwrap(),
                ),
                PricingOperation::Evaluate,
            )
            .unwrap(),
        ));
        // Pure stochastic volatility has no fixture: drop the Hull-White rates.
        let PricingSource::SingleAsset(hw) = out
            .iter()
            .find(|(n, _)| n == "bergomi_hull_white mc")
            .unwrap()
            .1
            .source()
            .clone()
        else {
            unreachable!()
        };
        let mut model = hw.model().clone();
        model.rates = RateModelSpec::Deterministic;
        let step = hw.numerics().maximum_step;
        out.push((
            "bergomi_sv mc".into(),
            PricingTask::new(
                hw.with_model(model, step).unwrap(),
                PricingOperation::Evaluate,
            )
            .unwrap(),
        ));
        out
    }

    /// The recorded units are the engine's own: their mean and standard
    /// error are the price and error the engine reports, for every engine
    /// and both samplers.
    #[test]
    fn recorded_units_reproduce_every_engine() {
        let mut checked = 0;
        for (name, task) in fixtures() {
            let compiled = task.compile().unwrap();
            let report = compiled.evaluate().unwrap().report;
            let (price, units) = compiled.price_and_units().unwrap();
            assert_eq!(price.to_bits(), report.value().to_bits(), "{name}");
            let units = units.unwrap_or_else(|| panic!("{name}: no unit readout"));
            assert_eq!(
                units.len() as u64,
                report.independent_sampling_units(),
                "{name}"
            );
            let statistics =
                crate::mc::DeterministicStatistics::from_ordered_values_two_pass(&units);
            let n = units.len() as f64;
            let mean = statistics.sum().total() / n;
            let se = (statistics.moments().sample_variance().unwrap() / n).sqrt();
            assert!(
                (mean - report.value()).abs() <= 1e-13 * report.value().abs(),
                "{name}: mean {mean} vs {}",
                report.value()
            );
            assert!(
                (se - report.standard_error()).abs() <= 1e-12 * report.standard_error(),
                "{name}: error {se} vs {}",
                report.standard_error()
            );
            checked += 1;
        }
        assert_eq!(checked, fixtures().len());
    }
}

/// Phase 2 acceptance (ADR 0043): bump-and-revalue against native Spot
/// Deltas that share the paths and seeds of the bumped sides.
#[cfg(test)]
mod native_acceptance_tests {
    use serde_json::{Value, json};

    use crate::rough_volatility::RoughFamilyLsvPricingPlan;
    use crate::stochastic_volatility::StochasticVolatilityPricingPlan;
    use crate::*;

    fn rqmc_engine() -> Value {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,
            "scramble_count":16,"master_scramble_seed":819,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    }
    fn fixture(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../fixtures/specification-v1/{name}.golden.json"
        ));
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }
    fn parse(v: &Value) -> PricingSpecification {
        parse_specification_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap()
    }
    fn measured(s: &PricingSpecification, request: MeasureRequest) -> (f64, f64) {
        let report = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap()
        .report;
        let row = report.risk(MeasureKind::SpotDelta).unwrap();
        (row.values[0], row.standard_errors.as_ref().unwrap()[0])
    }
    fn spot_bump(calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::SpotDelta,
            MeasureMethod::BumpAndRevalue {
                size: BumpSize::relative(1e-6).unwrap(),
                calibration,
            },
        )
    }
    /// Agreement within the combined errors of the two estimates.
    fn within(label: &str, (a, a_se): (f64, f64), (b, b_se): (f64, f64)) {
        let gap = (a - b).abs();
        println!(
            "NATIVE_ACCEPTANCE {label} bump={a:.12e} +/- {a_se:.3e} \
             native={b:.12e} +/- {b_se:.3e} gap={gap:.3e}"
        );
        assert!(a_se > 0.0 && b_se > 0.0, "{label}");
        assert!(gap <= a_se.hypot(b_se), "{label}: gap {gap}");
    }

    /// Local Volatility on the standard engine, against its pathwise Delta.
    #[test]
    fn local_volatility_bump_agrees_with_native_delta() {
        let mut v = fixture("black_scholes");
        v["model"]["equity"]["model"] = json!({"type":"local_volatility","local_variance_grid":{
            "time_nodes":[0.0,0.25,0.5,1.0],"log_forward_moneyness_nodes":[-1.0,0.0,1.0],
            "shape":[4,3],
            "values":[0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045,0.05,0.04,0.045],
            "floor":1e-8,"cap":4.0}});
        for engine in [None, Some(rqmc_engine())] {
            if let Some(engine) = engine {
                v["engine"] = engine;
            } else {
                v["engine"]["independent_sampling_units"] = json!(4096);
            }
            let s = parse(&v);
            within(
                &format!("local_volatility {}", v["engine"]["type"]),
                measured(&s, spot_bump(BumpCalibration::NotApplicable)),
                measured(&s, MeasureRequest::native(Measure::SpotDelta)),
            );
        }
    }

    /// Bergomi stochastic volatility, against its AAD Spot derivative. AAD
    /// reports an error only under RQMC.
    #[test]
    fn bergomi_sv_bump_agrees_with_native_aad_delta() {
        let mut v = fixture("bergomi_hull_white");
        v["model"]["rates"] = json!({"type":"deterministic"});
        v["engine"] = rqmc_engine();
        let s = parse(&v);
        let EquityModelSpec::StochasticVolatility {
            factor: VolatilityFactorSpec::Bergomi(factor),
            ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        let aad = StochasticVolatilityPricingPlan::compile_bergomi(
            s.carrier_request(),
            *factor,
            s.numerics().maximum_step.unwrap(),
            s.numerics().execution_policy,
        )
        .unwrap()
        .evaluate_aad()
        .unwrap();
        within(
            "bergomi_sv rqmc",
            measured(&s, spot_bump(BumpCalibration::NotApplicable)),
            (aad.delta(), aad.standard_errors.as_ref().unwrap()[0]),
        );
    }

    /// A rough-Heston LSV source on a skewed target. The bergomi_lsv fixture
    /// supplies the market, product and numerics.
    fn rough_lsv(rqmc: bool) -> PricingSpecification {
        let mut v = fixture("bergomi_lsv");
        let values = (0..15)
            .map(|i| 0.04 + 0.006 * f64::from(2 - i % 3) + 0.001 * f64::from(i / 3))
            .collect::<Vec<_>>();
        v["model"]["equity"] = json!({
            "type":"rough_local_stochastic_volatility",
            "target":{"local_variance_grid":{
                "time_nodes":[0.0,0.25,0.5,0.75,1.0],
                "log_forward_moneyness_nodes":[-0.6,0.0,0.6],
                "shape":[5,3],"values":values,"floor":1e-8,"cap":4.0}},
            "model":{"type":"rough_heston","hurst":0.2,"initial_variance":0.04,
                "mean_reversion":0.7,"long_run_variance":0.055,"vol_of_vol":0.15,
                "correlation":-0.6},
            "particles":{"particle_count":256,"seed":429,"log_bandwidth":0.4,
                "minimum_effective_samples":2.0,"retain_reverse_trace":false}});
        if rqmc {
            v["engine"] = rqmc_engine();
        } else {
            v["engine"]["independent_sampling_units"] = json!(2048);
        }
        parse(&v)
    }

    fn native_rough_lsv(s: &PricingSpecification) -> RoughFamilyLsvPricingPlan {
        let EquityModelSpec::RoughLocalStochasticVolatility {
            model, particles, ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        RoughFamilyLsvPricingPlan::compile(
            s.carrier_request(),
            model.clone(),
            particles.clone(),
            s.numerics().execution_policy,
        )
        .unwrap()
    }

    /// Recalibrated bumps hold the relative local-variance target in
    /// log-moneyness, the convention of the native sticky-moneyness Delta;
    /// fixed bumps hold the leverage in the absolute forward, the native
    /// frozen-leverage Delta (no cash dividends, so residual equity is Spot).
    /// Both pairs agree within the paired standard error, and in fact to the
    /// central difference's truncation, since the paths are shared. The bump
    /// is small (1e-4 on a Spot of 100) so that no path's payoff crosses the
    /// strike between the two sides.
    #[test]
    fn lsv_bumps_agree_with_native_recalibrated_and_frozen_deltas() {
        for rqmc in [false, true] {
            let s = rough_lsv(rqmc);
            let plan = native_rough_lsv(&s);
            for (calibration, reference) in [
                (
                    BumpCalibration::Recalibrated,
                    plan.evaluate_sticky_moneyness_delta().unwrap(),
                ),
                (
                    BumpCalibration::Fixed,
                    plan.evaluate_frozen_leverage_delta().unwrap(),
                ),
            ] {
                let label = format!("rough_lsv rqmc={rqmc} {calibration:?}");
                let (delta, se) = measured(&s, spot_bump(calibration));
                within(
                    &label,
                    (delta, se),
                    (reference.delta, reference.delta_standard_error),
                );
                let gap = (delta - reference.delta).abs();
                // The paired error is the native estimator's own error.
                assert!(
                    (se - reference.delta_standard_error).abs()
                        <= 1e-3 * reference.delta_standard_error,
                    "{label}: {se} vs {}",
                    reference.delta_standard_error
                );
                assert!(gap <= 1e-6 * reference.delta.abs(), "{label}: gap {gap}");
            }
        }
    }

    /// Parallel market-IV Vega of rough-family LSV: the recalibrated bump of
    /// every reporting-basis IV against the native particle-adjoint Vega,
    /// which differentiates through the same recalibration. The native error
    /// exists only under RQMC.
    #[test]
    fn lsv_market_iv_bump_agrees_with_native_vega() {
        use crate::engine::risk::measure_backend::rough_family_lsv_market_iv_tests as f;
        let vega = |s: &PricingSpecification, request: MeasureRequest| {
            let report = PricingTask::new(
                s.clone(),
                PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
            )
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
            let row = report.risk(MeasureKind::ParallelMarketIvVega).unwrap();
            (row.values[0], row.standard_errors.as_ref().unwrap()[0])
        };
        let bump = |calibration| {
            MeasureRequest::new(
                Measure::ParallelMarketIvVega,
                MeasureMethod::BumpAndRevalue {
                    size: BumpSize::absolute(1e-5).unwrap(),
                    calibration,
                },
            )
        };
        let s = f::source(true, Some(f::quotes()), true);
        let source = PricingSource::from(s.clone());
        let cost = source
            .bump_cost(&[bump(BumpCalibration::Recalibrated)])
            .unwrap();
        assert_eq!((cost.calibrations, cost.evaluations), (2, 2));
        let (bumped, native) = (
            vega(&s, bump(BumpCalibration::Recalibrated)),
            vega(&s, f::vega(BumpCalibration::Recalibrated)),
        );
        within("rough_family_lsv market_iv rqmc", bumped, native);
        // The paired error of the recalibrated sides is the native error.
        assert!((bumped.1 / native.1 - 1.0).abs() < 1e-3);
        assert!((bumped.0 - native.0).abs() <= 1e-6 * native.0.abs());
        // Bergomi LSV has no native market-IV estimator; its bump runs on the
        // same reproducing basis.
        let mut v = f::json_source(true, Some(f::quotes()), true);
        let target = v["model"]["equity"]["target"].take();
        let particles = v["model"]["equity"]["particles"].take();
        v["model"]["equity"] = json!({"type":"local_stochastic_volatility",
            "target":target,"particles":particles,
            "factor":fixture("bergomi_lsv")["model"]["equity"]["factor"].clone()});
        let bergomi = parse(&v);
        assert_eq!(
            PricingSource::from(bergomi.clone()).family(),
            SourceFamily::LocalStochasticVolatility
        );
        let (value, error) = vega(&bergomi, bump(BumpCalibration::Recalibrated));
        println!("NATIVE_ACCEPTANCE bergomi_lsv market_iv bump={value:.12e} +/- {error:.3e}");
        assert!(value.is_finite() && value > 0.0 && error > 0.0);
    }
}
