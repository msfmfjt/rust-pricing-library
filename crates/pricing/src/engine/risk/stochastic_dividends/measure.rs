//! The stochastic-dividend risks behind the measures facade (ADR 0043,
//! phase 3 steps D1 to D3). Every field is a dedicated API's result unchanged.
use super::hull_white::StochasticDividendHullWhitePricingPlan;
use super::*;
use crate::risk::GammaConfig;
use crate::{CorrelationId, ModelParameterId};

/// A parameter coordinate of the dividend AAD: its identifier, its native
/// label, and whether only the model's extended reverse produces it.
type NativeParameter = (ModelParameterId, String, bool);

/// The Buehler and residual-volatility coordinates of every `evaluate_aad`.
fn basic_parameters() -> Vec<NativeParameter> {
    [
        (ModelParameterId::InitialVolatility, "initial_volatility"),
        (
            ModelParameterId::DividendMeanReversion,
            "dividend_mean_reversion",
        ),
        (ModelParameterId::EquityLinkage, "equity_linkage"),
        (ModelParameterId::DividendVolatility, "dividend_volatility"),
    ]
    .into_iter()
    .map(|(id, name)| (id, name.to_owned(), false))
    .collect()
}

/// The volatility structure of a stochastic-dividend source without a
/// calibration step, which selects its parameter reverse.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DividendVolatility {
    /// Black-Scholes residual equity: `evaluate_aad` only.
    Flat,
    /// One- or two-factor Bergomi: `evaluate_bergomi_aad`.
    Bergomi { factors: u16 },
    /// Rough Bergomi: `evaluate_rough_aad`.
    Rough,
    /// Black-Scholes with Hull-White rates: `evaluate_hull_white_aad`.
    HullWhite { knots: u16 },
}

/// The parameters of a source in the native order: the basic ones, then
/// the extended ones its model's reverse adds.
fn dividend_parameters(volatility: DividendVolatility) -> Vec<NativeParameter> {
    let mut parameters = basic_parameters();
    let mut extend = |id, name: String| parameters.push((id, name, true));
    match volatility {
        DividendVolatility::Flat => {}
        DividendVolatility::Bergomi { factors } => {
            for factor in 0..factors {
                extend(
                    ModelParameterId::FactorMeanReversion { factor },
                    format!("bergomi_mean_reversion[{factor}]"),
                );
            }
            extend(ModelParameterId::VolOfVol, "bergomi_vol_of_vol".into());
            if factors == 2 {
                extend(
                    ModelParameterId::MixingWeight,
                    "bergomi_mixing_weight".into(),
                );
            }
        }
        DividendVolatility::Rough => {
            extend(ModelParameterId::Hurst, "rough_hurst".into());
            extend(ModelParameterId::VolOfVol, "rough_vol_of_vol".into());
        }
        DividendVolatility::HullWhite { knots } => {
            extend(
                ModelParameterId::RateMeanReversion,
                "rate_mean_reversion".into(),
            );
            for knot in 0..knots {
                extend(
                    ModelParameterId::RateVolatility { knot },
                    format!("rate_volatility[{knot}]"),
                );
            }
        }
    }
    parameters
}

/// The parameters a `model_parameters` measure may list on a source.
pub(crate) fn dividend_parameter_ids(volatility: DividendVolatility) -> Vec<ModelParameterId> {
    dividend_parameters(volatility)
        .into_iter()
        .map(|(id, ..)| id)
        .collect()
}

/// The coordinates of `ids` (identified by `name`) at their positions in
/// `labels`; `None` when one is missing.
fn positions<I: Copy>(labels: &[String], ids: &[(I, String)]) -> Option<Box<[usize]>> {
    ids.iter()
        .map(|(_, name)| labels.iter().position(|l| l == name))
        .collect()
}

/// A native parameter risk with the identifier of each parameter the source
/// has, in the native order (the task's rows follow the request). `risk` is
/// `evaluate_aad`, or the model's extended reverse (`evaluate_bergomi_aad`,
/// `evaluate_rough_aad`, `evaluate_hull_white_aad`) when a listed parameter
/// needs it; then `parameters` includes the extended coordinates.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StochasticDividendModelParameterRisk {
    pub parameters: Box<[ModelParameterId]>,
    pub risk: StochasticDividendAadRisk,
    indices: Box<[usize]>,
}

impl StochasticDividendModelParameterRisk {
    fn identify(risk: StochasticDividendAadRisk, parameters: &[NativeParameter]) -> Option<Self> {
        let named = parameters
            .iter()
            .map(|(id, name, _)| (*id, name.clone()))
            .collect::<Vec<_>>();
        let indices = positions(&risk.parameter_labels, &named)?;
        Some(Self {
            parameters: named.into_iter().map(|(id, _)| id).collect(),
            risk,
            indices,
        })
    }
    /// Derivative and sampling error of one parameter.
    #[must_use]
    pub fn coordinate(&self, id: ModelParameterId) -> Option<(f64, f64)> {
        let i = self.indices[self.parameters.iter().position(|&p| p == id)?];
        Some((self.risk.derivatives[i], self.risk.standard_errors[i]))
    }
}

/// The parameters a run covers: the basic ones, plus the extended ones
/// when `extended`.
fn covered(parameters: &[NativeParameter], extended: bool) -> Vec<NativeParameter> {
    parameters
        .iter()
        .filter(|(.., e)| extended || !e)
        .cloned()
        .collect()
}

fn needs_extended(parameters: &[NativeParameter], listed: &[ModelParameterId]) -> bool {
    parameters
        .iter()
        .any(|(id, _, extended)| *extended && listed.contains(id))
}

/// The Brownian correlations of a stochastic-dividend source without a
/// calibration step, in the order `evaluate_correlation_aad` reports them:
/// equity/dividend, then with Hull-White rates the two rate correlations,
/// or with a volatility driver its Spot and dividend correlations per factor
/// and, for two factors, the factor correlation.
pub(crate) fn dividend_correlations(
    volatility_factors: u16,
    hull_white: bool,
) -> Vec<CorrelationId> {
    let mut ids = vec![CorrelationId::EquityDividend];
    if hull_white {
        ids.extend([CorrelationId::EquityRate, CorrelationId::DividendRate]);
        return ids;
    }
    ids.extend((0..volatility_factors).map(|factor| CorrelationId::SpotVolatility { factor }));
    ids.extend((0..volatility_factors).map(|factor| CorrelationId::DividendVolatility { factor }));
    if volatility_factors == 2 {
        ids.push(CorrelationId::VolatilityFactors);
    }
    ids
}

/// The name `evaluate_correlation_aad` gives a correlation.
fn native_correlation_name(id: CorrelationId) -> String {
    match id {
        CorrelationId::EquityDividend => "equity_dividend_correlation".into(),
        CorrelationId::EquityRate => "equity_rate_correlation".into(),
        CorrelationId::DividendRate => "dividend_rate_correlation".into(),
        CorrelationId::SpotVolatility { factor } => {
            format!("spot_volatility_correlation[{factor}]")
        }
        CorrelationId::DividendVolatility { factor } => {
            format!("dividend_volatility_correlation[{factor}]")
        }
        CorrelationId::VolatilityFactors => "volatility_factor_correlation".into(),
    }
}

/// `evaluate_correlation_aad` with the position of each correlation among
/// its coordinates. The other coordinates (Spot, volatility, cash means,
/// curves and any model parameters) stay in `risk`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StochasticDividendCorrelationRisk {
    pub correlations: Box<[CorrelationId]>,
    pub risk: StochasticDividendAadRisk,
    indices: Box<[usize]>,
}

impl StochasticDividendCorrelationRisk {
    /// Identifies the correlation coordinates of `risk` by name; `None` when
    /// a correlation of the source is missing from it.
    fn identify(risk: StochasticDividendAadRisk, correlations: Vec<CorrelationId>) -> Option<Self> {
        let named = correlations
            .iter()
            .map(|&id| (id, native_correlation_name(id)))
            .collect::<Vec<_>>();
        let indices = positions(&risk.parameter_labels, &named)?;
        Some(Self {
            correlations: correlations.into(),
            risk,
            indices,
        })
    }
    /// Derivative and sampling error of one correlation.
    #[must_use]
    pub fn coordinate(&self, id: CorrelationId) -> Option<(f64, f64)> {
        let i = self.indices[self.correlations.iter().position(|&c| c == id)?];
        Some((self.risk.derivatives[i], self.risk.standard_errors[i]))
    }
}

fn unidentified() -> MonteCarloError {
    MonteCarloError::UnsupportedRiskForModel {
        model: "native correlation names do not match the source's correlation list",
    }
}

fn unidentified_parameters() -> MonteCarloError {
    MonteCarloError::UnsupportedRiskForModel {
        model: "native parameter names do not match the source's parameter list",
    }
}

/// The native Spot Delta of one stochastic-dividend source.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StochasticDividendDelta {
    /// `evaluate_aad` of a source without a calibration step; the Delta is
    /// its first coordinate.
    Aad(Box<StochasticDividendAadRisk>),
    /// `evaluate_lsv_spot_risk` of a residual-equity LSV source.
    Lsv(StochasticDividendLsvSpotRisk),
}

impl StochasticDividendDelta {
    #[must_use]
    pub fn price(&self) -> &StochasticDividendPrice {
        match self {
            Self::Aad(r) => &r.price,
            Self::Lsv(r) => &r.price,
        }
    }
    #[must_use]
    pub fn delta(&self) -> f64 {
        match self {
            Self::Aad(r) => r.delta(),
            Self::Lsv(r) => r.delta,
        }
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        match self {
            Self::Aad(r) => r.standard_errors[0],
            Self::Lsv(r) => r.delta_standard_error,
        }
    }
    #[must_use]
    pub fn method(&self) -> &'static str {
        match self {
            Self::Aad(r) => r.method,
            Self::Lsv(r) => r.method,
        }
    }
}

/// The native measures of one stochastic-dividend source, from a
/// `Measures` task. Each risk is its dedicated API's result unchanged.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StochasticDividendMeasureRisk {
    /// The Delta's price, else the plan's.
    pub price: StochasticDividendPrice,
    pub delta: Option<StochasticDividendDelta>,
    /// `evaluate_gamma` (`evaluate_lsv_gamma` on LSV) with the request's bump.
    pub gamma: Option<StochasticDividendGammaRisk>,
    /// `evaluate_correlation_aad`, with its correlations identified.
    pub correlations: Option<StochasticDividendCorrelationRisk>,
    pub model_parameters: Option<StochasticDividendModelParameterRisk>,
}

/// The listed native selection of a `Measures` task on a dividend source.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DividendMeasureSelection<'a> {
    pub delta: bool,
    pub gamma: Option<GammaConfig>,
    pub correlations: bool,
    pub parameters: Option<&'a [ModelParameterId]>,
}

impl DividendMeasureSelection<'_> {
    pub(crate) const fn is_empty(&self) -> bool {
        !self.delta && self.gamma.is_none() && !self.correlations && self.parameters.is_none()
    }
}

/// The price of the first run: Delta, correlations, parameters, else none.
fn first_price(
    delta: Option<&StochasticDividendDelta>,
    correlations: Option<&StochasticDividendCorrelationRisk>,
    parameters: Option<&StochasticDividendModelParameterRisk>,
) -> Option<StochasticDividendPrice> {
    delta
        .map(StochasticDividendDelta::price)
        .or_else(|| correlations.map(|c| &c.risk.price))
        .or_else(|| parameters.map(|p| &p.risk.price))
        .cloned()
}

/// Runs the parameter risk the listed parameters need: the extended reverse
/// when one of them needs it, else `evaluate_aad`, reusing the Delta's run.
fn parameter_risk(
    parameters: &[NativeParameter],
    listed: &[ModelParameterId],
    delta: Option<&StochasticDividendDelta>,
    basic: impl FnOnce() -> Result<StochasticDividendAadRisk, MonteCarloError>,
    extended: impl FnOnce() -> Result<StochasticDividendAadRisk, MonteCarloError>,
) -> Result<StochasticDividendModelParameterRisk, MonteCarloError> {
    let wide = needs_extended(parameters, listed);
    let risk = match (wide, delta) {
        (true, _) => extended()?,
        (false, Some(StochasticDividendDelta::Aad(r))) => (**r).clone(),
        (false, _) => basic()?,
    };
    StochasticDividendModelParameterRisk::identify(risk, &covered(parameters, wide))
        .ok_or_else(unidentified_parameters)
}

impl StochasticDividendPricingPlan {
    /// Whether the residual equity follows a calibrated LSV.
    pub(crate) const fn is_calibrated(&self) -> bool {
        self.lsv.is_some()
    }

    /// Whether the payoff admits the native pathwise estimators.
    pub(crate) const fn supports_spot_risk(&self) -> bool {
        self.risk_supported
    }

    /// The correlations `evaluate_correlation_aad` differentiates; empty on
    /// a residual-equity LSV, whose correlation risks are finite differences.
    pub(crate) fn correlation_ids(&self) -> Vec<CorrelationId> {
        if self.is_calibrated() {
            return Vec::new();
        }
        dividend_correlations(self.path.volatility_factor_count(), false)
    }

    /// The parameters of the native AAD with their native labels; empty on
    /// a residual-equity LSV, whose parameter risks are finite differences.
    fn native_parameters(&self) -> Vec<NativeParameter> {
        if self.is_calibrated() {
            return Vec::new();
        }
        dividend_parameters(if self.path.is_rough() {
            DividendVolatility::Rough
        } else {
            match self.path.volatility_factor_count() {
                0 => DividendVolatility::Flat,
                factors => DividendVolatility::Bergomi { factors },
            }
        })
    }

    /// The parameters a `model_parameters` measure may list.
    pub(crate) fn parameter_ids(&self) -> Vec<ModelParameterId> {
        self.native_parameters()
            .into_iter()
            .map(|(id, ..)| id)
            .collect()
    }

    /// The model's extended parameter reverse, else the basic one.
    fn evaluate_extended_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        if self.path.is_rough() {
            self.evaluate_rough_aad()
        } else if self.path.volatility_factor_count() > 0 {
            self.evaluate_bergomi_aad()
        } else {
            self.evaluate_aad()
        }
    }

    /// Runs the requested dedicated estimators.
    pub(crate) fn evaluate_measure_risk(
        &self,
        selection: DividendMeasureSelection<'_>,
    ) -> Result<StochasticDividendMeasureRisk, MonteCarloError> {
        let DividendMeasureSelection {
            delta,
            gamma,
            correlations,
            parameters,
        } = selection;
        let calibrated = self.is_calibrated();
        let delta = delta
            .then(|| {
                if calibrated {
                    self.evaluate_lsv_spot_risk()
                        .map(StochasticDividendDelta::Lsv)
                } else {
                    self.evaluate_aad()
                        .map(|r| StochasticDividendDelta::Aad(Box::new(r)))
                }
            })
            .transpose()?;
        let gamma = gamma
            .map(|config| {
                if calibrated {
                    self.evaluate_lsv_gamma(config)
                } else {
                    self.evaluate_gamma(config)
                }
            })
            .transpose()?;
        let correlations = correlations
            .then(|| {
                let risk = self.evaluate_correlation_aad()?;
                StochasticDividendCorrelationRisk::identify(risk, self.correlation_ids())
                    .ok_or_else(unidentified)
            })
            .transpose()?;
        let model_parameters = parameters
            .map(|listed| {
                parameter_risk(
                    &self.native_parameters(),
                    listed,
                    delta.as_ref(),
                    || self.evaluate_aad(),
                    || self.evaluate_extended_aad(),
                )
            })
            .transpose()?;
        let price = match first_price(
            delta.as_ref(),
            correlations.as_ref(),
            model_parameters.as_ref(),
        ) {
            Some(price) => price,
            None => self.evaluate()?,
        };
        Ok(StochasticDividendMeasureRisk {
            price,
            delta,
            gamma,
            correlations,
            model_parameters,
        })
    }
}

impl StochasticDividendHullWhitePricingPlan {
    /// The correlations `evaluate_correlation_aad` differentiates.
    pub(crate) fn correlation_ids(&self) -> Vec<CorrelationId> {
        dividend_correlations(0, true)
    }

    /// The parameters of the native AAD with their native labels.
    fn native_parameters(&self) -> Vec<NativeParameter> {
        let knots = u16::try_from(self.rate_volatility_knots()).unwrap_or(u16::MAX);
        dividend_parameters(DividendVolatility::HullWhite { knots })
    }

    /// The parameters a `model_parameters` measure may list.
    pub(crate) fn parameter_ids(&self) -> Vec<ModelParameterId> {
        self.native_parameters()
            .into_iter()
            .map(|(id, ..)| id)
            .collect()
    }

    /// Runs the requested dedicated estimators.
    pub(crate) fn evaluate_measure_risk(
        &self,
        selection: DividendMeasureSelection<'_>,
    ) -> Result<StochasticDividendMeasureRisk, MonteCarloError> {
        let DividendMeasureSelection {
            delta,
            gamma,
            correlations,
            parameters,
        } = selection;
        let delta = delta
            .then(|| {
                self.evaluate_aad()
                    .map(|r| StochasticDividendDelta::Aad(Box::new(r)))
            })
            .transpose()?;
        let gamma = gamma.map(|c| self.evaluate_gamma(c)).transpose()?;
        let correlations = correlations
            .then(|| {
                let risk = self.evaluate_correlation_aad()?;
                StochasticDividendCorrelationRisk::identify(risk, self.correlation_ids())
                    .ok_or_else(unidentified)
            })
            .transpose()?;
        let model_parameters = parameters
            .map(|listed| {
                parameter_risk(
                    &self.native_parameters(),
                    listed,
                    delta.as_ref(),
                    || self.evaluate_aad(),
                    || self.evaluate_hull_white_aad(),
                )
            })
            .transpose()?;
        let price = match first_price(
            delta.as_ref(),
            correlations.as_ref(),
            model_parameters.as_ref(),
        ) {
            Some(price) => price,
            None => self.evaluate()?,
        };
        Ok(StochasticDividendMeasureRisk {
            price,
            delta,
            gamma,
            correlations,
            model_parameters,
        })
    }
}
