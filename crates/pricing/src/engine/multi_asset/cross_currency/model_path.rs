//! Native foreign-equity processes. The first normal block drives the equity.
use super::CrossCurrencyHullWhite;
use super::dividend_path::DividendPath;
use crate::bass_lv::{BassMarketIvModel, BassSimulationPlan};
use crate::engine::processes::rough_volatility::{RoughFamilyLsvPlan, RoughVolatilityPathPlan};
use crate::market::{EquityForward, LocalVarianceGrid};
use crate::mc::hull_white::{HullWhiteEquityPlan, HybridEquityVolatility};
use crate::mc::lsv::*;
use crate::mc::{LocalVolLogEulerPlan, LocalVolTimeGrid};
use crate::models::BuehlerDividendModel;
use crate::models::{Bergomi2Factor, ModelSpec};
use crate::models::{HullWhite1Factor, HybridCorrelation};
use crate::multi_asset::MultiAssetError as Error;
use crate::{EquityModelSpec, VolatilityFactorSpec};

/// Instantaneous physical share-price loadings on the stock Brownian,
/// the orthogonal dividend Brownian and the foreign-rate Brownian.
pub(super) type StockLoadings = [f64; 3];

pub(super) enum StockPath {
    Equity(EquityPath),
    Dividend(DividendPath),
}

impl StockPath {
    pub(super) fn compile(
        model: &EquityModelSpec,
        grid: &LocalVolTimeGrid,
        market: &EquityForward,
        rates: Option<&CrossCurrencyHullWhite>,
        dividend: Option<BuehlerDividendModel>,
    ) -> Result<Self, Error> {
        if let Some(d) = dividend {
            return Ok(Self::Dividend(DividendPath::compile(
                model, d, grid, market, rates,
            )?));
        }
        let equity = match rates {
            Some(r) => EquityPath::compile_for_rates(model, grid, market, r)?,
            None => EquityPath::compile(model, grid, market.spot().get())?,
        };
        Ok(Self::Equity(equity))
    }

    pub(super) fn layout(&self, steps: usize) -> (usize, usize) {
        match self {
            Self::Equity(e) => e.layout(steps),
            Self::Dividend(d) => d.layout(steps),
        }
    }

    /// Post-event spot and optional pre-event spot, both in the foreign currency.
    pub(super) fn spots(
        &self,
        spot: f64,
        shocks: &[f64],
        carry: &[f64],
        log_rate_carry: Option<&[f64]>,
        rate_factors: &[f64],
    ) -> Result<Vec<(f64, Option<f64>)>, Error> {
        self.spots_with_loadings(spot, shocks, carry, log_rate_carry, rate_factors, None)
    }

    pub(super) fn spots_with_loadings(
        &self,
        spot: f64,
        shocks: &[f64],
        carry: &[f64],
        log_rate_carry: Option<&[f64]>,
        rate_factors: &[f64],
        mut loadings: Option<&mut Vec<StockLoadings>>,
    ) -> Result<Vec<(f64, Option<f64>)>, Error> {
        match self {
            Self::Dividend(p) => p.spots(
                shocks,
                log_rate_carry.unwrap_or_default(),
                rate_factors,
                loadings,
            ),
            Self::Equity(p) => {
                let mut variances = Vec::new();
                let states = p.states(
                    spot,
                    shocks,
                    log_rate_carry,
                    loadings.as_ref().map(|_| &mut variances),
                )?;
                let beta = match p {
                    EquityPath::Rough(path) => match path.model() {
                        crate::models::RoughVolatilityModel::RoughSabr(m) => m.beta,
                        _ => 1.,
                    },
                    _ => 1.,
                };
                Ok(states
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let growth = carry[i] * log_rate_carry.map_or(1., |r| r[i].exp());
                        if let Some(out) = &mut loadings
                            && i < variances.len()
                        {
                            let stock = if beta == 1. { s } else { s.powf(beta) };
                            out.push([growth * stock * variances[i].sqrt(), 0., 0.]);
                        }
                        (s * growth, None)
                    })
                    .collect())
            }
        }
    }
}

pub(super) fn simulation_grid(
    model: &EquityModelSpec,
    market: &EquityForward,
    events: Vec<f64>,
    equity_end: f64,
    maximum_step: f64,
) -> Result<LocalVolTimeGrid, Error> {
    joint_grid(
        std::iter::once((model, market, equity_end)),
        events,
        maximum_step,
    )
}

pub(super) fn joint_grid<'a>(
    stocks: impl Iterator<Item = (&'a EquityModelSpec, &'a EquityForward, f64)> + Clone,
    mut events: Vec<f64>,
    maximum_step: f64,
) -> Result<LocalVolTimeGrid, Error> {
    for (model, market, equity_end) in stocks.clone() {
        events.extend(
            model_times(model)
                .iter()
                .copied()
                .filter(|&t| t > 0. && t <= equity_end),
        );
        if let Some(schedule) = market.discrete_dividends() {
            events.extend(
                schedule
                    .events()
                    .iter()
                    .map(|e| e.ex_time())
                    .filter(|&t| t <= equity_end),
            );
        }
    }
    events.push(0.);
    events.sort_by(f64::total_cmp);
    events.dedup();
    let steps = |pair: &[f64]| ((pair[1] - pair[0]) / maximum_step).ceil().max(1.);
    if events.windows(2).map(steps).sum::<f64>() > u32::MAX as f64 {
        return Err(Error::Invalid("equity/FX random dimension is too large"));
    }
    for (model, _, equity_end) in stocks {
        if let EquityModelSpec::RoughVolatility(m)
        | EquityModelSpec::RoughLocalStochasticVolatility { model: m, .. } = model
        {
            let count = events
                .windows(2)
                .take_while(|w| w[1] <= equity_end)
                .map(steps)
                .sum::<f64>();
            if count > RoughVolatilityPathPlan::maximum_time_steps(m) as f64 {
                return Err(Error::Invalid(
                    "rough equity time grid exceeds the model step limit",
                ));
            }
        }
    }
    LocalVolTimeGrid::compile(events, maximum_step).map_err(Error::numerical)
}

pub(super) enum EquityPath {
    Black(f64, Vec<f64>),
    Pure(Box<HullWhiteEquityPlan>),
    Local(LocalVolLogEulerPlan, LocalVarianceGrid),
    Bass(BassSimulationPlan),
    Bergomi(BergomiLsvPlan),
    TwoFactor(BergomiLsvPlan<Bergomi2Factor>),
    RoughBergomi(RoughBergomiLsvPlan),
    Rough(RoughVolatilityPathPlan),
    RoughLsv(RoughFamilyLsvPlan),
}

pub(super) fn model_times(model: &EquityModelSpec) -> &[f64] {
    match model {
        EquityModelSpec::Standard(ModelSpec::BassLocalVolatility(spec)) => {
            spec.surface().maturity_nodes()
        }
        EquityModelSpec::Standard(ModelSpec::LocalVolatility(target))
        | EquityModelSpec::LocalStochasticVolatility { target, .. }
        | EquityModelSpec::RoughLocalStochasticVolatility { target, .. } => {
            target.local_variance_grid().time_nodes()
        }
        _ => &[],
    }
}

pub(super) fn refined_target(
    target: &LocalVarianceGrid,
    grid: &LocalVolTimeGrid,
) -> Result<LocalVarianceGrid, Error> {
    let nodes = target.log_moneyness_nodes();
    let mut values = Vec::new();
    for &t in grid.nodes() {
        for &k in nodes {
            values.push(target.interpolate(t, k)?.value);
        }
    }
    Ok(LocalVarianceGrid::new(
        grid.nodes().to_vec(),
        nodes.to_vec(),
        values,
        target.floor(),
        target.cap(),
    )?)
}

impl EquityPath {
    pub(super) fn compile_for_rates(
        model: &EquityModelSpec,
        grid: &LocalVolTimeGrid,
        market: &crate::market::EquityForward,
        rates: &super::CrossCurrencyHullWhite,
    ) -> Result<Self, Error> {
        use crate::mc::hull_white::{HybridVolatilityFactor, calibrate_hybrid_lsv_with_dividends};
        use crate::models::hull_white_dividends::HullWhiteDividendPlan;
        let EquityModelSpec::LocalStochasticVolatility {
            target,
            factor,
            particles,
            forward_density: Some(density),
        } = model
        else {
            return Self::compile(model, grid, market.spot().get());
        };
        if !crate::engine::risk::hull_white::matching_target_grid(
            target.local_variance_grid(),
            density.grid(),
        ) {
            return Err(Error::Invalid(
                "foreign forward-density and local variance target grids must agree",
            ));
        }
        let equity_rate = rates.correlations[0][3];
        let driver = match factor {
            VolatilityFactorSpec::Bergomi(f) => (*f).into(),
            VolatilityFactorSpec::RoughBergomi(f) => (*f).into(),
            VolatilityFactorSpec::BergomiTwoFactor(f) => HybridVolatilityFactor::BergomiTwoFactor {
                factor: *f,
                second_vol_rate_correlation: equity_rate * f.spot_correlations()[1],
            },
        };
        let correlation = HybridCorrelation::new(
            driver.correlation(),
            equity_rate,
            driver.correlation() * equity_rate,
        )
        .map_err(Error::numerical)?;
        let dividends =
            HullWhiteDividendPlan::new(&rates.foreign, market, density.grid().time_nodes())
                .map_err(Error::numerical)?;
        let surface = calibrate_hybrid_lsv_with_dividends(
            density,
            driver,
            &rates.foreign,
            correlation,
            market.spot().get(),
            particles,
            Some(&dividends),
        )
        .map_err(Error::numerical)?
        .surface;
        Ok(match factor {
            VolatilityFactorSpec::Bergomi(f) => {
                Self::Bergomi(BergomiLsvPlan::new(*f, surface, grid).map_err(Error::numerical)?)
            }
            VolatilityFactorSpec::BergomiTwoFactor(f) => {
                Self::TwoFactor(BergomiLsvPlan::new(*f, surface, grid).map_err(Error::numerical)?)
            }
            VolatilityFactorSpec::RoughBergomi(f) => Self::RoughBergomi(
                RoughBergomiLsvPlan::new(*f, surface, grid).map_err(Error::numerical)?,
            ),
        })
    }

    pub(super) fn compile(
        model: &EquityModelSpec,
        grid: &LocalVolTimeGrid,
        spot: f64,
    ) -> Result<Self, Error> {
        use EquityModelSpec as M;
        use VolatilityFactorSpec as V;
        let e = Error::numerical;
        Ok(match model {
            M::Standard(ModelSpec::LocalVolatility(m)) => Self::Local(
                LocalVolLogEulerPlan::with_constant_forward(grid.clone(), spot)
                    .map_err(Error::numerical)?,
                m.local_variance_grid().clone(),
            ),
            M::Standard(ModelSpec::BassLocalVolatility(m)) => {
                let calibrated = BassMarketIvModel::calibrate(
                    1.0,
                    m.surface().clone(),
                    m.projection_nodes().to_vec(),
                    m.config(),
                    m.projection_config(),
                )
                .map_err(Error::numerical)?;
                let plan = calibrated
                    .model()
                    .compile_simulation(grid.nodes().to_vec())
                    .map_err(Error::numerical)?;
                if plan.time_nodes() != grid.nodes() {
                    return Err(Error::Invalid("Bass and currency grids differ"));
                }
                Self::Bass(plan)
            }
            M::StochasticVolatility {
                initial_volatility,
                factor,
            } => {
                crate::models::BlackScholesSpec::new(*initial_volatility)?;
                let initial_volatility = *initial_volatility;
                let (volatility, spot_vol) = match factor {
                    V::Bergomi(f) => (
                        HybridEquityVolatility::Bergomi {
                            factor: *f,
                            initial_volatility,
                        },
                        f.correlation(),
                    ),
                    V::BergomiTwoFactor(f) => (
                        HybridEquityVolatility::Bergomi2Factor {
                            factor: *f,
                            initial_volatility,
                            second_vol_rate_correlation: 0.,
                        },
                        f.spot_correlations()[0],
                    ),
                    V::RoughBergomi(f) => (
                        HybridEquityVolatility::RoughBergomi {
                            factor: *f,
                            initial_volatility,
                        },
                        f.correlation(),
                    ),
                };
                let rates =
                    HullWhite1Factor::new(0., vec![0.], vec![0.]).map_err(Error::numerical)?;
                let correlation =
                    HybridCorrelation::new(spot_vol, 0., 0.).map_err(Error::numerical)?;
                Self::Pure(Box::new(
                    HullWhiteEquityPlan::new(rates, volatility, correlation, grid)
                        .map_err(Error::numerical)?,
                ))
            }
            M::LocalStochasticVolatility {
                target,
                factor,
                particles,
                ..
            } => {
                let target = refined_target(target.local_variance_grid(), grid)?;
                match factor {
                    V::Bergomi(f) => Self::Bergomi(
                        calibrate_bergomi_lsv(&target, *f, spot, particles.clone())
                            .map_err(e)?
                            .pricing_plan(grid)
                            .map_err(e)?,
                    ),
                    V::BergomiTwoFactor(f) => Self::TwoFactor(
                        calibrate_bergomi_lsv(&target, *f, spot, particles.clone())
                            .map_err(e)?
                            .pricing_plan(grid)
                            .map_err(e)?,
                    ),
                    V::RoughBergomi(f) => Self::RoughBergomi(
                        calibrate_rough_bergomi_lsv(&target, *f, spot, particles.clone())
                            .map_err(e)?
                            .pricing_plan(grid)
                            .map_err(e)?,
                    ),
                }
            }
            M::RoughVolatility(f) => Self::Rough(
                RoughVolatilityPathPlan::compile(f.clone(), grid.nodes().to_vec())
                    .map_err(Error::numerical)?,
            ),
            M::RoughLocalStochasticVolatility {
                model,
                target,
                particles,
            } => {
                let target = refined_target(target.local_variance_grid(), grid)?;
                Self::RoughLsv(
                    calibrate_rough_family_lsv(&target, model.clone(), spot, particles.clone())
                        .map_err(e)?
                        .pricing_plan(grid)
                        .map_err(e)?,
                )
            }
            M::Standard(ModelSpec::BlackScholes(m)) => Self::Black(
                m.volatility().get(),
                grid.nodes().windows(2).map(|w| w[1] - w[0]).collect(),
            ),
            M::Standard(ModelSpec::Black76(m)) => Self::Black(
                m.volatility().get(),
                grid.nodes().windows(2).map(|w| w[1] - w[0]).collect(),
            ),
        })
    }

    // Independent normal count and number of leading Brownian-increment blocks.
    pub(super) fn layout(&self, steps: usize) -> (usize, usize) {
        match self {
            Self::Pure(p) => (p.random_factor_count() * steps, p.random_factor_count()),
            Self::Black(..) | Self::Local(..) | Self::Bass(_) => (steps, 1),
            Self::Bergomi(_) => (2 * steps, 2),
            Self::TwoFactor(_) => (3 * steps, 3),
            Self::RoughBergomi(_) => (3 * steps, 2),
            Self::Rough(p) => (p.random_dimension() as usize, p.brownian_block_count()),
            Self::RoughLsv(p) => (p.random_dimension() as usize, p.brownian_block_count()),
        }
    }

    pub(super) fn states(
        &self,
        spot: f64,
        shocks: &[f64],
        log_carry: Option<&[f64]>,
        mut variances: Option<&mut Vec<f64>>,
    ) -> Result<Vec<f64>, Error> {
        let mut states = Vec::new();
        match self {
            Self::Black(vol, dt) => {
                if let Some(v) = &mut variances {
                    v.resize(dt.len(), vol * vol);
                }
                let mut state = spot;
                states.push(state);
                for (&h, &z) in dt.iter().zip(shocks) {
                    state *= (vol * h.sqrt() * z - 0.5 * vol * vol * h).exp();
                    states.push(state);
                }
            }
            Self::Pure(p) => {
                let path = p.evolve_path(spot, shocks).map_err(Error::numerical)?;
                if let Some(v) = &mut variances {
                    let initial = p
                        .volatility
                        .direct_volatility()
                        .expect("unlevered equity volatility");
                    let nu = p.volatility.log_vol_coefficient();
                    v.extend(
                        path[..path.len() - 1]
                            .iter()
                            .map(|s| initial.powi(2) * (2. * nu * s.volatility_factor).exp()),
                    );
                }
                return Ok(path.into_iter().map(|s| s.normalized_equity).collect());
            }
            Self::Local(p, g) => {
                p.evolve_with_rate_carry(g, spot, shocks, &mut states, log_carry)
                    .map_err(Error::numerical)?;
                if let Some(v) = &mut variances {
                    for (i, &s) in states[..shocks.len()].iter().enumerate() {
                        v.push(
                            g.interpolate(
                                p.time_grid().nodes()[i],
                                (s / spot).ln() + log_carry.map_or(0., |r| r[i]),
                            )?
                            .value,
                        );
                    }
                }
            }
            Self::Bass(p) => {
                return Ok(p
                    .path_with_variances(shocks, variances)
                    .map_err(Error::numerical)?
                    .into_iter()
                    .map(|s| s * spot)
                    .collect());
            }
            Self::Bergomi(p) => p
                .evolve_with_variances(spot, shocks, &mut states, log_carry, variances)
                .map_err(Error::numerical)?,
            Self::TwoFactor(p) => p
                .evolve_with_variances(spot, shocks, &mut states, log_carry, variances)
                .map_err(Error::numerical)?,
            Self::RoughBergomi(p) => p
                .evolve_with_variances(spot, shocks, &mut states, log_carry, variances)
                .map_err(Error::numerical)?,
            Self::Rough(p) => {
                let path = p.evolve_path(spot, shocks).map_err(Error::numerical)?;
                if let Some(v) = &mut variances {
                    v.extend_from_slice(&path.variances[..path.forwards.len() - 1]);
                }
                return Ok(path.forwards);
            }
            Self::RoughLsv(p) => p
                .evolve_with_variances(spot, shocks, &mut states, log_carry, variances)
                .map_err(Error::numerical)?,
        }
        Ok(states)
    }
}
