//! Physical foreign-stock observations with Buehler cash dividends.
use super::model_path::refined_target;
use super::rates::CrossCurrencyHullWhite;
use crate::engine::processes::stochastic_dividends::StochasticDividendPathPlan;
use crate::engine::processes::stochastic_dividends::hull_white::{
    StochasticDividendHullWhitePathPlan, StochasticDividendHullWhiteState,
};
use crate::market::EquityForward;
use crate::mc::LocalVolTimeGrid;
use crate::mc::lsv::{calibrate_bergomi_lsv, calibrate_rough_bergomi_lsv};
use crate::models::{BuehlerDividendModel, BuehlerDividendState, ModelSpec};
use crate::multi_asset::MultiAssetError as Error;
use crate::{EquityModelSpec, VolatilityFactorSpec};

pub(super) fn validate(model: &EquityModelSpec, market: &EquityForward) -> Result<(), Error> {
    if !matches!(
        model,
        EquityModelSpec::Standard(ModelSpec::BlackScholes(_) | ModelSpec::Black76(_))
            | EquityModelSpec::StochasticVolatility { .. }
            | EquityModelSpec::LocalStochasticVolatility {
                forward_density: None,
                ..
            }
    ) {
        return Err(Error::Invalid(
            "Buehler cash dividends require Black or a Bergomi volatility or residual-equity LSV model",
        ));
    }
    if market.discrete_dividends().is_some_and(|s| {
        s.events()
            .iter()
            .any(|e| e.beta() != 0. || e.ex_time() <= 0.)
    }) {
        return Err(Error::Invalid(
            "Buehler dividends require strictly future fixed-cash means",
        ));
    }
    Ok(())
}

pub(super) enum DividendPath {
    DeterministicForeignRate(StochasticDividendPathPlan),
    HullWhite {
        model: BuehlerDividendModel,
        volatility: f64,
        path: StochasticDividendHullWhitePathPlan,
        half_variances: Vec<f64>,
        rate_vols: Vec<f64>,
    },
}

impl DividendPath {
    pub(super) fn compile(
        model: &EquityModelSpec,
        dividend: BuehlerDividendModel,
        grid: &LocalVolTimeGrid,
        market: &EquityForward,
        rates: Option<&CrossCurrencyHullWhite>,
    ) -> Result<Self, Error> {
        use EquityModelSpec as M;
        use VolatilityFactorSpec as V;
        let equity_div = dividend.equity_dividend_correlation();
        let vol = match model {
            M::Standard(ModelSpec::BlackScholes(m)) => m.volatility().get(),
            M::Standard(ModelSpec::Black76(m)) => m.volatility().get(),
            M::StochasticVolatility {
                initial_volatility, ..
            } => *initial_volatility,
            M::LocalStochasticVolatility { .. } => 0.,
            _ => return Err(Error::Invalid("unsupported Buehler equity model")),
        };
        if let Some(rates) = rates.filter(|r| !r.foreign.is_deterministic()) {
            if !matches!(model, M::Standard(_)) {
                return Err(Error::Invalid(
                    "Buehler with stochastic foreign rates requires constant equity volatility",
                ));
            }
            let er = rates.correlations[0][3];
            let path = StochasticDividendHullWhitePathPlan::compile_bs(
                market,
                dividend,
                vol,
                &rates.foreign,
                er,
                er * equity_div,
                grid,
            )
            .map_err(Error::numerical)?;
            let half_variances = grid
                .nodes()
                .iter()
                .map(|&t| {
                    rates
                        .foreign
                        .integrated_variance(t)
                        .map(|v| v * 0.5)
                        .map_err(Error::numerical)
                })
                .collect::<Result<_, _>>()?;
            return Ok(Self::HullWhite {
                model: dividend,
                volatility: vol,
                path,
                half_variances,
                rate_vols: grid
                    .nodes()
                    .windows(2)
                    .map(|w| {
                        rates.foreign.volatilities()[rates
                            .foreign
                            .volatility_times()
                            .partition_point(|&t| t <= w[0])
                            - 1]
                    })
                    .collect(),
            });
        }
        let base = StochasticDividendPathPlan::compile(market, dividend, vol, grid)
            .map_err(Error::numerical)?;
        let risky = base.risky_spot();
        let path = match model {
            M::Standard(_) => base,
            M::StochasticVolatility { factor, .. } => match factor {
                V::Bergomi(f) => base
                    .with_bergomi(*f, equity_div * f.correlation())
                    .map_err(Error::numerical)?,
                V::BergomiTwoFactor(f) => base
                    .with_bergomi_two_factor(*f, f.spot_correlations().map(|rho| equity_div * rho))
                    .map_err(Error::numerical)?,
                V::RoughBergomi(f) => base
                    .with_rough_bergomi(*f, equity_div * f.correlation())
                    .map_err(Error::numerical)?,
            },
            M::LocalStochasticVolatility {
                factor,
                target,
                particles,
                ..
            } => {
                let target = refined_target(target.local_variance_grid(), grid)?;
                match factor {
                    V::Bergomi(f) => {
                        let cal = calibrate_bergomi_lsv(&target, *f, risky, particles.clone())
                            .map_err(Error::numerical)?;
                        base.with_bergomi_lsv(
                            *f,
                            equity_div * f.correlation(),
                            cal.surface().clone(),
                        )
                        .map_err(Error::numerical)?
                    }
                    V::BergomiTwoFactor(f) => {
                        let cal = calibrate_bergomi_lsv(&target, *f, risky, particles.clone())
                            .map_err(Error::numerical)?;
                        base.with_bergomi_two_factor_lsv(
                            *f,
                            f.spot_correlations().map(|rho| equity_div * rho),
                            cal.surface().clone(),
                        )
                        .map_err(Error::numerical)?
                    }
                    V::RoughBergomi(f) => {
                        let cal =
                            calibrate_rough_bergomi_lsv(&target, *f, risky, particles.clone())
                                .map_err(Error::numerical)?;
                        base.with_rough_bergomi_lsv(
                            *f,
                            equity_div * f.correlation(),
                            cal.surface().clone(),
                        )
                        .map_err(Error::numerical)?
                    }
                }
            }
            _ => return Err(Error::Invalid("unsupported Buehler equity model")),
        };
        Ok(Self::DeterministicForeignRate(path))
    }

    pub(super) fn layout(&self, steps: usize) -> (usize, usize) {
        match self {
            // The stock and orthogonal dividend increments are Brownian blocks.
            // Exact OU noise or rough near-cell residuals retain native time order.
            Self::DeterministicForeignRate(path) => (path.random_dimension() as usize, 2),
            Self::HullWhite { .. } => (2 * steps, 2),
        }
    }

    pub(super) fn spots(
        &self,
        shocks: &[f64],
        rate_carry: &[f64],
        rate_factors: &[f64],
        mut loadings: Option<&mut Vec<[f64; 3]>>,
    ) -> Result<Vec<(f64, Option<f64>)>, Error> {
        match self {
            Self::DeterministicForeignRate(path) => {
                let steps = path.times().len() - 1;
                let mut step_major = Vec::with_capacity(shocks.len());
                for i in 0..steps {
                    step_major.extend(shocks.chunks_exact(steps).map(|block| block[i]));
                }
                let states = if let Some(out) = loadings {
                    let (states, vols) = path
                        .path_with_equity_volatilities(&step_major)
                        .map_err(Error::numerical)?;
                    for ((node, &s), sigma) in path.nodes().iter().zip(&states).zip(vols) {
                        let [a, b, _] = node.coefficients();
                        out.push(dividend_loadings(
                            path.model(),
                            a * s.equity() * sigma,
                            b * s.dividend(),
                            0.,
                        ));
                    }
                    states
                } else {
                    path.evolve_path(&step_major).map_err(Error::numerical)?
                };
                states
                    .into_iter()
                    .zip(path.nodes())
                    .map(|(s, node)| node.spots(s).map_err(Error::numerical))
                    .collect()
            }
            Self::HullWhite {
                model,
                volatility,
                path,
                half_variances,
                rate_vols,
            } => {
                let steps = path.times().len() - 1;
                let mut factors = BuehlerDividendState::initial();
                let mut spots = Vec::with_capacity(steps + 1);
                for i in 0..=steps {
                    if i > 0 {
                        factors = model
                            .evolve(
                                factors,
                                *volatility,
                                path.times()[i] - path.times()[i - 1],
                                [shocks[i - 1], shocks[steps + i - 1]],
                            )
                            .map_err(Error::numerical)?;
                    }
                    // The native reserve expects the integral of the OU factor.
                    // The shared foreign rate carry includes the curve-fitting shift.
                    let state = StochasticDividendHullWhiteState::new(
                        factors,
                        rate_factors[i],
                        rate_carry[i] - half_variances[i],
                    )
                    .map_err(Error::numerical)?;
                    if let Some(out) = &mut loadings
                        && i < steps
                    {
                        let [e, d, r] = path
                            .spot_factor_loadings(i, state)
                            .map_err(Error::numerical)?;
                        out.push(dividend_loadings(
                            *model,
                            e * volatility,
                            d,
                            r * rate_vols[i],
                        ));
                    }
                    spots.push(path.spots(i, state).map_err(Error::numerical)?);
                }
                Ok(spots)
            }
        }
    }
}

fn dividend_loadings(
    model: BuehlerDividendModel,
    stock: f64,
    dividend_factor: f64,
    rate: f64,
) -> [f64; 3] {
    let rho = model.equity_dividend_correlation();
    let d = dividend_factor * model.dividend_volatility();
    [stock + rho * d, ((1. - rho) * (1. + rho)).sqrt() * d, rate]
}
