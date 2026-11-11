//! Physical foreign-stock observations with Buehler cash dividends.
use super::model_path::refined_target;
use super::rates::CrossCurrencyHullWhite;
use crate::engine::processes::stochastic_dividends::StochasticDividendPathPlan;
pub(super) use crate::engine::processes::stochastic_dividends::hull_white::BergomiAsOf;
use crate::engine::processes::stochastic_dividends::hull_white::cash_volatility::{
    CashVolatility, OuStep,
};
use crate::engine::processes::stochastic_dividends::hull_white::{
    StochasticDividendHullWhitePathPlan, StochasticDividendHullWhiteState,
};
use crate::market::EquityForward;
use crate::mc::LocalVolTimeGrid;
use crate::mc::lsv::{calibrate_bergomi_lsv, calibrate_rough_bergomi_lsv};
use crate::models::{BuehlerDividendModel, ModelSpec};
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

pub(super) fn validate_stochastic_rate(
    model: &EquityModelSpec,
) -> Result<Option<CashVolatility>, Error> {
    match model {
        EquityModelSpec::Standard(ModelSpec::BlackScholes(_) | ModelSpec::Black76(_)) => Ok(None),
        EquityModelSpec::StochasticVolatility {
            factor: VolatilityFactorSpec::Bergomi(f),
            ..
        } => Ok(Some(CashVolatility::One(*f))),
        EquityModelSpec::StochasticVolatility {
            factor: VolatilityFactorSpec::BergomiTwoFactor(f),
            ..
        } => Ok(Some(CashVolatility::Two(*f))),
        _ => Err(Error::Invalid(
            "stochastic rates and cash dividends support Black, one-factor Bergomi or two-factor Bergomi residual-equity volatility",
        )),
    }
}

pub(super) struct DividendVolatility {
    factor: CashVolatility,
    // Exact OU transition conditional on the shared stock Brownian increment
    // and the foreign-measure variance centering at the left endpoint.
    steps: Vec<(OuStep, f64)>,
}
pub(super) enum DividendPath {
    DeterministicForeignRate(StochasticDividendPathPlan),
    HullWhite {
        model: BuehlerDividendModel,
        volatility: f64,
        path: StochasticDividendHullWhitePathPlan,
        half_variances: Vec<f64>,
        rate_vols: Vec<f64>,
        bergomi: Option<DividendVolatility>,
        snapshot: BergomiAsOf,
    },
}

impl DividendPath {
    pub(super) fn compile(
        model: &EquityModelSpec,
        dividend: BuehlerDividendModel,
        grid: &LocalVolTimeGrid,
        market: &EquityForward,
        rates: Option<&CrossCurrencyHullWhite>,
        snapshot: Option<BergomiAsOf>,
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
        let zero_rate;
        let foreign = match rates {
            Some(r) if !r.foreign.is_deterministic() || snapshot.is_some() => {
                Some((&r.foreign, r.correlations[0][3]))
            }
            None if snapshot.is_some() => {
                zero_rate = crate::models::HullWhite1Factor::new(0., vec![0.], vec![0.])
                    .map_err(Error::numerical)?;
                Some((&zero_rate, 0.))
            }
            _ => None,
        };
        if let Some((rate, er)) = foreign {
            let snapshot = snapshot.unwrap_or_default();
            let factor = validate_stochastic_rate(model)?;
            let path = StochasticDividendHullWhitePathPlan::compile(
                market,
                dividend,
                vol,
                rate,
                er,
                er * equity_div,
                grid,
                factor.map(|f| (f, snapshot)),
            )
            .map_err(Error::numerical)?;
            let half_variances = grid
                .nodes()
                .iter()
                .map(|&t| {
                    rate.integrated_variance(t)
                        .map(|v| v * 0.5)
                        .map_err(Error::numerical)
                })
                .collect::<Result<_, _>>()?;
            let bergomi = factor
                .map(|factor| {
                    let steps = grid
                        .nodes()
                        .windows(2)
                        .map(|w| {
                            Ok((
                                factor.transition(w[1] - w[0]).map_err(Error::numerical)?,
                                factor
                                    .centering(snapshot.age + w[0])
                                    .map_err(Error::numerical)?,
                            ))
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    Ok::<_, Error>(DividendVolatility { factor, steps })
                })
                .transpose()?;
            return Ok(Self::HullWhite {
                model: dividend,
                volatility: vol,
                bergomi,
                snapshot,
                path,
                half_variances,
                rate_vols: grid
                    .nodes()
                    .windows(2)
                    .map(|w| {
                        rate.volatilities()
                            [rate.volatility_times().partition_point(|&t| t <= w[0]) - 1]
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
            Self::HullWhite { bergomi, .. } => (
                (2 + bergomi.as_ref().map_or(0, |v| v.factor.dimension())) * steps,
                2,
            ),
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
                bergomi,
                snapshot,
            } => {
                let steps = path.times().len() - 1;
                let mut factors = snapshot.factors;
                let mut spots = Vec::with_capacity(steps + 1);
                let mut x = snapshot.x;
                for i in 0..=steps {
                    let sigma = if let Some(v) =
                        bergomi.as_ref().filter(|_| i < steps && *volatility > 0.)
                    {
                        v.factor.volatility(*volatility, x, v.steps[i].1)
                    } else {
                        *volatility
                    };
                    // The native reserve expects the integral of the OU factor.
                    // The shared foreign rate carry includes the curve-fitting shift.
                    let mut state = StochasticDividendHullWhiteState::new(
                        factors,
                        rate_factors[i],
                        rate_carry.get(i).copied().unwrap_or(0.) - half_variances[i],
                    )
                    .map_err(Error::numerical)?;
                    state.volatility_factor = x;
                    if let Some(out) = &mut loadings
                        && i < steps
                    {
                        let [e, d, r, v1, v2] = path
                            .spot_factor_loadings(i, state)
                            .map_err(Error::numerical)?;
                        let mut drivers = dividend_loadings(*model, e * sigma, d, r * rate_vols[i]);
                        if let Some(v) = bergomi {
                            let [shared, orthogonal] = v.factor.diffusion([v1, v2]);
                            drivers[0] += shared;
                            drivers[1] = drivers[1].hypot(orthogonal);
                        }
                        out.push(drivers);
                    }
                    spots.push(path.spots(i, state).map_err(Error::numerical)?);
                    if i < steps {
                        factors = model
                            .evolve(
                                factors,
                                sigma,
                                path.times()[i + 1] - path.times()[i],
                                [shocks[i], shocks[steps + i]],
                            )
                            .map_err(Error::numerical)?;
                        if let Some(v) = bergomi {
                            x = v.steps[i]
                                .0
                                .evolve(x, shocks[i], &shocks[2 * steps + i..], steps);
                        }
                    }
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
