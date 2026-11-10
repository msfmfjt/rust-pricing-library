//! Native foreign-equity processes. The first normal block drives the equity.
use crate::bass_lv::{BassMarketIvModel, BassSimulationPlan};
use crate::engine::processes::rough_volatility::{RoughFamilyLsvPlan, RoughVolatilityPathPlan};
use crate::market::LocalVarianceGrid;
use crate::mc::hull_white::{HullWhiteEquityPlan, HybridEquityVolatility};
use crate::mc::lsv::*;
use crate::mc::{LocalVolLogEulerPlan, LocalVolTimeGrid};
use crate::models::{Bergomi2Factor, ModelSpec};
use crate::models::{HullWhite1Factor, HybridCorrelation};
use crate::multi_asset::MultiAssetError as Error;
use crate::{EquityModelSpec, VolatilityFactorSpec};

pub(super) enum EquityPath {
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

fn refined_target(
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
            M::Standard(_) => {
                return Err(Error::Invalid(
                    "constant volatility uses the Black pricing path",
                ));
            }
        })
    }

    // Independent normal count and number of leading Brownian-increment blocks.
    pub(super) fn layout(&self, steps: usize) -> (usize, usize) {
        match self {
            Self::Pure(p) => (p.random_factor_count() * steps, p.random_factor_count()),
            Self::Local(..) | Self::Bass(_) => (steps, 1),
            Self::Bergomi(_) => (2 * steps, 2),
            Self::TwoFactor(_) => (3 * steps, 3),
            Self::RoughBergomi(_) => (3 * steps, 2),
            Self::Rough(p) => (p.random_dimension() as usize, p.brownian_block_count()),
            Self::RoughLsv(p) => (p.random_dimension() as usize, p.brownian_block_count()),
        }
    }

    pub(super) fn states(&self, spot: f64, shocks: &[f64]) -> Result<Vec<f64>, Error> {
        let mut states = Vec::new();
        match self {
            Self::Pure(p) => {
                return Ok(p
                    .evolve_path(spot, shocks)
                    .map_err(Error::numerical)?
                    .into_iter()
                    .map(|s| s.normalized_equity)
                    .collect());
            }
            Self::Local(p, g) => p
                .evolve_states(g, spot, shocks, &mut states)
                .map_err(Error::numerical)?,
            Self::Bass(p) => {
                return Ok(p
                    .path_from_normals(shocks)
                    .map_err(Error::numerical)?
                    .into_iter()
                    .map(|s| s * spot)
                    .collect());
            }
            Self::Bergomi(p) => p
                .evolve_states(spot, shocks, &mut states)
                .map_err(Error::numerical)?,
            Self::TwoFactor(p) => p
                .evolve_states(spot, shocks, &mut states)
                .map_err(Error::numerical)?,
            Self::RoughBergomi(p) => p
                .evolve_states(spot, shocks, &mut states)
                .map_err(Error::numerical)?,
            Self::Rough(p) => {
                return Ok(p
                    .evolve_path(spot, shocks)
                    .map_err(Error::numerical)?
                    .forwards);
            }
            Self::RoughLsv(p) => p
                .evolve_states(spot, shocks, &mut states)
                .map_err(Error::numerical)?,
        }
        Ok(states)
    }
}
