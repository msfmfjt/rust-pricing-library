//! compile / local vol implementation.

use crate::MonteCarloError;
use crate::core::{Date, DayCountConvention};
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::LocalVolRuntimeInputs;
use crate::engine::plan::simulation::LocalVolVegaKtRuntime;
use crate::engine::plan::simulation::ReportingIvSurface;
use crate::market::{AffineDividendTransform, EquityForward};
use crate::mc::{LocalVolDividendCheckpointSchedule, LocalVolLogEulerPlan, LocalVolTimeGrid};
use crate::models::LocalVolatilityReportingBasis;
use crate::risk::{ReportingIvBasis, VegaKtConfig, analytic_call_density_rows_from_surface};

pub(in crate::engine) fn compile_local_vol_runtime(
    inputs: LocalVolRuntimeInputs<'_>,
) -> Result<LocalVolRuntime, MonteCarloError> {
    let LocalVolRuntimeInputs {
        grid,
        reporting_iv_basis,
        vega_kt,
        market_forward,
        valuation_date,
        expiry_time,
        event_times,
    } = inputs;
    let first_time = grid.time_nodes()[0];
    let last_time = grid.time_nodes()[grid.time_nodes().len() - 1];
    if first_time.to_bits() != 0.0_f64.to_bits() || last_time.to_bits() != expiry_time.to_bits() {
        return Err(MonteCarloError::InvalidLocalVolatilityTimeGrid {
            expiry_bits: expiry_time.to_bits(),
            first_bits: first_time.to_bits(),
            last_bits: last_time.to_bits(),
        });
    }
    let maximum_step = grid
        .time_nodes()
        .windows(2)
        .map(|times| times[1] - times[0])
        .fold(0.0_f64, f64::max);
    let mut required_times = grid.time_nodes().to_vec();
    required_times.extend_from_slice(event_times);
    let time_grid = if let Some(dividends) = market_forward.discrete_dividends() {
        LocalVolTimeGrid::compile_with_dividends(required_times, dividends, maximum_step)?
    } else {
        LocalVolTimeGrid::compile(required_times, maximum_step)?
    };
    let dividends = market_forward.discrete_dividends().cloned();
    let dividend_timeline = dividends
        .as_ref()
        .map(AffineDividendTransform::event_timeline)
        .transpose()?
        .unwrap_or_default();
    let mut forwards = Vec::with_capacity(time_grid.nodes().len());
    let mut node_affine_coordinates = Vec::with_capacity(time_grid.nodes().len());
    let mut node_pre_dividend_coordinates = Vec::with_capacity(time_grid.nodes().len());
    for time in time_grid.nodes().iter().copied() {
        let evaluation = market_forward.evaluate(time)?;
        forwards.push(evaluation.forward);
        node_affine_coordinates.push(evaluation.affine_coordinate);
        node_pre_dividend_coordinates.push(
            dividend_timeline
                .iter()
                .find(|entry| entry.ex_time().to_bits() == time.to_bits())
                .map(|entry| entry.before()),
        );
    }
    let plan = LocalVolLogEulerPlan::new(time_grid, forwards)?;
    let dividend_schedule = dividends
        .as_ref()
        .map(|dividends| LocalVolDividendCheckpointSchedule::compile(plan.time_grid(), dividends))
        .transpose()?;
    let vega_kt = vega_kt
        .map(|config| {
            compile_local_vol_vega_kt_runtime(
                config,
                reporting_iv_basis,
                market_forward,
                valuation_date,
            )
        })
        .transpose()?;
    Ok(LocalVolRuntime {
        grid,
        plan,
        node_affine_coordinates: node_affine_coordinates.into_boxed_slice(),
        node_pre_dividend_coordinates: node_pre_dividend_coordinates.into_boxed_slice(),
        dividends,
        dividend_schedule,
        vega_kt,
    })
}

pub(in crate::engine) fn compile_local_vol_vega_kt_runtime(
    config: &VegaKtConfig,
    reporting_iv_basis: Option<&LocalVolatilityReportingBasis>,
    market_forward: &EquityForward,
    valuation_date: Date,
) -> Result<LocalVolVegaKtRuntime, MonteCarloError> {
    let reporting_iv_basis =
        reporting_iv_basis.ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
    let maturity_nodes = config
        .maturity_nodes()
        .iter()
        .map(|maturity| DayCountConvention::Act365F.year_fraction(valuation_date, *maturity))
        .collect::<Vec<_>>();
    let log_moneyness_nodes = config
        .log_forward_moneyness_nodes()
        .iter()
        .map(|node| node.get())
        .collect::<Vec<_>>();
    if reporting_iv_basis.maturity_nodes() != maturity_nodes
        || reporting_iv_basis.log_forward_moneyness_nodes() != log_moneyness_nodes
    {
        return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
    }
    let basis = ReportingIvBasis::new(
        maturity_nodes.clone(),
        log_moneyness_nodes.clone(),
        reporting_iv_basis.implied_volatilities().to_vec(),
    )?;
    let surface = ReportingIvSurface::new(reporting_iv_basis);
    let mut forwards = Vec::with_capacity(maturity_nodes.len());
    for maturity in maturity_nodes.iter().copied() {
        forwards.push(market_forward.evaluate(maturity)?.forward);
    }
    let density_rows = analytic_call_density_rows_from_surface(
        &surface,
        &maturity_nodes,
        &forwards,
        log_moneyness_nodes,
        config.relative_density_threshold().get(),
    )?;
    Ok(LocalVolVegaKtRuntime {
        basis,
        density_rows,
        full_bucket_covariance: config.full_bucket_covariance(),
    })
}
