//! Select price-only or recalibrated risk once at the API boundary. Neither
//! mode introduces virtual calls, extra records or a different reduction order.
use super::*;

pub(super) trait Evaluation<C: CalibratedModel> {
    const RISK: bool;
    fn observe(
        core: &LsvPricingCore<C>,
        shocks: &[f64],
        path: u64,
        weight: f64,
        output: &mut [f64],
    ) -> Result<(), MonteCarloError>;
    fn target_gradient(
        core: &LsvPricingCore<C>,
        stats: &[DeterministicStatistics],
        units: u64,
    ) -> Result<Option<Vec<f64>>, MonteCarloError>;
}

pub(super) struct PriceOnly;
pub(super) struct Recalibrated;

impl<C: CalibratedModel> Evaluation<C> for PriceOnly {
    const RISK: bool = false;
    fn observe(
        core: &LsvPricingCore<C>,
        shocks: &[f64],
        path: u64,
        weight: f64,
        output: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let initial_f = core.calibration.surface().initial_f();
        let mut states = Vec::new();
        core.path_plan
            .evolve_states(initial_f, shocks, &mut states)?;
        let (price, _) = core.base.lsv_payoff(&states, PathIndex::new(path), false)?;
        output[0] += weight * price;
        Ok(())
    }
    fn target_gradient(
        _: &LsvPricingCore<C>,
        _: &[DeterministicStatistics],
        _: u64,
    ) -> Result<Option<Vec<f64>>, MonteCarloError> {
        Ok(None)
    }
}

impl<C> Evaluation<C> for Recalibrated
where
    C: CalibratedModel + CalibrationReverse<Adjoints = Vec<f64>, Error = LsvError>,
    C::Plan: LeveragePathModel,
{
    const RISK: bool = true;
    fn observe(
        core: &LsvPricingCore<C>,
        shocks: &[f64],
        path: u64,
        weight: f64,
        output: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let initial_f = core.calibration.surface().initial_f();
        let recorded = core.path_plan.evolve_path(initial_f, shocks)?;
        let (price, seeds) =
            core.base
                .lsv_payoff(C::Plan::path_states(&recorded), PathIndex::new(path), true)?;
        output[0] += weight * price;
        if let Some(seeds) = seeds {
            let adjoints = C::Plan::leverage_adjoints(recorded.path_pullback(&seeds)?);
            for (v, &a) in output[1..].iter_mut().zip(adjoints.iter()) {
                *v += weight * a;
            }
        }
        Ok(())
    }
    fn target_gradient(
        core: &LsvPricingCore<C>,
        stats: &[DeterministicStatistics],
        units: u64,
    ) -> Result<Option<Vec<f64>>, MonteCarloError> {
        let leverage = stats[1..]
            .iter()
            .map(|s| s.sum().total() / units as f64)
            .collect::<Vec<_>>();
        Ok(Some(core.target_reverse(&leverage)?))
    }
}
