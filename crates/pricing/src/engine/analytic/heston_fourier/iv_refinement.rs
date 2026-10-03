//! Finite-grid checks and bounded staged recalibration, not a continuum error estimator.
use super::*;
use crate::rough_volatility::HestonFourierPlan;

/// Tolerances are absolute annualized IV, independent of every quote's iv_scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonIvRefinementOptions {
    pub fit_tolerance: f64,
    pub grid_tolerance: f64,
    /// Maximum calibration stages. Each uses the previous joint probe as its grid.
    /// Optimizer budgets passed to calibrate_refined apply PER STAGE.
    pub max_stages: usize,
}
impl Default for HestonIvRefinementOptions {
    fn default() -> Self {
        Self {
            fit_tolerance: 5e-4,
            grid_tolerance: 1e-4,
            max_stages: 2,
        }
    }
}
impl HestonIvRefinementOptions {
    pub fn validate(self) -> Result<(), HestonCalibrationError> {
        if [self.fit_tolerance, self.grid_tolerance]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
            || !(1..=3).contains(&self.max_stages)
        {
            return Err(HestonCalibrationError::InvalidInput(
                "positive finite IV tolerances and 1..=3 stages required",
            ));
        }
        Ok(())
    }
}

/// Column order in every report: base, time-only, frequency-only, cutoff-only, joint.
/// Each probe prices the SAME model. No refitting takes place inside a probe.
#[derive(Clone, Debug, PartialEq)]
pub struct HestonIvGridValidation {
    pub configurations: [HestonFourierConfig; 5],
    /// Quote-major matrix, with the five grid columns in the documented order.
    pub model_implied_volatilities: Vec<[f64; 5]>,
    /// Unscaled model IV minus target IV, on all five grids.
    pub iv_residuals: Vec<[f64; 5]>,
    /// Each probe IV minus base IV. Columns: time, frequency, cutoff, joint.
    pub iv_differences: Vec<[f64; 4]>,
    pub fit_tolerance: f64,
    pub grid_tolerance: f64,
    pub max_abs_iv_residual: f64,
    pub max_abs_iv_difference: f64,
    pub fit_within_tolerance: bool,
    pub grid_stable: bool,
    /// Both finite-grid conditions hold. NOT a total-error or global-fit certificate.
    pub accepted: bool,
    pub plan_compilations: usize,
}
#[derive(Clone, Debug)]
pub struct HestonIvRefinementStage {
    pub initial_parameters: Vec<f64>,
    pub calibration: HestonIvCalibrationResult,
    pub validation: HestonIvGridValidation,
}
#[derive(Clone, Debug)]
pub struct HestonIvRefinementResult {
    pub stages: Vec<HestonIvRefinementStage>,
    /// The final stage passed the finite-grid policy; distinct from optimizer fit_achieved.
    pub accepted: bool,
    /// Sum of the parent calibration evaluation counts, including final evaluations.
    pub optimizer_evaluations: usize,
    /// All pricing-only plan compilations in validation (five per distinct maturity/stage).
    pub validation_plan_compilations: usize,
}

fn probe_grids(c: HestonFourierConfig) -> Result<[HestonFourierConfig; 5], FourierError> {
    let (n, m, u) = (c.time_steps(), c.integration_intervals(), c.cutoff());
    // The config constructor bounds n,m, so the small integer multipliers cannot overflow.
    Ok([
        c,
        HestonFourierConfig::new(2 * n, m, u)?,
        HestonFourierConfig::new(n, 2 * m, u)?,
        HestonFourierConfig::new(n, 2 * m, 2. * u)?,
        HestonFourierConfig::new(2 * n, 4 * m, 2. * u)?,
    ])
}
fn work(model: &RoughVolatilityModel, c: HestonFourierConfig) -> u128 {
    let n = c.time_steps() as u128;
    let kernel = match model {
        RoughVolatilityModel::RoughHeston(_) => n * n,
        RoughVolatilityModel::LiftedHeston(m) => n * m.weights().len() as u128,
        _ => unreachable!("validated calibration model"),
    };
    kernel * (c.integration_intervals() as u128 + 1)
}
impl HestonIvCalibrationProblem {
    /// Price-only validation on five explicit grids. Reuses one plan per maturity/grid.
    /// Any invalid price/IV is an error; no quote or probe is silently discarded.
    /// policy.max_stages does not change this single-point validation.
    pub fn validate_grid(
        &self,
        parameters: &[f64],
        policy: HestonIvRefinementOptions,
    ) -> Result<HestonIvGridValidation, HestonCalibrationError> {
        policy.validate()?;
        let model = self.price_problem.model_at(parameters)?;
        validate_quote_grid(&model, &self.quotes, self.fourier_config(), policy)
    }

    /// Evaluate disjoint holdout IV sites at frozen parameters on the same five
    /// grids as validate_grid. NEVER calibrates, chooses parameters, or changes
    /// the original problem. max_stages is ignored for this single validation.
    ///
    /// Reject overlap with fitting sites and duplicate holdout sites, comparing
    /// (T, log(K/F)) within 64*EPSILON*max(1,abs(a),abs(b)) per coordinate.
    /// Changing side, currency scale, discount, target or iv_scale does not create
    /// a new site. The caller must freeze this panel before inspecting its output;
    /// disjointness alone does not establish statistical independence or prevent
    /// selection leakage outside this problem.
    pub fn validate_holdout(
        &self,
        parameters: &[f64],
        quotes: &[HestonIvCalibrationQuote],
        policy: HestonIvRefinementOptions,
    ) -> Result<HestonIvGridValidation, HestonCalibrationError> {
        policy.validate()?;
        if quotes.is_empty() || quotes.len() > 4096 {
            return Err(HestonCalibrationError::InvalidInput(
                "need 1..=4096 holdout IV quotes",
            ));
        }
        let model = self.price_problem.model_at(parameters)?;
        for (i, q) in quotes.iter().enumerate() {
            q.to_price_quote()?;
            if self.quotes.iter().any(|train| same_iv_site(q, train)) {
                return Err(HestonCalibrationError::InvalidInput(
                    "holdout IV site overlaps a fitting site",
                ));
            }
            if quotes[..i].iter().any(|previous| same_iv_site(q, previous)) {
                return Err(HestonCalibrationError::InvalidInput(
                    "duplicate holdout IV site",
                ));
            }
        }
        validate_quote_grid(&model, quotes, self.fourier_config(), policy)
    }

    /// Calibrate, freeze parameters, check all probes, and if necessary warm-start
    /// the next stage on the joint grid. All stages use identical quotes, bounds,
    /// objective scales and optimizer options. The starting problem is never mutated.
    /// A failed numerical evaluation aborts with an error, never an accepted partial report.
    pub fn calibrate_refined(
        &self,
        options: LeastSquaresOptions,
        policy: HestonIvRefinementOptions,
    ) -> Result<HestonIvRefinementResult, HestonCalibrationError> {
        policy.validate()?;
        options
            .validate()
            .map_err(HestonCalibrationError::InvalidInput)?;
        if options.max_evaluations < 2 {
            return Err(HestonCalibrationError::InvalidInput(
                "reserve at least two evaluations per stage",
            ));
        }
        // Validate the ENTIRE requested ladder and aggregate worst-case work before solving.
        // An impossible later grid is not silently downgraded to a shorter schedule.
        let initial_model = self.price_problem.model_at(&self.initial_parameters())?;
        let scalar = self
            .variables()
            .iter()
            .any(|v| v.parameter != super::super::HestonCalibrationParameter::Hurst);
        let hurst = self
            .variables()
            .iter()
            .any(|v| v.parameter == super::super::HestonCalibrationParameter::Hurst);
        let multiplier = 1 + 6 * u128::from(scalar) + 2 * u128::from(hurst);
        let mut config = self.fourier_config();
        let mut budget = 0u128;
        for _ in 0..policy.max_stages {
            let probes = probe_grids(config)?;
            if probes
                .iter()
                .any(|&c| work(&initial_model, c) > 8_000_000_000)
            {
                return Err(HestonCalibrationError::InvalidInput(
                    "IV refinement per-plan work limit",
                ));
            }
            budget += work(&initial_model, config) * multiplier * options.max_evaluations as u128
                + probes
                    .iter()
                    .map(|&c| work(&initial_model, c))
                    .sum::<u128>();
            config = probes[4];
        }
        if budget * self.maturity_count() as u128 > 1_000_000_000_000 {
            return Err(HestonCalibrationError::InvalidInput(
                "IV refinement aggregate work limit",
            ));
        }
        let mut problem = self.clone();
        let mut out = HestonIvRefinementResult {
            stages: Vec::with_capacity(policy.max_stages),
            accepted: false,
            optimizer_evaluations: 0,
            validation_plan_compilations: 0,
        };
        for stage in 0..policy.max_stages {
            let initial_parameters = problem.initial_parameters();
            let calibration = problem.calibrate(options)?;
            let validation = problem.validate_grid(&calibration.optimizer.parameters, policy)?;
            let next_config = validation.configurations[4];
            let next_model = calibration.model.clone();
            out.accepted = validation.accepted;
            out.optimizer_evaluations += calibration.evaluations;
            out.validation_plan_compilations += validation.plan_compilations;
            out.stages.push(HestonIvRefinementStage {
                initial_parameters,
                calibration,
                validation,
            });
            if out.accepted {
                break;
            }
            if stage + 1 < policy.max_stages {
                problem = Self::new(
                    next_model,
                    self.quotes.clone(),
                    self.variables().to_vec(),
                    next_config,
                )?;
            }
        }
        Ok(out)
    }
}

/// Evaluate arbitrary already-validated quotes, without building an optimizer or Jacobian.
fn validate_quote_grid(
    model: &RoughVolatilityModel,
    quotes: &[HestonIvCalibrationQuote],
    fourier: HestonFourierConfig,
    policy: HestonIvRefinementOptions,
) -> Result<HestonIvGridValidation, HestonCalibrationError> {
    let mut maturities = Vec::new();
    for q in quotes {
        if !maturities.contains(&q.maturity) {
            maturities.push(q.maturity);
        }
    }
    let maturity_count = maturities.len();
    if maturity_count > 64 {
        return Err(HestonCalibrationError::InvalidInput(
            "at most 64 holdout maturities",
        ));
    }
    let configs = probe_grids(fourier)?;
    let cost: u128 = configs.iter().map(|&c| work(model, c)).sum();
    if cost * maturity_count as u128 > 1_000_000_000_000
        || configs.iter().any(|&c| work(model, c) > 8_000_000_000)
    {
        return Err(HestonCalibrationError::InvalidInput(
            "IV validation work limit",
        ));
    }
    let mut ivs = vec![[0.; 5]; quotes.len()];
    // Preserve the original order, including duplicate strikes and repeated maturities.
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for (i, q) in quotes.iter().enumerate() {
        if let Some((_, indices)) = groups.iter_mut().find(|(t, _)| *t == q.maturity) {
            indices.push(i);
        } else {
            groups.push((q.maturity, vec![i]));
        }
    }
    for (column, &config) in configs.iter().enumerate() {
        for (t, indices) in &groups {
            let plan = HestonFourierPlan::compile(model.clone(), *t, config)?;
            for &i in indices {
                let q = quotes[i];
                let p = plan.price(q.forward, q.strike, q.discount)?;
                let price = if q.is_call { p.call } else { p.put };
                let b =
                    BlackCoordinates::new(q.forward, q.strike, q.discount, q.maturity, q.is_call)?;
                ivs[i][column] = b.invert(price)?.0;
            }
        }
    }
    let residuals: Vec<[f64; 5]> = ivs
        .iter()
        .zip(quotes)
        .map(|(row, q)| row.map(|v| v - q.target_volatility))
        .collect();
    let differences: Vec<[f64; 4]> = ivs
        .iter()
        .map(|row| std::array::from_fn(|i| row[i + 1] - row[0]))
        .collect();
    let max_residual = residuals
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0., f64::max);
    let max_difference = differences
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0., f64::max);
    let fit = max_residual <= policy.fit_tolerance;
    let stable = max_difference <= policy.grid_tolerance;
    Ok(HestonIvGridValidation {
        configurations: configs,
        model_implied_volatilities: ivs,
        iv_residuals: residuals,
        iv_differences: differences,
        fit_tolerance: policy.fit_tolerance,
        grid_tolerance: policy.grid_tolerance,
        max_abs_iv_residual: max_residual,
        max_abs_iv_difference: max_difference,
        fit_within_tolerance: fit,
        grid_stable: stable,
        accepted: fit && stable,
        plan_compilations: 5 * groups.len(),
    })
}

// An IV site is (maturity, log(K/F)), independent of discount, units, side,
// target and residual scale. Conservative roundoff-sized exclusion, not clustering.
fn same_iv_site(a: &HestonIvCalibrationQuote, b: &HestonIvCalibrationQuote) -> bool {
    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() <= 64. * f64::EPSILON * a.abs().max(b.abs()).max(1.)
    }
    near(a.maturity, b.maturity)
        && near(
            a.strike.ln() - a.forward.ln(),
            b.strike.ln() - b.forward.ln(),
        )
}
