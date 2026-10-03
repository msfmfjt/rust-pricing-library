//! Sequential survival conditioning for hard discrete Barrier Spot risk.
use super::*;
use crate::models::BuehlerDividendState;
use crate::product::{BarrierDirection, BarrierMonitoring, BarrierStyle, OptionSide};
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};

#[cfg(test)]
mod tests;

pub(super) const METHOD: &str = "buehler-rough-residual-lsv-hard-barrier-survival-spot-v2";

pub(super) struct HardBarrierPlan {
    pub fingerprint: Fingerprint,
    monitors: Vec<bool>,
    rows: Vec<usize>,
    growth: Vec<f64>,
    cash: Vec<f64>,
    weights: Vec<Box<[f64]>>,
    variances: Vec<f64>,
    factor: RoughBergomi,
    dv: f64,
    orth: f64,
    conditional_root: f64,
}

impl StochasticDividendPricingPlan {
    /// Hard-payoff physical-Spot Delta by survival conditioning. Supports rough
    /// residual LSV, discrete up/down knock-in/out calls/puts, future monitoring and no
    /// rebate or smoothing. Requires positive conditional equity variance.
    /// Leverage is re-anchored under Spot; calibration uncertainty and time-grid
    /// bias are excluded from the reported MC/RQMC standard errors.
    pub fn evaluate_lsv_hard_barrier_spot_risk(
        &self,
    ) -> Result<StochasticDividendLsvSpotRisk, MonteCarloError> {
        self.lsv_spot_risk(Some(&HardBarrierPlan::compile(self)?))
    }
}

fn unsupported() -> MonteCarloError {
    MonteCarloError::UnsupportedRiskForModel {
        model: "hard Barrier Spot risk requires rough residual LSV, an unsmoothed discrete Barrier call/put, future monitoring, no rebate, and positive conditional equity variance",
    }
}

impl HardBarrierPlan {
    fn compile(plan: &StochasticDividendPricingPlan) -> Result<Self, MonteCarloError> {
        let Some(StochasticDividendLsvCalibration::Rough {
            calibration,
            dividend_volatility_correlation,
            ..
        }) = &plan.lsv
        else {
            return Err(unsupported());
        };
        let barrier = plan.barrier.as_ref().ok_or_else(unsupported)?;
        if barrier.monitoring() != BarrierMonitoring::Discrete
            || barrier.rebate().is_some()
            || plan.base.payoff_smoothing.is_some()
        {
            return Err(unsupported());
        }
        let factor = calibration.model();
        let dv = *dividend_volatility_correlation;
        let sd = plan.path.model().equity_dividend_correlation();
        if dv.abs() >= 1.0 {
            return Err(unsupported());
        }
        let orth = (factor.correlation() - sd * dv) / (1.0 - dv * dv).sqrt();
        let variance = 1.0 - sd * sd - orth * orth;
        if !variance.is_finite() || variance <= 0.0 {
            return Err(unsupported());
        }
        let times = plan.path.times();
        let mut monitors = vec![false; times.len()];
        for &date in barrier.monitoring_dates() {
            let time = DayCountConvention::Act365F.year_fraction(plan.base.valuation_date, date);
            if time <= 0.0 {
                return Err(unsupported());
            }
            let i = times
                .binary_search_by(|t| t.total_cmp(&time))
                .map_err(|_| invalid("hard_barrier_monitoring_time"))?;
            monitors[i] = true;
        }
        let surface = plan.path.lsv_surface().ok_or_else(unsupported)?;
        let rows = times
            .iter()
            .map(|t| surface.times().partition_point(|x| x <= t) - 1)
            .collect();
        let growth = times
            .iter()
            .map(|&t| Ok(plan.market.forward(t)? / plan.market.spot().get()))
            .collect::<Result<_, MonteCarloError>>()?;
        let cash = plan
            .path
            .nodes()
            .iter()
            .map(|n| n.cash_paid(BuehlerDividendState::initial()))
            .collect::<Result<_, _>>()?;
        let (weights, variances) = factor.volterra_weights(times)?;
        let mut hash = blake3::Hasher::new();
        hash.update(plan.fingerprint.as_bytes());
        hash.update(METHOD.as_bytes());
        Ok(Self {
            fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            monitors,
            rows,
            growth,
            cash,
            weights,
            variances,
            factor,
            dv,
            orth,
            conditional_root: variance.sqrt(),
        })
    }

    pub(super) fn sample(
        &self,
        plan: &StochasticDividendPricingPlan,
        z: &[f64],
    ) -> Result<[f64; 2], MonteCarloError> {
        // Independent coordinates: dividend, orthogonal volatility, newest
        // Volterra residual, free equity. The terminal equity input is reserved.
        let times = plan.path.times();
        if z.len() != 4 * (times.len() - 1) || z.iter().any(|v| !v.is_finite()) {
            return Err(invalid("hard_barrier_normals").into());
        }
        let h = self.factor.hurst();
        let mut increments = Vec::with_capacity(times.len() - 1);
        let mut loadings = vec![1.0];
        for (i, step) in times.windows(2).enumerate() {
            let dt = step[1] - step[0];
            let dw =
                dt.sqrt() * (self.dv * z[4 * i] + (1.0 - self.dv * self.dv).sqrt() * z[4 * i + 1]);
            let mut driver = NeumaierSum::new();
            driver.add(
                self.factor.average_kernel(0.0, dt)? * dw
                    + dt.powf(h) * (0.5 - h) / (h + 0.5) * z[4 * i + 2],
            );
            for (&w, &v) in self.weights[i + 1].iter().zip(&increments) {
                driver.add(w * v);
            }
            let eta = self.factor.vol_of_vol();
            loadings.push(
                (0.5 * eta * driver.total() - 0.25 * eta * eta * self.variances[i + 1]).exp(),
            );
            increments.push(dw);
        }
        let ko = self.leg(plan, z, &loadings, true)?;
        let barrier = plan.barrier.as_ref().expect("validated Barrier");
        let value = if barrier.style() == BarrierStyle::KnockIn {
            let vanilla = self.leg(plan, z, &loadings, false)?;
            [vanilla[0] - ko[0], vanilla[1] - ko[1]]
        } else {
            ko
        };
        let scale = plan.base.discount * barrier.notional().get();
        let result = value.map(|v| scale * v);
        if result.iter().any(|v| !v.is_finite()) {
            return Err(invalid("hard_barrier_sample").into());
        }
        Ok(result)
    }

    fn leg(
        &self,
        plan: &StochasticDividendPricingPlan,
        z: &[f64],
        loadings: &[f64],
        knockout: bool,
    ) -> Result<[f64; 2], MonteCarloError> {
        let surface = plan.path.lsv_surface().expect("validated LSV");
        let barrier = plan.barrier.as_ref().expect("validated Barrier");
        let model = plan.path.model();
        let alpha = model.equity_linkage();
        let mut f: f64 = 1.0;
        let mut y = 1.0;
        let (mut df, mut dy, mut weight, mut dweight) = (0.0, 0.0, 1.0, 0.0);
        for (i, step) in plan.path.times().windows(2).enumerate() {
            let dt = step[1] - step[0];
            let lookup = surface.lookup_row(self.rows[i], f.ln());
            let sigma = lookup.value.sqrt() * loadings[i];
            let dsigma = sigma * lookup.derivative_log_f * df / (2.0 * lookup.value * f);
            let mean_normal =
                model.equity_dividend_correlation() * z[4 * i] + self.orth * z[4 * i + 1];
            let mu = f.ln() - 0.5 * sigma * sigma * dt + sigma * dt.sqrt() * mean_normal;
            let dmu = df / f + dsigma * (dt.sqrt() * mean_normal - sigma * dt);
            let s = sigma * dt.sqrt() * self.conditional_root;
            let ds = dsigma * dt.sqrt() * self.conditional_root;
            if !s.is_finite() || s <= 0.0 {
                return Err(invalid("hard_barrier_conditional_scale").into());
            }
            let exponent = -0.5 * model.mean_reversion() * dt;
            let half = exponent.exp();
            let complement = -exponent.exp_m1();
            let noise = (model.dividend_volatility() * dt.sqrt() * z[4 * i]
                - 0.5 * model.dividend_volatility().powi(2) * dt)
                .exp();
            let link = complement * alpha;
            let u = half * (half * y + complement * (alpha * f + 1.0 - alpha)) * noise
                + complement * (1.0 - alpha);
            let du = half * (half * dy + complement * alpha * df) * noise;
            if !noise.is_finite() || noise <= 0.0 || !u.is_finite() || !du.is_finite() {
                return Err(invalid("hard_barrier_dividend_diffusion").into());
            }
            let [a, b, c] = plan.path.nodes()[i + 1].coefficients();
            let growth = self.growth[i + 1];
            // Cash is positive: an up hit checks the pre-cash maximum, while
            // a down hit checks the post-cash minimum at the same observation.
            let up = barrier.direction() == BarrierDirection::Up;
            let monitor_b = b + if up { self.cash[i + 1] } else { 0.0 };
            let monitor_q = a + monitor_b * link;
            let boundary = (barrier.barrier().get() - monitor_b * u - c) / monitor_q;
            let dboundary = -monitor_b * du / monitor_q - boundary * growth / monitor_q;
            let monitored = knockout && self.monitors[i + 1];
            if monitored && up && boundary <= 0.0 {
                return Ok([0.0, 0.0]);
            }
            if i + 2 == plan.path.times().len() {
                let q = a + b * link;
                let constant = b * u + c - barrier.strike().get();
                let dconstant = b * du;
                let exercise = -constant / q;
                let dexercise = -dconstant / q - exercise * growth / q;
                let call = barrier.side() == OptionSide::Call;
                let mut lower = (0.0, 0.0);
                let mut upper = (f64::INFINITY, 0.0);
                if call {
                    if exercise > 0.0 {
                        lower = (exercise, dexercise);
                    }
                } else {
                    upper = (exercise, dexercise);
                }
                if monitored {
                    if up && boundary < upper.0 {
                        upper = (boundary, dboundary);
                    }
                    if !up && boundary > lower.0 {
                        lower = (boundary, dboundary);
                    }
                }
                if upper.0 <= lower.0 {
                    return Ok([0.0, 0.0]);
                }
                let mean = (mu + 0.5 * s * s).exp();
                if !mean.is_finite() || mean <= 0.0 {
                    return Err(invalid("hard_barrier_terminal_mean").into());
                }
                // Call uses upper lognormal tails; Put uses lower tails to
                // avoid subtracting a nearly complete moment for OTM Puts.
                let tail = |(cut, dcut): (f64, f64)| {
                    let (z, dz) = if cut > 0.0 && cut.is_finite() {
                        let z = (mu - cut.ln()) / s;
                        (z, (dmu - dcut / cut - z * ds) / s)
                    } else if cut <= 0.0 {
                        (f64::INFINITY, 0.0)
                    } else {
                        (f64::NEG_INFINITY, 0.0)
                    };
                    let sign = if call { 1.0 } else { -1.0 };
                    let prob = cdf(sign * z);
                    let moment = mean * cdf(sign * (z + s));
                    let dmoment = mean
                        * ((dmu + s * ds) * cdf(sign * (z + s)) + sign * pdf(z + s) * (dz + ds));
                    [
                        sign * (q * moment + constant * prob),
                        sign * (growth * moment + q * dmoment + dconstant * prob)
                            + constant * pdf(z) * dz,
                    ]
                };
                let (first, second) = if call {
                    (tail(lower), tail(upper))
                } else {
                    (tail(upper), tail(lower))
                };
                let value = [first[0] - second[0], first[1] - second[1]];
                return Ok([weight * value[0], dweight * value[0] + weight * value[1]]);
            }
            let mut equity = z[4 * i + 3];
            let mut dequity = 0.0;
            if monitored && boundary > 0.0 {
                // Down survival Z>cut is upper truncation of the reflected
                // normal -Z. A nonpositive down boundary imposes no restriction.
                let sign = if up { 1.0 } else { -1.0 };
                equity *= sign;
                let raw_cutoff = (boundary.ln() - mu) / s;
                let cutoff = sign * raw_cutoff;
                let dcutoff = sign * (dboundary / boundary - dmu - raw_cutoff * ds) / s;
                let probability = cdf(cutoff);
                let dprobability = pdf(cutoff) * dcutoff;
                let uniform = cdf(equity);
                let p = uniform * probability;
                // Use complementary tails near one; never clamp quantiles or
                // floor positive survival probabilities. Underflow is an error.
                equity = if p <= 0.5 {
                    inverse_standard_normal(p)
                } else {
                    inverse_standard_normal(cdf(-equity) + uniform * cdf(-cutoff)).map(|v| -v)
                }
                .map_err(|_| invalid("hard_barrier_survival_quantile"))?;
                dequity = sign * uniform * dprobability / pdf(equity);
                equity *= sign;
                dweight = dweight * probability + weight * dprobability;
                weight *= probability;
            }
            f = (mu + s * equity).exp();
            df = f * (dmu + ds * equity + s * dequity);
            y = u + link * f;
            dy = du + link * df;
            if !f.is_finite()
                || f <= 0.0
                || !y.is_finite()
                || y <= 0.0
                || !df.is_finite()
                || !dy.is_finite()
                || !weight.is_finite()
                || weight <= 0.0
                || !dweight.is_finite()
            {
                return Err(invalid("hard_barrier_state").into());
            }
        }
        Err(invalid("hard_barrier_terminal_step").into())
    }
}
