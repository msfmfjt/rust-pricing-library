use crate::engine::sampling::random::{Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::bass_lv::{BassError, BassLvConfig, BassMarginal, interpolate};
use pricing_numerics::{standard_normal_cdf as cdf, standard_normal_pdf as pdf};

mod mapping_risk;
pub use mapping_risk::{BassDeterministicMappingRisk, BassMappingRisk, BassMappingRiskPlan};
mod market_risk;
pub(crate) mod request;
pub use market_risk::{
    BASS_VEGA_KT_METHOD, BassMarketIvModel, BassVegaKtRisk, BassVegaKtRiskPlan,
    BassVegaKtScenarioDiagnostics,
};

#[derive(Clone, Debug)]
struct Table {
    x: Vec<f64>,
    y: Vec<f64>,
}
impl Table {
    fn at(&self, w: f64) -> f64 {
        interpolate(&self.x, &self.y, w)
    }
    fn inverse(&self, s: f64) -> Result<f64, BassError> {
        if !s.is_finite() || s <= self.y[0] || s >= self.y[self.y.len() - 1] {
            return Err(BassError::Numerical(format!(
                "spot {s} is outside the invertible Bass mapping; widen/refine the grid"
            )));
        }
        let i = self.y.partition_point(|v| *v <= s) - 1;
        Ok(self.x[i] + (s - self.y[i]) / (self.y[i + 1] - self.y[i]) * (self.x[i + 1] - self.x[i]))
    }
    fn slope(&self, w: f64) -> f64 {
        if w < self.x[0] || w >= self.x[self.x.len() - 1] {
            return 0.0;
        }
        let i = self.x.partition_point(|x| *x <= w).saturating_sub(1);
        (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
    }
    // Exact Gaussian convolution of the represented piecewise-linear function,
    // including constant tails. Tail integrals use the small CDF tail directly.
    fn heat_value_slope(&self, w: f64, variance: f64) -> (f64, f64) {
        if variance == 0.0 {
            return (self.at(w), self.slope(w));
        }
        let sd = variance.sqrt();
        let n = self.x.len();
        let mut value =
            self.y[0] * cdf((self.x[0] - w) / sd) + self.y[n - 1] * cdf((w - self.x[n - 1]) / sd);
        let mut slope = 0.0;
        let first = self
            .x
            .partition_point(|x| *x < w - 10.0 * sd)
            .saturating_sub(1);
        let last = self.x.partition_point(|x| *x <= w + 10.0 * sd).min(n - 1);
        for i in first..last {
            let a = (self.x[i] - w) / sd;
            let b = (self.x[i + 1] - w) / sd;
            let probability = normal_interval(a, b);
            let m = (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i]);
            value += self.y[i] * probability
                + m * ((w - self.x[i]) * probability + sd * (pdf(a) - pdf(b)));
            slope += m * probability;
        }
        (value, slope)
    }
    fn heated(&self, variance: f64) -> Self {
        if variance == 0.0 {
            return self.clone();
        }
        let k = HeatKernel::new(self.x[1] - self.x[0], variance, self.x.len());
        Self {
            x: self.x.clone(),
            y: k.apply(&self.y),
        }
    }
}
fn normal_interval(a: f64, b: f64) -> f64 {
    if a >= 0.0 {
        cdf(-a) - cdf(-b)
    } else {
        cdf(b) - cdf(a)
    }
}

// On a uniform grid the exact integrals of linear interpolation hats form a
// Toeplitz stencil. Constant endpoint extensions are included, with no FFT
// periodic wraparound and no Gauss-Hermite quadrature error.
struct HeatKernel {
    weights: Vec<f64>,
}
impl HeatKernel {
    fn new(h: f64, variance: f64, n: usize) -> Self {
        let sd = variance.sqrt();
        let radius = ((10.0 * sd / h).ceil() as usize + 1).min(n);
        let mut weights = Vec::with_capacity(radius + 1);
        for j in 0..=radius {
            let c = j as f64 * h;
            let a = (c - h) / sd;
            let b = c / sd;
            let d = (c + h) / sd;
            let left = (sd * (pdf(a) - pdf(b)) - (c - h) * normal_interval(a, b)) / h;
            let right = ((c + h) * normal_interval(b, d) - sd * (pdf(b) - pdf(d))) / h;
            weights.push((left + right).max(0.0));
        }
        // The maximum radius covers the entire finite grid. Probability beyond
        // that radius belongs to the constant endpoint extensions.
        let mass = weights[0] + 2.0 * weights[1..].iter().sum::<f64>();
        let last = weights.len() - 1;
        weights[last] += (1.0 - mass) * 0.5;
        Self { weights }
    }
    fn apply(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        (0..n)
            .map(|i| {
                let mut s = self.weights[0] * y[i];
                for (j, p) in self.weights.iter().enumerate().skip(1) {
                    s += p * (y[i.saturating_sub(j)] + y[(i + j).min(n - 1)]);
                }
                s
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct BassCalibrationDiagnostics {
    pub start_time: f64,
    pub end_time: f64,
    pub iterations: usize,
    /// Unshifted ||A F - F|| infinity norm, evaluated on the Brownian grid.
    pub cdf_residual: f64,
    /// Estimated marginal CDF error after propagating through preceding resets.
    /// Checked at input CDF knots and nine central/tail quantiles; includes
    /// interpolation error and is distinct from fixed-point convergence.
    pub marginal_cdf_error: f64,
    pub boundary_tail_probability: f64,
    pub brownian_min: f64,
    pub brownian_max: f64,
    pub grid_spacing: f64,
}

#[derive(Clone, Debug)]
struct Interval {
    end: f64,
    terminal: Table,
    initial: Table,
}

/// Immutable calibrated model in a zero-drift martingale coordinate.
#[derive(Clone, Debug)]
pub struct BassLvModel {
    spot: f64,
    marginals: Vec<BassMarginal>,
    intervals: Vec<Interval>,
    diagnostics: Vec<BassCalibrationDiagnostics>,
    initial_w: f64,
    initial_spot_error: f64,
}

impl BassLvModel {
    pub fn calibrate(
        spot: f64,
        marginals: Vec<BassMarginal>,
        config: BassLvConfig,
    ) -> Result<Self, BassError> {
        config.validate()?;
        if !spot.is_finite() || spot <= 0.0 || marginals.is_empty() {
            return Err(BassError::InvalidInput(
                "positive finite spot and at least one marginal are required".into(),
            ));
        }
        for (i, m) in marginals.iter().enumerate() {
            if (m.mean() - spot).abs() > 1e-9 * spot {
                return Err(BassError::InvalidInput(format!(
                    "marginal {i} mean {} differs from martingale spot {spot}",
                    m.mean()
                )));
            }
            if i > 0 {
                if m.expiry() <= marginals[i - 1].expiry() {
                    return Err(BassError::InvalidInput(
                        "marginal expiries must strictly increase".into(),
                    ));
                }
                check_convex_order(&marginals[i - 1], m, i, spot)?;
                if m.variance() - marginals[i - 1].variance() <= 1e-10 * spot * spot {
                    return Err(BassError::InvalidInput("Bass requires strictly spreading marginals; identical/degenerate consecutive marginals are unsupported".into()));
                }
            }
        }
        let mut intervals = Vec::with_capacity(marginals.len());
        let mut diagnostics = Vec::with_capacity(marginals.len());
        for (i, m) in marginals.iter().enumerate() {
            let start = if i == 0 {
                0.0
            } else {
                marginals[i - 1].expiry()
            };
            let dt = m.expiry() - start;
            // The variance-ratio estimate is exact for additive Gaussian
            // marginals and supplies a useful scale for positive marginals.
            let latent_variance = if i == 0 {
                0.0
            } else {
                dt * marginals[i - 1].variance() / (m.variance() - marginals[i - 1].variance())
            };
            let half = config.grid_width * (latent_variance + dt).sqrt();
            if !half.is_finite() || half <= 0.0 {
                return Err(BassError::InvalidInput(
                    "unrepresentable Bass Brownian grid scale".into(),
                ));
            }
            let n = config.grid_points;
            let x: Vec<_> = (0..n)
                .map(|j| -half + 2.0 * half * j as f64 / (n - 1) as f64)
                .collect();
            let h = x[1] - x[0];
            let (terminal, initial, iterations, residual, tail) = if i == 0 {
                let terminal = Table {
                    x: x.clone(),
                    y: x.iter()
                        .map(|w| m.quantile_unchecked(cdf(w / dt.sqrt())))
                        .collect(),
                };
                let initial = terminal.heated(dt);
                (terminal, initial, 0, 0.0, 2.0 * cdf(-config.grid_width))
            } else {
                let prev = &marginals[i - 1];
                let k = HeatKernel::new(h, dt, n);
                let mut f: Vec<_> = x.iter().map(|w| cdf(w / latent_variance.sqrt())).collect();
                let mut result = None;
                let mut last_residual = f64::INFINITY;
                for iteration in 1..=config.max_iterations {
                    let end_cdf = k.apply(&f);
                    let g: Vec<_> = end_cdf
                        .iter()
                        .map(|p| m.quantile_unchecked(p.clamp(0.0, 1.0)))
                        .collect();
                    let start_map = k.apply(&g);
                    let next: Vec<_> = start_map.iter().map(|s| prev.cdf(*s)).collect();
                    let residual = next
                        .iter()
                        .zip(&f)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    let tail = f[0].max(1.0 - f[n - 1]);
                    if !residual.is_finite() || start_map.iter().any(|s| !s.is_finite()) {
                        return Err(BassError::Numerical(
                            "nonfinite Bass calibration iterate".into(),
                        ));
                    }
                    if residual <= config.cdf_tolerance {
                        result = Some((
                            Table { x: x.clone(), y: g },
                            Table {
                                x: x.clone(),
                                y: start_map,
                            },
                            iteration,
                            residual,
                            tail,
                        ));
                        break;
                    }
                    last_residual = residual;
                    f = next;
                }
                result.ok_or(BassError::Calibration {
                    interval: i,
                    iterations: config.max_iterations,
                    residual: last_residual,
                })?
            };
            if tail > config.tail_tolerance {
                return Err(BassError::GridTooNarrow {
                    interval: i,
                    tail_probability: tail,
                });
            }
            diagnostics.push(BassCalibrationDiagnostics {
                start_time: start,
                end_time: m.expiry(),
                iterations,
                cdf_residual: residual,
                marginal_cdf_error: 0.0,
                boundary_tail_probability: tail,
                brownian_min: -half,
                brownian_max: half,
                grid_spacing: h,
            });
            intervals.push(Interval {
                end: m.expiry(),
                terminal,
                initial,
            });
        }
        let initial_spot_error = intervals[0].initial.at(0.0) - spot;
        // Section 5's fixed-spot convention also removes the finite-grid mean
        // error of the first mapping. Diagnostics retain the uncorrected error.
        let initial_w = intervals[0].initial.inverse(spot)?;
        let mut model = Self {
            spot,
            marginals,
            intervals,
            diagnostics,
            initial_w,
            initial_spot_error,
        };
        model.propagate_cdf_diagnostics()?;
        Ok(model)
    }
    // Paper Eq. (12), applied to the represented maps, including the initial
    // mean correction. This diagnostic is numerical, not a certified bound.
    fn propagate_cdf_diagnostics(&mut self) -> Result<(), BassError> {
        let mut previous: Option<Table> = None;
        for (i, interval) in self.intervals.iter().enumerate() {
            let x = &interval.terminal.x;
            let end_cdf = if let Some(prev) = previous {
                let old = &self.intervals[i - 1].terminal;
                let mut probabilities = Vec::with_capacity(x.len());
                for s in &interval.initial.y {
                    probabilities.push(if *s <= old.y[0] {
                        0.0
                    } else if *s >= *old.y.last().unwrap() {
                        1.0
                    } else {
                        prev.at(old.inverse(*s)?)
                    });
                }
                Table {
                    x: x.clone(),
                    y: probabilities,
                }
                .heated(interval.end - self.intervals[i - 1].end)
            } else {
                Table {
                    x: x.clone(),
                    y: x.iter()
                        .map(|w| cdf((w - self.initial_w) / interval.end.sqrt()))
                        .collect(),
                }
            };
            let mut error: f64 = 0.0;
            for p in self.marginals[i]
                .probabilities()
                .iter()
                .copied()
                .chain([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            {
                let s = self.marginals[i].quantile_unchecked(p);
                let actual = if s <= interval.terminal.y[0] {
                    0.0
                } else if s >= *interval.terminal.y.last().unwrap() {
                    1.0
                } else {
                    end_cdf.at(interval.terminal.inverse(s)?)
                };
                error = error.max((actual - p).abs());
            }
            self.diagnostics[i].marginal_cdf_error = error;
            previous = Some(end_cdf);
        }
        Ok(())
    }
    pub fn spot(&self) -> f64 {
        self.spot
    }
    pub fn marginals(&self) -> &[BassMarginal] {
        &self.marginals
    }
    pub fn diagnostics(&self) -> &[BassCalibrationDiagnostics] {
        &self.diagnostics
    }
    pub fn initial_spot_error(&self) -> f64 {
        self.initial_spot_error
    }
    pub fn initial_brownian_state(&self) -> f64 {
        self.initial_w
    }
    fn interval(&self, time: f64) -> Result<&Interval, BassError> {
        if !time.is_finite() || time < 0.0 || time > self.intervals.last().unwrap().end {
            return Err(BassError::InvalidInput(
                "time must lie between zero and the last calibrated expiry".into(),
            ));
        }
        let i = self
            .intervals
            .partition_point(|i| i.end <= time)
            .min(self.intervals.len() - 1);
        Ok(&self.intervals[i])
    }
    /// Right-continuous interval convention at interior calibration expiries.
    pub fn mapping(&self, time: f64, brownian_state: f64) -> Result<f64, BassError> {
        if !brownian_state.is_finite() {
            return Err(BassError::InvalidInput(
                "Brownian state must be finite".into(),
            ));
        }
        let i = self.interval(time)?;
        Ok(i.terminal.heat_value_slope(brownian_state, i.end - time).0)
    }
    /// Relative instantaneous volatility f_w/f; undefined at final expiry.
    pub fn local_volatility(&self, time: f64, spot: f64) -> Result<f64, BassError> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err(BassError::InvalidInput(
                "spot must be positive and finite".into(),
            ));
        }
        let i = self.interval(time)?;
        if time == i.end {
            return Err(BassError::InvalidInput(
                "local volatility is undefined at the final expiry".into(),
            ));
        }
        // Root-finding against the same exact heat-convolved map used below.
        let mut lo = i.terminal.x[0];
        let mut hi = *i.terminal.x.last().unwrap();
        let variance = i.end - time;
        if spot <= i.terminal.heat_value_slope(lo, variance).0
            || spot >= i.terminal.heat_value_slope(hi, variance).0
        {
            return Err(BassError::Numerical(
                "local volatility spot lies outside the Bass grid".into(),
            ));
        }
        for _ in 0..60 {
            let mid = (lo + hi) * 0.5;
            if i.terminal.heat_value_slope(mid, variance).0 < spot {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(i.terminal.heat_value_slope((lo + hi) * 0.5, variance).1 / spot)
    }
    /// Precompute maps at observations and all intervening market expiries.
    pub fn compile_simulation(
        &self,
        observation_times: Vec<f64>,
    ) -> Result<BassSimulationPlan, BassError> {
        if observation_times.is_empty()
            || observation_times.iter().any(|t| !t.is_finite() || *t < 0.0)
            || observation_times.windows(2).any(|t| t[0] >= t[1])
        {
            return Err(BassError::InvalidInput(
                "observation times must be nonempty, finite, nonnegative and strictly increasing"
                    .into(),
            ));
        }
        let last = *observation_times.last().unwrap();
        self.interval(last)?;
        let mut times = observation_times.clone();
        times.push(0.0);
        times.extend(
            self.intervals
                .iter()
                .filter(|i| i.end <= last)
                .map(|i| i.end),
        );
        times.sort_by(f64::total_cmp);
        times.dedup();
        if times.len() > u32::MAX as usize {
            return Err(BassError::InvalidInput("too many observation dates".into()));
        }
        let observation_indices = observation_times
            .iter()
            .map(|t| times.binary_search_by(|v| v.total_cmp(t)).unwrap())
            .collect();
        let mut steps = Vec::with_capacity(times.len() - 1);
        for pair in times.windows(2) {
            let index = self.intervals.partition_point(|i| i.end < pair[1]);
            let interval = &self.intervals[index];
            let map = interval.terminal.heated(interval.end - pair[1]);
            let reset =
                if pair[1] == interval.end && index + 1 < self.intervals.len() && pair[1] < last {
                    Some(self.intervals[index + 1].initial.clone())
                } else {
                    None
                };
            steps.push(Step {
                sqrt_dt: (pair[1] - pair[0]).sqrt(),
                map,
                reset,
            });
        }
        Ok(BassSimulationPlan {
            spot: self.spot,
            initial_w: self.initial_w,
            observation_times,
            time_nodes: times,
            observation_indices,
            steps,
        })
    }
}

// Call-price differences are quadratic between the union of marginal knots.
// Test each knot AND every interior stationary point: grid-only validation
// can miss calendar arbitrage between quotes.
fn check_convex_order(
    a: &BassMarginal,
    b: &BassMarginal,
    interval: usize,
    spot: f64,
) -> Result<(), BassError> {
    let mut knots = a.spots().to_vec();
    knots.extend_from_slice(b.spots());
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let mut candidates = knots.clone();
    for s in knots.windows(2) {
        let d0 = b.cdf(s[0]) - a.cdf(s[0]);
        let d1 = b.cdf(s[1]) - a.cdf(s[1]);
        if d0 * d1 < 0.0 {
            candidates.push(s[0] + (s[1] - s[0]) * (-d0) / (d1 - d0));
        }
    }
    for strike in candidates {
        let gap = b.call_unchecked(strike) - a.call_unchecked(strike);
        if gap < -1e-10 * spot {
            return Err(BassError::ConvexOrder {
                interval,
                strike,
                gap,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Step {
    sqrt_dt: f64,
    map: Table,
    reset: Option<Table>,
}

#[derive(Clone, Debug)]
pub struct BassSimulationPlan {
    spot: f64,
    initial_w: f64,
    observation_times: Vec<f64>,
    time_nodes: Vec<f64>,
    observation_indices: Vec<usize>,
    steps: Vec<Step>,
}

#[derive(Clone, Debug)]
pub struct BassEstimate {
    pub price: f64,
    pub standard_error: f64,
    pub paths: usize,
    pub seed: u64,
}

impl BassSimulationPlan {
    pub fn observation_times(&self) -> &[f64] {
        &self.observation_times
    }
    pub fn time_nodes(&self) -> &[f64] {
        &self.time_nodes
    }
    pub fn normal_count(&self) -> usize {
        self.steps.len()
    }
    /// One independent standard normal per augmented time step. Supplying
    /// correlated arrays allows external multi-asset coupling on a common grid.
    pub fn path_from_normals(&self, normals: &[f64]) -> Result<Vec<f64>, BassError> {
        if normals.len() != self.normal_count() || normals.iter().any(|z| !z.is_finite()) {
            return Err(BassError::InvalidInput(
                "normals must be finite and match the augmented time grid".into(),
            ));
        }
        let mut full = vec![0.0; self.time_nodes.len()];
        let mut output = vec![0.0; self.observation_times.len()];
        self.fill_path(|d| normals[d], &mut full, &mut output)?;
        Ok(output)
    }
    fn fill_path<F: FnMut(usize) -> f64>(
        &self,
        mut normal: F,
        full: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BassError> {
        full[0] = self.spot;
        let mut w = self.initial_w;
        for (d, step) in self.steps.iter().enumerate() {
            w += step.sqrt_dt * normal(d);
            // Never silently clip a simulated tail to an endpoint value.
            if w < step.map.x[0] || w > *step.map.x.last().unwrap() {
                return Err(BassError::Numerical("simulated Brownian state left the Bass grid; increase grid_width and grid_points".into()));
            }
            let s = step.map.at(w);
            full[d + 1] = s;
            if let Some(next) = &step.reset {
                w = next.inverse(s)?;
            }
        }
        for (out, i) in output.iter_mut().zip(&self.observation_indices) {
            *out = full[*i];
        }
        Ok(())
    }
    pub fn sample_paths(&self, paths: usize, seed: u64) -> Result<Vec<Vec<f64>>, BassError> {
        if paths == 0 {
            return Err(BassError::InvalidInput("paths must be positive".into()));
        }
        let rng = Philox4x32::from_seed(seed);
        let mut full = vec![0.0; self.time_nodes.len()];
        (0..paths)
            .map(|p| {
                let mut output = vec![0.0; self.observation_times.len()];
                self.fill_path(
                    |d| {
                        rng.standard_normal(RandomCoordinate::new(
                            p as u64,
                            d as u32,
                            RandomDomain::Valuation,
                        ))
                    },
                    &mut full,
                    &mut output,
                )?;
                Ok(output)
            })
            .collect()
    }
    /// Generic undiscounted path payoff with an explicit deterministic discount
    /// factor. Standard error is the ordinary IID sample-mean error.
    pub fn price<F: Fn(&[f64]) -> f64>(
        &self,
        paths: usize,
        seed: u64,
        discount_factor: f64,
        payoff: F,
    ) -> Result<BassEstimate, BassError> {
        if paths < 2 || !discount_factor.is_finite() || discount_factor <= 0.0 {
            return Err(BassError::InvalidInput(
                "pricing requires at least two paths and a positive finite discount factor".into(),
            ));
        }
        let rng = Philox4x32::from_seed(seed);
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut full = vec![0.0; self.time_nodes.len()];
        let mut output = vec![0.0; self.observation_times.len()];
        for p in 0..paths {
            self.fill_path(
                |d| {
                    rng.standard_normal(RandomCoordinate::new(
                        p as u64,
                        d as u32,
                        RandomDomain::Valuation,
                    ))
                },
                &mut full,
                &mut output,
            )?;
            let x = discount_factor * payoff(&output);
            if !x.is_finite() {
                return Err(BassError::Numerical("payoff must be finite".into()));
            }
            let delta = x - mean;
            mean += delta / (p + 1) as f64;
            m2 += delta * (x - mean);
        }
        if !mean.is_finite() || !m2.is_finite() {
            return Err(BassError::Numerical(
                "Bass payoff statistics overflowed".into(),
            ));
        }
        Ok(BassEstimate {
            price: mean,
            standard_error: (m2 / ((paths - 1) as f64 * paths as f64)).sqrt(),
            paths,
            seed,
        })
    }
    pub fn price_european(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassEstimate, BassError> {
        if !strike.is_finite() || strike < 0.0 {
            return Err(BassError::InvalidInput(
                "strike must be finite and nonnegative".into(),
            ));
        }
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s| {
            (sign * (s[s.len() - 1] - strike)).max(0.0)
        })
    }
    pub fn price_asian(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassEstimate, BassError> {
        if !strike.is_finite() || strike < 0.0 {
            return Err(BassError::InvalidInput(
                "strike must be finite and nonnegative".into(),
            ));
        }
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s| {
            (sign * (s.iter().sum::<f64>() / s.len() as f64 - strike)).max(0.0)
        })
    }
}

#[cfg(test)]
mod extension_tests;
#[cfg(test)]
mod tests;
