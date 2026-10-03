//! Sections 5.2-5.4: reverse path sensitivities and differentiated CDF transport.
//! Differentiate the same finite maps as the price engine, including inverses
//! and the fixed-spot initial coordinate. These are mapping risks, not IV Vega.
use super::*;
use crate::models::bass_lv::BassMappingBump;

#[derive(Clone, Debug)]
pub struct BassMappingRisk {
    pub estimate: BassEstimate,
    pub sensitivities: Vec<f64>,
    pub standard_errors: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct BassDeterministicMappingRisk {
    pub price: f64,
    pub sensitivities: Vec<f64>,
}

#[derive(Clone, Debug)]
struct RiskStep {
    map_bumps: Vec<(usize, Table)>,
    reset_bumps: Vec<(usize, Table)>,
}

#[derive(Clone, Debug)]
pub struct BassMappingRiskPlan {
    model: BassLvModel,
    simulation: BassSimulationPlan,
    bumps: Vec<BassMappingBump>,
    terminal_bumps: Vec<Table>,
    initial_bumps: Vec<Table>,
    steps: Vec<RiskStep>,
}

impl BassLvModel {
    pub fn compile_mapping_risk(
        &self,
        observation_times: Vec<f64>,
        bumps: Vec<BassMappingBump>,
    ) -> Result<BassMappingRiskPlan, BassError> {
        if bumps.is_empty() || bumps.len() > 256 {
            return Err(BassError::InvalidInput(
                "mapping risk requires 1..256 hat functions".into(),
            ));
        }
        let simulation = self.compile_simulation(observation_times)?;
        let mut terminal_bumps = Vec::with_capacity(bumps.len());
        let mut initial_bumps = Vec::with_capacity(bumps.len());
        for b in &bumps {
            let interval = self.intervals.get(b.interval()).ok_or_else(|| {
                BassError::InvalidInput("mapping bump interval is out of range".into())
            })?;
            let x = &interval.terminal.x;
            if b.left() < x[0] || b.right() > *x.last().unwrap() {
                return Err(BassError::InvalidInput(
                    "mapping hat must be contained in its interval's Brownian grid".into(),
                ));
            }
            let table = Table {
                x: x.clone(),
                y: x.iter().map(|w| b.value(*w)).collect(),
            };
            if !table.y.iter().any(|v| *v > 0.0) {
                return Err(BassError::InvalidInput(
                    "mapping hat is unresolved by the grid; widen the hat or refine grid_points"
                        .into(),
                ));
            }
            let start = if b.interval() == 0 {
                0.0
            } else {
                self.intervals[b.interval() - 1].end
            };
            initial_bumps.push(table.heated(interval.end - start));
            terminal_bumps.push(table);
        }
        let mut steps = Vec::with_capacity(simulation.normal_count());
        for (d, step) in simulation.steps.iter().enumerate() {
            let time = simulation.time_nodes[d + 1];
            let index = self.intervals.partition_point(|i| i.end < time);
            let mut map_bumps = Vec::new();
            let mut reset_bumps = Vec::new();
            for (j, b) in bumps.iter().enumerate() {
                if b.interval() == index {
                    map_bumps.push((
                        j,
                        terminal_bumps[j].heated(self.intervals[index].end - time),
                    ));
                }
                if step.reset.is_some() && b.interval() == index + 1 {
                    reset_bumps.push((j, initial_bumps[j].clone()));
                }
            }
            steps.push(RiskStep {
                map_bumps,
                reset_bumps,
            });
        }
        Ok(BassMappingRiskPlan {
            model: self.clone(),
            simulation,
            bumps,
            terminal_bumps,
            initial_bumps,
            steps,
        })
    }
}

struct Workspace {
    full: Vec<f64>,
    observed: Vec<f64>,
    before: Vec<f64>,
    after: Vec<f64>,
    partials: Vec<f64>,
    spot_adjoints: Vec<f64>,
    gradient: Vec<f64>,
}
impl Workspace {
    fn new(plan: &BassMappingRiskPlan) -> Self {
        let s = &plan.simulation;
        Self {
            full: vec![0.0; s.time_nodes.len()],
            observed: vec![0.0; s.observation_times.len()],
            before: vec![0.0; s.normal_count()],
            after: vec![0.0; s.normal_count()],
            partials: vec![0.0; s.observation_times.len()],
            spot_adjoints: vec![0.0; s.time_nodes.len()],
            gradient: vec![0.0; plan.bumps.len()],
        }
    }
}

impl BassMappingRiskPlan {
    pub fn simulation(&self) -> &BassSimulationPlan {
        &self.simulation
    }
    pub fn bumps(&self) -> &[BassMappingBump] {
        &self.bumps
    }
    fn forward<F: FnMut(usize) -> f64>(
        &self,
        mut normal: F,
        work: &mut Workspace,
    ) -> Result<(), BassError> {
        let s = &self.simulation;
        let mut w = s.initial_w;
        work.full[0] = s.spot;
        for (d, step) in s.steps.iter().enumerate() {
            w += step.sqrt_dt * normal(d);
            if !w.is_finite() || w < step.map.x[0] || w > *step.map.x.last().unwrap() {
                return Err(BassError::Numerical(
                    "mapping-risk path left the Brownian grid".into(),
                ));
            }
            work.before[d] = w;
            let spot = step.map.at(w);
            work.full[d + 1] = spot;
            if let Some(next) = &step.reset {
                w = next.inverse(spot)?;
            }
            work.after[d] = w;
        }
        for (out, i) in work.observed.iter_mut().zip(&s.observation_indices) {
            *out = work.full[*i];
        }
        Ok(())
    }
    fn reverse(&self, work: &mut Workspace) -> Result<(), BassError> {
        work.spot_adjoints.fill(0.0);
        work.gradient.fill(0.0);
        for (i, v) in self
            .simulation
            .observation_indices
            .iter()
            .zip(&work.partials)
        {
            work.spot_adjoints[*i] += v;
        }
        let mut w_adjoint = 0.0;
        for d in (0..self.steps.len()).rev() {
            let step = &self.simulation.steps[d];
            let risk = &self.steps[d];
            let mut spot_adjoint = work.spot_adjoints[d + 1];
            if let Some(next) = &step.reset {
                let slope = positive_slope(next, work.after[d])?;
                let ratio = w_adjoint / slope;
                spot_adjoint += ratio;
                for (j, table) in &risk.reset_bumps {
                    work.gradient[*j] -= ratio * table.at(work.after[d]);
                }
                w_adjoint = 0.0;
            }
            for (j, table) in &risk.map_bumps {
                work.gradient[*j] += spot_adjoint * table.at(work.before[d]);
            }
            w_adjoint += spot_adjoint * step.map.slope(work.before[d]);
        }
        let ratio = w_adjoint
            / positive_slope(&self.model.intervals[0].initial, self.simulation.initial_w)?;
        for (j, b) in self.bumps.iter().enumerate() {
            if b.interval() == 0 {
                work.gradient[j] -= ratio * self.initial_bumps[j].at(self.simulation.initial_w);
            }
        }
        if work.gradient.iter().any(|v| !v.is_finite()) {
            return Err(BassError::Numerical("nonfinite mapping sensitivity".into()));
        }
        Ok(())
    }
    /// Derivative of a path functional with supplied observation partials.
    /// The time-zero spot is fixed even if it appears in the payoff.
    pub fn path_sensitivities(
        &self,
        normals: &[f64],
        payoff_partials: &[f64],
    ) -> Result<Vec<f64>, BassError> {
        if normals.len() != self.simulation.normal_count()
            || normals.iter().any(|v| !v.is_finite())
            || payoff_partials.len() != self.simulation.observation_times.len()
            || payoff_partials.iter().any(|v| !v.is_finite())
        {
            return Err(BassError::InvalidInput(
                "path risk needs finite normals/partials matching the simulation grid/observations"
                    .into(),
            ));
        }
        let mut work = Workspace::new(self);
        self.forward(|d| normals[d], &mut work)?;
        work.partials.copy_from_slice(payoff_partials);
        self.reverse(&mut work)?;
        Ok(work.gradient)
    }
    /// Caller fills undiscounted payoff derivatives in the supplied zeroed buffer.
    pub fn price<F: Fn(&[f64], &mut [f64]) -> f64>(
        &self,
        paths: usize,
        seed: u64,
        discount_factor: f64,
        payoff: F,
    ) -> Result<BassMappingRisk, BassError> {
        validate_pricing(paths, discount_factor)?;
        let mut work = Workspace::new(self);
        let rng = Philox4x32::from_seed(seed);
        let mut means = vec![0.0; self.bumps.len() + 1];
        let mut m2 = means.clone();
        for p in 0..paths {
            self.forward(
                |d| {
                    rng.standard_normal(RandomCoordinate::new(
                        p as u64,
                        d as u32,
                        RandomDomain::Valuation,
                    ))
                },
                &mut work,
            )?;
            work.partials.fill(0.0);
            let value = payoff(&work.observed, &mut work.partials);
            if !value.is_finite() || work.partials.iter().any(|v| !v.is_finite()) {
                return Err(BassError::Numerical(
                    "payoff and its observation partials must be finite".into(),
                ));
            }
            self.reverse(&mut work)?;
            for (j, x) in std::iter::once(value)
                .chain(work.gradient.iter().copied())
                .enumerate()
            {
                let x = x * discount_factor;
                let delta = x - means[j];
                means[j] += delta / (p + 1) as f64;
                m2[j] += delta * (x - means[j]);
            }
        }
        if means.iter().chain(&m2).any(|v| !v.is_finite()) {
            return Err(BassError::Numerical(
                "mapping-risk statistics overflowed".into(),
            ));
        }
        let se: Vec<_> = m2
            .iter()
            .map(|v| (v / ((paths - 1) as f64 * paths as f64)).sqrt())
            .collect();
        Ok(BassMappingRisk {
            estimate: BassEstimate {
                price: means[0],
                standard_error: se[0],
                paths,
                seed,
            },
            sensitivities: means[1..].to_vec(),
            standard_errors: se[1..].to_vec(),
        })
    }
    pub fn price_european(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassMappingRisk, BassError> {
        validate_strike(strike)?;
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s, d| {
            let payoff = sign * (s[s.len() - 1] - strike);
            if payoff > 0.0 {
                d[s.len() - 1] = sign;
                payoff
            } else {
                0.0
            }
        })
    }
    pub fn price_asian(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassMappingRisk, BassError> {
        validate_strike(strike)?;
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s, d| {
            let payoff = sign * (s.iter().sum::<f64>() / s.len() as f64 - strike);
            if payoff > 0.0 {
                d.fill(sign / s.len() as f64);
                payoff
            } else {
                0.0
            }
        })
    }
    fn perturbed_model(&self, amplitudes: &[f64]) -> Result<BassLvModel, BassError> {
        if amplitudes.len() != self.bumps.len() || amplitudes.iter().any(|v| !v.is_finite()) {
            return Err(BassError::InvalidInput(
                "finite mapping amplitudes must match the hat count".into(),
            ));
        }
        let mut m = self.model.clone();
        for (j, b) in self.bumps.iter().enumerate() {
            for (y, h) in m.intervals[b.interval()]
                .terminal
                .y
                .iter_mut()
                .zip(&self.terminal_bumps[j].y)
            {
                *y += amplitudes[j] * h;
            }
        }
        let mut start = 0.0;
        for interval in &mut m.intervals {
            if interval
                .terminal
                .y
                .iter()
                .any(|y| !y.is_finite() || *y < 0.0)
                || interval.terminal.y.windows(2).any(|p| p[0] > p[1])
            {
                return Err(BassError::InvalidInput(
                    "mapping perturbation must preserve nonnegative nondecreasing terminal maps"
                        .into(),
                ));
            }
            interval.initial = interval.terminal.heated(interval.end - start);
            start = interval.end;
        }
        m.initial_w = m.intervals[0].initial.inverse(m.spot)?;
        Ok(m)
    }
    /// CRN validation plan: preserve grids and spot, perturb maps without a new
    /// marginal calibration. Returned plan makes no calibrated-marginal claim.
    pub fn bumped_simulation(&self, amplitudes: &[f64]) -> Result<BassSimulationPlan, BassError> {
        self.perturbed_model(amplitudes)?
            .compile_simulation(self.simulation.observation_times.clone())
    }
    /// Semi-analytic call price and mapping gradients at a calibration expiry.
    /// CDF transport uses the same finite maps, differentiated as Eq. (13).
    pub fn vanilla_call(
        &self,
        interval: usize,
        strike: f64,
        discount_factor: f64,
    ) -> Result<BassDeterministicMappingRisk, BassError> {
        self.deterministic_call(&self.model, interval, strike, discount_factor, true)
    }
    pub fn bumped_vanilla_call(
        &self,
        amplitudes: &[f64],
        interval: usize,
        strike: f64,
        discount_factor: f64,
    ) -> Result<f64, BassError> {
        let m = self.perturbed_model(amplitudes)?;
        Ok(self
            .deterministic_call(&m, interval, strike, discount_factor, false)?
            .price)
    }
    fn deterministic_call(
        &self,
        model: &BassLvModel,
        last: usize,
        strike: f64,
        discount: f64,
        derivatives: bool,
    ) -> Result<BassDeterministicMappingRisk, BassError> {
        validate_strike(strike)?;
        validate_pricing(2, discount)?;
        if last >= model.intervals.len() {
            return Err(BassError::InvalidInput(
                "vanilla expiry interval is out of range".into(),
            ));
        }
        let count = if derivatives { self.bumps.len() } else { 0 };
        let first = &model.intervals[0];
        let sd = first.end.sqrt();
        let mut distribution = Table {
            x: first.terminal.x.clone(),
            y: first
                .terminal
                .x
                .iter()
                .map(|w| cdf((w - model.initial_w) / sd))
                .collect(),
        };
        let initial_slope = positive_slope(&first.initial, model.initial_w)?;
        let mut tangents: Vec<Table> = (0..count)
            .map(|j| {
                let coefficient = if self.bumps[j].interval() == 0 {
                    self.initial_bumps[j].at(model.initial_w) / initial_slope
                } else {
                    0.0
                };
                Table {
                    x: first.terminal.x.clone(),
                    y: first
                        .terminal
                        .x
                        .iter()
                        .map(|w| coefficient * pdf((w - model.initial_w) / sd) / sd)
                        .collect(),
                }
            })
            .collect();
        for i in 1..=last {
            let previous = &model.intervals[i - 1].terminal;
            let next = &model.intervals[i];
            let mut start_cdf = Table {
                x: next.terminal.x.clone(),
                y: Vec::with_capacity(next.initial.y.len()),
            };
            let mut start_tangents: Vec<Table> = (0..count)
                .map(|_| Table {
                    x: next.terminal.x.clone(),
                    y: Vec::with_capacity(next.initial.y.len()),
                })
                .collect();
            for (k, s) in next.initial.y.iter().enumerate() {
                if *s <= previous.y[0] || *s >= *previous.y.last().unwrap() {
                    start_cdf
                        .y
                        .push(if *s <= previous.y[0] { 0.0 } else { 1.0 });
                    for t in &mut start_tangents {
                        t.y.push(0.0);
                    }
                    continue;
                }
                let u = previous.inverse(*s)?;
                start_cdf.y.push(distribution.at(u));
                let ratio = distribution.slope(u) / positive_slope(previous, u)?;
                for j in 0..count {
                    let b = self.bumps[j];
                    let next_bump = if b.interval() == i {
                        self.initial_bumps[j].y[k]
                    } else {
                        0.0
                    };
                    let previous_bump = if b.interval() == i - 1 {
                        self.terminal_bumps[j].at(u)
                    } else {
                        0.0
                    };
                    start_tangents[j]
                        .y
                        .push(tangents[j].at(u) + ratio * (next_bump - previous_bump));
                }
            }
            let dt = next.end - model.intervals[i - 1].end;
            distribution = start_cdf.heated(dt);
            tangents = start_tangents.iter().map(|t| t.heated(dt)).collect();
        }
        let map = &model.intervals[last].terminal;
        let payoff: Vec<_> = map.y.iter().map(|s| (s - strike).max(0.0)).collect();
        let n = map.y.len();
        let mut price =
            distribution.y[0] * payoff[0] + (1.0 - distribution.y[n - 1]) * payoff[n - 1];
        let mut risk: Vec<_> = tangents
            .iter()
            .map(|t| t.y[0] * payoff[0] - t.y[n - 1] * payoff[n - 1])
            .collect();
        // Hats vanish at endpoints, so endpoint map derivatives are zero.
        for k in 0..n - 1 {
            let a = map.y[k];
            let b = map.y[k + 1];
            let (average, da, db) = if strike <= a {
                ((a + b) * 0.5 - strike, 0.5, 0.5)
            } else if strike >= b {
                (0.0, 0.0, 0.0)
            } else {
                let q = (b - strike) / (b - a);
                ((b - strike) * q * 0.5, 0.5 * q * q, q - 0.5 * q * q)
            };
            let mass = distribution.y[k + 1] - distribution.y[k];
            price += mass * average;
            for j in 0..count {
                risk[j] += (tangents[j].y[k + 1] - tangents[j].y[k]) * average;
                if self.bumps[j].interval() == last {
                    risk[j] += mass
                        * (da * self.terminal_bumps[j].y[k] + db * self.terminal_bumps[j].y[k + 1]);
                }
            }
        }
        price *= discount;
        for r in &mut risk {
            *r *= discount;
        }
        if !price.is_finite() || risk.iter().any(|v| !v.is_finite()) {
            return Err(BassError::Numerical(
                "nonfinite deterministic mapping risk".into(),
            ));
        }
        Ok(BassDeterministicMappingRisk {
            price,
            sensitivities: risk,
        })
    }
}

fn positive_slope(table: &Table, w: f64) -> Result<f64, BassError> {
    let slope = table.slope(w);
    if !slope.is_finite() || slope <= 0.0 {
        Err(BassError::Numerical(
            "mapping sensitivity needs a positive finite inverse slope".into(),
        ))
    } else {
        Ok(slope)
    }
}
fn validate_strike(strike: f64) -> Result<(), BassError> {
    if !strike.is_finite() || strike < 0.0 {
        Err(BassError::InvalidInput(
            "strike must be finite and nonnegative".into(),
        ))
    } else {
        Ok(())
    }
}
fn validate_pricing(paths: usize, discount: f64) -> Result<(), BassError> {
    if paths < 2 || !discount.is_finite() || discount <= 0.0 {
        Err(BassError::InvalidInput(
            "pricing requires two or more paths and positive finite discount".into(),
        ))
    } else {
        Ok(())
    }
}
