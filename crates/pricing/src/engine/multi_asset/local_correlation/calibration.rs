use super::*;
use crate::core::DayCountConvention;
use pricing_numerics::NeumaierSum;

impl LocalCorrelationCalibration {
    pub(in crate::engine::multi_asset) fn compile(
        plan: &MultiAssetPricingPlan,
        config: LocalCorrelationConfig,
    ) -> Result<Self, E> {
        let n = plan.assets.len();
        let weights = &config.basket_weights;
        let second = &config.second_correlation;
        if n < 2
            || weights.len() != n
            || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(E::Invalid(
                "local correlation requires at least two positive basket weights summing to one",
            ));
        }
        if plan.hull_white.is_some() || plan.assets.iter().any(Asset::has_lsv) {
            return Err(E::Invalid(
                "local correlation v0.1 supports deterministic-rate BS/LV; stochastic-volatility and HW joint drivers are not connected",
            ));
        }
        if config.minimum_variance_span <= 0.0 || !config.minimum_variance_span.is_finite() {
            return Err(E::Invalid(
                "minimum local correlation variance span must be finite and positive",
            ));
        }
        if second.underlyings() != plan.correlation.underlyings()
            || second.tolerances() != plan.correlation.tolerances()
            || second.entries().len() != plan.correlation.entries().len()
            || second
                .entries()
                .iter()
                .zip(plan.correlation.entries())
                .any(|(a, b)| a.0 != b.0)
        {
            return Err(E::Invalid(
                "local correlation endpoints must share asset order, dates and tolerances, including future dates",
            ));
        }
        let target_times = config.target.time_nodes();
        let xs = config.target.log_moneyness_nodes();
        if target_times[0] != 0.0
            || target_times[target_times.len() - 1] < *plan.times.last().unwrap()
            || xs[0] > 0.0
            || xs[xs.len() - 1] < 0.0
        {
            return Err(E::Invalid(
                "local correlation target must cover the horizon and log basket zero",
            ));
        }
        let endpoints = plan
            .correlation
            .entries()
            .iter()
            .zip(second.entries())
            .map(|(a, b)| [a.1.clone(), b.1.clone()])
            .collect();
        let entries = plan
            .times
            .iter()
            .map(|t| {
                plan.correlation.entries().partition_point(|(d, _)| {
                    DayCountConvention::Act365F.year_fraction(plan.valuation_date, *d) <= *t
                }) - 1
            })
            .collect();
        let nt = plan.times.len();
        let m = xs.len();
        let np = config.particles.particle_count();
        let trace = config
            .particles
            .retain_reverse_trace()
            .then(|| std::sync::Arc::new(Vec::new()));
        let mut result = Self {
            config,
            times: plan.times.clone(),
            models: plan.assets.iter().map(|a| a.model.clone()).collect(),
            endpoints,
            entries,
            mixing: vec![0.0; nt * m],
            diagnostics: Vec::new(),
            weight_sums: Vec::new(),
            particle_means: Vec::new(),
            trace,
        };
        let mut states = vec![vec![1.0; n]; np];
        for row in 0..nt {
            result.calibrate_row(row, &states)?;
            result.particle_means.push(
                (0..n)
                    .map(|i| {
                        let mut sum = NeumaierSum::new();
                        for state in &states {
                            sum.add(state[i]);
                        }
                        sum.total() / np as f64
                    })
                    .collect(),
            );
            if let Some(trace) = &mut result.trace {
                std::sync::Arc::make_mut(trace).push(states.clone());
            }
            if row + 1 < nt {
                for (p, state) in states.iter_mut().enumerate() {
                    *state = result.step(row, state, &result.calibration_normals(p, row)?)?;
                }
            }
        }
        Ok(result)
    }
    fn calibrate_row(&mut self, row: usize, states: &[Vec<f64>]) -> Result<(), E> {
        let xs = self.log_nodes().to_vec();
        let m = xs.len();
        let np = states.len();
        let moments = states
            .iter()
            .map(|s| self.endpoint_variances(row, s))
            .collect::<Result<Vec<_>, _>>()?;
        let log_baskets: Vec<_> = states.iter().map(|s| self.basket(s).ln()).collect();
        let mut sums = Vec::with_capacity(m);
        let mut effective = Vec::with_capacity(m);
        let mut means = Vec::with_capacity(m);
        for &x in &xs {
            let mut accum = [NeumaierSum::new(); 4];
            for (p, &log_basket) in log_baskets.iter().enumerate() {
                let w = if row == 0 {
                    1.0
                } else {
                    quartic((log_basket - x) / self.config.particles.log_bandwidth())
                };
                for (sum, value) in
                    accum
                        .iter_mut()
                        .zip([w, w * w, w * moments[p][0], w * moments[p][1]])
                {
                    sum.add(value);
                }
            }
            let [s, s2, a, b] = accum.map(|v| v.total());
            sums.push(s);
            effective.push(if s2 > 0.0 { s * s / s2 } else { 0.0 });
            means.push(if s > 0.0 { [a / s, b / s] } else { [0.0; 2] });
        }
        let usable: Vec<_> = (0..m)
            .filter(|&j| {
                sums[j] > 0.0 && effective[j] >= self.config.particles.minimum_effective_samples()
            })
            .collect();
        if usable.is_empty() {
            return Err(E::Invalid(
                "local correlation particle support is insufficient at every basket node",
            ));
        }
        for (j, &x) in xs.iter().enumerate() {
            let source = if usable.contains(&j) {
                j
            } else {
                *usable
                    .iter()
                    .min_by(|&&a, &&b| {
                        (xs[a] - x)
                            .abs()
                            .total_cmp(&(xs[b] - x).abs())
                            .then(a.cmp(&b))
                    })
                    .unwrap()
            };
            let [a, b] = means[source];
            let target = self.config.target.interpolate(self.times[row], x)?.value;
            let span = b - a;
            let unidentifiable = span.abs() <= self.config.minimum_variance_span;
            let raw = if unidentifiable {
                0.0
            } else {
                (target - a) / span
            };
            if !raw.is_finite() || !a.is_finite() || !b.is_finite() {
                return Err(E::Invalid(
                    "nonfinite local correlation conditional variance",
                ));
            }
            let projected = if unidentifiable {
                (target - a).abs() > self.config.minimum_variance_span
            } else {
                !(0.0..=1.0).contains(&raw)
            };
            if projected && self.config.feasibility == LocalCorrelationFeasibility::Reject {
                return Err(E::numerical(format!(
                    "infeasible local correlation target at time {}, log basket {}: target {}, endpoints [{}, {}]",
                    self.times[row], x, target, a, b
                )));
            }
            let lambda = raw.clamp(0.0, 1.0);
            self.mixing[row * m + j] = lambda;
            self.weight_sums.push(sums[source]);
            self.diagnostics.push(LocalCorrelationNodeDiagnostics {
                endpoint_variances: [a, b],
                target_variance: target,
                attained_variance: a + lambda * span,
                raw_mixing: raw,
                effective_samples: if row == 0 { np as f64 } else { effective[j] },
                source_node: source,
                fallback: source != j,
                projected,
                unidentifiable,
            });
        }
        Ok(())
    }
}
