use super::*;
use pricing_numerics::NeumaierSum;

impl LocalCorrelationCalibration {
    pub(super) fn joint_discount_rate(&self, row: usize, s: &[f64]) -> Result<(f64, f64), E> {
        let j = self.joint.as_ref().unwrap();
        if let Some(r) = j.rate {
            let rates = j.rates.as_ref().unwrap();
            Ok((
                rates
                    .relative_discount(self.times[row], s[r + 1])
                    .map_err(E::numerical)?,
                s[r] + rates.rate_shift(self.times[row]).map_err(E::numerical)?,
            ))
        } else {
            Ok((1.0, 0.0))
        }
    }
    pub(super) fn correction_active(&self, row: usize) -> bool {
        row > 0
            && self
                .joint
                .as_ref()
                .unwrap()
                .rates
                .as_ref()
                .is_some_and(|r| !r.is_deterministic())
    }
    pub(super) fn calibrate_joint(mut self) -> Result<Self, E> {
        let np = self.config.particles.particle_count();
        let nt = self.times.len();
        let mut histories = vec![vec![self.initial_joint_state()]; np];
        for row in 0..nt {
            self.calibrate_joint_row(row, &histories)?;
            let mut means = vec![NeumaierSum::new(); self.models.len()];
            let states: Vec<_> = histories.iter().map(|h| h[row].clone()).collect();
            for s in &states {
                let (d, _) = self.joint_discount_rate(row, s)?;
                for (sum, u) in means.iter_mut().zip(s) {
                    sum.add(d * u);
                }
            }
            self.particle_means
                .push(means.iter().map(|s| s.total() / np as f64).collect());
            if let Some(trace) = &mut self.trace {
                std::sync::Arc::make_mut(trace).push(states);
            }
            if row + 1 < nt {
                for (p, h) in histories.iter_mut().enumerate() {
                    h.push(self.joint_step(row, h, &self.calibration_normals(p, row)?)?);
                }
            }
        }
        Ok(self)
    }
    fn calibrate_joint_row(&mut self, row: usize, histories: &[Vec<Vec<f64>>]) -> Result<(), E> {
        let xs = self.log_nodes().to_vec();
        let m = xs.len();
        let np = histories.len();
        let moments = histories
            .iter()
            .map(|h| self.joint_moments(row, h))
            .collect::<Result<Vec<_>, _>>()?;
        let logs: Vec<_> = histories
            .iter()
            .map(|h| self.basket(&h[row]).ln())
            .collect();
        let dr = histories
            .iter()
            .map(|h| self.joint_discount_rate(row, &h[row]))
            .collect::<Result<Vec<_>, _>>()?;
        let total_d = dr.iter().map(|v| v.0).sum::<f64>();
        let total_y = dr.iter().map(|v| v.0 * v.1).sum::<f64>();
        let active = self.correction_active(row);
        let target = self.joint.as_ref().unwrap().target.as_ref();
        let mut sums = vec![0.0; m];
        let mut ess = vec![0.0; m];
        let mut means = vec![[0.0; 2]; m];
        let mut corrections = vec![0.0; m];
        for (k, &x) in xs.iter().enumerate() {
            let mut acc = [NeumaierSum::new(); 6];
            for (p, &log) in logs.iter().enumerate() {
                let kernel = if row == 0 {
                    1.0
                } else {
                    quartic((log - x) / self.config.particles.log_bandwidth())
                };
                let (d, r) = dr[p];
                let w = d * kernel;
                let above = if log > x { d } else { 0.0 };
                for (s, v) in acc.iter_mut().zip([
                    w,
                    w * w,
                    w * moments[p][0],
                    w * moments[p][1],
                    above,
                    above * r,
                ]) {
                    s.add(v);
                }
            }
            let [s, s2, a, b, da, ya] = acc.map(|s| s.total());
            sums[k] = s;
            ess[k] = if s2 > 0.0 { s * s / s2 } else { 0.0 };
            if s > 0.0 {
                means[k] = [a / s, b / s];
            }
            if active {
                let density = target.unwrap().log_densities()[row * m + k];
                if density > 0.0 {
                    corrections[k] = 2.0 * (ya - da / total_d * total_y) / (np as f64 * density);
                } else {
                    ess[k] = 0.0;
                }
            }
        }
        let usable: Vec<_> = (0..m)
            .filter(|&k| {
                sums[k] > 0.0 && ess[k] >= self.config.particles.minimum_effective_samples()
            })
            .collect();
        if usable.is_empty() {
            return Err(E::Invalid(
                "joint local correlation particle/density support is insufficient at every basket node",
            ));
        }
        for (k, &x) in xs.iter().enumerate() {
            let source = if usable.contains(&k) {
                k
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
            let correction = corrections[source];
            let variance = if let Some(t) = target {
                t.grid().values()[row * m + k]
            } else {
                self.config.target.interpolate(self.times[row], x)?.value
            };
            let span = b - a;
            let unidentifiable = span.abs() <= self.config.minimum_variance_span;
            let raw = if unidentifiable {
                0.0
            } else {
                (variance - correction - a) / span
            };
            let projected = if unidentifiable {
                (variance - correction - a).abs() > self.config.minimum_variance_span
            } else {
                !(0.0..=1.0).contains(&raw)
            };
            if !raw.is_finite() || ![a, b, correction].iter().all(|v| v.is_finite()) {
                return Err(E::Invalid("nonfinite discounted local correlation moments"));
            }
            if projected && self.config.feasibility == LocalCorrelationFeasibility::Reject {
                return Err(E::numerical(format!(
                    "infeasible joint local correlation target at time {}, log basket {}: target {}, correction {}, endpoints [{}, {}]",
                    self.times[row], x, variance, correction, a, b
                )));
            }
            let lambda = raw.clamp(0.0, 1.0);
            self.mixing[row * m + k] = lambda;
            self.weight_sums.push(sums[source]);
            self.diagnostics.push(LocalCorrelationNodeDiagnostics {
                endpoint_variances: [a, b],
                target_variance: variance,
                attained_variance: a + lambda * span + correction,
                rate_correction: correction,
                raw_mixing: raw,
                effective_samples: ess[k],
                source_node: source,
                fallback: source != k,
                projected,
                unidentifiable,
            });
        }
        Ok(())
    }
}
