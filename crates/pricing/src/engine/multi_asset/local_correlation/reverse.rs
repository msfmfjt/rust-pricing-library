use super::*;

impl LocalCorrelationCalibration {
    fn reverse_sigma(
        &self,
        asset: usize,
        row: usize,
        m: f64,
        sigma_bar: f64,
        state_bar: &mut f64,
        parameters: &mut [f64],
    ) -> Result<(), E> {
        match &self.models[asset] {
            ModelSpec::BlackScholes(_) => parameters[0] += sigma_bar,
            ModelSpec::LocalVolatility(lv) => {
                let grid = lv.local_variance_grid();
                let value = grid.interpolate(self.times[row], m.ln())?;
                let variance_bar = sigma_bar / (2.0 * value.value.sqrt());
                *state_bar += variance_bar * grid.interpolation_log_moneyness_derivative(value) / m;
                value.transpose_accumulate(
                    variance_bar,
                    parameters,
                    grid.log_moneyness_nodes().len(),
                );
            }
            _ => unreachable!("validated model"),
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn reverse_step(
        &self,
        row: usize,
        states: &[f64],
        next: &[f64],
        independent: &[f64],
        next_bar: &[f64],
        parameter_bar: &mut [Vec<f64>],
        mixing_bar: &mut [f64],
    ) -> Result<Vec<f64>, E> {
        let b = self.basket(states);
        let lookup = self.lookup(row, b.ln());
        let lambda = lookup.value;
        let z = self.endpoint_normals(row, independent);
        let dt = self.times[row + 1] - self.times[row];
        let mut bars = vec![0.0; states.len()];
        let mut lambda_bar = 0.0;
        for i in 0..states.len() {
            let sigma = self.sigma(i, row, states[i])?;
            let noise = (1.0 - lambda).sqrt() * z[i][0] + lambda.sqrt() * z[i][1];
            let exponent_bar = next_bar[i] * next[i];
            bars[i] += next_bar[i] * next[i] / states[i];
            self.reverse_sigma(
                i,
                row,
                states[i],
                exponent_bar * (-sigma * dt + dt.sqrt() * noise),
                &mut bars[i],
                &mut parameter_bar[i],
            )?;
            // Strictly projected nodes are locally constant. Exact active-set
            // transitions are rejected by the public AAD entry point.
            if lambda > 0.0 && lambda < 1.0 {
                lambda_bar += exponent_bar
                    * sigma
                    * dt.sqrt()
                    * (-z[i][0] / (2.0 * (1.0 - lambda).sqrt()) + z[i][1] / (2.0 * lambda.sqrt()));
            }
        }
        lookup.transpose(lambda_bar, mixing_bar);
        for (i, bar) in bars.iter_mut().enumerate() {
            *bar += lambda_bar * lookup.derivative * self.config.basket_weights[i] / b;
        }
        Ok(bars)
    }
    pub(in crate::engine::multi_asset) fn reverse_path(
        &self,
        path: &LocalCorrelationPath,
        seeds: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), E> {
        let n = self.models.len();
        let nt = self.times.len();
        if seeds.len() != n || seeds.iter().any(|v| v.len() != nt) {
            return Err(E::Invalid("local correlation state-adjoint dimensions"));
        }
        let mut parameter_bar = self.empty_asset_adjoints();
        let mut mixing_bar = vec![0.0; self.mixing.len()];
        let mut state_bar: Vec<_> = seeds.iter().map(|v| v[nt - 1]).collect();
        for row in (0..nt - 1).rev() {
            let z: Vec<_> = path.independent.iter().map(|v| v[row]).collect();
            state_bar = self.reverse_step(
                row,
                &path.states[row],
                &path.states[row + 1],
                &z,
                &state_bar,
                &mut parameter_bar,
                &mut mixing_bar,
            )?;
            for i in 0..n {
                state_bar[i] += seeds[i][row];
            }
        }
        Ok((parameter_bar, mixing_bar))
    }
    /// Transpose of the finite interacting-particle calibration. Model axes,
    /// correlation endpoints, weights, bandwidth, seed and donor choices are fixed.
    pub(in crate::engine::multi_asset) fn reverse_calibration(
        &self,
        seeds: &[f64],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), E> {
        let trace = self.trace.as_ref().ok_or(E::Invalid(
            "local correlation AAD requires retain_reverse_trace=true",
        ))?;
        if seeds.len() != self.mixing.len() || seeds.iter().any(|v| !v.is_finite()) {
            return Err(E::Invalid(
                "local correlation mixing-adjoint dimensions/values",
            ));
        }
        let n = self.models.len();
        let m = self.log_nodes().len();
        let nt = self.times.len();
        let np = self.config.particles.particle_count();
        let mut parameter_bar = self.empty_asset_adjoints();
        let mut basket_bar = vec![0.0; self.config.target.values().len()];
        let mut mixing_bar = seeds.to_vec();
        let mut state_bar = vec![vec![0.0; n]; np];
        for row in (0..nt).rev() {
            if row + 1 < nt {
                for (p, bar) in state_bar.iter_mut().enumerate() {
                    *bar = self.reverse_step(
                        row,
                        &trace[row][p],
                        &trace[row + 1][p],
                        &self.calibration_normals(p, row)?,
                        bar,
                        &mut parameter_bar,
                        &mut mixing_bar,
                    )?;
                }
            }
            let mut moment_bar = vec![[0.0; 2]; m];
            for j in 0..m {
                let index = row * m + j;
                let diagnostic = &self.diagnostics[index];
                if diagnostic.projected || diagnostic.unidentifiable {
                    continue;
                }
                let lambda = self.mixing[index];
                let span = diagnostic.endpoint_variances[1] - diagnostic.endpoint_variances[0];
                let ratio = mixing_bar[index] / span;
                self.config
                    .target
                    .interpolate(self.times[row], self.log_nodes()[j])?
                    .transpose_accumulate(ratio, &mut basket_bar, m);
                moment_bar[diagnostic.source_node][0] -= ratio * (1.0 - lambda);
                moment_bar[diagnostic.source_node][1] -= ratio * lambda;
            }
            for (j, moment_seed) in moment_bar
                .into_iter()
                .enumerate()
                .filter(|(_, v)| v.iter().any(|b| *b != 0.0))
            {
                let sum = self.weight_sums[row * m + j];
                let mean = self.diagnostics[row * m + j].endpoint_variances;
                for (p, states) in trace[row].iter().enumerate() {
                    let b = self.basket(states);
                    let u = (b.ln() - self.log_nodes()[j]) / self.config.particles.log_bandwidth();
                    let w = if row == 0 { 1.0 } else { quartic(u) };
                    if w == 0.0 {
                        continue;
                    }
                    let q = self.endpoint_variances(row, states)?;
                    let qbar = moment_seed.map(|v| v * w / sum);
                    self.reverse_endpoint_variances(
                        row,
                        states,
                        q,
                        qbar,
                        &mut state_bar[p],
                        &mut parameter_bar,
                    )?;
                    if row > 0 {
                        let wb = (moment_seed[0] * (q[0] - mean[0])
                            + moment_seed[1] * (q[1] - mean[1]))
                            / sum;
                        let log_bar =
                            wb * quartic_derivative(u) / self.config.particles.log_bandwidth();
                        for (i, bar) in state_bar[p].iter_mut().enumerate() {
                            *bar += log_bar * self.config.basket_weights[i] / b;
                        }
                    }
                }
            }
        }
        if basket_bar
            .iter()
            .chain(parameter_bar.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(E::Invalid("nonfinite calibrated local correlation adjoint"));
        }
        Ok((parameter_bar, basket_bar))
    }
    fn reverse_endpoint_variances(
        &self,
        row: usize,
        states: &[f64],
        q: [f64; 2],
        seeds: [f64; 2],
        state_bar: &mut [f64],
        parameter_bar: &mut [Vec<f64>],
    ) -> Result<(), E> {
        let n = states.len();
        let b = self.basket(states);
        let sigmas = (0..n)
            .map(|i| self.sigma(i, row, states[i]))
            .collect::<Result<Vec<_>, _>>()?;
        let a: Vec<_> = (0..n)
            .map(|i| self.config.basket_weights[i] * states[i] * sigmas[i])
            .collect();
        let ends = &self.endpoints[self.entries[row]];
        for e in 0..2 {
            let basket_bar = -2.0 * seeds[e] * q[e] / b;
            for i in 0..n {
                let asset_bar = 2.0 * seeds[e] / (b * b)
                    * (0..n)
                        .map(|j| ends[e].canonical()[i * n + j] * a[j])
                        .sum::<f64>();
                let w = self.config.basket_weights[i];
                state_bar[i] += asset_bar * w * sigmas[i] + basket_bar * w;
                self.reverse_sigma(
                    i,
                    row,
                    states[i],
                    asset_bar * w * states[i],
                    &mut state_bar[i],
                    &mut parameter_bar[i],
                )?;
            }
        }
        Ok(())
    }
}
