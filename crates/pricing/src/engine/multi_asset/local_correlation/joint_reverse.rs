//! Reverse the finite coupled simulation, including OU/Volterra histories,
//! stochastic discount weights and the centered rate-tail correction.
use super::*;

impl LocalCorrelationCalibration {
    #[allow(clippy::too_many_arguments)]
    fn reverse_joint_quote(
        &self,
        i: usize,
        row: usize,
        state: &[f64],
        seeds: [f64; 3],
        bars: &mut [Vec<f64>],
        parameters: &mut [f64],
    ) -> Result<(), E> {
        let j = self.joint.as_ref().unwrap();
        let Some(hw) = &j.assets[i].hw else {
            bars[row][i] += seeds[0];
            return Ok(());
        };
        let spot = hw.dividends.initial_spot();
        let r = j.rate.unwrap();
        let node = &hw.dividends.nodes()[row];
        let [f, zeta, h] = j.quote_state(i, row, state)?;
        let [fb, zb, hb] = seeds;
        let mut coefficients = node.zero_adjoints();
        let residual = node
            .reverse_target(
                spot * state[i],
                state[r],
                fb / spot,
                zb / spot,
                &mut coefficients,
            )
            .map_err(E::numerical)?;
        bars[row][i] += residual * spot;
        bars[row][r] += (fb * node.reserve_rate_derivative(state[r])
            + zb * node.reserve_loading_rate_derivative(state[r]))
            / (spot * node.scale());
        parameters[hw.volatility_parameter_count()] +=
            residual * state[i] - (fb * f + zb * zeta + hb * h) / spot;
        coefficients.deterministic_reserve += hb / (spot * node.scale());
        coefficients.scale -= hb * h / node.scale();
        hw.add_node_parameters(row, &coefficients, parameters);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn reverse_joint_sigma(
        &self,
        i: usize,
        row: usize,
        h: &[Vec<f64>],
        seed: f64,
        bars: &mut [Vec<f64>],
        parameters: &mut [f64],
    ) -> Result<(), E> {
        let j = self.joint.as_ref().unwrap();
        let Some(surface) = j.surface(i) else {
            return self.reverse_sigma(i, row, h[row][i], seed, &mut bars[row][i], parameters);
        };
        let f = if let Some(hw) = &j.assets[i].hw {
            hw.dividends.nodes()[row]
                .target_state(
                    h[row][i] * hw.dividends.initial_spot(),
                    h[row][j.rate.unwrap()],
                )
                .map_err(E::numerical)?
                .0
        } else {
            h[row][i]
        };
        let lookup = crate::engine::processes::hull_white::reverse::lookup(surface, row, f);
        let sigma = self.joint_sigma(i, row, h)?;
        let log_bar = seed * sigma;
        let lbar = 0.5 * log_bar / lookup.value;
        crate::engine::processes::hull_white::reverse::transpose_lookup(lookup, lbar, parameters);
        let fbar = lbar * lookup.slope / f;
        if let Some(hw) = &j.assets[i].hw {
            let spot = hw.dividends.initial_spot();
            let r = j.rate.unwrap();
            let node = &hw.dividends.nodes()[row];
            let mut coefficients = node.zero_adjoints();
            let residual_bar = node
                .reverse_target(spot * h[row][i], h[row][r], fbar, 0.0, &mut coefficients)
                .map_err(E::numerical)?;
            bars[row][i] += spot * residual_bar;
            bars[row][r] += fbar * node.reserve_rate_derivative(h[row][r]) / node.scale();
            parameters[hw.volatility_parameter_count()] +=
                residual_bar * h[row][i] - lbar * lookup.slope / spot;
            hw.add_node_parameters(row, &coefficients, parameters);
        } else {
            bars[row][i] += fbar;
        }
        let v = j.offsets[i];
        match j.configs[i].as_ref().unwrap() {
            MultiAssetBergomiLsvConfig::OneFactor(c) => {
                bars[row][v] += log_bar * c.factor.vol_of_vol()
            }
            MultiAssetBergomiLsvConfig::TwoFactor(c) => {
                let w = c.factor.normalized_weights();
                for k in 0..2 {
                    bars[row][v + k] += log_bar * c.factor.vol_of_vol() * w[k];
                }
            }
            MultiAssetBergomiLsvConfig::Rough(c) => {
                let seed = 0.5 * c.factor.vol_of_vol() * log_bar;
                if row > 0 {
                    if c.factor.hurst() == 0.5 {
                        bars[row][v] += seed;
                        bars[row - 1][v] -= seed;
                    } else {
                        bars[row][j.near[i].unwrap()] += seed;
                    }
                }
                for (k, &w) in j.rough_weights[i][row].iter().enumerate() {
                    bars[k + 1][v] += seed * w;
                    bars[k][v] -= seed * w;
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn reverse_joint_step(
        &self,
        row: usize,
        h: &[Vec<f64>],
        z: &[f64],
        bars: &mut [Vec<f64>],
        parameters: &mut [Vec<f64>],
        mixing: &mut [f64],
    ) -> Result<(), E> {
        let j = self.joint.as_ref().unwrap();
        let n = self.models.len();
        let dt = self.times[row + 1] - self.times[row];
        let b = self.joint_basket(row, &h[row])?;
        let lookup = self.lookup(row, b.ln());
        let lambda = lookup.value;
        let endpoints = j.endpoint_noises(row, z);
        let noise: Vec<_> = endpoints
            .iter()
            .map(|v| (1.0 - lambda).sqrt() * v[0] + lambda.sqrt() * v[1])
            .collect();
        let next_bar = bars[row + 1].clone();
        let mut noise_bar = vec![0.0; j.width];
        let mut integral_bar = 0.0;
        for i in 0..n {
            let sigma = self.joint_sigma(i, row, h)?;
            let exp_bar = next_bar[i] * h[row + 1][i];
            bars[row][i] += exp_bar / h[row][i];
            noise_bar[i] += exp_bar * sigma;
            integral_bar += exp_bar;
            self.reverse_joint_sigma(
                i,
                row,
                h,
                exp_bar * (-sigma * dt + noise[i]),
                bars,
                &mut parameters[i],
            )?;
            if let Some(c) = &j.configs[i] {
                for (k, f) in c.components().iter().enumerate() {
                    let v = j.offsets[i] + k;
                    bars[row][v] += next_bar[v] * (-f.mean_reversion() * dt).exp();
                    noise_bar[v] += next_bar[v];
                }
                if let Some(near) = j.near[i] {
                    noise_bar[near] += next_bar[near];
                }
            }
        }
        if let Some(r) = j.rate {
            let kernel = &j.assets[0].hw.as_ref().unwrap().process.kernels[row];
            integral_bar += next_bar[r + 1];
            bars[row][r + 1] += next_bar[r + 1];
            bars[row][r] += integral_bar * kernel.transition.integral_loading
                + next_bar[r] * kernel.transition.rate_decay;
            noise_bar[r] += next_bar[r];
            noise_bar[r + 1] += integral_bar;
        }
        let lambda_bar = if lambda > 0.0 && lambda < 1.0 {
            noise_bar
                .iter()
                .zip(&endpoints)
                .map(|(b, z)| {
                    b * (-z[0] / (2.0 * (1.0 - lambda).sqrt()) + z[1] / (2.0 * lambda.sqrt()))
                })
                .sum::<f64>()
        } else {
            0.0
        };
        lookup.transpose(lambda_bar, mixing);
        for (i, &w) in self.config.basket_weights.iter().enumerate() {
            self.reverse_joint_quote(
                i,
                row,
                &h[row],
                [lambda_bar * lookup.derivative * w / b, 0.0, 0.0],
                bars,
                &mut parameters[i],
            )?;
        }
        Ok(())
    }
    pub(super) fn reverse_joint_path(
        &self,
        path: &LocalCorrelationPath,
        seeds: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), E> {
        let width = self.joint.as_ref().unwrap().width;
        let nt = self.times.len();
        // Seeds are factor-major; absent auxiliary seeds are zero in deterministic mode.
        if !(seeds.len() == self.models.len() || seeds.len() == width)
            || seeds
                .iter()
                .any(|v| v.len() != nt || v.iter().any(|x| !x.is_finite()))
        {
            return Err(E::Invalid(
                "joint local correlation adjoint dimensions/values",
            ));
        }
        let mut bars = vec![vec![0.0; width]; nt];
        for (i, s) in seeds.iter().enumerate() {
            for r in 0..nt {
                bars[r][i] = s[r];
            }
        }
        let mut parameters = self.empty_asset_adjoints();
        let mut mixing = vec![0.0; self.mixing.len()];
        for row in (0..nt - 1).rev() {
            let z = path.independent.iter().map(|v| v[row]).collect::<Vec<_>>();
            self.reverse_joint_step(
                row,
                &path.states,
                &z,
                &mut bars,
                &mut parameters,
                &mut mixing,
            )?;
        }
        Ok((parameters, mixing))
    }
    fn reverse_joint_moments(
        &self,
        row: usize,
        h: &[Vec<f64>],
        q: [f64; 2],
        seeds: [f64; 2],
        bars: &mut [Vec<f64>],
        parameters: &mut [Vec<f64>],
    ) -> Result<(), E> {
        let n = self.models.len();
        let j = self.joint.as_ref().unwrap();
        let state = &h[row];
        let basket = self.joint_basket(row, state)?;
        let sigmas = (0..n)
            .map(|i| self.joint_sigma(i, row, h))
            .collect::<Result<Vec<_>, _>>()?;
        let a: Vec<_> = (0..n)
            .map(|i| self.config.basket_weights[i] * state[i] * sigmas[i])
            .collect();
        let mut rate_loading = 0.0;
        for (i, w) in self.config.basket_weights.iter().enumerate() {
            rate_loading += w * j.quote_state(i, row, state)?[1];
        }
        for e in 0..2 {
            let matrix = &self.endpoints[self.entries[row]][e];
            let cross: Vec<_> = (0..n)
                .map(|i| {
                    j.rate.map_or(0.0, |r| {
                        let c = &j.drivers[e].entries[self.entries[row]];
                        c.canonical()[i * c.dimension() + r]
                    })
                })
                .collect();
            let multiplier = 2.0 * seeds[e] / (basket * basket);
            let basket_bar = -2.0 * seeds[e] * q[e] / basket;
            let rate_bar =
                multiplier * (rate_loading + a.iter().zip(&cross).map(|(a, r)| a * r).sum::<f64>());
            for i in 0..n {
                let ab = multiplier
                    * ((0..n)
                        .map(|k| matrix.canonical()[i * n + k] * a[k])
                        .sum::<f64>()
                        + rate_loading * cross[i]);
                let w = self.config.basket_weights[i];
                bars[row][i] += ab * w * sigmas[i];
                self.reverse_joint_sigma(i, row, h, ab * w * state[i], bars, &mut parameters[i])?;
                self.reverse_joint_quote(
                    i,
                    row,
                    state,
                    [basket_bar * w, rate_bar * w, 0.0],
                    bars,
                    &mut parameters[i],
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn reverse_joint_calibration(
        &self,
        seeds: &[f64],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), E> {
        let trace = self.trace.as_ref().ok_or(E::Invalid(
            "local correlation AAD requires retain_reverse_trace=true",
        ))?;
        if seeds.len() != self.mixing.len() || seeds.iter().any(|v| !v.is_finite()) {
            return Err(E::Invalid("joint mixing adjoint dimensions/values"));
        }
        let j = self.joint.as_ref().unwrap();
        let nt = self.times.len();
        let np = self.config.particles.particle_count();
        let m = self.log_nodes().len();
        let histories: Vec<Vec<Vec<f64>>> = (0..np)
            .map(|p| trace.iter().map(|row| row[p].clone()).collect())
            .collect();
        let mut bars = vec![vec![vec![0.0; j.width]; nt]; np];
        let mut parameters = self.empty_asset_adjoints();
        let mut mixing = seeds.to_vec();
        let mut variance = vec![
            0.0;
            j.target
                .as_ref()
                .map_or(self.config.target.values().len(), |t| t
                    .grid()
                    .values()
                    .len())
        ];
        let mut density = vec![0.0; nt * m];
        for row in (0..nt).rev() {
            if row + 1 < nt {
                for p in 0..np {
                    self.reverse_joint_step(
                        row,
                        &histories[p],
                        &self.calibration_normals(p, row)?,
                        &mut bars[p],
                        &mut parameters,
                        &mut mixing,
                    )?;
                }
            }
            let mut moment_bar = vec![[0.0; 2]; m];
            let mut correction_bar = vec![0.0; m];
            for k in 0..m {
                let index = row * m + k;
                let d = &self.diagnostics[index];
                if d.projected || d.unidentifiable {
                    continue;
                }
                let lambda = self.mixing[index];
                let ratio = mixing[index] / (d.endpoint_variances[1] - d.endpoint_variances[0]);
                if j.target.is_some() {
                    variance[index] += ratio;
                } else {
                    self.config
                        .target
                        .interpolate(self.times[row], self.log_nodes()[k])?
                        .transpose_accumulate(ratio, &mut variance, m);
                }
                moment_bar[d.source_node][0] -= ratio * (1.0 - lambda);
                moment_bar[d.source_node][1] -= ratio * lambda;
                correction_bar[d.source_node] -= ratio;
            }
            let dr = histories
                .iter()
                .map(|h| self.joint_discount_rate(row, &h[row]))
                .collect::<Result<Vec<_>, _>>()?;
            let total_d = dr.iter().map(|v| v.0).sum::<f64>();
            let total_y = dr.iter().map(|v| v.0 * v.1).sum::<f64>();
            for k in 0..m {
                let mb = moment_bar[k];
                let cb = correction_bar[k];
                if mb == [0.0; 2] && cb == 0.0 {
                    continue;
                }
                let sum = self.weight_sums[row * m + k];
                let mean = self.diagnostics[row * m + k].endpoint_variances;
                let x = self.log_nodes()[k];
                for p in 0..np {
                    let h = &histories[p];
                    let b = self.joint_basket(row, &h[row])?;
                    let u = (b.ln() - x) / self.config.particles.log_bandwidth();
                    let kernel = if row == 0 { 1.0 } else { quartic(u) };
                    if kernel == 0.0 {
                        continue;
                    }
                    let (d, _) = dr[p];
                    let q = self.joint_moments(row, h)?;
                    let weight = d * kernel;
                    self.reverse_joint_moments(
                        row,
                        h,
                        q,
                        mb.map(|v| v * weight / sum),
                        &mut bars[p],
                        &mut parameters,
                    )?;
                    let wb = (mb[0] * (q[0] - mean[0]) + mb[1] * (q[1] - mean[1])) / sum;
                    if let Some(r) = j.rate {
                        bars[p][row][r + 1] -= wb * weight;
                    }
                    if row > 0 {
                        let log_bar =
                            wb * d * quartic_derivative(u) / self.config.particles.log_bandwidth();
                        for (i, &w) in self.config.basket_weights.iter().enumerate() {
                            self.reverse_joint_quote(
                                i,
                                row,
                                &h[row],
                                [log_bar * w / b, 0.0, 0.0],
                                &mut bars[p],
                                &mut parameters[i],
                            )?;
                        }
                    }
                }
                if self.correction_active(row) && cb != 0.0 {
                    let r = j.rate.unwrap();
                    let pdf = j.target.as_ref().unwrap().log_densities()[row * m + k];
                    let above: Vec<_> = histories
                        .iter()
                        .map(|h| self.joint_basket(row, &h[row]).map(|b| b.ln() > x))
                        .collect::<Result<_, _>>()?;
                    let da = dr
                        .iter()
                        .zip(&above)
                        .filter(|(_, a)| **a)
                        .map(|(v, _)| v.0)
                        .sum::<f64>();
                    let ya = dr
                        .iter()
                        .zip(&above)
                        .filter(|(_, a)| **a)
                        .map(|(v, _)| v.0 * v.1)
                        .sum::<f64>();
                    let q = ya - da / total_d * total_y;
                    let factor = 1.0 + self.joint_basket_shift(row) / x.exp();
                    let qbar = 2.0 * cb * factor / (np as f64 * pdf);
                    density[row * m + k] -= 2.0 * cb * factor * q / (np as f64 * pdf * pdf);
                    let hbar = 2.0 * cb * q / (np as f64 * pdf * x.exp());
                    for (i, &w) in self.config.basket_weights.iter().enumerate() {
                        self.reverse_joint_quote(
                            i,
                            row,
                            &histories[0][row],
                            [0.0, 0.0, hbar * w],
                            &mut bars[0],
                            &mut parameters[i],
                        )?;
                    }
                    // Empirically centered tails. Frozen indicators/donors define
                    // the finite-program derivative away from exact ties.
                    for p in 0..np {
                        let (d, residual) = dr[p];
                        let a = if above[p] { 1.0 } else { 0.0 };
                        let ybar = qbar * (a - da / total_d);
                        let dbar =
                            qbar * (-a * total_y / total_d + da * total_y / (total_d * total_d));
                        bars[p][row][r] += ybar * d;
                        bars[p][row][r + 1] -= d * (dbar + ybar * residual);
                    }
                }
            }
        }
        let basket = if let Some(t) = &j.target {
            let risk = j.risk_target.as_ref().unwrap();
            let g = risk.grid();
            let mut v = vec![0.0; g.values().len()];
            let mut d = v.clone();
            for (row, &time) in self.times.iter().enumerate() {
                for (k, &x) in self.log_nodes().iter().enumerate() {
                    let w = g.interpolate(time, x)?;
                    w.transpose_accumulate(variance[row * m + k], &mut v, m);
                    w.transpose_accumulate(density[row * m + k], &mut d, m);
                }
            }
            debug_assert_eq!(t.grid().values().len(), nt * m);
            let quotes = risk
                .market_iv_surface()
                .map(|_| risk.reverse_market_iv(&v, &d))
                .transpose()
                .map_err(E::numerical)?;
            v.extend(d);
            if let Some(q) = quotes {
                let parallel = q.iter().sum();
                v.extend(q);
                v.push(parallel);
            }
            v
        } else {
            variance
        };
        if basket
            .iter()
            .chain(parameters.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(E::Invalid(
                "nonfinite joint local correlation calibration adjoint",
            ));
        }
        Ok((parameters, basket))
    }
}
