//! Fixed, simultaneous linear moment constraints on the Tikhonov inverse.
use super::{VegaKtConvolutionPlan, dot, invalid};
use crate::risk::RiskConfigError;
use pricing_numerics::NeumaierSum;

impl VegaKtConvolutionPlan {
    /// Minimize the original ridge objective subject to D' B q = D' b.
    /// Each supplied row is one local-volatility direction on source nodes.
    /// Optionally preserve the total in the SAME solve. Constraints must be
    /// independent in the operator metric; redundant/degenerate ones fail.
    /// No payoff samples, mass rescaling or positivity assumptions enter here.
    pub fn conserve_local_vega_directions(
        mut self,
        directions: Vec<Vec<f64>>,
        preserve_total: bool,
    ) -> Result<Self, RiskConfigError> {
        if self.preserves_local_vega_total || !self.conservation_directions.is_empty() {
            return Err(invalid(
                "compile all Gamma conservation constraints together",
            ));
        }
        let (m, n) = (self.forward.len(), self.inverse.len());
        if directions.is_empty() || directions.len() > 8 {
            return Err(invalid("1..8 Gamma conservation directions required"));
        }
        let mut d = directions.clone();
        if preserve_total {
            d.insert(0, vec![1.; m]);
        }
        // Normalization changes neither the constraint nor its mathematical
        // solution, and prevents the caller's units from deciding numerical rank.
        for row in &mut d {
            let norm = row.iter().fold(0., |a, &b| f64::hypot(a, b));
            if row.len() != m
                || row.iter().any(|v| !v.is_finite())
                || !norm.is_finite()
                || norm <= 0.
            {
                return Err(invalid(
                    "finite nonzero Gamma direction with source-node width required",
                ));
            }
            row.iter_mut().for_each(|v| *v /= norm);
        }
        let k = d.len();
        if k > n.min(m) {
            return Err(invalid(
                "too many independent Gamma conservation constraints",
            ));
        }
        // R = H^-1 B'; Q = R D; C = B' D; G = C' Q = C' H^-1 C.
        // R_D = R + Q G^-1 (D' - C' R). Columns are stored as rows here.
        let q: Vec<Vec<f64>> = d
            .iter()
            .map(|v| self.inverse.iter().map(|r| dot(r, v)).collect())
            .collect();
        let c: Vec<Vec<f64>> = d
            .iter()
            .map(|v| {
                (0..n)
                    .map(|j| {
                        (0..m)
                            .map(|i| self.forward[i][j] * v[i])
                            .collect::<NeumaierSum>()
                            .total()
                    })
                    .collect()
            })
            .collect();
        let mut scale = Vec::with_capacity(k);
        for i in 0..k {
            let g = dot(&c[i], &q[i]);
            if !g.is_finite() || g <= 0. {
                return Err(invalid("degenerate Gamma conservation constraint"));
            }
            scale.push(g.sqrt());
        }
        let mut lower = vec![vec![0.; k]; k];
        for i in 0..k {
            for j in 0..=i {
                let g = (0.5 * dot(&c[i], &q[j]) + 0.5 * dot(&c[j], &q[i])) / scale[i] / scale[j];
                let pivot = g - dot(&lower[i][..j], &lower[j][..j]);
                if i == j {
                    // Fixed rank guard in the equilibrated operator metric.
                    if !pivot.is_finite() || pivot <= 1e-10 {
                        return Err(invalid("rank deficient Gamma conservation directions"));
                    }
                    lower[i][j] = pivot.sqrt();
                } else {
                    lower[i][j] = pivot / lower[j][j];
                }
            }
        }
        for column in 0..m {
            let original: Vec<_> = self.inverse.iter().map(|r| r[column]).collect();
            let mut y = vec![0.; k];
            for i in 0..k {
                let rhs = (d[i][column] - dot(&c[i], &original)) / scale[i];
                y[i] = (rhs - dot(&lower[i][..i], &y[..i])) / lower[i][i];
            }
            let mut z = vec![0.; k];
            for i in (0..k).rev() {
                let s = (i + 1..k)
                    .map(|j| lower[j][i] * z[j])
                    .collect::<NeumaierSum>()
                    .total();
                z[i] = (y[i] - s) / lower[i][i];
            }
            for (j, row) in self.inverse.iter_mut().enumerate() {
                row[column] += (0..k)
                    .map(|i| q[i][j] * z[i] / scale[i])
                    .collect::<NeumaierSum>()
                    .total();
                if !row[column].is_finite() {
                    return Err(invalid("nonfinite direction-constrained Gamma inverse"));
                }
            }
        }
        // Verify the linear identities on every input column, not only on a
        // conveniently chosen payoff or on its aggregate Monte Carlo mean.
        for i in 0..k {
            for (column, &target) in d[i].iter().enumerate() {
                let value = (0..n)
                    .map(|j| c[i][j] * self.inverse[j][column])
                    .collect::<NeumaierSum>()
                    .total();
                if !value.is_finite() || (value - target).abs() > 1e-9 {
                    return Err(invalid(
                        "Gamma conservation matrix residual exceeds tolerance",
                    ));
                }
            }
        }
        self.inverse_noise_gain_bound = self
            .inverse
            .iter()
            .flatten()
            .fold(0., |n, &v| f64::hypot(n, v));
        self.resolution_trace = (0..n)
            .map(|j| {
                (0..m)
                    .map(|i| self.inverse[j][i] * self.forward[i][j])
                    .collect::<NeumaierSum>()
                    .total()
            })
            .collect::<NeumaierSum>()
            .total();
        if !self.inverse_noise_gain_bound.is_finite() || !self.resolution_trace.is_finite() {
            return Err(invalid(
                "nonfinite direction-constrained inverse diagnostics",
            ));
        }
        self.preserves_local_vega_total = preserve_total;
        self.conservation_directions = directions;
        Ok(self)
    }
}
