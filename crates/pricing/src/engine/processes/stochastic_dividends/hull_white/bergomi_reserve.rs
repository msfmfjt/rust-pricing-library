//! Conditional discounted cash for a one-factor Bergomi residual share.
//! Under the dividend-date forward measure E[Y(T)] = A(t,X) f + B(t) Y + C(t).
//! Only A depends on the volatility factor. Propagate its difference from the
//! constant-volatility A, retaining the exact existing B, C and Black limit.
use super::*;
use crate::models::Bergomi1Factor;

const GRID_POINTS: usize = 257;
// Seven-point Gauss-Hermite quadrature for a standard normal.
const GAUSSIAN: [(f64, f64); 7] = [
    (-3.7504397177257425, 0.000548268855972218),
    (-2.3667594107345415, 0.0307571239675865),
    (-1.1544053947399682, 0.2401231786050127),
    (0., 0.4571428571428571),
    (1.1544053947399682, 0.2401231786050127),
    (2.3667594107345415, 0.0307571239675865),
    (3.7504397177257425, 0.000548268855972218),
];

#[derive(Clone, Debug)]
pub(super) struct ReserveRow {
    width: f64,
    values: Box<[f64]>,
}

impl ReserveRow {
    fn interpolate(&self, x: f64) -> (f64, f64) {
        let last = self.values.len() - 1;
        let scale = last as f64 / (2. * self.width);
        let position = ((x + self.width) * scale).clamp(0., last as f64);
        let i = (position as usize).min(last - 1);
        let diff = self.values[i + 1] - self.values[i];
        (self.values[i] + (position - i as f64) * diff, diff * scale)
    }

    pub(super) fn at(&self, x: f64) -> Result<(f64, f64), StochasticDividendError> {
        if !x.is_finite() || x.abs() > self.width {
            return Err(invalid(
                "conditional dividend volatility factor outside reserve grid",
            ));
        }
        Ok(self.interpolate(x))
    }
}

pub(super) struct ReserveModel<'a> {
    pub factor: Bergomi1Factor,
    pub snapshot: BergomiAsOf,
    pub initial_volatility: f64,
    pub dividend: BuehlerDividendModel,
    pub rates: &'a HullWhite1Factor,
    pub stock_rate: f64,
    pub dividend_rate: f64,
}

impl ReserveModel<'_> {
    pub fn rows(
        &self,
        simulation: &[f64],
        maturity: f64,
    ) -> Result<Vec<Option<ReserveRow>>, MonteCarloError> {
        let count = simulation.partition_point(|&t| t < maturity);
        let mut rows = vec![None; count];
        let eta = self.factor.vol_of_vol();
        let sigma0 = self.initial_volatility;
        let k = self.dividend.mean_reversion();
        let alpha = self.dividend.equity_linkage();
        let a = self.factor.mean_reversion();
        let width = self.snapshot.x.abs() + 9. * b(2. * a, maturity).sqrt();
        let mut next = ReserveRow {
            width,
            values: vec![0.; GRID_POINTS].into_boxed_slice(),
        };
        let mut boundaries = simulation[..count].to_vec();
        boundaries.extend(
            self.rates
                .volatility_times()
                .iter()
                .copied()
                .filter(|&t| t > 0. && t < maturity),
        );
        boundaries.push(maturity);
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup();
        // Cover reserve cash beyond the last payoff fixing. Smaller simulation
        // steps are retained, and long gaps in the cash tail are subdivided.
        let mut baseline_a = 0.;
        let mut baseline_b = 1.;
        for pair in boundaries.windows(2).rev() {
            let steps = ((pair[1] - pair[0]) * 52.).ceil().max(1.);
            if steps > 1_000_000. {
                return Err(invalid("dividend reserve time grid limit").into());
            }
            let h = (pair[1] - pair[0]) / steps;
            let rate_sigma = self.rates.volatilities()[self
                .rates
                .volatility_times()
                .partition_point(|&t| t <= pair[0])
                - 1];
            let tr = self
                .factor
                .transition(h)
                .map_err(|_| invalid("Bergomi reserve OU transition"))?;
            let a_rate = self.rates.mean_reversion();
            let half_decay = (-0.5 * k * h).exp();
            let blend = -(-0.5 * k * h).exp_m1();
            for j in (0..steps as usize).rev() {
                let start = pair[0] + j as f64 * h;
                let tail = (maturity - start - h).max(0.);
                // Integral over this step of sigma_r B_r(u,T).
                let bond_loading = rate_sigma
                    * (crate::models::hull_white::integrated_b(a_rate, h)
                        + b(a_rate, tail) * b(a_rate, h));
                let stock_shift = -self.stock_rate * bond_loading;
                let dividend_mean =
                    (-self.dividend_rate * self.dividend.dividend_volatility() * bond_loading)
                        .exp();
                let baseline_mean = (sigma0 * stock_shift).exp();
                let carry = baseline_a + baseline_b * blend * alpha;
                let center = eta * b(2. * a, self.snapshot.age + start);
                let mut current = Vec::with_capacity(GRID_POINTS);
                for i in 0..GRID_POINTS {
                    let x = width * (2. * i as f64 / (GRID_POINTS - 1) as f64 - 1.);
                    let sigma = sigma0 * (eta * (x - center)).exp();
                    // Exponential tilt by the residual equity increment. The FX
                    // quanto shift belongs only to the OUTER settlement-measure path.
                    let mean = tr.decay * x + tr.covariance * (sigma + stock_shift / h);
                    let expectation: f64 = GAUSSIAN
                        .iter()
                        .map(|(z, w)| w * next.interpolate(mean + tr.variance.sqrt() * z).0)
                        .sum();
                    let value = (sigma * stock_shift).exp() * expectation
                        + baseline_mean * ((sigma - sigma0) * stock_shift).exp_m1() * carry;
                    if !value.is_finite() {
                        return Err(invalid("nonfinite conditional dividend coefficient").into());
                    }
                    current.push(value);
                }
                next.values = current.into_boxed_slice();
                baseline_a =
                    baseline_mean * carry + baseline_b * blend * alpha * half_decay * dividend_mean;
                baseline_b *= half_decay * half_decay * dividend_mean;
            }
            if let Ok(i) = simulation[..count].binary_search_by(|t| t.total_cmp(&pair[0])) {
                rows[i] = Some(next.clone());
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::{Philox4x32, RandomCoordinate, RandomDomain};

    #[test]
    fn conditional_cash_at_nonzero_volatility_state_against_forward_simulation() {
        // Independent forward Feynman–Kac integral for the f coefficient of Y.
        // Simulate under the stock tilt, and integrate the rate-covariance weight.
        let factor = Bergomi1Factor::new(0.85, 0.82, -0.55).unwrap();
        let rate = HullWhite1Factor::new(0.32, vec![0.], vec![0.22]).unwrap();
        let model = ReserveModel {
            factor,
            snapshot: BergomiAsOf {
                age: 0.43,
                x: 0.82,
                ..Default::default()
            },
            initial_volatility: 0.45,
            dividend: BuehlerDividendModel::new(1.1, 0.8, 0.31, 0.45).unwrap(),
            rates: &rate,
            stock_rate: 0.72,
            dividend_rate: 0.72 * 0.45,
        };
        let t_end = 1.7;
        let row = model.rows(&[0., 0.28], t_end).unwrap().remove(0).unwrap();
        let (calculated, slope) = row.at(model.snapshot.x).unwrap();
        assert!(calculated < -0.005 && slope < 0.);
        let n = 240;
        let dt = t_end / n as f64;
        let terms: Vec<_> = (0..n)
            .map(|j| {
                let t = j as f64 * dt;
                let remaining = t_end - t - 0.5 * dt;
                let k = factor.mean_reversion();
                let q = 0.22 * b(0.32, remaining);
                let by = (-1.1 * remaining
                    - model.dividend_rate
                        * 0.31
                        * 0.22
                        * crate::models::hull_white::integrated_b(0.32, remaining))
                .exp();
                (
                    q,
                    0.8 * 1.1 * dt * by,
                    factor.vol_of_vol() * b(2. * k, model.snapshot.age + t),
                )
            })
            .collect();
        let rng = Philox4x32::from_seed(9182);
        let mut sum = 0.;
        let mut sum2 = 0.;
        let paths = 4000;
        let tr = factor.transition(dt).unwrap();
        for path in 0..paths {
            let mut x = [model.snapshot.x; 2];
            let mut weight = [1.; 2];
            let mut base = 1.;
            let mut integral = 0.;
            for (j, &(q, source, center)) in terms.iter().enumerate() {
                let z = rng.standard_normal(RandomCoordinate::new(
                    path,
                    j as u32,
                    RandomDomain::Valuation,
                ));
                let ref_step = (-model.stock_rate * model.initial_volatility * q * dt).exp();
                for (i, signed) in [z, -z].iter().enumerate() {
                    let vol =
                        model.initial_volatility * (factor.vol_of_vol() * (x[i] - center)).exp();
                    let step = (-model.stock_rate * vol * q * dt).exp();
                    integral += 0.5 * source * (weight[i] * step.sqrt() - base * ref_step.sqrt());
                    weight[i] *= step;
                    x[i] = tr.decay * x[i]
                        + b(factor.mean_reversion(), dt)
                            * factor.correlation()
                            * (vol - model.stock_rate * q)
                        + tr.variance.sqrt() * signed;
                }
                base *= ref_step;
            }
            sum += integral;
            sum2 += integral * integral;
        }
        let average = sum / paths as f64;
        let error = ((sum2 / paths as f64 - average * average) / (paths - 1) as f64).sqrt();
        assert!(
            (calculated - average).abs() < 5. * error + 0.00025,
            "backward {calculated}, forward {average} +/- {error}"
        );
    }
}
