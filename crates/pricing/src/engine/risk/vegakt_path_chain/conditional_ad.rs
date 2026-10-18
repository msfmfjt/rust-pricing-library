//! Exact last-shock integration of native pathwise AD, independent of Gamma.
use super::measure::{invalid, sum};
use crate::MonteCarloError;
use crate::mc::LocalVolPath;
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};

#[derive(Clone, Copy)]
struct Moments {
    probability: f64,
    exponential: f64,
    z_exponential: f64,
    squared_exponential: f64,
    z_squared_exponential: f64,
    z2_squared_exponential: f64,
}
fn normal_moments(lo: f64, hi: f64) -> [f64; 3] {
    let p = if lo >= 0. {
        cdf(-lo) - cdf(-hi)
    } else {
        cdf(hi) - cdf(lo)
    };
    let edge = |z: f64| if z.is_finite() { z * pdf(z) } else { 0. };
    [p, pdf(lo) - pdf(hi), p + edge(lo) - edge(hi)]
}
impl Moments {
    fn new(source: f64, variance: f64, lo: f64, hi: f64) -> Self {
        let sd = variance.sqrt();
        let z = |s: f64| (s.ln() - source.ln() + variance / 2.) / sd;
        let a = z(lo);
        let b = z(hi);
        let [p, _, _] = normal_moments(a, b);
        let [p1, q1, _] = normal_moments(a - sd, b - sd);
        let [p2, q2, r2] = normal_moments(a - 2. * sd, b - 2. * sd);
        let e = variance.exp();
        Self {
            probability: p,
            exponential: p1,
            z_exponential: sd * p1 + q1,
            squared_exponential: e * p2,
            z_squared_exponential: e * (2. * sd * p2 + q2),
            z2_squared_exponential: e * (4. * variance * p2 + 4. * sd * q2 + r2),
        }
    }
    fn integrate(self, a: f64, b: f64, c: f64) -> (f64, f64) {
        // X(Z)=a+exp(-V/2+sqrt(V) Z)*(b+c Z), on one payoff branch.
        let mean = sum([
            a * self.probability,
            b * self.exponential,
            c * self.z_exponential,
        ]
        .into_iter());
        let second = sum([
            a * a * self.probability,
            2. * a * b * self.exponential,
            2. * a * c * self.z_exponential,
            b * b * self.squared_exponential,
            2. * b * c * self.z_squared_exponential,
            c * c * self.z2_squared_exponential,
        ]
        .into_iter());
        (mean, second)
    }
}

pub(super) struct ConditionalAdjoints {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
}

pub(super) fn integrate(
    path: &LocalVolPath,
    grid_value_count: usize,
    nodes: usize,
    boundaries: &[f64],
    mut seeds: impl FnMut(f64) -> Result<Vec<f64>, MonteCarloError>,
) -> Result<ConditionalAdjoints, MonteCarloError> {
    let n = path.step_cache().len();
    let last = path.step_cache()[n - 1];
    let source = last.state_before;
    let variance = last.local_variance * last.dt;
    let sd = variance.sqrt();
    if !variance.is_finite() || variance <= 0. || sd > 4. {
        return Err(invalid("conditional AD variance outside precision domain").into());
    }
    let mut means = vec![NeumaierSum::new(); grid_value_count];
    let mut seconds = means.clone();
    let mut left = 0.;
    for right in boundaries
        .iter()
        .copied()
        .chain(std::iter::once(f64::INFINITY))
    {
        let moments = Moments::new(source, variance, left, right);
        if moments.probability == 0.
            && moments.exponential == 0.
            && moments.squared_exponential == 0.
        {
            left = right;
            continue;
        }
        let point = if right.is_finite() {
            left + (right - left) / 2.
        } else {
            left + left.max(source)
        };
        if !point.is_finite() || point <= left || point >= right {
            return Err(invalid("conditional AD branch has no representable interior").into());
        }
        let mut local = seeds(point)?;
        let terminal_seed = local[n];
        local[n] = 0.;
        let a = path.reverse_state_adjoints(&local, grid_value_count, nodes)?;
        local.fill(0.);
        local[n - 1] = terminal_seed;
        let h = path.reverse_state_adjoints(&local, grid_value_count, nodes)?;
        let log_variance_derivative_dt = last.local_variance_state_derivative * source * last.dt;
        let mut b: Vec<_> = h
            .local_variance_value_adjoints()
            .iter()
            .map(|h| h * (1. - log_variance_derivative_dt / 2.))
            .collect();
        let mut c: Vec<_> = h
            .local_variance_value_adjoints()
            .iter()
            .map(|h| h * log_variance_derivative_dt / (2. * sd))
            .collect();
        last.interpolation.transpose_accumulate(
            -terminal_seed * source * last.dt / 2.,
            &mut b,
            nodes,
        );
        last.interpolation.transpose_accumulate(
            terminal_seed * source * last.dt / (2. * sd),
            &mut c,
            nodes,
        );
        for i in 0..grid_value_count {
            let (mean, second) =
                moments.integrate(a.local_variance_value_adjoints()[i], b[i], c[i]);
            means[i].add(mean);
            seconds[i].add(second);
        }
        left = right;
    }
    let means: Vec<_> = means.into_iter().map(NeumaierSum::total).collect();
    let variances = seconds
        .into_iter()
        .zip(&means)
        .map(|(second, mean)| {
            let second = second.total();
            let variance = second - mean * mean;
            let roundoff = 256. * f64::EPSILON * (second.abs() + mean * mean);
            if !mean.is_finite() || !variance.is_finite() || variance < -roundoff {
                return Err(invalid("nonfinite or negative conditional AD variance").into());
            }
            Ok(variance.max(0.))
        })
        .collect::<Result<Vec<_>, MonteCarloError>>()?;
    Ok(ConditionalAdjoints { means, variances })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::gaussian_legendre;

    #[test]
    fn all_node_conditional_adjoints_match_direct_path_reverse_quadrature() {
        use crate::market::LocalVarianceGrid;
        use crate::mc::{LocalVolLogEulerPlan, LocalVolTimeGrid};
        let times = vec![0., 0.2, 0.4, 0.6, 0.8];
        let grid = LocalVarianceGrid::new(
            times.clone(),
            vec![-0.8, 0., 0.8],
            vec![
                0.035, 0.04, 0.045, 0.045, 0.037, 0.032, 0.032, 0.042, 0.047, 0.03, 0.039, 0.052,
                0.03, 0.039, 0.052,
            ],
            1e-6,
            1.,
        )
        .unwrap();
        let plan =
            LocalVolLogEulerPlan::new(LocalVolTimeGrid::compile(times, 1.).unwrap(), vec![100.; 5])
                .unwrap();
        let mut shocks = [-1.2, 0.7, -0.4, 0.35];
        let path = plan.evolve_path(&grid, 100., &shocks).unwrap();
        let fixed = 0.2 * path.states()[1] + 0.3 * path.states()[2];
        let asian_strike = (100. - fixed) / 0.5;
        let mut cuts = vec![asian_strike, 115.];
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let seeds = |s: f64| {
            let active = if s > asian_strike { 1. } else { 0. };
            vec![
                0.,
                0.2 * active,
                0.3 * active,
                0.,
                0.5 * active - if s > 115. { 0.4 } else { 0. },
            ]
        };
        let actual = integrate(&path, 15, 3, &cuts, |s| Ok(seeds(s))).unwrap();
        let source = path.states()[3];
        let v = path.step_cache()[3].local_variance * 0.2;
        let mut zcuts = vec![-14., 14.];
        zcuts.extend(
            cuts.iter()
                .map(|k| (k.ln() - source.ln() + v / 2.) / v.sqrt()),
        );
        zcuts.sort_by(f64::total_cmp);
        let mut first = vec![NeumaierSum::new(); 15];
        let mut second = first.clone();
        for pair in zcuts.windows(2) {
            for (x, w) in gaussian_legendre(128).unwrap() {
                let z = (pair[0] + pair[1]) / 2. + (pair[1] - pair[0]) * x / 2.;
                shocks[3] = z;
                let sample = plan.evolve_path(&grid, 100., &shocks).unwrap();
                let ad = sample
                    .reverse_state_adjoints(&seeds(sample.states()[4]), 15, 3)
                    .unwrap();
                let weight = w * (pair[1] - pair[0]) / 2. * pdf(z);
                for (i, &value) in ad.local_variance_value_adjoints().iter().enumerate() {
                    first[i].add(weight * value);
                    second[i].add(weight * value * value);
                }
            }
        }
        for i in 0..15 {
            let mean = first[i].total();
            let variance = second[i].total() - mean * mean;
            assert!(
                (actual.means[i] - mean).abs() < 1e-9 * (1. + mean.abs()),
                "mean node {i}"
            );
            assert!(
                (actual.variances[i] - variance).abs() < 1e-8 * (1. + variance.abs()),
                "variance node {i}"
            );
        }
    }

    #[test]
    fn exponential_polynomial_moments_match_independent_quadrature() {
        for (source, variance, lo, hi) in [
            (100_f64, 0.013_f64, 0., 105.),
            (70., 0.03, 100., f64::INFINITY),
            (110., 0.007, 88., 124.),
        ] {
            for (a, b, c) in [(2_f64, -3_f64, 0.7_f64), (0., 1., -2.), (-17., 0., 0.)] {
                let moments = Moments::new(source, variance, lo, hi);
                let exact = moments.integrate(a, b, c);
                let sd = variance.sqrt();
                let z = |s: f64| (s.ln() - source.ln() + variance / 2.) / sd;
                let left = z(lo).max(-14.);
                let right = z(hi).min(14.);
                let mut first = NeumaierSum::new();
                let mut second = NeumaierSum::new();
                for (x, w) in gaussian_legendre(128).unwrap() {
                    let q = (left + right) / 2. + (right - left) * x / 2.;
                    let value = a + (-variance / 2. + sd * q).exp() * (b + c * q);
                    let weight = w * (right - left) / 2. * pdf(q);
                    first.add(weight * value);
                    second.add(weight * value * value);
                }
                assert!((first.total() - exact.0).abs() < 1e-10);
                assert!((second.total() - exact.1).abs() < 1e-8);
            }
        }
    }
}
