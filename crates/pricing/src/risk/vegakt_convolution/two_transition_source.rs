//! Non-Gaussian marginal from two log-Euler transitions, without moment matching.
use super::gaussian_variance::legendre;
use super::{VegaKtConvolutionPlan, invalid};
use crate::risk::RiskConfigError;
use pricing_numerics::{NeumaierSum, standard_normal_cdf, standard_normal_pdf};

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtTwoTransitionSourceAccuracy {
    pub quadrature_order: usize,
    pub source_log_window: [f64; 2],
    pub source_probability: f64,
    pub source_first_spot_moment: f64,
    pub source_probability_tail_bound: f64,
    pub source_spot_tail_bound: f64,
    pub probability_refinement_difference: f64,
    pub spot_refinement_difference: f64,
    pub operator_refinement_difference: f64,
    pub omitted_operator_entry_bound: f64,
    pub operator_entry_tolerance: f64,
}

fn hat(nodes: &[f64], x: f64) -> (usize, f64) {
    let i = nodes
        .partition_point(|v| *v <= x)
        .saturating_sub(1)
        .min(nodes.len() - 2);
    (
        i,
        ((x - nodes[i]) / (nodes[i + 1] - nodes[i])).clamp(0., 1.),
    )
}
fn rule(nodes: &[f64], lo: f64, hi: f64, order: usize) -> Result<Vec<(f64, f64)>, RiskConfigError> {
    let mut cuts = vec![lo];
    cuts.extend(nodes.iter().copied().filter(|x| lo < *x && *x < hi));
    cuts.push(hi);
    let gl = legendre(order)?;
    Ok(cuts
        .windows(2)
        .flat_map(|c| {
            gl.iter().map(move |(x, w)| {
                (
                    (c[0] + c[1]) / 2. + (c[1] - c[0]) * x / 2.,
                    (c[1] - c[0]) * w / 2.,
                )
            })
        })
        .collect())
}

impl VegaKtConvolutionPlan {
    /// The source is the actual two-transition Gaussian mixture. The third
    /// transition uses its own state-dependent annual variance. No renormalizing
    /// or fitting source moments is permitted. Model rows use clamped linear hats.
    pub fn from_two_transition_variance_source(
        spot: f64,
        first_variance: f64,
        subsequent_dts: [f64; 2],
        nodes: &[f64],
        variance_rows: [&[f64]; 2],
        strikes: Vec<f64>,
        lambda: f64,
    ) -> Result<(Self, VegaKtTwoTransitionSourceAccuracy), RiskConfigError> {
        let m = nodes.len();
        let n = strikes.len();
        let [second, third] = variance_rows;
        let [dt2, dt3] = subsequent_dts;
        if !spot.is_finite()
            || spot <= 0.
            || !first_variance.is_finite()
            || first_variance <= 0.
            || subsequent_dts.iter().any(|v| !v.is_finite() || *v <= 0.)
            || !(2..=65).contains(&m)
            || !(2..=65).contains(&n)
            || nodes.iter().any(|x| !x.is_finite())
            || nodes.windows(2).any(|p| p[0] >= p[1])
            || variance_rows
                .iter()
                .any(|r| r.len() != m || r.iter().any(|v| !v.is_finite() || *v <= 0.))
            || strikes.iter().any(|k| !k.is_finite() || *k <= 0.)
            || strikes.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(invalid(
                "invalid two-transition source variance model or 2..65 node budget",
            ));
        }
        let sd1 = first_variance.sqrt();
        let mu1 = -first_variance / 2.;
        let vmax = first_variance + second.iter().copied().fold(0., f64::max) * dt2;
        let lo = -vmax / 2. - 12. * vmax.sqrt();
        let hi = vmax / 2. + 12. * vmax.sqrt();
        if !lo.is_finite() || !hi.is_finite() || hi > 100. {
            return Err(invalid(
                "two-transition source window or tail budget overflow",
            ));
        }
        // Chernoff bound for the adapted exponential martingale with total
        // quadratic variation <= vmax. First integration truncation is separate.
        let probability_tail = 2. * standard_normal_cdf(-12.) + 2. * (-72_f64).exp();
        let stock_tail = spot
            * (standard_normal_cdf(-12. - sd1)
                + standard_normal_cdf(-12. + sd1)
                + (1. + lo.exp()) * (-72_f64).exp());
        let min3 = third.iter().copied().fold(f64::INFINITY, f64::min);
        let tail = stock_tail * standard_normal_pdf(0.) * dt3.sqrt() / (2. * min3.sqrt());
        let integrate = |order| -> Result<(Vec<Vec<f64>>, f64, f64), RiskConfigError> {
            let first: Vec<_> = rule(nodes, mu1 - 12. * sd1, mu1 + 12. * sd1, order)?
                .into_iter()
                .map(|(x, w)| {
                    let (i, h) = hat(nodes, x);
                    let v = ((1. - h) * second[i] + h * second[i + 1]) * dt2;
                    (
                        x - v / 2.,
                        v.sqrt(),
                        w * standard_normal_pdf((x - mu1) / sd1) / sd1,
                    )
                })
                .collect();
            let mut sums = vec![vec![NeumaierSum::new(); n]; m];
            let mut mass = NeumaierSum::new();
            let mut stock = NeumaierSum::new();
            for (x, w) in rule(nodes, lo, hi, order)? {
                let density = first
                    .iter()
                    .map(|(mu, sd, p)| p * standard_normal_pdf((x - mu) / sd) / sd)
                    .collect::<NeumaierSum>()
                    .total();
                let probability = w * density;
                let s = spot * x.exp();
                mass.add(probability);
                stock.add(probability * s);
                let (i, h) = hat(nodes, x);
                let sd = (((1. - h) * third[i] + h * third[i + 1]) * dt3).sqrt();
                for (j, k) in strikes.iter().enumerate() {
                    let d1 = (spot.ln() - k.ln() + x) / sd + sd / 2.;
                    let value = probability * s * standard_normal_pdf(d1) * dt3 / (2. * sd);
                    if !value.is_finite() {
                        return Err(invalid("nonfinite two-transition source integrand"));
                    }
                    sums[i][j].add((1. - h) * value);
                    sums[i + 1][j].add(h * value);
                }
            }
            Ok((
                sums.into_iter()
                    .map(|r| r.into_iter().map(NeumaierSum::total).collect())
                    .collect(),
                mass.total(),
                stock.total(),
            ))
        };
        let (coarse, cm, cs) = integrate(32)?;
        let (fine, fm, fs) = integrate(64)?;
        let difference = coarse
            .iter()
            .flatten()
            .zip(fine.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let scale = fine.iter().flatten().copied().fold(0., f64::max);
        let tolerance = 1e-10 * scale;
        if [fm, fs, tail, stock_tail].iter().any(|v| !v.is_finite())
            || scale <= 0.
            || difference > tolerance
            || tail > tolerance
            || (cm - fm).abs() > 1e-10
            || (cs - fs).abs() > 1e-10 * spot
            || (fm - 1.).abs() > 1e-10 + probability_tail
            || (fs - spot).abs() > 1e-10 * spot + stock_tail
        {
            return Err(invalid(
                "two-transition source quadrature, mass, martingale or tail budget failed",
            ));
        }
        let accuracy = VegaKtTwoTransitionSourceAccuracy {
            quadrature_order: 64,
            source_log_window: [lo, hi],
            source_probability: fm,
            source_first_spot_moment: fs,
            source_probability_tail_bound: probability_tail,
            source_spot_tail_bound: stock_tail,
            probability_refinement_difference: (cm - fm).abs(),
            spot_refinement_difference: (cs - fs).abs(),
            operator_refinement_difference: difference,
            omitted_operator_entry_bound: tail,
            operator_entry_tolerance: tolerance,
        };
        Ok((Self::from_operator(fine, strikes, lambda)?, accuracy))
    }

    pub(crate) fn with_unconstrained_regularization(
        &self,
        lambda: f64,
    ) -> Result<Self, RiskConfigError> {
        Self::from_operator(self.forward.clone(), self.strikes.clone(), lambda)
    }
}
