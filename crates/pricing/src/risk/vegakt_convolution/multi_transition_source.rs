//! Successive integration of the actual log-Euler transition densities.
use super::gaussian_variance::legendre;
use super::{VegaKtConvolutionPlan, invalid};
use crate::risk::RiskConfigError;
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtSourceStepAccuracy {
    pub transition_count: usize,
    pub log_window: [f64; 2],
    pub probability: f64,
    pub first_spot_moment: f64,
    pub probability_tail_bound: f64,
    pub spot_tail_bound: f64,
    pub probability_refinement_difference: f64,
    pub spot_refinement_difference: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtMultiTransitionSourceAccuracy {
    pub quadrature_order: usize,
    pub steps: Vec<VegaKtSourceStepAccuracy>,
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
fn value(nodes: &[f64], row: &[f64], x: f64) -> f64 {
    let (i, h) = hat(nodes, x);
    (1. - h) * row[i] + h * row[i + 1]
}
fn rule(
    nodes: &[f64],
    window: [f64; 2],
    order: usize,
    max_width: f64,
) -> Result<Vec<(f64, f64)>, RiskConfigError> {
    let [lo, hi] = window;
    if !lo.is_finite()
        || !hi.is_finite()
        || hi > 100.
        || hi - lo <= 128. * f64::EPSILON * (1. + lo.abs() + hi.abs())
    {
        return Err(invalid(
            "multi-transition source window exceeds floating-point domain",
        ));
    }
    let mut cuts = vec![lo];
    cuts.extend(nodes.iter().copied().filter(|x| lo < *x && *x < hi));
    cuts.push(hi);
    let mut resolved = vec![lo];
    for cell in cuts.windows(2) {
        let pieces = ((cell[1] - cell[0]) / max_width).ceil().max(1.);
        if !pieces.is_finite()
            || pieces > 8192.
            || (resolved.len() + pieces as usize) * order > 8192
        {
            return Err(invalid(
                "multi-transition source quadrature node budget exceeded",
            ));
        }
        for j in 1..=pieces as usize {
            resolved.push(cell[0] + (cell[1] - cell[0]) * j as f64 / pieces);
        }
    }
    let cuts = resolved;
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
struct Integration {
    matrix: Vec<Vec<f64>>,
    steps: Vec<VegaKtSourceStepAccuracy>,
}
impl VegaKtConvolutionPlan {
    /// A terminal-row operator after any budgeted number of actual transitions.
    /// First-row variance is sampled at log spot zero; all earlier rows stay fixed.
    pub fn from_multi_transition_variance_source(
        spot: f64,
        times: &[f64],
        nodes: &[f64],
        variance_rows: &[Vec<f64>],
        strikes: Vec<f64>,
        lambda: f64,
    ) -> Result<(Self, VegaKtMultiTransitionSourceAccuracy), RiskConfigError> {
        let m = nodes.len();
        let count = variance_rows.len();
        let n = strikes.len();
        if !(2..=16).contains(&count)
            || times.len() != count + 1
            || times[0] != 0.
            || times.iter().any(|x| !x.is_finite())
            || times.windows(2).any(|p| p[0] >= p[1])
            || !(2..=33).contains(&m)
            || !(2..=65).contains(&n)
            || !spot.is_finite()
            || spot <= 0.
            || nodes.iter().any(|x| !x.is_finite())
            || nodes.windows(2).any(|p| p[0] >= p[1])
            || nodes[0] > 0.
            || nodes[m - 1] < 0.
            || variance_rows
                .iter()
                .any(|r| r.len() != m || r.iter().any(|x| !x.is_finite() || *x <= 0.))
            || strikes.iter().any(|x| !x.is_finite() || *x <= 0.)
            || strikes.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(invalid(
                "multi-transition source requires 2..16 intervals, 2..33 local nodes, positive ordered model and strikes",
            ));
        }
        let first = value(nodes, &variance_rows[0], 0.) * times[1];
        let sd0 = first.sqrt();
        if !first.is_finite() || first <= 0. || sd0 > 4. {
            return Err(invalid(
                "multi-transition first variance precision domain failed",
            ));
        }
        let integrate = |order| -> Result<Integration, RiskConfigError> {
            let mut window = [-first / 2. - 12. * sd0, -first / 2. + 12. * sd0];
            let mut points = rule(nodes, window, order, 4. * sd0)?;
            for (x, w) in &mut points {
                *w *= pdf((*x + first / 2.) / sd0) / sd0;
            }
            let mut vmax = first;
            let mut ptail = 2. * cdf(-12.);
            let mut stail = spot * (cdf(-12. - sd0) + cdf(-12. + sd0));
            let mut steps = Vec::new();
            let mut work = 0u128;
            for row in 0..count - 1 {
                if row > 0 {
                    let dt = times[row + 1] - times[row];
                    vmax += variance_rows[row].iter().copied().fold(0., f64::max) * dt;
                    window = [
                        -vmax / 2. - 12. * vmax.sqrt(),
                        vmax / 2. + 12. * vmax.sqrt(),
                    ];
                    let min_sd = (variance_rows[row]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
                        * dt)
                        .sqrt();
                    let mut next = rule(nodes, window, order, 4. * min_sd)?;
                    work += next.len() as u128 * points.len() as u128;
                    if work > 400_000_000 {
                        return Err(invalid("multi-transition integration work budget exceeded"));
                    }
                    let source: Vec<_> = points
                        .iter()
                        .map(|(x, p)| {
                            let v = value(nodes, &variance_rows[row], *x) * dt;
                            (*x - v / 2., v.sqrt(), *p)
                        })
                        .collect();
                    for (y, w) in &mut next {
                        *w *= source
                            .iter()
                            .map(|(mu, sd, p)| p * pdf((*y - mu) / sd) / sd)
                            .collect::<NeumaierSum>()
                            .total();
                    }
                    points = next;
                    // Union bound over discarded marginals. The stock-weighted
                    // loss is propagated unchanged by later martingale steps.
                    ptail += 2. * (-72_f64).exp();
                    stail += spot * (1. + window[0].exp()) * (-72_f64).exp();
                }
                let probability = points
                    .iter()
                    .map(|(_, p)| *p)
                    .collect::<NeumaierSum>()
                    .total();
                let stock = points
                    .iter()
                    .map(|(x, p)| spot * x.exp() * p)
                    .collect::<NeumaierSum>()
                    .total();
                if !probability.is_finite()
                    || !stock.is_finite()
                    || (probability - 1.).abs() > 1e-10 + ptail
                    || (stock - spot).abs() > 1e-10 * spot + stail
                {
                    return Err(invalid(
                        "multi-transition source probability or martingale integration failed",
                    ));
                }
                steps.push(VegaKtSourceStepAccuracy {
                    transition_count: row + 1,
                    log_window: window,
                    probability,
                    first_spot_moment: stock,
                    probability_tail_bound: ptail,
                    spot_tail_bound: stail,
                    probability_refinement_difference: 0.,
                    spot_refinement_difference: 0.,
                });
            }
            let dt = times[count] - times[count - 1];
            let mut matrix = vec![vec![NeumaierSum::new(); n]; m];
            for (x, p) in points {
                let (i, h) = hat(nodes, x);
                let sd = (value(nodes, &variance_rows[count - 1], x) * dt).sqrt();
                let s = spot * x.exp();
                for (j, k) in strikes.iter().enumerate() {
                    let v = p * s * pdf((spot.ln() - k.ln() + x) / sd + sd / 2.) * dt / (2. * sd);
                    if !v.is_finite() {
                        return Err(invalid("nonfinite multi-transition variance operator"));
                    }
                    matrix[i][j].add((1. - h) * v);
                    matrix[i + 1][j].add(h * v);
                }
            }
            Ok(Integration {
                matrix: matrix
                    .into_iter()
                    .map(|r| r.into_iter().map(NeumaierSum::total).collect())
                    .collect(),
                steps,
            })
        };
        let coarse = integrate(32)?;
        let mut fine = integrate(64)?;
        for (a, b) in coarse.steps.iter().zip(&mut fine.steps) {
            b.probability_refinement_difference = (a.probability - b.probability).abs();
            b.spot_refinement_difference = (a.first_spot_moment - b.first_spot_moment).abs();
            if b.probability_refinement_difference > 1e-10
                || b.spot_refinement_difference > 1e-10 * spot
            {
                return Err(invalid("multi-transition source refinement failed"));
            }
        }
        let difference = coarse
            .matrix
            .iter()
            .flatten()
            .zip(fine.matrix.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let scale = fine.matrix.iter().flatten().copied().fold(0., f64::max);
        let tolerance = 1e-10 * scale;
        let min_last = variance_rows[count - 1]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let tail = fine.steps.last().unwrap().spot_tail_bound
            * pdf(0.)
            * (times[count] - times[count - 1]).sqrt()
            / (2. * min_last.sqrt());
        if !tail.is_finite() || scale <= 0. || difference > tolerance || tail > tolerance {
            return Err(invalid(
                "multi-transition operator refinement or tail budget failed",
            ));
        }
        Ok((
            Self::from_operator(fine.matrix, strikes, lambda)?,
            VegaKtMultiTransitionSourceAccuracy {
                quadrature_order: 64,
                steps: fine.steps,
                operator_refinement_difference: difference,
                omitted_operator_entry_bound: tail,
                operator_entry_tolerance: tolerance,
            },
        ))
    }
}
