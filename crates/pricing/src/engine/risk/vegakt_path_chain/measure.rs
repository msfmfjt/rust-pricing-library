//! One-dimensional conditional continuation curves; no bump of the variance input.
use crate::{MonteCarloError, risk::RiskConfigError};
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};

pub(super) fn invalid(s: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity: s }
}
pub(super) fn sum(v: impl Iterator<Item = f64>) -> f64 {
    v.collect::<NeumaierSum>().total()
}
pub(super) fn call(s: f64, k: f64, v: f64) -> f64 {
    let sd = v.sqrt();
    let d1 = (s.ln() - k.ln()) / sd + sd / 2.;
    let d2 = d1 - sd;
    if s >= k {
        s - k + k * cdf(-d2) - s * cdf(-d1)
    } else {
        s * cdf(d1) - k * cdf(d2)
    }
}
fn tail(sd: f64) -> f64 {
    cdf(-12. - sd) + cdf(-12. + sd)
}
fn tail_moment(sd: f64) -> f64 {
    (sd * sd + 2.) * tail(sd) + (12. + sd) * (pdf(12. - sd) + pdf(12. + sd))
}

/// Conditional second moment of the raw terminal pathwise variance adjoint.
/// Integrating this moment prevents a zero observed hit count from implying
/// zero uncertainty in remote source nodes. Antithetic averaging uses the
/// Cauchy-Schwarz upper bound, not an independence assumption for mates.
pub(super) fn terminal_ad_variance(
    payoff: &crate::risk::VegaKtTerminalPayoff,
    source: f64,
    variance: f64,
    dt: f64,
    mean: f64,
) -> Result<f64, MonteCarloError> {
    let sd = variance.sqrt();
    let mut slope = payoff.slope();
    let mut lo = f64::NEG_INFINITY;
    let mut second = NeumaierSum::new();
    for atom in payoff
        .gamma()
        .atoms()
        .iter()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let hi = atom.map_or(f64::INFINITY, |a| {
            (a.strike().ln() - source.ln() + variance / 2.) / sd - 2. * sd
        });
        let p = if lo >= 0. {
            cdf(-lo) - cdf(-hi)
        } else {
            cdf(hi) - cdf(lo)
        };
        let p1 = pdf(lo) - pdf(hi);
        let edge = |z: f64| if z.is_finite() { z * pdf(z) } else { 0. };
        let p2 = p + edge(lo) - edge(hi);
        second.add(slope * slope * (p2 / variance + 2. * p1 / sd + p));
        if let Some(a) = atom {
            slope += a.mass();
        }
        lo = hi;
    }
    let value = source * source * variance.exp() * dt * dt / 4. * second.total() - mean * mean;
    if !value.is_finite() {
        return Err(invalid("nonfinite terminal conditional adjoint variance").into());
    }
    Ok(value.max(0.))
}
#[derive(Clone)]
pub(super) struct Slice {
    pub price: f64,
    pub risk: f64,
    pub masses: Vec<f64>,
    pub bin_risks: Vec<f64>,
    pub tail_price: f64,
    pub tail_risk: f64,
    pub negative: f64,
}
impl Slice {
    fn extrapolate(a: &Self, b: &Self) -> Self {
        let pair = |a: f64, b: f64| (4. * b - a) / 3.;
        Self {
            price: pair(a.price, b.price),
            risk: pair(a.risk, b.risk),
            masses: a
                .masses
                .iter()
                .zip(&b.masses)
                .map(|(a, b)| pair(*a, *b))
                .collect(),
            bin_risks: a
                .bin_risks
                .iter()
                .zip(&b.bin_risks)
                .map(|(a, b)| pair(*a, *b))
                .collect(),
            tail_price: (4. * b.tail_price + a.tail_price) / 3.,
            tail_risk: (4. * b.tail_risk + a.tail_risk) / 3.,
            negative: b.negative,
        }
    }
}
pub(super) struct CurveInput<'a> {
    pub source: f64,
    pub variance: f64,
    pub dt: f64,
    pub growth: [f64; 2],
    pub spot: f64,
    pub nodes: &'a [f64],
    pub boundaries: &'a [f64],
    pub subdivisions: usize,
}
pub(super) fn recover(
    input: &CurveInput<'_>,
    mut continuation: impl FnMut(f64) -> Result<f64, MonteCarloError>,
) -> Result<(Slice, Slice), MonteCarloError> {
    let CurveInput {
        source,
        variance,
        dt,
        growth,
        spot,
        nodes,
        boundaries,
        subdivisions,
    } = *input;
    let sd = variance.sqrt();
    let lo = source.ln() - variance / 2. - 12. * sd;
    let hi = source.ln() - variance / 2. + 12. * sd;
    if !lo.is_finite()
        || !hi.is_finite()
        || hi > 100.
        || hi - lo <= 128. * f64::EPSILON * (1. + lo.abs() + hi.abs())
        || sd > 4.
    {
        return Err(invalid("conditional Gamma window exceeds precision domain").into());
    }
    let mut xs = (0..=subdivisions)
        .map(|i| lo + (hi - lo) * i as f64 / subdivisions as f64)
        .chain(nodes.iter().map(|x| spot.ln() + x))
        .chain(boundaries.iter().map(|s| s.ln()))
        .filter(|x| lo <= *x && *x <= hi)
        .collect::<Vec<_>>();
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() <= 32. * f64::EPSILON * (1. + a.abs() + b.abs()));
    let mut points = xs
        .into_iter()
        .map(|x| Ok((x, continuation(x.exp())?)))
        .collect::<Result<Vec<_>, MonteCarloError>>()?;
    let evaluate = |points: &[(f64, f64)]| -> Result<Slice, MonteCarloError> {
        let slopes = points
            .windows(2)
            .map(|p| (p[1].1 - p[0].1) / (p[1].0.exp() - p[0].0.exp()))
            .collect::<Vec<_>>();
        let cash = points[0].1 - slopes[0] * points[0].0.exp();
        let last = points.len() - 1;
        let end_cash = points[last].1 - slopes[last - 1] * points[last].0.exp();
        let a = growth[0] + cash.abs().max(end_cash.abs());
        let b = growth[1] + slopes[0].abs().max(slopes[last - 1].abs());
        let mut price = NeumaierSum::new();
        price.add(cash);
        price.add(slopes[0] * source);
        let mut masses = vec![NeumaierSum::new(); boundaries.len() + 1];
        let mut risks = masses.clone();
        let mut negative = 0.;
        for i in 1..last {
            let k = points[i].0.exp();
            let mass = slopes[i] - slopes[i - 1];
            let bin = boundaries.partition_point(|b| *b <= k);
            masses[bin].add(mass);
            risks[bin]
                .add(mass * source * pdf((source.ln() - k.ln()) / sd + sd / 2.) * dt / (2. * sd));
            price.add(mass * call(source, k, variance));
            if mass < 0. {
                negative += 1.;
            }
        }
        let masses = masses
            .into_iter()
            .map(NeumaierSum::total)
            .collect::<Vec<_>>();
        let bin_risks = risks
            .into_iter()
            .map(NeumaierSum::total)
            .collect::<Vec<_>>();
        let factor = dt * (1. / (2. * variance) + 1. / (4. * sd));
        let out = Slice {
            price: price.total(),
            risk: sum(bin_risks.iter().copied()),
            masses,
            bin_risks,
            tail_price: a * tail(0.) + b * source * tail(sd),
            tail_risk: factor * (a * tail_moment(0.) + b * source * tail_moment(sd)),
            negative,
        };
        if [out.price, out.risk, out.tail_price, out.tail_risk]
            .iter()
            .chain(&out.masses)
            .chain(&out.bin_risks)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite conditional Gamma reconstruction").into());
        }
        Ok(out)
    };
    let mut levels = vec![evaluate(&points)?];
    for _ in 0..2 {
        let mut next = Vec::with_capacity(2 * points.len() - 1);
        for p in points.windows(2) {
            next.push(p[0]);
            let x = (p[0].0 + p[1].0) / 2.;
            next.push((x, continuation(x.exp())?));
        }
        next.push(*points.last().unwrap());
        points = next;
        levels.push(evaluate(&points)?);
    }
    Ok((
        Slice::extrapolate(&levels[0], &levels[1]),
        Slice::extrapolate(&levels[1], &levels[2]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_ad_second_moment_matches_split_gaussian_quadrature() {
        use crate::risk::{VegaKtGammaAtoms, VegaKtTerminalPayoff, gaussian_legendre};
        let payoff = VegaKtTerminalPayoff {
            cash: 2.,
            slope: 0.3,
            gamma: VegaKtGammaAtoms::new(vec![(95., 1.), (110., -0.4)]).unwrap(),
        };
        for source in [70_f64, 85., 100., 120.] {
            let v = 0.007_f64;
            let sd = v.sqrt();
            let dt = 0.175;
            let mut cuts = vec![-12., 12.];
            cuts.extend(
                payoff
                    .gamma()
                    .atoms()
                    .iter()
                    .map(|a| (a.strike().ln() - source.ln() + v / 2.) / sd)
                    .filter(|x| x.abs() < 12.),
            );
            cuts.sort_by(f64::total_cmp);
            let mut first = NeumaierSum::new();
            let mut second = NeumaierSum::new();
            for cell in cuts.windows(2) {
                for (z, w) in gaussian_legendre(64).unwrap() {
                    let x = (cell[0] + cell[1]) / 2. + (cell[1] - cell[0]) * z / 2.;
                    let s = source * (-v / 2. + sd * x).exp();
                    let delta = payoff.slope()
                        + sum(payoff
                            .gamma()
                            .atoms()
                            .iter()
                            .filter(|a| s > a.strike())
                            .map(|a| a.mass()));
                    let risk = delta * s * dt * (-0.5 + x / (2. * sd));
                    let weight = w * (cell[1] - cell[0]) / 2. * pdf(x);
                    first.add(weight * risk);
                    second.add(weight * risk * risk);
                }
            }
            let mean = sum(payoff.gamma().atoms().iter().map(|a| {
                a.mass() * source * pdf((source.ln() - a.strike().ln()) / sd + sd / 2.) * dt
                    / (2. * sd)
            }));
            assert!((first.total() - mean).abs() < 1e-10);
            let actual = terminal_ad_variance(&payoff, source, v, dt, mean).unwrap();
            assert!((actual - (second.total() - first.total().powi(2))).abs() < 1e-8);
        }
    }
    #[test]
    fn conditional_measure_matches_black_semigroup_variance_derivative() {
        let nodes = [-0.8, -0.2, 0., 0.2, 0.8];
        let bins = [80., 90., 100., 110., 120.];
        let input = CurveInput {
            source: 100.,
            variance: 0.02,
            dt: 0.5,
            growth: [100., 1.],
            spot: 100.,
            nodes: &nodes,
            boundaries: &bins,
            subdivisions: 256,
        };
        let (a, b) = recover(&input, |s| Ok(call(s, 103., 0.03))).unwrap();
        let expected =
            100. * pdf((100_f64 / 103.).ln() / 0.05_f64.sqrt() + 0.05_f64.sqrt() / 2.) * 0.5
                / (2. * 0.05_f64.sqrt());
        assert!((b.price - call(100., 103., 0.05)).abs() < 2e-6);
        assert!((b.risk - expected).abs() < 2e-6, "{}", b.risk - expected);
        assert!((a.risk - b.risk).abs() < 2e-5);
        assert!((sum(b.bin_risks.iter().copied()) - b.risk).abs() < 1e-12);
    }
    #[test]
    fn negative_payoff_kink_and_affine_part_are_preserved() {
        let input = CurveInput {
            source: 100.,
            variance: 0.04,
            dt: 1.,
            growth: [2., 3.],
            spot: 100.,
            nodes: &[-0.5, 0., 0.5],
            boundaries: &[100.],
            subdivisions: 32,
        };
        let (_, b) = recover(&input, |s| Ok(2. + 0.5 * s - (s - 100.).max(0.))).unwrap();
        assert!((sum(b.masses.iter().copied()) + 1.).abs() < 1e-9);
        assert!(b.negative > 0.);
        assert!((b.risk + 100. * pdf(0.1) / 0.4).abs() < 1e-8);
        assert!((b.price - (52. - call(100., 100., 0.04))).abs() < 1e-9);
    }
}
