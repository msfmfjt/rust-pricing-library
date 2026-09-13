//! One-factor Hull–White with constant mean reversion and piecewise constant
//! rate volatility. The deterministic shift is integrated from the input
//! discount curve; no numerical differentiation of that curve is needed.

use pricing_market::{DiscountCurve, LogLinearDiscountCurve, MarketError};
use pricing_numerics::standard_normal_cdf;
use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum HullWhiteError {
    InvalidInput { field: &'static str, index: usize },
    InvalidCorrelation,
    NonPositiveState { step: usize },
    Unsupported { feature: &'static str },
    Market(MarketError),
}
impl fmt::Display for HullWhiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, index } => {
                write!(f, "invalid Hull–White {field} at index {index}")
            }
            Self::InvalidCorrelation => write!(
                f,
                "Hull–White hybrid correlation matrix is not positive semidefinite"
            ),
            Self::NonPositiveState { step } => write!(
                f,
                "non-finite or non-positive hybrid asset/discount state at step {step}"
            ),
            Self::Unsupported { feature } => {
                write!(f, "Hull–White hybrid does not support {feature}")
            }
            Self::Market(e) => e.fmt(f),
        }
    }
}
impl Error for HullWhiteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Market(e) => Some(e),
            _ => None,
        }
    }
}
impl From<MarketError> for HullWhiteError {
    fn from(e: MarketError) -> Self {
        Self::Market(e)
    }
}

pub fn hw_valid(
    value: f64,
    field: &'static str,
    index: usize,
    nonnegative: bool,
) -> Result<(), HullWhiteError> {
    if !value.is_finite() || (nonnegative && value < 0.0) {
        return Err(HullWhiteError::InvalidInput { field, index });
    }
    Ok(())
}

/// Brownian correlations, ordered (equity, Bergomi OU, short rate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridCorrelation {
    pub equity_vol: f64,
    pub equity_rate: f64,
    pub vol_rate: f64,
}
impl HybridCorrelation {
    pub fn new(equity_vol: f64, equity_rate: f64, vol_rate: f64) -> Result<Self, HullWhiteError> {
        let values = [equity_vol, equity_rate, vol_rate];
        if values.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
            || 1.0 + 2.0 * equity_vol * equity_rate * vol_rate
                - equity_vol * equity_vol
                - equity_rate * equity_rate
                - vol_rate * vol_rate
                < -64.0 * f64::EPSILON
        {
            return Err(HullWhiteError::InvalidCorrelation);
        }
        Ok(Self {
            equity_vol,
            equity_rate,
            vol_rate,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HullWhite1Factor {
    mean_reversion: f64,
    volatility_times: Box<[f64]>,
    volatilities: Box<[f64]>,
}

/// Conditional innovations (dW_equity, OU_vol, OU_rate, integrated_OU_rate).
/// These four Gaussian coordinates are required even though there are three
/// continuous Brownian drivers. Covariance includes every volatility breakpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullWhiteHybridTransition {
    pub rate_decay: f64,
    pub vol_decay: f64,
    pub integral_loading: f64,
    pub covariance: [[f64; 4]; 4],
}

impl HullWhite1Factor {
    /// Values apply on [time_i, time_{i+1}); the last volatility extrapolates flat.
    pub fn new(
        mean_reversion: f64,
        volatility_times: Vec<f64>,
        volatilities: Vec<f64>,
    ) -> Result<Self, HullWhiteError> {
        hw_valid(mean_reversion, "mean_reversion", 0, true)?;
        if volatility_times.is_empty()
            || volatility_times.len() != volatilities.len()
            || volatility_times[0] != 0.0
        {
            return Err(HullWhiteError::InvalidInput {
                field: "volatility_grid",
                index: 0,
            });
        }
        for (i, (&t, &v)) in volatility_times.iter().zip(&volatilities).enumerate() {
            hw_valid(t, "volatility_time", i, true)?;
            hw_valid(v, "rate_volatility", i, true)?;
            if i > 0 && t <= volatility_times[i - 1] {
                return Err(HullWhiteError::InvalidInput {
                    field: "volatility_time_order",
                    index: i,
                });
            }
        }
        Ok(Self {
            mean_reversion,
            volatility_times: volatility_times.into(),
            volatilities: volatilities.into(),
        })
    }
    #[must_use]
    pub const fn mean_reversion(&self) -> f64 {
        self.mean_reversion
    }
    #[must_use]
    pub fn volatility_times(&self) -> &[f64] {
        &self.volatility_times
    }
    #[must_use]
    pub fn volatilities(&self) -> &[f64] {
        &self.volatilities
    }
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        self.volatilities.iter().all(|v| *v == 0.0)
    }

    pub fn transition(
        &self,
        start: f64,
        end: f64,
        vol_reversion: f64,
        correlation: HybridCorrelation,
    ) -> Result<HullWhiteHybridTransition, HullWhiteError> {
        hw_valid(start, "start", 0, true)?;
        hw_valid(end, "end", 0, true)?;
        hw_valid(vol_reversion, "vol_reversion", 0, true)?;
        HybridCorrelation::new(
            correlation.equity_vol,
            correlation.equity_rate,
            correlation.vol_rate,
        )?;
        if end < start {
            return Err(HullWhiteError::InvalidInput {
                field: "interval",
                index: 0,
            });
        }
        let a = self.mean_reversion;
        let k = vol_reversion;
        let h = end - start;
        let mut c = [[0.0; 4]; 4];
        c[0][0] = h;
        c[1][1] = b(2.0 * k, h);
        c[1][0] = correlation.equity_vol * b(k, h);
        for (i, &left) in self.volatility_times.iter().enumerate() {
            let lo = start.max(left);
            let hi = end.min(self.volatility_times.get(i + 1).copied().unwrap_or(end));
            if hi <= lo {
                continue;
            }
            let v = self.volatilities[i];
            let d = end - hi;
            let length = hi - lo;
            let e = (-a * d).exp();
            let ev = (-k * d).exp();
            let bd = b(a, d);
            let bh = b(a, length);
            let j = integrated_b(a, length);
            c[2][2] += v * v * e * e * b(2.0 * a, length);
            c[3][3] += v
                * v
                * (bd * bd * length + 2.0 * bd * e * j + e * e * integrated_b_squared(a, length));
            c[3][2] += v * v * e * (bd * bh + e * 0.5 * bh * bh);
            c[2][0] += correlation.equity_rate * v * e * bh;
            c[3][0] += correlation.equity_rate * v * (bd * length + e * j);
            c[2][1] += correlation.vol_rate * v * ev * e * b(k + a, length);
            c[3][1] +=
                correlation.vol_rate * v * ev * (bd * b(k, length) + e * mixed_b(k, a, length));
        }
        let lower = c;
        for (i, row) in c.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate().skip(i + 1) {
                *value = lower[j][i];
            }
            for &value in row.iter() {
                hw_valid(value, "transition_covariance", i, false)?;
            }
        }
        Ok(HullWhiteHybridTransition {
            rate_decay: (-a * h).exp(),
            vol_decay: (-k * h).exp(),
            integral_loading: b(a, h),
            covariance: c,
        })
    }

    /// Var[integral_0^t x_s ds] with x_0=0.
    pub fn integrated_variance(&self, time: f64) -> Result<f64, HullWhiteError> {
        Ok(self
            .transition(0.0, time, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?
            .covariance[3][3])
    }
    /// The convexity shift r_t - f(0,t) - x_t, also valid for piecewise sigma.
    pub fn rate_shift(&self, time: f64) -> Result<f64, HullWhiteError> {
        Ok(self
            .transition(0.0, time, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?
            .covariance[2][3])
    }
    pub fn integrated_shift(&self, start: f64, end: f64) -> Result<f64, HullWhiteError> {
        if end < start {
            return Err(HullWhiteError::InvalidInput {
                field: "interval",
                index: 0,
            });
        }
        Ok(0.5 * (self.integrated_variance(end)? - self.integrated_variance(start)?))
    }
    /// Discount factor divided by the deterministic input discount factor.
    pub fn relative_discount(&self, time: f64, integrated_x: f64) -> Result<f64, HullWhiteError> {
        hw_valid(integrated_x, "integrated_x", 0, false)?;
        checked_positive(
            (-integrated_x - 0.5 * self.integrated_variance(time)?).exp(),
            "relative_discount",
        )
    }
    /// P(t,T) / [P(0,T)/P(0,t)].
    pub fn relative_bond(&self, time: f64, maturity: f64, x: f64) -> Result<f64, HullWhiteError> {
        hw_valid(x, "rate_state", 0, false)?;
        let tr = self.transition(time, maturity, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?;
        checked_positive(
            (-tr.integral_loading * x - self.integrated_shift(time, maturity)?
                + 0.5 * tr.covariance[3][3])
                .exp(),
            "relative_bond",
        )
    }
    pub fn bond_price(
        &self,
        curve: &LogLinearDiscountCurve,
        time: f64,
        maturity: f64,
        x: f64,
    ) -> Result<f64, HullWhiteError> {
        checked_positive(
            (curve.evaluate(maturity)?.log_discount - curve.evaluate(time)?.log_discount).exp()
                * self.relative_bond(time, maturity, x)?,
            "bond_price",
        )
    }
    pub fn bond_option(
        &self,
        curve: &LogLinearDiscountCurve,
        expiry: f64,
        maturity: f64,
        strike: f64,
        is_call: bool,
    ) -> Result<f64, HullWhiteError> {
        if maturity < expiry {
            return Err(HullWhiteError::InvalidInput {
                field: "bond_option_maturity",
                index: 0,
            });
        }
        let c = self
            .transition(0.0, expiry, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?
            .covariance;
        let d = curve.evaluate(expiry)?.discount;
        let forward = curve.evaluate(maturity)?.discount / d;
        black_value(
            forward,
            strike,
            b(self.mean_reversion, maturity - expiry).powi(2) * c[2][2],
            d,
            is_call,
        )
    }
}

pub fn black_value(
    forward: f64,
    strike: f64,
    variance: f64,
    discount: f64,
    is_call: bool,
) -> Result<f64, HullWhiteError> {
    checked_positive(forward, "forward")?;
    checked_positive(strike, "strike")?;
    checked_positive(discount, "discount")?;
    hw_valid(variance, "total_variance", 0, true)?;
    let sign = if is_call { 1.0 } else { -1.0 };
    if variance == 0.0 {
        return Ok(discount * (sign * (forward - strike)).max(0.0));
    }
    let root = variance.sqrt();
    let d1 = (forward / strike).ln() / root + 0.5 * root;
    let d2 = d1 - root;
    let value = discount
        * sign
        * (forward * standard_normal_cdf(sign * d1) - strike * standard_normal_cdf(sign * d2));
    hw_valid(value, "option_price", 0, false)?;
    Ok(value)
}

fn checked_positive(value: f64, field: &'static str) -> Result<f64, HullWhiteError> {
    hw_valid(value, field, 0, true)?;
    if value == 0.0 {
        return Err(HullWhiteError::InvalidInput { field, index: 0 });
    }
    Ok(value)
}

/// Integral_0^h exp(-a*u) du, including the a=0 Ho–Lee limit.
#[must_use]
pub fn b(a: f64, h: f64) -> f64 {
    if a == 0.0 { h } else { -(-a * h).exp_m1() / a }
}
fn integrated_b(a: f64, h: f64) -> f64 {
    let z = a * h;
    if z.abs() < 0.1 {
        let mut term = 0.5;
        let mut sum = term;
        for n in 1..16 {
            term *= -z / (n + 2) as f64;
            sum += term;
        }
        h * h * sum
    } else {
        (h - b(a, h)) / a
    }
}
fn integrated_b_squared(a: f64, h: f64) -> f64 {
    let z = a * h;
    if z.abs() < 0.1 {
        let mut power = 1.0;
        let mut factorial = 2.0;
        let mut two = 4.0;
        let mut sum = 0.0;
        for n in 0..18 {
            sum += power * (two - 2.0) / (factorial * (n + 3) as f64);
            power *= -z;
            factorial *= (n + 3) as f64;
            two *= 2.0;
        }
        h * h * h * sum
    } else {
        (h - 2.0 * b(a, h) + b(2.0 * a, h)) / (a * a)
    }
}
fn exponential_moment(n: usize, z: f64) -> f64 {
    if z < 1.0 {
        let mut term = 1.0;
        let mut sum = 1.0 / (n + 1) as f64;
        for m in 1..40 {
            term *= -z / m as f64;
            sum += term / (n + m + 1) as f64;
        }
        sum
    } else {
        let mut value = -(-z).exp_m1() / z;
        for m in 1..=n {
            value = (m as f64 * value - (-z).exp()) / z;
        }
        value
    }
}
fn mixed_b(k: f64, a: f64, h: f64) -> f64 {
    if a * h < 0.001 {
        let mut term = 1.0;
        let mut sum = exponential_moment(1, k * h);
        for n in 1..7 {
            term *= -a * h / (n + 1) as f64;
            sum += term * exponential_moment(n + 1, k * h);
        }
        h * h * sum
    } else {
        (b(k, h) - b(k + a, h)) / a
    }
}
