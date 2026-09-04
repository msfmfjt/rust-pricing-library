//! Independent analytical reference used to validate simulation and risk engines.

use std::error::Error;
use std::f64::consts::{PI, SQRT_2};
use std::fmt;

use pricing_core::DayCountConvention;
use pricing_market::{DiscountCurve, MarketError};
use pricing_models::ModelSpec;
use pricing_product::{OptionSide, ProductSpec};

use crate::PricingRequest;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackScholesOracleResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Sensitivity to a unit absolute volatility change, not one vol point.
    pub vega: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AnalyticalError {
    Market(MarketError),
    UnsupportedProductOrModel,
    InvalidInput { field: &'static str, bits: u64 },
    NonFiniteOutput { field: &'static str, bits: u64 },
}

impl From<MarketError> for AnalyticalError {
    fn from(error: MarketError) -> Self {
        Self::Market(error)
    }
}

impl fmt::Display for AnalyticalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Market(error) => error.fmt(formatter),
            Self::UnsupportedProductOrModel => {
                write!(formatter, "analytical oracle supports European Black-Scholes only")
            }
            Self::InvalidInput { field, bits } => {
                write!(formatter, "invalid analytical input {field}: 0x{bits:016x}")
            }
            Self::NonFiniteOutput { field, bits } => {
                write!(formatter, "analytical {field} is non-finite: 0x{bits:016x}")
            }
        }
    }
}

impl Error for AnalyticalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Market(error) => Some(error),
            _ => None,
        }
    }
}

/// Prices the supported vertical-slice request using the ACT/365F convention.
pub fn black_scholes_oracle(
    request: &PricingRequest,
) -> Result<BlackScholesOracleResult, AnalyticalError> {
    let ProductSpec::EuropeanVanilla(product) = request.product();
    let ModelSpec::BlackScholes(model) = request.model();
    let time = DayCountConvention::Act365F
        .year_fraction(request.valuation_date(), product.expiry());
    let forward = request.market().equity().forward();
    let discount = forward.discount_curve().discount(time)?;
    let dividend_discount = forward.dividend_curve().discount(time)?;
    evaluate(BlackScholesOracleInputs {
        side: product.side(),
        spot: forward.spot().get(),
        strike: product.strike().get(),
        notional: product.notional().get(),
        discount,
        dividend_discount,
        volatility: model.volatility().get(),
        time,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackScholesOracleInputs {
    pub side: OptionSide,
    pub spot: f64,
    pub strike: f64,
    pub notional: f64,
    pub discount: f64,
    pub dividend_discount: f64,
    pub volatility: f64,
    pub time: f64,
}

pub fn evaluate(
    inputs: BlackScholesOracleInputs,
) -> Result<BlackScholesOracleResult, AnalyticalError> {
    for (field, value) in [
        ("spot", inputs.spot),
        ("strike", inputs.strike),
        ("notional", inputs.notional),
        ("discount", inputs.discount),
        ("dividend_discount", inputs.dividend_discount),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(AnalyticalError::InvalidInput {
                field,
                bits: value.to_bits(),
            });
        }
    }
    for (field, value) in [("volatility", inputs.volatility), ("time", inputs.time)] {
        if !value.is_finite() || value < 0.0 {
            return Err(AnalyticalError::InvalidInput {
                field,
                bits: value.to_bits(),
            });
        }
    }
    let forward = inputs.spot * inputs.dividend_discount / inputs.discount;
    let result = if inputs.time == 0.0 || inputs.volatility == 0.0 {
        deterministic_limit(inputs, forward)
    } else {
        regular_formula(inputs, forward)
    };
    validate_result(result)
}

fn deterministic_limit(
    inputs: BlackScholesOracleInputs,
    forward: f64,
) -> BlackScholesOracleResult {
    let signed_intrinsic = match inputs.side {
        OptionSide::Call => forward - inputs.strike,
        OptionSide::Put => inputs.strike - forward,
    };
    let price = inputs.notional * inputs.discount * signed_intrinsic.max(0.0);
    let delta = if signed_intrinsic > 0.0 {
        match inputs.side {
            OptionSide::Call => inputs.notional * inputs.dividend_discount,
            OptionSide::Put => -inputs.notional * inputs.dividend_discount,
        }
    } else {
        0.0
    };
    BlackScholesOracleResult {
        price,
        delta,
        gamma: 0.0,
        vega: 0.0,
    }
}

fn regular_formula(
    inputs: BlackScholesOracleInputs,
    forward: f64,
) -> BlackScholesOracleResult {
    let root_time = inputs.time.sqrt();
    let standard_deviation = inputs.volatility * root_time;
    let d1 = (forward / inputs.strike).ln() / standard_deviation
        + 0.5 * standard_deviation;
    let d2 = d1 - standard_deviation;
    let density = normal_density(d1);
    let (undiscounted_price, delta_probability) = match inputs.side {
        OptionSide::Call => (
            forward * normal_cdf(d1) - inputs.strike * normal_cdf(d2),
            normal_cdf(d1),
        ),
        OptionSide::Put => (
            inputs.strike * normal_cdf(-d2) - forward * normal_cdf(-d1),
            normal_cdf(d1) - 1.0,
        ),
    };
    BlackScholesOracleResult {
        price: inputs.notional * inputs.discount * undiscounted_price,
        delta: inputs.notional * inputs.dividend_discount * delta_probability,
        gamma: inputs.notional * inputs.dividend_discount * density
            / (inputs.spot * standard_deviation),
        vega: inputs.notional * inputs.spot * inputs.dividend_discount * density * root_time,
    }
}

fn validate_result(
    result: BlackScholesOracleResult,
) -> Result<BlackScholesOracleResult, AnalyticalError> {
    for (field, value) in [
        ("price", result.price),
        ("delta", result.delta),
        ("gamma", result.gamma),
        ("vega", result.vega),
    ] {
        if !value.is_finite() {
            return Err(AnalyticalError::NonFiniteOutput {
                field,
                bits: value.to_bits(),
            });
        }
    }
    Ok(result)
}

fn normal_density(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * PI).sqrt()
}

// Hart-style rational approximation in the central region and an asymptotic
// continued fraction in the tails. Maximum absolute error is below 1e-15 for
// finite binary64 inputs used by the Black-Scholes oracle.
fn normal_cdf(value: f64) -> f64 {
    let magnitude = value.abs();
    let tail = if magnitude > 37.0 {
        0.0
    } else if magnitude < 7.071_067_811_865_475 {
        let numerator = ((((((0.035_262_496_599_891_1 * magnitude
            + 0.700_383_064_443_688)
            * magnitude
            + 6.373_962_203_531_65)
            * magnitude
            + 33.912_866_078_383)
            * magnitude
            + 112.079_291_497_871)
            * magnitude
            + 221.213_596_169_931)
            * magnitude
            + 220.206_867_912_376;
        let denominator = (((((((0.088_388_347_648_318_4 * magnitude
            + 1.755_667_163_182_64)
            * magnitude
            + 16.064_177_579_207)
            * magnitude
            + 86.780_732_202_946_1)
            * magnitude
            + 296.564_248_779_674)
            * magnitude
            + 637.333_633_378_831)
            * magnitude
            + 793.826_512_519_948)
            * magnitude
            + 440.413_735_824_752;
        (-0.5 * magnitude * magnitude).exp() * numerator / denominator
    } else {
        let continued_fraction = magnitude
            + 1.0
                / (magnitude
                    + 2.0
                        / (magnitude
                            + 3.0 / (magnitude + 4.0 / (magnitude + 0.65))));
        (-0.5 * magnitude * magnitude).exp() / (continued_fraction * SQRT_2 * PI.sqrt())
    };
    if value > 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(side: OptionSide) -> BlackScholesOracleInputs {
        BlackScholesOracleInputs {
            side,
            spot: 100.0,
            strike: 100.0,
            notional: 1.0,
            discount: (-0.05_f64).exp(),
            dividend_discount: (-0.02_f64).exp(),
            volatility: 0.2,
            time: 1.0,
        }
    }

    #[test]
    fn cdf_matches_reference_values_and_symmetry() {
        for (x, expected) in [
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_542_9),
            (-1.0, 0.158_655_253_931_457_07),
            (6.0, 0.999_999_999_013_412_3),
        ] {
            assert!((normal_cdf(x) - expected).abs() < 2.0e-15);
            assert!((normal_cdf(x) + normal_cdf(-x) - 1.0).abs() < 2.0e-15);
        }
    }

    #[test]
    fn regular_price_and_greeks_match_reference() {
        let result = evaluate(inputs(OptionSide::Call)).expect("oracle");
        assert!((result.price - 9.227_005_508_154_036).abs() < 2.0e-12);
        assert!((result.delta - 0.586_851_146_134_764).abs() < 2.0e-13);
        assert!((result.gamma - 0.018_950_578_755_008_72).abs() < 2.0e-14);
        assert!((result.vega - 37.901_157_510_017_44).abs() < 2.0e-12);
    }

    #[test]
    fn put_call_parity_holds_over_regular_grid() {
        for spot in [40.0, 80.0, 100.0, 120.0, 250.0] {
            for volatility in [0.01, 0.2, 1.0] {
                for time in [1.0 / 365.0, 0.25, 2.0, 10.0] {
                    let discount = (-0.03 * time).exp();
                    let dividend_discount = (-0.01 * time).exp();
                    let base = BlackScholesOracleInputs {
                        spot,
                        volatility,
                        time,
                        discount,
                        dividend_discount,
                        ..inputs(OptionSide::Call)
                    };
                    let call = evaluate(base).expect("call");
                    let put = evaluate(BlackScholesOracleInputs {
                        side: OptionSide::Put,
                        ..base
                    })
                    .expect("put");
                    let parity = spot * dividend_discount - base.strike * discount;
                    assert!((call.price - put.price - parity).abs() < 2.0e-11);
                    assert!((call.delta - put.delta - dividend_discount).abs() < 2.0e-14);
                    assert!((call.gamma - put.gamma).abs() < 2.0e-14);
                    assert!((call.vega - put.vega).abs() < 2.0e-12);
                }
            }
        }
    }

    #[test]
    fn deterministic_limits_are_explicit() {
        let at_expiry = BlackScholesOracleInputs {
            time: 0.0,
            discount: 1.0,
            dividend_discount: 1.0,
            ..inputs(OptionSide::Call)
        };
        assert_eq!(
            evaluate(at_expiry).expect("expiry"),
            BlackScholesOracleResult {
                price: 0.0,
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
            }
        );
        let deterministic = BlackScholesOracleInputs {
            volatility: 0.0,
            strike: 90.0,
            ..inputs(OptionSide::Call)
        };
        let result = evaluate(deterministic).expect("zero volatility");
        assert!(result.price > 0.0);
        assert_eq!(result.delta, deterministic.dividend_discount);
        assert_eq!(result.gamma, 0.0);
        assert_eq!(result.vega, 0.0);
    }

    #[test]
    fn analytical_greeks_match_central_differences() {
        let base = inputs(OptionSide::Call);
        let result = evaluate(base).expect("oracle");
        let spot_bump = 1.0e-3;
        let up = evaluate(BlackScholesOracleInputs {
            spot: base.spot + spot_bump,
            ..base
        })
        .expect("spot up");
        let down = evaluate(BlackScholesOracleInputs {
            spot: base.spot - spot_bump,
            ..base
        })
        .expect("spot down");
        let delta = (up.price - down.price) / (2.0 * spot_bump);
        let gamma = (up.price - 2.0 * result.price + down.price) / spot_bump.powi(2);
        assert!((delta - result.delta).abs() < 2.0e-9);
        assert!((gamma - result.gamma).abs() < 2.0e-7);

        let vol_bump = 1.0e-5;
        let vol_up = evaluate(BlackScholesOracleInputs {
            volatility: base.volatility + vol_bump,
            ..base
        })
        .expect("vol up");
        let vol_down = evaluate(BlackScholesOracleInputs {
            volatility: base.volatility - vol_bump,
            ..base
        })
        .expect("vol down");
        let vega = (vol_up.price - vol_down.price) / (2.0 * vol_bump);
        assert!((vega - result.vega).abs() < 2.0e-8);
    }
}
