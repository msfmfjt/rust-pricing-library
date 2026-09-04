use std::sync::Arc;

use pricing_core::{PositiveF64, UnderlyingId};

use crate::{CurveRegion, DiscountCurve, LogLinearDiscountCurve, MarketError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForwardEvaluation {
    pub forward: f64,
    pub discount_region: CurveRegion,
    pub dividend_region: CurveRegion,
}

/// Standard equity forward from Spot and continuous carry curves:
/// `F(t) = S(0) D_q(t) / D_r(t)`.
#[derive(Clone, Debug)]
pub struct EquityForward {
    underlying: UnderlyingId,
    spot: PositiveF64,
    discount_curve: Arc<LogLinearDiscountCurve>,
    dividend_curve: Arc<LogLinearDiscountCurve>,
}

impl EquityForward {
    #[must_use]
    pub fn new(
        underlying: UnderlyingId,
        spot: PositiveF64,
        discount_curve: Arc<LogLinearDiscountCurve>,
        dividend_curve: Arc<LogLinearDiscountCurve>,
    ) -> Self {
        Self {
            underlying,
            spot,
            discount_curve,
            dividend_curve,
        }
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn spot(&self) -> PositiveF64 {
        self.spot
    }

    #[must_use]
    pub fn discount_curve(&self) -> &LogLinearDiscountCurve {
        self.discount_curve.as_ref()
    }

    #[must_use]
    pub fn dividend_curve(&self) -> &LogLinearDiscountCurve {
        self.dividend_curve.as_ref()
    }

    pub fn evaluate(&self, time: f64) -> Result<ForwardEvaluation, MarketError> {
        let discount = self.discount_curve.evaluate(time)?;
        let dividend = self.dividend_curve.evaluate(time)?;
        let forward = self.spot.get() * (dividend.log_discount - discount.log_discount).exp();
        if !forward.is_finite() || forward <= 0.0 {
            return Err(MarketError::NonFiniteForward {
                underlying: self.underlying,
                time_bits: time.to_bits(),
                forward_bits: forward.to_bits(),
            });
        }
        Ok(ForwardEvaluation {
            forward,
            discount_region: discount.region,
            dividend_region: dividend.region,
        })
    }

    pub fn forward(&self, time: f64) -> Result<f64, MarketError> {
        Ok(self.evaluate(time)?.forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pricing_core::CurveId;

    fn flat_curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(id),
                vec![0.0, 1.0],
                vec![1.0, (-rate).exp()],
            )
            .expect("valid flat curve"),
        )
    }

    #[test]
    fn forward_uses_one_canonical_carry_identity() {
        let forward = EquityForward::new(
            UnderlyingId::new(3),
            PositiveF64::new(100.0, "spot").expect("positive spot"),
            flat_curve(10, 0.05),
            flat_curve(11, 0.02),
        );
        let expected = 100.0 * 0.03_f64.exp();
        assert!((forward.forward(1.0).expect("forward") - expected).abs() < 1.0e-13);
        assert_eq!(forward.forward(0.0), Ok(100.0));
    }

    #[test]
    fn curve_regions_are_retained_for_diagnostics() {
        let forward = EquityForward::new(
            UnderlyingId::new(3),
            PositiveF64::new(100.0, "spot").expect("positive spot"),
            flat_curve(10, 0.05),
            flat_curve(11, 0.02),
        );
        let value = forward.evaluate(2.0).expect("forward");
        assert_eq!(value.discount_region, CurveRegion::RightExtrapolated);
        assert_eq!(value.dividend_region, CurveRegion::RightExtrapolated);
    }
}
