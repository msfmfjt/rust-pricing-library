use pricing_core::CurveId;

use crate::MarketError;

pub trait DiscountCurve: Send + Sync {
    fn evaluate(&self, time: f64) -> Result<CurveEvaluation, MarketError>;

    fn discount(&self, time: f64) -> Result<f64, MarketError> {
        Ok(self.evaluate(time)?.discount)
    }

    fn log_discount(&self, time: f64) -> Result<f64, MarketError> {
        Ok(self.evaluate(time)?.log_discount)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurveRegion {
    Pillar,
    Interpolated,
    RightExtrapolated,
}

impl CurveRegion {
    #[must_use]
    pub const fn is_extrapolated(self) -> bool {
        matches!(self, Self::RightExtrapolated)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveEvaluation {
    pub discount: f64,
    pub log_discount: f64,
    pub region: CurveRegion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogLinearDiscountCurve {
    id: CurveId,
    times: Box<[f64]>,
    discount_factors: Box<[f64]>,
    log_discounts: Box<[f64]>,
}

impl LogLinearDiscountCurve {
    pub fn new(
        id: CurveId,
        mut times: Vec<f64>,
        discount_factors: Vec<f64>,
    ) -> Result<Self, MarketError> {
        if times.len() != discount_factors.len() {
            return Err(MarketError::PillarLengthMismatch {
                curve: id,
                times: times.len(),
                discount_factors: discount_factors.len(),
            });
        }
        if times.len() < 2 {
            return Err(MarketError::InvalidPillarCount {
                curve: id,
                count: times.len(),
            });
        }
        for (index, time) in times.iter().copied().enumerate() {
            if !time.is_finite() {
                return Err(MarketError::NonFinitePillarTime {
                    curve: id,
                    index,
                    bits: time.to_bits(),
                });
            }
            if time < 0.0 {
                return Err(MarketError::NegativePillarTime {
                    curve: id,
                    index,
                    bits: time.to_bits(),
                });
            }
        }
        for (index, discount) in discount_factors.iter().copied().enumerate() {
            if !discount.is_finite() {
                return Err(MarketError::NonFiniteDiscountFactor {
                    curve: id,
                    index,
                    bits: discount.to_bits(),
                });
            }
            if discount <= 0.0 {
                return Err(MarketError::NonPositiveDiscountFactor {
                    curve: id,
                    index,
                    bits: discount.to_bits(),
                });
            }
        }
        if times[0] != 0.0 {
            return Err(MarketError::MissingValuationAnchor {
                curve: id,
                first_time_bits: times[0].to_bits(),
            });
        }
        if discount_factors[0] != 1.0 {
            return Err(MarketError::InvalidValuationDiscount {
                curve: id,
                discount_bits: discount_factors[0].to_bits(),
            });
        }
        for (left_index, pair) in times.windows(2).enumerate() {
            if pair[1] <= pair[0] {
                return Err(MarketError::UnsortedPillars {
                    curve: id,
                    left_index,
                    left_bits: pair[0].to_bits(),
                    right_bits: pair[1].to_bits(),
                });
            }
        }

        times[0] = 0.0;
        let log_discounts = discount_factors
            .iter()
            .copied()
            .map(f64::ln)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            id,
            times: times.into_boxed_slice(),
            discount_factors: discount_factors.into_boxed_slice(),
            log_discounts,
        })
    }

    #[must_use]
    pub const fn id(&self) -> CurveId {
        self.id
    }

    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    #[must_use]
    pub fn log_discounts(&self) -> &[f64] {
        &self.log_discounts
    }

    /// Returns the exact finite values supplied at construction time.
    #[must_use]
    pub fn discount_factors(&self) -> &[f64] {
        &self.discount_factors
    }

    fn segment_value(&self, left: usize, right: usize, time: f64) -> f64 {
        let weight = (time - self.times[left]) / (self.times[right] - self.times[left]);
        self.log_discounts[left] + weight * (self.log_discounts[right] - self.log_discounts[left])
    }
}

impl DiscountCurve for LogLinearDiscountCurve {
    fn evaluate(&self, time: f64) -> Result<CurveEvaluation, MarketError> {
        if !time.is_finite() {
            return Err(MarketError::InvalidQueryTime {
                curve: self.id,
                bits: time.to_bits(),
            });
        }
        if time < 0.0 {
            return Err(MarketError::NegativeQueryTime {
                curve: self.id,
                bits: time.to_bits(),
            });
        }
        let time = if time == 0.0 { 0.0 } else { time };
        let (log_discount, region) =
            match self.times.binary_search_by(|value| value.total_cmp(&time)) {
                Ok(index) => (self.log_discounts[index], CurveRegion::Pillar),
                Err(right) if right < self.times.len() => (
                    self.segment_value(right - 1, right, time),
                    CurveRegion::Interpolated,
                ),
                Err(_) => {
                    let right = self.times.len() - 1;
                    (
                        self.segment_value(right - 1, right, time),
                        CurveRegion::RightExtrapolated,
                    )
                }
            };
        let discount = log_discount.exp();
        if !log_discount.is_finite() || !discount.is_finite() || discount <= 0.0 {
            return Err(MarketError::NonFiniteCurveValue {
                curve: self.id,
                time_bits: time.to_bits(),
                log_discount_bits: log_discount.to_bits(),
            });
        }
        Ok(CurveEvaluation {
            discount,
            log_discount,
            region,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurveExtrapolationStats {
    curve: CurveId,
    count: u64,
    minimum_time: Option<f64>,
    maximum_time: Option<f64>,
}

impl CurveExtrapolationStats {
    #[must_use]
    pub const fn new(curve: CurveId) -> Self {
        Self {
            curve,
            count: 0,
            minimum_time: None,
            maximum_time: None,
        }
    }

    pub fn record(&mut self, time: f64, region: CurveRegion) -> Result<(), MarketError> {
        if !region.is_extrapolated() {
            return Ok(());
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or(MarketError::DiagnosticCountOverflow { curve: self.curve })?;
        self.minimum_time = Some(self.minimum_time.map_or(time, |value| value.min(time)));
        self.maximum_time = Some(self.maximum_time.map_or(time, |value| value.max(time)));
        Ok(())
    }

    #[must_use]
    pub const fn curve(&self) -> CurveId {
        self.curve
    }

    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn minimum_time(&self) -> Option<f64> {
        self.minimum_time
    }

    #[must_use]
    pub const fn maximum_time(&self) -> Option<f64> {
        self.maximum_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve() -> LogLinearDiscountCurve {
        LogLinearDiscountCurve::new(CurveId::new(2), vec![0.0, 1.0, 2.0], vec![1.0, 0.95, 0.90])
            .expect("valid curve")
    }

    #[test]
    fn validation_rejects_invalid_pillars() {
        let id = CurveId::new(1);
        assert!(LogLinearDiscountCurve::new(id, vec![0.0], vec![1.0]).is_err());
        assert!(LogLinearDiscountCurve::new(id, vec![0.0, 1.0], vec![1.0]).is_err());
        assert!(LogLinearDiscountCurve::new(id, vec![0.1, 1.0], vec![1.0, 0.9]).is_err());
        assert!(LogLinearDiscountCurve::new(id, vec![0.0, 1.0], vec![0.99, 0.9]).is_err());
        assert!(LogLinearDiscountCurve::new(id, vec![0.0, 1.0], vec![1.0, 0.0]).is_err());
        assert!(LogLinearDiscountCurve::new(id, vec![0.0, 0.0], vec![1.0, 0.9]).is_err());
    }

    #[test]
    fn interpolation_is_linear_in_log_discount() {
        let curve = curve();
        let value = curve.evaluate(0.5).expect("inside curve");
        assert_eq!(value.region, CurveRegion::Interpolated);
        assert!((value.discount - 0.95_f64.sqrt()).abs() < 1.0e-15);
        assert_eq!(
            curve.evaluate(1.0).expect("pillar").region,
            CurveRegion::Pillar
        );
    }

    #[test]
    fn right_extrapolation_extends_boundary_forward() {
        let curve = curve();
        let value = curve.evaluate(3.0).expect("extrapolated curve");
        let expected = 0.90 * (0.90 / 0.95);
        assert_eq!(value.region, CurveRegion::RightExtrapolated);
        assert!((value.discount - expected).abs() < 1.0e-15);
    }

    #[test]
    fn invalid_query_times_are_errors() {
        let curve = curve();
        assert!(curve.evaluate(-0.1).is_err());
        assert!(curve.evaluate(f64::NAN).is_err());
        assert_eq!(curve.evaluate(-0.0).expect("zero").discount, 1.0);
    }

    #[test]
    fn extrapolation_statistics_are_explicit() {
        let curve = curve();
        let mut stats = CurveExtrapolationStats::new(curve.id());
        for time in [0.5, 4.0, 3.0] {
            let value = curve.evaluate(time).expect("curve value");
            stats
                .record(time, value.region)
                .expect("diagnostic capacity");
        }
        assert_eq!(stats.count(), 2);
        assert_eq!(stats.minimum_time(), Some(3.0));
        assert_eq!(stats.maximum_time(), Some(4.0));
    }
}
