use crate::core::CurveId;

use crate::market::MarketError;

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

    #[test]
    fn construction_preserves_inputs_and_normalizes_the_zero_anchor() {
        let id = CurveId::new(17);
        let curve = LogLinearDiscountCurve::new(id, vec![-0.0, 2.0, 5.0], vec![1.0, 0.1, 0.125])
            .expect("positive, nonmonotone discount factors are valid");
        assert_eq!(curve.id(), id);
        assert_eq!(curve.times(), &[0.0, 2.0, 5.0]);
        assert_eq!(curve.times()[0].to_bits(), 0.0_f64.to_bits());
        // In particular, 0.1 must not be reconstructed through exp(ln(0.1)).
        assert_eq!(curve.discount_factors(), &[1.0, 0.1, 0.125]);
        let expected_logs = [0.0, -std::f64::consts::LN_10, -3.0 * std::f64::consts::LN_2];
        assert_eq!(curve.log_discounts().len(), expected_logs.len());
        for (&actual, expected) in curve.log_discounts().iter().zip(expected_logs) {
            assert!((actual - expected).abs() < 1.0e-15);
        }
    }

    #[test]
    fn invalid_pillars_report_indices_and_exact_input_bits() {
        let id = CurveId::new(17);
        for index in 0..2 {
            for value in [
                f64::from_bits(0x7ff8_0000_0000_0017),
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                let mut times = vec![0.0, 1.0];
                times[index] = value;
                assert_eq!(
                    LogLinearDiscountCurve::new(id, times, vec![1.0, 0.9]),
                    Err(MarketError::NonFinitePillarTime {
                        curve: id,
                        index,
                        bits: value.to_bits()
                    })
                );
                let mut discounts = vec![1.0, 0.9];
                discounts[index] = value;
                assert_eq!(
                    LogLinearDiscountCurve::new(id, vec![0.0, 1.0], discounts),
                    Err(MarketError::NonFiniteDiscountFactor {
                        curve: id,
                        index,
                        bits: value.to_bits()
                    })
                );
            }
            let mut times = vec![0.0, 1.0];
            times[index] = -1.0;
            assert_eq!(
                LogLinearDiscountCurve::new(id, times, vec![1.0, 0.9]),
                Err(MarketError::NegativePillarTime {
                    curve: id,
                    index,
                    bits: (-1.0_f64).to_bits()
                })
            );
            for value in [0.0_f64, -0.0, -1.0] {
                let mut discounts = vec![1.0, 0.9];
                discounts[index] = value;
                assert_eq!(
                    LogLinearDiscountCurve::new(id, vec![0.0, 1.0], discounts),
                    Err(MarketError::NonPositiveDiscountFactor {
                        curve: id,
                        index,
                        bits: value.to_bits()
                    })
                );
            }
        }
    }

    #[test]
    fn pillar_count_anchor_and_order_errors_are_specific() {
        let id = CurveId::new(17);
        for (times, discounts, expected) in [
            (
                vec![],
                vec![],
                MarketError::InvalidPillarCount {
                    curve: id,
                    count: 0,
                },
            ),
            (
                vec![0.0],
                vec![1.0],
                MarketError::InvalidPillarCount {
                    curve: id,
                    count: 1,
                },
            ),
            (
                vec![0.0, 1.0],
                vec![1.0],
                MarketError::PillarLengthMismatch {
                    curve: id,
                    times: 2,
                    discount_factors: 1,
                },
            ),
            (
                vec![0.5, 1.0],
                vec![1.0, 0.9],
                MarketError::MissingValuationAnchor {
                    curve: id,
                    first_time_bits: 0.5_f64.to_bits(),
                },
            ),
            (
                vec![0.0, 1.0],
                vec![0.5, 0.9],
                MarketError::InvalidValuationDiscount {
                    curve: id,
                    discount_bits: 0.5_f64.to_bits(),
                },
            ),
            (
                vec![0.0, 1.0, 1.0],
                vec![1.0, 0.9, 0.8],
                MarketError::UnsortedPillars {
                    curve: id,
                    left_index: 1,
                    left_bits: 1.0_f64.to_bits(),
                    right_bits: 1.0_f64.to_bits(),
                },
            ),
            (
                vec![0.0, 2.0, 1.0],
                vec![1.0, 0.9, 0.8],
                MarketError::UnsortedPillars {
                    curve: id,
                    left_index: 1,
                    left_bits: 2.0_f64.to_bits(),
                    right_bits: 1.0_f64.to_bits(),
                },
            ),
        ] {
            assert_eq!(
                LogLinearDiscountCurve::new(id, times, discounts),
                Err(expected)
            );
        }
    }

    #[test]
    fn irregular_pillars_interpolate_and_extrapolate_the_local_log_slope() {
        let curve = LogLinearDiscountCurve::new(
            CurveId::new(17),
            vec![0.0, 2.0, 5.0],
            vec![1.0, 0.25, 0.015625],
        )
        .expect("valid curve");
        // Exact powers of two give independent values on two different slopes.
        let cases = [
            (0.0, 1.0, 0.0, CurveRegion::Pillar),
            (1.0, 0.5, -1.0, CurveRegion::Interpolated),
            (2.0, 0.25, -2.0, CurveRegion::Pillar),
            (2.75, 0.125, -3.0, CurveRegion::Interpolated),
            (3.5, 0.0625, -4.0, CurveRegion::Interpolated),
            (4.25, 0.03125, -5.0, CurveRegion::Interpolated),
            (5.0, 0.015625, -6.0, CurveRegion::Pillar),
            (5.75, 0.0078125, -7.0, CurveRegion::RightExtrapolated),
        ];
        for (time, discount, log2_discount, region) in cases {
            let actual = curve.evaluate(time).expect("finite curve value");
            let log_discount = log2_discount * std::f64::consts::LN_2;
            assert_eq!(actual.region, region, "time={time}");
            assert!((actual.discount - discount).abs() < 1.0e-15, "time={time}");
            assert!(
                (actual.log_discount - log_discount).abs() < 1.0e-15 * log_discount.abs().max(1.0),
                "time={time}"
            );
        }
    }

    #[test]
    fn negative_rates_and_flat_terminal_segments_are_supported() {
        let curve =
            LogLinearDiscountCurve::new(CurveId::new(17), vec![0.0, 2.0, 5.0], vec![1.0, 2.0, 2.0])
                .expect("negative forward rates are valid");
        let value = curve.evaluate(1.0).expect("negative-rate segment");
        assert_eq!(value.region, CurveRegion::Interpolated);
        assert!((value.discount - std::f64::consts::SQRT_2).abs() < 1.0e-15);
        assert!((value.log_discount - 0.5 * std::f64::consts::LN_2).abs() < 1.0e-15);
        for (time, region) in [
            (2.0, CurveRegion::Pillar),
            (3.5, CurveRegion::Interpolated),
            (5.0, CurveRegion::Pillar),
            (8.0, CurveRegion::RightExtrapolated),
        ] {
            let value = curve.evaluate(time).expect("flat terminal segment");
            assert_eq!(value.region, region);
            assert!((value.discount - 2.0).abs() < 1.0e-15);
            assert!((value.log_discount - std::f64::consts::LN_2).abs() < 1.0e-15);
        }
    }

    #[test]
    fn discount_accessors_preserve_values_and_errors() {
        let curve = curve();
        let discount_curve: &dyn DiscountCurve = &curve;
        for time in [0.0, 0.5, 1.0, 3.0] {
            let value = discount_curve.evaluate(time).expect("valid query");
            assert_eq!(discount_curve.discount(time), Ok(value.discount));
            assert_eq!(discount_curve.log_discount(time), Ok(value.log_discount));
        }
        for time in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = discount_curve.evaluate(time).expect_err("invalid query");
            assert_eq!(discount_curve.discount(time), Err(error.clone()));
            assert_eq!(discount_curve.log_discount(time), Err(error));
        }
    }

    #[test]
    fn invalid_query_errors_preserve_curve_and_input_bits() {
        let curve = curve();
        let id = CurveId::new(2);
        for time in [
            f64::from_bits(0x7ff8_0000_0000_0017),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                curve.evaluate(time),
                Err(MarketError::InvalidQueryTime {
                    curve: id,
                    bits: time.to_bits()
                })
            );
        }
        for time in [-f64::from_bits(1), -1.0, -f64::MAX] {
            assert_eq!(
                curve.evaluate(time),
                Err(MarketError::NegativeQueryTime {
                    curve: id,
                    bits: time.to_bits()
                })
            );
        }
        for time in [-0.0, 0.0] {
            assert_eq!(
                curve.evaluate(time),
                Ok(CurveEvaluation {
                    discount: 1.0,
                    log_discount: 0.0,
                    region: CurveRegion::Pillar
                })
            );
        }
    }

    #[test]
    fn finite_log_values_reject_discount_overflow_and_underflow() {
        let id = CurveId::new(17);
        for discount in [0.5, 2.0] {
            let curve = LogLinearDiscountCurve::new(id, vec![0.0, 1.0], vec![1.0, discount])
                .expect("valid curve");
            let time = 2048.0_f64;
            let error = curve.evaluate(time).expect_err("unrepresentable discount");
            let MarketError::NonFiniteCurveValue {
                curve,
                time_bits,
                log_discount_bits,
            } = error
            else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(curve, id);
            assert_eq!(time_bits, time.to_bits());
            let log_discount = f64::from_bits(log_discount_bits);
            assert!(log_discount.is_finite());
            assert!(log_discount.abs() > 1000.0);
            assert_eq!(log_discount.is_sign_negative(), discount < 1.0);
        }
        let curve = LogLinearDiscountCurve::new(id, vec![0.0, 1.0], vec![1.0, f64::MIN_POSITIVE])
            .expect("positive finite pillar");
        let value = curve
            .evaluate(1.01)
            .expect("positive subnormal discounts remain valid");
        assert!(value.discount.is_subnormal());
        assert!(value.discount > 0.0);
        assert!(value.log_discount.is_finite());
    }

    #[test]
    fn nonfinite_log_values_are_reported_with_query_context() {
        let id = CurveId::new(17);
        for (discount, expected_log) in [
            (0.5, f64::NEG_INFINITY),
            (1.0, f64::NAN),
            (2.0, f64::INFINITY),
        ] {
            let curve =
                LogLinearDiscountCurve::new(id, vec![0.0, f64::MIN_POSITIVE], vec![1.0, discount])
                    .expect("positive finite pillars");
            let error = curve
                .evaluate(f64::MAX)
                .expect_err("nonfinite log discount");
            let MarketError::NonFiniteCurveValue {
                curve,
                time_bits,
                log_discount_bits,
            } = error
            else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(curve, id);
            assert_eq!(time_bits, f64::MAX.to_bits());
            let actual_log = f64::from_bits(log_discount_bits);
            if expected_log.is_nan() {
                assert!(actual_log.is_nan());
            } else {
                assert_eq!(actual_log, expected_log);
            }
        }
    }

    #[test]
    fn extrapolation_statistics_ignore_other_regions_and_keep_extrema() {
        let id = CurveId::new(17);
        let mut stats = CurveExtrapolationStats::new(id);
        assert_eq!(stats.curve(), id);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.minimum_time(), None);
        assert_eq!(stats.maximum_time(), None);
        for (time, region) in [(0.0, CurveRegion::Pillar), (1.0, CurveRegion::Interpolated)] {
            assert!(!region.is_extrapolated());
            stats.record(time, region).expect("not extrapolated");
        }
        assert_eq!(stats, CurveExtrapolationStats::new(id));
        assert!(CurveRegion::RightExtrapolated.is_extrapolated());
        for (index, (time, minimum, maximum)) in [
            (7.0, 7.0, 7.0),
            (3.0, 3.0, 7.0),
            (5.0, 3.0, 7.0),
            (9.0, 3.0, 9.0),
            (3.0, 3.0, 9.0),
        ]
        .into_iter()
        .enumerate()
        {
            stats
                .record(time, CurveRegion::RightExtrapolated)
                .expect("diagnostic capacity");
            assert_eq!(stats.count(), index as u64 + 1);
            assert_eq!(stats.minimum_time(), Some(minimum));
            assert_eq!(stats.maximum_time(), Some(maximum));
        }
        let before = stats.clone();
        stats
            .record(100.0, CurveRegion::Pillar)
            .expect("pillar is not counted");
        stats
            .record(0.0, CurveRegion::Interpolated)
            .expect("interpolation is not counted");
        assert_eq!(stats, before);
    }

    #[test]
    fn extrapolation_count_overflow_preserves_the_previous_diagnostics() {
        let id = CurveId::new(17);
        // Seed the reachable boundary directly instead of recording u64::MAX times.
        let mut stats = CurveExtrapolationStats {
            curve: id,
            count: u64::MAX,
            minimum_time: Some(3.0),
            maximum_time: Some(7.0),
        };
        let before = stats.clone();
        assert_eq!(
            stats.record(10.0, CurveRegion::RightExtrapolated),
            Err(MarketError::DiagnosticCountOverflow { curve: id })
        );
        assert_eq!(stats, before);
        assert_eq!(stats.record(10.0, CurveRegion::Pillar), Ok(()));
        assert_eq!(stats, before);
    }
}
