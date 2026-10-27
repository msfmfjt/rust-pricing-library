use std::sync::Arc;

use crate::core::{PositiveF64, UnderlyingId};

use crate::market::{
    AffineDividendCoordinate, AffineDividendTransform, CurveRegion, DiscountCurve, DividendEvent,
    LogLinearDiscountCurve, MarketError,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForwardEvaluation {
    /// Canonical continuous-martingale `f` forward used for Local Volatility and VegaKT.
    pub forward: f64,
    /// Spot-contract forward reconstructed from `S=A*S0+B*f`.
    pub spot_contract_forward: f64,
    pub affine_coordinate: AffineDividendCoordinate,
    pub discount_region: CurveRegion,
    pub dividend_region: CurveRegion,
}

/// Standard equity forward from Spot and continuous carry curves:
/// `F(t) = S(0) D_q(t) / D_r(t)`.
#[derive(Clone, Debug, PartialEq)]
pub struct EquityForward {
    underlying: UnderlyingId,
    spot: PositiveF64,
    discount_curve: Arc<LogLinearDiscountCurve>,
    dividend_curve: Arc<LogLinearDiscountCurve>,
    discrete_dividends: Option<AffineDividendTransform>,
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
            discrete_dividends: None,
        }
    }

    pub fn with_discrete_dividends(
        underlying: UnderlyingId,
        spot: PositiveF64,
        discount_curve: Arc<LogLinearDiscountCurve>,
        dividend_curve: Arc<LogLinearDiscountCurve>,
        dividends: Vec<DividendEvent>,
    ) -> Result<Self, MarketError> {
        let discrete_dividends = AffineDividendTransform::new_with_carry_curves(
            underlying,
            spot,
            dividends,
            Arc::clone(&discount_curve),
            Arc::clone(&dividend_curve),
        )?;
        Ok(Self {
            underlying,
            spot,
            discount_curve,
            dividend_curve,
            discrete_dividends: Some(discrete_dividends),
        })
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

    #[must_use]
    pub const fn discrete_dividends(&self) -> Option<&AffineDividendTransform> {
        self.discrete_dividends.as_ref()
    }

    pub fn with_spot(&self, spot: PositiveF64) -> Result<Self, MarketError> {
        let discrete_dividends = self
            .discrete_dividends
            .as_ref()
            .map(|dividends| dividends.with_spot(spot))
            .transpose()?;
        Ok(Self {
            underlying: self.underlying,
            spot,
            discount_curve: Arc::clone(&self.discount_curve),
            dividend_curve: Arc::clone(&self.dividend_curve),
            discrete_dividends,
        })
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
        let affine_coordinate = self
            .discrete_dividends
            .as_ref()
            .map_or(Ok(AffineDividendCoordinate::identity()), |dividends| {
                dividends.coordinate_after_time(time)
            })?;
        let spot_contract_forward = affine_coordinate.reconstruct_spot(self.spot, forward);
        if !spot_contract_forward.is_finite() || spot_contract_forward <= 0.0 {
            return Err(MarketError::NonFiniteForward {
                underlying: self.underlying,
                time_bits: time.to_bits(),
                forward_bits: spot_contract_forward.to_bits(),
            });
        }
        Ok(ForwardEvaluation {
            forward,
            spot_contract_forward,
            affine_coordinate,
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
    use crate::core::{CurveId, EventId};

    use crate::market::DividendQuote;

    fn flat_curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
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
        let value = forward.evaluate(1.0).expect("forward");
        assert!((value.forward - expected).abs() < 1.0e-13);
        assert_eq!(value.forward, value.spot_contract_forward);
        assert_eq!(
            value.affine_coordinate,
            AffineDividendCoordinate::identity()
        );
        assert_eq!(forward.forward(0.0), Ok(100.0));
    }

    #[test]
    fn discrete_dividends_keep_f_forward_canonical_and_map_spot_contract_forward() {
        let forward = EquityForward::with_discrete_dividends(
            UnderlyingId::new(3),
            PositiveF64::new(100.0, "spot").expect("positive spot"),
            flat_curve(10, 0.05),
            flat_curve(11, 0.02),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.5,
                    DividendQuote::fixed_cash_and_proportional(4.0, 0.1, EventId::new(1))
                        .expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("forward");
        let value = forward.evaluate(1.0).expect("evaluation");
        let f_forward = 100.0 * 0.03_f64.exp();
        assert!((value.forward - f_forward).abs() < 1.0e-13);
        let cash_carry = 0.015_f64.exp();
        assert!((value.affine_coordinate.a()).abs() < 1.0e-15);
        assert!((value.affine_coordinate.b() - (0.9 - 0.04 / cash_carry)).abs() < 1.0e-15);
        assert!(
            (value.spot_contract_forward - (0.9 * f_forward - 4.0 * cash_carry)).abs() < 1.0e-13
        );
        assert!(forward.discrete_dividends().is_some());
    }

    #[test]
    fn bumped_spot_recompiles_discrete_dividends_with_fixed_cash_held_constant() {
        let forward = EquityForward::with_discrete_dividends(
            UnderlyingId::new(3),
            PositiveF64::new(100.0, "spot").expect("positive spot"),
            flat_curve(10, 0.05),
            flat_curve(11, 0.02),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.5,
                    DividendQuote::fixed_cash_and_proportional(4.0, 0.1, EventId::new(1))
                        .expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("forward");
        let bumped = forward
            .with_spot(PositiveF64::new(125.0, "spot").expect("positive spot"))
            .expect("bumped");

        let event = bumped.discrete_dividends().expect("dividends").events()[0];
        assert_eq!(event.fixed_cash(), 4.0);
        assert!((event.alpha() - 4.0 / 125.0).abs() < 1.0e-15);
        assert_eq!(event.beta(), 0.1);
        assert_eq!(bumped.spot().get(), 125.0);
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

    #[test]
    fn spot_replacement_preserves_identity_curves_and_independent_regions() {
        let discount = Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(10), vec![0.0, 1.0], vec![1.0, 0.5])
                .expect("discount curve"),
        );
        let dividend = Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(20), vec![0.0, 2.0, 4.0], vec![1.0, 1.0, 1.0])
                .expect("dividend curve"),
        );
        let original = EquityForward::new(
            UnderlyingId::new(7),
            PositiveF64::new(128.0, "spot").unwrap(),
            Arc::clone(&discount),
            Arc::clone(&dividend),
        );
        let bumped = original
            .with_spot(PositiveF64::new(160.0, "spot").unwrap())
            .unwrap();
        assert_eq!(bumped.underlying(), UnderlyingId::new(7));
        assert_eq!(original.spot().get(), 128.0);
        assert_eq!(bumped.spot().get(), 160.0);
        assert_eq!(bumped.discount_curve(), discount.as_ref());
        assert_eq!(bumped.dividend_curve(), dividend.as_ref());
        assert!(bumped.discrete_dividends().is_none());
        assert_eq!(bumped.forward(0.0), Ok(160.0));
        assert!((bumped.forward(1.0).unwrap() - 320.0).abs() < 1.0e-13);
        for (time, discount_region, dividend_region) in [
            (1.0, CurveRegion::Pillar, CurveRegion::Interpolated),
            (2.0, CurveRegion::RightExtrapolated, CurveRegion::Pillar),
            (
                3.0,
                CurveRegion::RightExtrapolated,
                CurveRegion::Interpolated,
            ),
            (
                5.0,
                CurveRegion::RightExtrapolated,
                CurveRegion::RightExtrapolated,
            ),
        ] {
            let value = bumped.evaluate(time).unwrap();
            assert_eq!(value.discount_region, discount_region);
            assert_eq!(value.dividend_region, dividend_region);
        }
    }

    #[test]
    fn multiple_dividends_follow_independent_cash_flow_recursion() {
        let discount = Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(10),
                vec![0.0, 1.0, 2.0],
                vec![1.0, 0.25, 0.0625],
            )
            .unwrap(),
        );
        let dividend = Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(20),
                vec![0.0, 1.0, 2.0],
                vec![1.0, 0.5, 0.25],
            )
            .unwrap(),
        );
        let forward = EquityForward::with_discrete_dividends(
            UnderlyingId::new(7),
            PositiveF64::new(128.0, "spot").unwrap(),
            discount,
            dividend,
            vec![
                DividendEvent::new(
                    EventId::new(11),
                    1.0,
                    DividendQuote::FixedCashAndProportional {
                        fixed_cash: 8.0,
                        beta: 0.25,
                    },
                )
                .unwrap(),
                DividendEvent::new(
                    EventId::new(12),
                    2.0,
                    DividendQuote::FixedCashAndProportional {
                        fixed_cash: 12.0,
                        beta: 0.5,
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        // Funded reserve = 8/(3/4)/2 + 12/(3/8)/4 = 40/3.
        let transform = forward.discrete_dividends().unwrap();
        assert!((transform.initial_reserve() - 40.0 / 3.0).abs() < 1.0e-13);
        assert_eq!(transform.underlying(), UnderlyingId::new(7));
        assert_eq!(transform.spot().get(), 128.0);
        for (time, canonical, spot_forward, a, b, scale) in [
            (0.0, 128.0, 128.0, 5.0 / 48.0, 43.0 / 48.0, 1.0),
            (1.0, 256.0, 184.0, 3.0 / 32.0, 43.0 / 64.0, 0.75),
            (2.0, 512.0, 172.0, 0.0, 43.0 / 128.0, 0.375),
            (3.0, 1024.0, 344.0, 0.0, 43.0 / 128.0, 0.375),
        ] {
            let value = forward.evaluate(time).unwrap();
            assert!((value.forward - canonical).abs() < 1.0e-12, "time={time}");
            assert!((forward.forward(time).unwrap() - canonical).abs() < 1.0e-12);
            assert!(
                (value.spot_contract_forward - spot_forward).abs() < 1.0e-12,
                "time={time}"
            );
            assert!((value.affine_coordinate.a() - a).abs() < 1.0e-15);
            assert!((value.affine_coordinate.b() - b).abs() < 1.0e-15);
            assert_eq!(value.affine_coordinate.spot_scale(), scale);
        }
        let timeline = transform.event_timeline().unwrap();
        for (index, (event, time, a, b, pre_spot, post_spot)) in [
            (EventId::new(11), 1.0, 5.0 / 24.0, 43.0 / 48.0, 256.0, 184.0),
            (EventId::new(12), 2.0, 3.0 / 16.0, 43.0 / 64.0, 368.0, 172.0),
        ]
        .into_iter()
        .enumerate()
        {
            let entry = timeline[index];
            assert_eq!(entry.event(), event);
            assert_eq!(entry.ex_time(), time);
            assert!((entry.before().a() - a).abs() < 1.0e-15);
            assert!((entry.before().b() - b).abs() < 1.0e-15);
            assert_eq!(
                entry.before(),
                transform.coordinate_before_event(index).unwrap()
            );
            assert_eq!(
                entry.after(),
                transform.coordinate_after_time(time).unwrap()
            );
            let compiled = transform.events()[index];
            assert_eq!(compiled.event(), event);
            assert_eq!(compiled.ex_time(), time);
            assert!((compiled.post_spot(pre_spot, forward.spot()) - post_spot).abs() < 1.0e-13);
        }
        let bumped = forward
            .with_spot(PositiveF64::new(160.0, "spot").unwrap())
            .unwrap();
        assert_eq!(
            bumped.discrete_dividends().unwrap().initial_reserve(),
            transform.initial_reserve()
        );
        assert!((bumped.evaluate(3.0).unwrap().spot_contract_forward - 440.0).abs() < 1.0e-12);
        assert!((forward.evaluate(3.0).unwrap().spot_contract_forward - 344.0).abs() < 1.0e-12);
    }

    #[test]
    fn canonical_forward_overflow_reports_underlying_before_dividend_carry() {
        let id = UnderlyingId::new(7);
        let discount = Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(10),
                vec![0.0, 1.0, 2.0],
                vec![1.0, 1.0e-300, 1.0],
            )
            .unwrap(),
        );
        let dividend = Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(20),
                vec![0.0, 1.0, 2.0],
                vec![1.0, 1.0e300, 1.0],
            )
            .unwrap(),
        );
        let forward = EquityForward::with_discrete_dividends(
            id,
            PositiveF64::new(100.0, "spot").unwrap(),
            discount,
            dividend,
            vec![DividendEvent::new(EventId::new(11), 2.0, DividendQuote::FixedCash(1.0)).unwrap()],
        )
        .expect("the future cash is funded at its actual ex-date");
        let expected = MarketError::NonFiniteForward {
            underlying: id,
            time_bits: 1.0_f64.to_bits(),
            forward_bits: f64::INFINITY.to_bits(),
        };
        assert_eq!(forward.evaluate(1.0), Err(expected.clone()));
        assert_eq!(forward.forward(1.0), Err(expected));
    }

    #[test]
    fn positive_canonical_forward_can_underflow_after_a_proportional_dividend() {
        let id = UnderlyingId::new(7);
        let forward = EquityForward::with_discrete_dividends(
            id,
            PositiveF64::new(f64::from_bits(1), "spot").unwrap(),
            flat_curve(10, 0.0),
            flat_curve(20, 0.0),
            vec![
                DividendEvent::new(EventId::new(11), 0.0, DividendQuote::Proportional(0.75))
                    .unwrap(),
            ],
        )
        .expect("a zero-cash proportional schedule is funded");
        assert_eq!(
            forward.evaluate(0.0),
            Err(MarketError::NonFiniteForward {
                underlying: id,
                time_bits: 0.0_f64.to_bits(),
                forward_bits: 0.0_f64.to_bits(),
            })
        );
    }

    #[test]
    fn zero_cash_coordinates_do_not_require_representable_carry_growth() {
        let discount = Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(10), vec![0.0, 1.0], vec![1.0, 1.0e-300])
                .unwrap(),
        );
        let dividend = Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(20), vec![0.0, 1.0], vec![1.0, 1.0e300])
                .unwrap(),
        );
        for (events, expected_b) in [
            (vec![], 1.0),
            (
                vec![
                    DividendEvent::new(EventId::new(11), 0.0, DividendQuote::Proportional(0.25))
                        .unwrap(),
                ],
                0.75,
            ),
        ] {
            let forward = EquityForward::with_discrete_dividends(
                UnderlyingId::new(7),
                PositiveF64::new(100.0, "spot").unwrap(),
                Arc::clone(&discount),
                Arc::clone(&dividend),
                events,
            )
            .unwrap();
            let transform = forward.discrete_dividends().unwrap();
            assert_eq!(transform.initial_reserve(), 0.0);
            let coordinate = transform
                .coordinate_after_time(1.0)
                .expect("zero cash remains zero");
            assert_eq!(coordinate.a(), 0.0);
            assert_eq!(coordinate.b(), expected_b);
            assert_eq!(coordinate.spot_scale(), expected_b);
        }
    }

    #[test]
    fn curve_errors_are_propagated_with_their_original_identity() {
        for (discount_rate, dividend_rate, expected_curve) in [
            (-700.0, 0.0, CurveId::new(10)),
            (0.0, -700.0, CurveId::new(20)),
        ] {
            let forward = EquityForward::new(
                UnderlyingId::new(7),
                PositiveF64::new(100.0, "spot").unwrap(),
                flat_curve(10, discount_rate),
                flat_curve(20, dividend_rate),
            );
            let error = forward
                .evaluate(2.0)
                .expect_err("individual curve overflow");
            let MarketError::NonFiniteCurveValue {
                curve,
                time_bits,
                log_discount_bits,
            } = error
            else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(curve, expected_curve);
            assert_eq!(time_bits, 2.0_f64.to_bits());
            assert!(f64::from_bits(log_discount_bits) > 1000.0);
            assert_eq!(forward.forward(2.0), Err(error));
        }
    }
}
