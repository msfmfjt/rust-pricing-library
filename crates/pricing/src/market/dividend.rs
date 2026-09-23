use crate::core::{EventId, PathIndex, PositiveF64, UnderlyingId};

use crate::market::{DiscountCurve, LogLinearDiscountCurve, MarketError};
use std::sync::Arc;

pub const DIVIDEND_EVENT_ORDER: &str = "dividend_before_expiry_v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DividendQuote {
    FixedCash(f64),
    Proportional(f64),
    FixedCashAndProportional { fixed_cash: f64, beta: f64 },
}

impl DividendQuote {
    pub fn fixed_cash(amount: f64, event: EventId) -> Result<Self, MarketError> {
        validate_fixed_cash(event, amount)?;
        Ok(Self::FixedCash(amount))
    }

    pub fn proportional(beta: f64, event: EventId) -> Result<Self, MarketError> {
        validate_beta(event, beta)?;
        Ok(Self::Proportional(beta))
    }

    pub fn fixed_cash_and_proportional(
        fixed_cash: f64,
        beta: f64,
        event: EventId,
    ) -> Result<Self, MarketError> {
        validate_fixed_cash(event, fixed_cash)?;
        validate_beta(event, beta)?;
        Ok(Self::FixedCashAndProportional { fixed_cash, beta })
    }

    #[must_use]
    pub const fn fixed_cash_amount(self) -> f64 {
        match self {
            Self::FixedCash(amount)
            | Self::FixedCashAndProportional {
                fixed_cash: amount, ..
            } => amount,
            Self::Proportional(_) => 0.0,
        }
    }

    #[must_use]
    pub const fn beta(self) -> f64 {
        match self {
            Self::Proportional(beta) | Self::FixedCashAndProportional { beta, .. } => beta,
            Self::FixedCash(_) => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DividendEvent {
    event: EventId,
    ex_time: f64,
    quote: DividendQuote,
}

impl DividendEvent {
    pub fn new(event: EventId, ex_time: f64, quote: DividendQuote) -> Result<Self, MarketError> {
        if !ex_time.is_finite() || ex_time < 0.0 {
            return Err(MarketError::InvalidDividendTime {
                event,
                index: 0,
                bits: ex_time.to_bits(),
            });
        }
        Ok(Self {
            event,
            ex_time,
            quote,
        })
    }

    #[must_use]
    pub const fn event(self) -> EventId {
        self.event
    }

    #[must_use]
    pub const fn ex_time(self) -> f64 {
        self.ex_time
    }

    #[must_use]
    pub const fn quote(self) -> DividendQuote {
        self.quote
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledDividendEvent {
    event: EventId,
    ex_time: f64,
    fixed_cash: f64,
    alpha: f64,
    beta: f64,
}

impl CompiledDividendEvent {
    #[must_use]
    pub const fn event(self) -> EventId {
        self.event
    }

    #[must_use]
    pub const fn ex_time(self) -> f64 {
        self.ex_time
    }

    #[must_use]
    pub const fn fixed_cash(self) -> f64 {
        self.fixed_cash
    }

    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    #[must_use]
    pub const fn beta(self) -> f64 {
        self.beta
    }

    #[must_use]
    pub fn post_spot(self, pre_spot: f64, initial_spot: PositiveF64) -> f64 {
        (1.0 - self.beta) * pre_spot - self.alpha * initial_spot.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineDividendCoordinate {
    a: f64,
    b: f64,
    spot_scale: f64,
}

impl AffineDividendCoordinate {
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            spot_scale: 1.0,
        }
    }

    #[must_use]
    pub const fn a(self) -> f64 {
        self.a
    }

    #[must_use]
    pub const fn b(self) -> f64 {
        self.b
    }

    /// Spot derivative at a fixed normalized residual-equity path. Cash amounts
    /// are fixed, including the cash funded in the initial escrow reserve.
    #[must_use]
    pub const fn spot_scale(self) -> f64 {
        self.spot_scale
    }

    /// Refit only the Spot-dependent escrow scale, holding all cash fixed.
    #[must_use]
    pub(crate) fn at_spot(self, original: f64, spot: f64) -> Self {
        if original == spot {
            return self;
        }
        Self {
            a: self.a * original / spot,
            b: self.spot_scale - (self.spot_scale - self.b) * original / spot,
            spot_scale: self.spot_scale,
        }
    }

    #[must_use]
    pub fn reconstruct_spot(self, initial_spot: PositiveF64, f_value: f64) -> f64 {
        self.a * initial_spot.get() + self.b * f_value
    }

    fn after(self, event: CompiledDividendEvent) -> Result<Self, MarketError> {
        let next = Self {
            a: (1.0 - event.beta) * self.a - event.alpha,
            b: (1.0 - event.beta) * self.b,
            spot_scale: (1.0 - event.beta) * self.spot_scale,
        };
        if !next.a.is_finite() {
            return Err(MarketError::NonFiniteDividendTransform {
                event: event.event,
                field: "A",
                bits: next.a.to_bits(),
            });
        }
        if !next.b.is_finite() {
            return Err(MarketError::NonFiniteDividendTransform {
                event: event.event,
                field: "B",
                bits: next.b.to_bits(),
            });
        }
        Ok(next)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineDividendTimelineEntry {
    event: EventId,
    ex_time: f64,
    before: AffineDividendCoordinate,
    after: AffineDividendCoordinate,
}

impl AffineDividendTimelineEntry {
    #[must_use]
    pub const fn event(self) -> EventId {
        self.event
    }

    #[must_use]
    pub const fn ex_time(self) -> f64 {
        self.ex_time
    }

    #[must_use]
    pub const fn before(self) -> AffineDividendCoordinate {
        self.before
    }

    #[must_use]
    pub const fn after(self) -> AffineDividendCoordinate {
        self.after
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DividendMatchingTolerance {
    abs_price_tol: f64,
    rel_price_tol: f64,
}

impl DividendMatchingTolerance {
    pub fn new(abs_price_tol: f64, rel_price_tol: f64) -> Result<Self, MarketError> {
        validate_tolerance("dividend_matching_abs_price_tol", abs_price_tol)?;
        validate_tolerance("dividend_matching_rel_price_tol", rel_price_tol)?;
        Ok(Self {
            abs_price_tol,
            rel_price_tol,
        })
    }

    pub fn default_for_spot(spot: PositiveF64) -> Self {
        Self {
            abs_price_tol: 1.0e-12 * spot.get(),
            rel_price_tol: 1.0e-11,
        }
    }

    #[must_use]
    pub const fn abs_price_tol(self) -> f64 {
        self.abs_price_tol
    }

    #[must_use]
    pub const fn rel_price_tol(self) -> f64 {
        self.rel_price_tol
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineDividendTransform {
    underlying: UnderlyingId,
    spot: PositiveF64,
    events: Box<[CompiledDividendEvent]>,
    carry_curves: Option<(Arc<LogLinearDiscountCurve>, Arc<LogLinearDiscountCurve>)>,
    initial_reserve: f64,
}

impl AffineDividendTransform {
    pub fn new(
        underlying: UnderlyingId,
        spot: PositiveF64,
        events: Vec<DividendEvent>,
    ) -> Result<Self, MarketError> {
        validate_events(&events)?;
        let compiled = events
            .into_iter()
            .map(|event| compile_event(event, spot))
            .collect::<Result<Box<[_]>, _>>()?;
        let mut transform = Self {
            underlying,
            spot,
            events: compiled,
            carry_curves: None,
            initial_reserve: 0.0,
        };
        transform.compile_reserve()?;
        Ok(transform)
    }

    /// Compile the funded reserve using the market curves. The standalone
    /// constructor uses zero rates; validate funding only in the chosen curves.
    pub(crate) fn new_with_carry_curves(
        underlying: UnderlyingId,
        spot: PositiveF64,
        events: Vec<DividendEvent>,
        discount: Arc<LogLinearDiscountCurve>,
        dividend: Arc<LogLinearDiscountCurve>,
    ) -> Result<Self, MarketError> {
        validate_events(&events)?;
        let mut result = Self {
            underlying,
            spot,
            events: events
                .into_iter()
                .map(|e| compile_event(e, spot))
                .collect::<Result<_, _>>()?,
            carry_curves: Some((discount, dividend)),
            initial_reserve: 0.0,
        };
        result.compile_reserve()?;
        Ok(result)
    }

    fn compile_reserve(&mut self) -> Result<(), MarketError> {
        let mut proportional = 1.0;
        let mut reserve = 0.0;
        for event in &self.events {
            proportional *= 1.0 - event.beta;
            let discount = if let Some((p, q)) = &self.carry_curves {
                p.discount(event.ex_time)? / q.discount(event.ex_time)?
            } else {
                1.0
            };
            reserve += event.fixed_cash / proportional * discount;
        }
        let residual = self.spot.get() - reserve;
        if !reserve.is_finite() || !residual.is_finite() || residual <= 0.0 {
            return Err(MarketError::NonPositiveEscrowedSpot {
                underlying: self.underlying,
                spot_bits: self.spot.get().to_bits(),
                reserve_bits: reserve.to_bits(),
            });
        }
        self.initial_reserve = reserve;
        Ok(())
    }

    #[must_use]
    pub const fn initial_reserve(&self) -> f64 {
        self.initial_reserve
    }

    #[must_use]
    pub fn initial_coordinate(&self) -> AffineDividendCoordinate {
        AffineDividendCoordinate {
            a: self.initial_reserve / self.spot.get(),
            b: (self.spot.get() - self.initial_reserve) / self.spot.get(),
            spot_scale: 1.0,
        }
    }

    fn carry(
        &self,
        mut coordinate: AffineDividendCoordinate,
        start: f64,
        end: f64,
    ) -> Result<AffineDividendCoordinate, MarketError> {
        if coordinate.a == 0.0 || start == end {
            return Ok(coordinate);
        }
        if let Some((discount, dividend)) = &self.carry_curves {
            let log_growth = dividend.evaluate(end)?.log_discount
                - discount.evaluate(end)?.log_discount
                - dividend.evaluate(start)?.log_discount
                + discount.evaluate(start)?.log_discount;
            coordinate.a *= log_growth.exp();
            if !coordinate.a.is_finite() {
                return Err(MarketError::NonFiniteDividendTransform {
                    event: EventId::new(0),
                    field: "carried_affine_A",
                    bits: coordinate.a.to_bits(),
                });
            }
        }
        Ok(coordinate)
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
    pub fn events(&self) -> &[CompiledDividendEvent] {
        &self.events
    }

    pub fn with_spot(&self, spot: PositiveF64) -> Result<Self, MarketError> {
        let events = self
            .events
            .iter()
            .map(|event| {
                DividendEvent::new(
                    event.event,
                    event.ex_time,
                    DividendQuote::fixed_cash_and_proportional(
                        event.fixed_cash,
                        event.beta,
                        event.event,
                    )?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some((p, q)) = &self.carry_curves {
            Self::new_with_carry_curves(self.underlying, spot, events, Arc::clone(p), Arc::clone(q))
        } else {
            Self::new(self.underlying, spot, events)
        }
    }

    pub fn coordinate_after_time(
        &self,
        time: f64,
    ) -> Result<AffineDividendCoordinate, MarketError> {
        if !time.is_finite() || time < 0.0 {
            return Err(MarketError::InvalidDividendTime {
                event: EventId::new(0),
                index: self.events.len(),
                bits: time.to_bits(),
            });
        }
        let mut coordinate = self.initial_coordinate();
        let mut previous_time = 0.0;
        for event in self.events.iter().take_while(|e| e.ex_time <= time) {
            coordinate = self.carry(coordinate, previous_time, event.ex_time)?;
            coordinate = coordinate.after(*event)?;
            previous_time = event.ex_time;
        }
        coordinate = self.carry(coordinate, previous_time, time)?;
        // Exact zero after the final cash event, avoiding cancellation residue.
        if self
            .events
            .iter()
            .all(|e| e.ex_time <= time || e.fixed_cash == 0.0)
        {
            coordinate.a = 0.0;
        }
        Ok(coordinate)
    }

    pub fn coordinate_before_event(
        &self,
        event_index: usize,
    ) -> Result<AffineDividendCoordinate, MarketError> {
        let event = self.events[event_index];
        let mut coordinate = self.initial_coordinate();
        let mut previous_time = 0.0;
        for earlier in &self.events[..event_index] {
            coordinate = self.carry(coordinate, previous_time, earlier.ex_time)?;
            coordinate = coordinate.after(*earlier)?;
            previous_time = earlier.ex_time;
        }
        coordinate = self.carry(coordinate, previous_time, event.ex_time)?;
        if self.events[event_index..]
            .iter()
            .all(|e| e.fixed_cash == 0.0)
        {
            coordinate.a = 0.0;
        }
        if coordinate.a.is_finite() && coordinate.b.is_finite() {
            Ok(coordinate)
        } else {
            Err(MarketError::NonFiniteDividendTransform {
                event: event.event,
                field: "event_coordinate_before",
                bits: coordinate.a.to_bits(),
            })
        }
    }

    pub fn event_timeline(&self) -> Result<Box<[AffineDividendTimelineEntry]>, MarketError> {
        let mut entries = Vec::with_capacity(self.events.len());
        let mut coordinate = self.initial_coordinate();
        let mut previous_time = 0.0;
        let last_cash = self.events.iter().rposition(|e| e.fixed_cash != 0.0);
        for (index, event) in self.events.iter().enumerate() {
            let before = self.carry(coordinate, previous_time, event.ex_time)?;
            let mut after = before.after(*event)?;
            if last_cash.is_none_or(|last| index >= last) {
                after.a = 0.0;
            }
            entries.push(AffineDividendTimelineEntry {
                event: event.event,
                ex_time: event.ex_time,
                before,
                after,
            });
            coordinate = after;
            previous_time = event.ex_time;
        }
        Ok(entries.into_boxed_slice())
    }

    pub fn spot_contract_strike(&self, time: f64, f_strike: f64) -> Result<f64, MarketError> {
        let coordinate = self.coordinate_after_time(time)?;
        let strike = coordinate.reconstruct_spot(self.spot, f_strike);
        if !strike.is_finite() {
            return Err(MarketError::NonFiniteDividendTransform {
                event: EventId::new(0),
                field: "spot_contract_strike",
                bits: strike.to_bits(),
            });
        }
        Ok(strike)
    }

    pub fn apply_event_to_spot(
        &self,
        event_index: usize,
        path: PathIndex,
        pre_spot: f64,
    ) -> Result<f64, MarketError> {
        let event = self.events[event_index];
        let post_spot = event.post_spot(pre_spot, self.spot);
        if !post_spot.is_finite() || post_spot <= 0.0 {
            return Err(MarketError::NonPositivePostDividendSpot {
                underlying: self.underlying,
                event: event.event,
                path,
                pre_spot_bits: pre_spot.to_bits(),
                alpha_bits: event.alpha.to_bits(),
                beta_bits: event.beta.to_bits(),
                fixed_cash_bits: event.fixed_cash.to_bits(),
                post_spot_bits: post_spot.to_bits(),
            });
        }
        Ok(post_spot)
    }

    pub fn validate_post_event_f_state(
        &self,
        event_index: usize,
        path: PathIndex,
        f_value: f64,
    ) -> Result<f64, MarketError> {
        let event = self.events[event_index];
        let before = self.coordinate_before_event(event_index)?;
        let pre_spot = before.reconstruct_spot(self.spot, f_value);
        let post_spot = event.post_spot(pre_spot, self.spot);
        if !post_spot.is_finite() || post_spot <= 0.0 {
            return Err(MarketError::NonPositivePostDividendSpot {
                underlying: self.underlying,
                event: event.event,
                path,
                pre_spot_bits: pre_spot.to_bits(),
                alpha_bits: event.alpha.to_bits(),
                beta_bits: event.beta.to_bits(),
                fixed_cash_bits: event.fixed_cash.to_bits(),
                post_spot_bits: post_spot.to_bits(),
            });
        }
        Ok(post_spot)
    }
}

pub fn validate_dividend_call_price_matching(
    event: CompiledDividendEvent,
    before_shifted_call: f64,
    after_call: f64,
    tolerance: DividendMatchingTolerance,
) -> Result<(), MarketError> {
    if !before_shifted_call.is_finite() {
        return Err(MarketError::NonFiniteDividendTransform {
            event: event.event,
            field: "before_shifted_call",
            bits: before_shifted_call.to_bits(),
        });
    }
    if !after_call.is_finite() {
        return Err(MarketError::NonFiniteDividendTransform {
            event: event.event,
            field: "after_call",
            bits: after_call.to_bits(),
        });
    }
    let expected = (1.0 - event.beta) * before_shifted_call;
    let abs_error = (after_call - expected).abs();
    let allowed = tolerance
        .abs_price_tol
        .max(tolerance.rel_price_tol * expected.abs().max(after_call.abs()));
    if abs_error > allowed {
        return Err(MarketError::DividendMatchingConditionViolation {
            event: event.event,
            expected_bits: expected.to_bits(),
            actual_bits: after_call.to_bits(),
            abs_error_bits: abs_error.to_bits(),
            abs_tol_bits: tolerance.abs_price_tol.to_bits(),
            rel_tol_bits: tolerance.rel_price_tol.to_bits(),
        });
    }
    Ok(())
}

fn compile_event(
    event: DividendEvent,
    spot: PositiveF64,
) -> Result<CompiledDividendEvent, MarketError> {
    let fixed_cash = event.quote.fixed_cash_amount();
    let alpha = fixed_cash / spot.get();
    if !alpha.is_finite() {
        return Err(MarketError::NonFiniteDividendTransform {
            event: event.event,
            field: "alpha",
            bits: alpha.to_bits(),
        });
    }
    Ok(CompiledDividendEvent {
        event: event.event,
        ex_time: event.ex_time,
        fixed_cash,
        alpha,
        beta: event.quote.beta(),
    })
}

fn validate_events(events: &[DividendEvent]) -> Result<(), MarketError> {
    let mut previous_time = None;
    for (index, event) in events.iter().enumerate() {
        if !event.ex_time.is_finite() || event.ex_time < 0.0 {
            return Err(MarketError::InvalidDividendTime {
                event: event.event,
                index,
                bits: event.ex_time.to_bits(),
            });
        }
        validate_fixed_cash(event.event, event.quote.fixed_cash_amount())?;
        validate_beta(event.event, event.quote.beta())?;
        if let Some(left) = previous_time
            && event.ex_time <= left
        {
            return Err(MarketError::UnsortedDividendEvents {
                left_index: index - 1,
                left_bits: left.to_bits(),
                right_bits: event.ex_time.to_bits(),
            });
        }
        previous_time = Some(event.ex_time);
    }
    Ok(())
}

fn validate_fixed_cash(event: EventId, amount: f64) -> Result<(), MarketError> {
    if amount.is_finite() && amount >= 0.0 {
        Ok(())
    } else {
        Err(MarketError::InvalidDividendCash {
            event,
            bits: amount.to_bits(),
        })
    }
}

fn validate_beta(event: EventId, beta: f64) -> Result<(), MarketError> {
    if beta.is_finite() && (0.0..1.0).contains(&beta) {
        Ok(())
    } else {
        Err(MarketError::InvalidDividendProportion {
            event,
            bits: beta.to_bits(),
        })
    }
}

fn validate_tolerance(field: &'static str, value: f64) -> Result<(), MarketError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(MarketError::NonFiniteDividendTransform {
            event: EventId::new(0),
            field,
            bits: value.to_bits(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot(value: f64) -> PositiveF64 {
        PositiveF64::new(value, "spot").expect("positive spot")
    }

    #[test]
    fn affine_transform_preserves_event_formula_and_spot_strike_mapping() {
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.25,
                    DividendQuote::fixed_cash(3.0, EventId::new(1)).expect("cash"),
                )
                .expect("event"),
                DividendEvent::new(
                    EventId::new(2),
                    0.5,
                    DividendQuote::proportional(0.1, EventId::new(2)).expect("proportional"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");

        let after_first = transform.coordinate_after_time(0.25).expect("coordinate");
        assert_eq!(after_first.a(), 0.0);
        assert_eq!(after_first.b(), 0.97);
        assert!((after_first.reconstruct_spot(spot(100.0), 103.0) - 99.91).abs() < 1e-13);

        let after_second = transform.coordinate_after_time(0.5).expect("coordinate");
        assert!((after_second.a()).abs() < 1.0e-15);
        assert!((after_second.b() - 0.873).abs() < 1.0e-15);
        assert!(
            (transform.spot_contract_strike(0.5, 120.0).expect("strike") - 104.76).abs() < 1.0e-14
        );
        assert_eq!(DIVIDEND_EVENT_ORDER, "dividend_before_expiry_v1");
    }

    #[test]
    fn event_timeline_records_pre_and_post_jump_coordinates() {
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.25,
                    DividendQuote::fixed_cash(3.0, EventId::new(1)).expect("cash"),
                )
                .expect("event"),
                DividendEvent::new(
                    EventId::new(2),
                    0.5,
                    DividendQuote::fixed_cash_and_proportional(5.0, 0.2, EventId::new(2))
                        .expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");

        let timeline = transform.event_timeline().expect("timeline");
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].before(), transform.initial_coordinate());
        assert_eq!(timeline[0].after().a(), 0.0625);
        assert_eq!(timeline[0].after().b(), 0.9075);
        assert_eq!(timeline[1].before(), timeline[0].after());
        assert!((timeline[1].after().a()).abs() < 1.0e-15);
        assert!((timeline[1].after().b() - 0.726).abs() < 1.0e-15);
    }

    #[test]
    fn unfunded_schedule_is_rejected_before_simulation() {
        let id = EventId::new(1);
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(id, 0.5, DividendQuote::fixed_cash(110.0, id).unwrap()).unwrap(),
            ],
        );
        assert!(matches!(
            transform,
            Err(MarketError::NonPositiveEscrowedSpot { .. })
        ));
    }

    #[test]
    fn fixed_cash_is_held_constant_when_spot_is_bumped() {
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.25,
                    DividendQuote::fixed_cash_and_proportional(5.0, 0.2, EventId::new(1))
                        .expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");
        let bumped = transform.with_spot(spot(125.0)).expect("bumped");
        assert_eq!(bumped.events()[0].fixed_cash(), 5.0);
        assert_eq!(bumped.events()[0].beta(), 0.2);
        assert!((bumped.events()[0].alpha() - 0.04).abs() < 1.0e-15);
    }

    #[test]
    fn dividend_events_validate_quotes_and_order() {
        assert_eq!(
            DividendQuote::fixed_cash(-1.0, EventId::new(9)),
            Err(MarketError::InvalidDividendCash {
                event: EventId::new(9),
                bits: (-1.0_f64).to_bits()
            })
        );
        assert_eq!(
            DividendQuote::proportional(1.0, EventId::new(9)),
            Err(MarketError::InvalidDividendProportion {
                event: EventId::new(9),
                bits: 1.0_f64.to_bits()
            })
        );
        assert_eq!(
            DividendQuote::fixed_cash_and_proportional(-1.0, 0.2, EventId::new(9)),
            Err(MarketError::InvalidDividendCash {
                event: EventId::new(9),
                bits: (-1.0_f64).to_bits()
            })
        );
        assert_eq!(
            DividendQuote::fixed_cash_and_proportional(1.0, -0.2, EventId::new(9)),
            Err(MarketError::InvalidDividendProportion {
                event: EventId::new(9),
                bits: (-0.2_f64).to_bits()
            })
        );
        assert!(matches!(
            AffineDividendTransform::new(
                UnderlyingId::new(7),
                spot(100.0),
                vec![
                    DividendEvent::new(
                        EventId::new(1),
                        0.5,
                        DividendQuote::fixed_cash(1.0, EventId::new(1)).expect("quote")
                    )
                    .expect("event"),
                    DividendEvent::new(
                        EventId::new(2),
                        0.5,
                        DividendQuote::fixed_cash(1.0, EventId::new(2)).expect("quote")
                    )
                    .expect("event")
                ]
            ),
            Err(MarketError::UnsortedDividendEvents { left_index: 0, .. })
        ));
    }

    #[test]
    fn non_positive_post_dividend_spot_is_typed_error() {
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.25,
                    DividendQuote::fixed_cash(60.0, EventId::new(1)).expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");
        assert!(matches!(
            transform.apply_event_to_spot(0, PathIndex::new(12), 50.0),
            Err(MarketError::NonPositivePostDividendSpot {
                underlying,
                event,
                path,
                ..
            }) if underlying == UnderlyingId::new(7)
                && event == EventId::new(1)
                && path == PathIndex::new(12)
        ));
    }

    #[test]
    fn matching_condition_uses_absolute_and_relative_tolerances() {
        let event = AffineDividendTransform::new(
            UnderlyingId::new(7),
            spot(100.0),
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    0.25,
                    DividendQuote::proportional(0.1, EventId::new(1)).expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform")
        .events()[0];
        let tolerance = DividendMatchingTolerance::new(1.0e-12, 1.0e-12).expect("tolerance");
        validate_dividend_call_price_matching(event, 12.0, 10.8, tolerance).expect("matches");
        assert!(matches!(
            validate_dividend_call_price_matching(event, 12.0, 10.7, tolerance),
            Err(MarketError::DividendMatchingConditionViolation {
                event,
                ..
            }) if event == EventId::new(1)
        ));
    }
}
