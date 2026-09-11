use pricing_core::{
    BusinessDayAdjustment, Calendar, CoreError, CurrencyId, Date, FiniteF64, NonNegativeF64,
    PositiveF64, UnderlyingId,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptionSide {
    Call,
    Put,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EuropeanVanillaSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    expiry: Date,
    strike: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmericanVanillaSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    expiry: Date,
    strike: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
    exercise_dates: Box<[Date]>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExerciseObservationTiming {
    PostDividendSpot,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NormalizedExerciseEvent {
    date: Date,
    exercise_index: usize,
    terminal: bool,
    dividend_collision: bool,
    observation_timing: ExerciseObservationTiming,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DigitalPayout {
    Cash,
    Asset,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierDirection {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierStyle {
    KnockIn,
    KnockOut,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierMonitoring {
    Discrete,
    Continuous,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DigitalSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    expiry: Date,
    strike: PositiveF64,
    payout: PositiveF64,
    side: OptionSide,
    payout_kind: DigitalPayout,
    payment_date: Date,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BarrierSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    expiry: Date,
    strike: PositiveF64,
    barrier: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
    direction: BarrierDirection,
    style: BarrierStyle,
    monitoring: BarrierMonitoring,
    monitoring_dates: Box<[Date]>,
    rebate: Option<PositiveF64>,
    payment_date: Date,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsianObservationValue {
    Known(PositiveF64),
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsianObservation {
    date: Date,
    weight: NonNegativeF64,
    value: AsianObservationValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArithmeticAsianSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    strike: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
    observations: Box<[AsianObservation]>,
    payment_date: Date,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedLookbackSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    strike: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
    monitoring_dates: Box<[Date]>,
    historical_extremum: Option<PositiveF64>,
    payment_date: Date,
}

impl EuropeanVanillaSpec {
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        notional: f64,
        side: OptionSide,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            underlying,
            currency,
            expiry,
            strike: PositiveF64::new(strike, "strike")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }
}

impl AmericanVanillaSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        notional: f64,
        side: OptionSide,
        exercise_dates: Vec<Date>,
    ) -> Result<Self, CoreError> {
        if exercise_dates.is_empty() {
            return Err(CoreError::EmptyInput {
                field: "american_exercise_dates",
            });
        }
        for pair in exercise_dates.windows(2) {
            if pair[0] >= pair[1] {
                return Err(CoreError::InvalidOrdering {
                    field: "american_exercise_dates",
                });
            }
        }
        if exercise_dates.last().copied() != Some(expiry) {
            return Err(CoreError::InvalidOrdering {
                field: "american_exercise_dates_must_end_at_expiry",
            });
        }
        Ok(Self {
            underlying,
            currency,
            expiry,
            strike: PositiveF64::new(strike, "strike")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
            exercise_dates: exercise_dates.into_boxed_slice(),
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }

    #[must_use]
    pub const fn exercise_dates(&self) -> &[Date] {
        &self.exercise_dates
    }

    #[must_use]
    pub fn normalized_exercise_events(
        &self,
        dividend_dates: &[Date],
    ) -> Box<[NormalizedExerciseEvent]> {
        self.exercise_dates
            .iter()
            .enumerate()
            .map(|(exercise_index, &date)| NormalizedExerciseEvent {
                date,
                exercise_index,
                terminal: date == self.expiry,
                dividend_collision: dividend_dates.contains(&date),
                observation_timing: ExerciseObservationTiming::PostDividendSpot,
            })
            .collect()
    }
}

impl NormalizedExerciseEvent {
    #[must_use]
    pub const fn date(self) -> Date {
        self.date
    }

    #[must_use]
    pub const fn exercise_index(self) -> usize {
        self.exercise_index
    }

    #[must_use]
    pub const fn terminal(self) -> bool {
        self.terminal
    }

    #[must_use]
    pub const fn dividend_collision(self) -> bool {
        self.dividend_collision
    }

    #[must_use]
    pub const fn observation_timing(self) -> ExerciseObservationTiming {
        self.observation_timing
    }
}

pub fn every_business_day_exercise_schedule(
    start: Date,
    expiry: Date,
    calendar: &Calendar,
    adjustment: BusinessDayAdjustment,
    max_dates: usize,
) -> Result<Box<[Date]>, CoreError> {
    if max_dates == 0 {
        return Err(CoreError::EmptyInput {
            field: "american_exercise_schedule_limit",
        });
    }
    if start > expiry {
        return Err(CoreError::InvalidOrdering {
            field: "american_exercise_schedule_range",
        });
    }
    let adjusted_start = calendar.adjust(start, adjustment)?;
    let adjusted_expiry = calendar.adjust(expiry, adjustment)?;
    if adjusted_start > adjusted_expiry {
        return Err(CoreError::InvalidOrdering {
            field: "american_adjusted_exercise_schedule_range",
        });
    }
    if start != expiry && adjusted_start == adjusted_expiry {
        return Err(CoreError::InvalidOrdering {
            field: "american_exercise_schedule_adjustment_collision",
        });
    }

    let span = usize::try_from(adjusted_start.days_until(adjusted_expiry))
        .ok()
        .and_then(|days| days.checked_add(1))
        .ok_or(CoreError::InvalidOrdering {
            field: "american_adjusted_exercise_schedule_range",
        })?;
    let capacity = span.min(max_dates);
    let mut dates = Vec::new();
    dates
        .try_reserve_exact(capacity)
        .map_err(|_| CoreError::InvalidOrdering {
            field: "american_exercise_schedule_capacity",
        })?;
    let mut date = adjusted_start;
    loop {
        let is_boundary = date == adjusted_start || date == adjusted_expiry;
        if (is_boundary || calendar.is_business_day(date)) && dates.last() != Some(&date) {
            if dates.len() == max_dates {
                return Err(CoreError::InvalidOrdering {
                    field: "american_exercise_schedule_limit",
                });
            }
            dates.push(date);
        }
        if date == adjusted_expiry {
            break;
        }
        date = date.next_day()?;
    }
    Ok(dates.into_boxed_slice())
}

impl DigitalSpec {
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        payout: f64,
        side: OptionSide,
        payout_kind: DigitalPayout,
    ) -> Result<Self, CoreError> {
        Self::with_payment_date(
            underlying,
            currency,
            expiry,
            strike,
            payout,
            side,
            payout_kind,
            expiry,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_payment_date(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        payout: f64,
        side: OptionSide,
        payout_kind: DigitalPayout,
        payment_date: Date,
    ) -> Result<Self, CoreError> {
        if payment_date < expiry {
            return Err(CoreError::InvalidOrdering {
                field: "digital_payment_date",
            });
        }
        Ok(Self {
            underlying,
            currency,
            expiry,
            strike: PositiveF64::new(strike, "strike")?,
            payout: PositiveF64::new(payout, "payout")?,
            side,
            payout_kind,
            payment_date,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn payout(&self) -> PositiveF64 {
        self.payout
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }

    #[must_use]
    pub const fn payout_kind(&self) -> DigitalPayout {
        self.payout_kind
    }

    #[must_use]
    pub const fn payment_date(&self) -> Date {
        self.payment_date
    }
}

impl BarrierSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        barrier: f64,
        notional: f64,
        side: OptionSide,
        direction: BarrierDirection,
        style: BarrierStyle,
        monitoring: BarrierMonitoring,
        monitoring_dates: Vec<Date>,
        rebate: Option<f64>,
        payment_date: Date,
    ) -> Result<Self, CoreError> {
        if monitoring_dates.is_empty() {
            return Err(CoreError::EmptyInput {
                field: "barrier_monitoring_dates",
            });
        }
        for pair in monitoring_dates.windows(2) {
            if pair[0] >= pair[1] {
                return Err(CoreError::InvalidOrdering {
                    field: "barrier_monitoring_dates",
                });
            }
        }
        if *monitoring_dates.last().expect("non-empty monitoring dates") > expiry {
            return Err(CoreError::InvalidOrdering {
                field: "barrier_monitoring_dates",
            });
        }
        if payment_date < expiry {
            return Err(CoreError::InvalidOrdering {
                field: "barrier_payment_date",
            });
        }
        Ok(Self {
            underlying,
            currency,
            expiry,
            strike: PositiveF64::new(strike, "strike")?,
            barrier: PositiveF64::new(barrier, "barrier")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
            direction,
            style,
            monitoring,
            monitoring_dates: monitoring_dates.into_boxed_slice(),
            rebate: rebate
                .map(|value| PositiveF64::new(value, "barrier_rebate"))
                .transpose()?,
            payment_date,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn barrier(&self) -> PositiveF64 {
        self.barrier
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }

    #[must_use]
    pub const fn direction(&self) -> BarrierDirection {
        self.direction
    }

    #[must_use]
    pub const fn style(&self) -> BarrierStyle {
        self.style
    }

    #[must_use]
    pub const fn monitoring(&self) -> BarrierMonitoring {
        self.monitoring
    }

    #[must_use]
    pub const fn monitoring_dates(&self) -> &[Date] {
        &self.monitoring_dates
    }

    #[must_use]
    pub const fn rebate(&self) -> Option<PositiveF64> {
        self.rebate
    }

    #[must_use]
    pub const fn payment_date(&self) -> Date {
        self.payment_date
    }
}

impl AsianObservation {
    pub fn unknown(date: Date, weight: f64) -> Result<Self, CoreError> {
        Ok(Self {
            date,
            weight: NonNegativeF64::new(weight, "asian_observation_weight")?,
            value: AsianObservationValue::Unknown,
        })
    }

    pub fn known(date: Date, weight: f64, fixing: f64) -> Result<Self, CoreError> {
        Ok(Self {
            date,
            weight: NonNegativeF64::new(weight, "asian_observation_weight")?,
            value: AsianObservationValue::Known(PositiveF64::new(fixing, "asian_fixing")?),
        })
    }

    #[must_use]
    pub const fn date(self) -> Date {
        self.date
    }

    #[must_use]
    pub const fn weight(self) -> NonNegativeF64 {
        self.weight
    }

    #[must_use]
    pub const fn value(self) -> AsianObservationValue {
        self.value
    }
}

impl ArithmeticAsianSpec {
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        strike: f64,
        notional: f64,
        side: OptionSide,
        observations: Vec<AsianObservation>,
        payment_date: Date,
    ) -> Result<Self, CoreError> {
        if observations.is_empty() {
            return Err(CoreError::EmptyInput {
                field: "asian_observations",
            });
        }
        for pair in observations.windows(2) {
            if pair[0].date() >= pair[1].date() {
                return Err(CoreError::InvalidOrdering {
                    field: "asian_observations",
                });
            }
        }
        let fixing_date = observations.last().expect("non-empty observations").date();
        if payment_date < fixing_date {
            return Err(CoreError::InvalidOrdering {
                field: "asian_payment_date",
            });
        }
        let weight_sum = observations
            .iter()
            .map(|obs| obs.weight().get())
            .sum::<f64>();
        let weight_error = (weight_sum - 1.0).abs();
        let tolerance = 1.0e-12_f64.max(1.0e-12 * observations.len() as f64);
        if !FiniteF64::new(weight_sum, "asian_weight_sum").is_ok() || weight_error > tolerance {
            return Err(CoreError::InvalidWeights {
                field: "asian_observation_weights",
            });
        }
        Ok(Self {
            underlying,
            currency,
            strike: PositiveF64::new(strike, "strike")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
            observations: observations.into_boxed_slice(),
            payment_date,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub fn expiry(&self) -> Date {
        self.observations
            .last()
            .expect("constructor rejects empty observations")
            .date()
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }

    #[must_use]
    pub const fn observations(&self) -> &[AsianObservation] {
        &self.observations
    }

    #[must_use]
    pub const fn payment_date(&self) -> Date {
        self.payment_date
    }
}

impl FixedLookbackSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        strike: f64,
        notional: f64,
        side: OptionSide,
        monitoring_dates: Vec<Date>,
        historical_extremum: Option<f64>,
        payment_date: Date,
    ) -> Result<Self, CoreError> {
        if monitoring_dates.is_empty() {
            return Err(CoreError::EmptyInput {
                field: "lookback_monitoring_dates",
            });
        }
        for pair in monitoring_dates.windows(2) {
            if pair[0] >= pair[1] {
                return Err(CoreError::InvalidOrdering {
                    field: "lookback_monitoring_dates",
                });
            }
        }
        let expiry = *monitoring_dates.last().expect("non-empty monitoring dates");
        if payment_date < expiry {
            return Err(CoreError::InvalidOrdering {
                field: "lookback_payment_date",
            });
        }
        Ok(Self {
            underlying,
            currency,
            strike: PositiveF64::new(strike, "strike")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
            monitoring_dates: monitoring_dates.into_boxed_slice(),
            historical_extremum: historical_extremum
                .map(|value| PositiveF64::new(value, "lookback_historical_extremum"))
                .transpose()?,
            payment_date,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub fn expiry(&self) -> Date {
        *self
            .monitoring_dates
            .last()
            .expect("constructor rejects empty monitoring dates")
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }

    #[must_use]
    pub const fn monitoring_dates(&self) -> &[Date] {
        &self.monitoring_dates
    }

    #[must_use]
    pub const fn historical_extremum(&self) -> Option<PositiveF64> {
        self.historical_extremum
    }

    #[must_use]
    pub const fn payment_date(&self) -> Date {
        self.payment_date
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductSpec {
    EuropeanVanilla(EuropeanVanillaSpec),
    Digital(DigitalSpec),
    Barrier(BarrierSpec),
    ArithmeticAsian(ArithmeticAsianSpec),
    FixedLookback(FixedLookbackSpec),
}

impl ProductSpec {
    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        match self {
            Self::EuropeanVanilla(spec) => spec.underlying(),
            Self::Digital(spec) => spec.underlying(),
            Self::Barrier(spec) => spec.underlying(),
            Self::ArithmeticAsian(spec) => spec.underlying(),
            Self::FixedLookback(spec) => spec.underlying(),
        }
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        match self {
            Self::EuropeanVanilla(spec) => spec.currency(),
            Self::Digital(spec) => spec.currency(),
            Self::Barrier(spec) => spec.currency(),
            Self::ArithmeticAsian(spec) => spec.currency(),
            Self::FixedLookback(spec) => spec.currency(),
        }
    }

    #[must_use]
    pub fn expiry(&self) -> Date {
        match self {
            Self::EuropeanVanilla(spec) => spec.expiry(),
            Self::Digital(spec) => spec.expiry(),
            Self::Barrier(spec) => spec.expiry(),
            Self::ArithmeticAsian(spec) => spec.expiry(),
            Self::FixedLookback(spec) => spec.expiry(),
        }
    }

    #[must_use]
    pub const fn payment_date(&self) -> Date {
        match self {
            Self::EuropeanVanilla(spec) => spec.expiry(),
            Self::Digital(spec) => spec.payment_date(),
            Self::Barrier(spec) => spec.payment_date(),
            Self::ArithmeticAsian(spec) => spec.payment_date(),
            Self::FixedLookback(spec) => spec.payment_date(),
        }
    }

    #[must_use]
    pub const fn supports_pathwise_risk(&self) -> bool {
        matches!(
            self,
            Self::EuropeanVanilla(_) | Self::ArithmeticAsian(_) | Self::FixedLookback(_)
        ) || matches!(
            self,
            Self::Barrier(spec)
                if matches!(spec.monitoring(), BarrierMonitoring::Continuous)
        )
    }

    #[must_use]
    pub fn payoff_determined_by(&self, valuation_date: Date) -> bool {
        match self {
            Self::ArithmeticAsian(spec) => spec
                .observations()
                .iter()
                .all(|observation| matches!(observation.value(), AsianObservationValue::Known(_))),
            Self::FixedLookback(spec) => {
                spec.historical_extremum().is_some()
                    && spec
                        .monitoring_dates()
                        .iter()
                        .all(|date| *date < valuation_date)
            }
            Self::EuropeanVanilla(_) | Self::Digital(_) | Self::Barrier(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(value: &str) -> Date {
        value.parse().expect("valid date")
    }

    #[test]
    fn american_contract_requires_strict_schedule_ending_at_expiry() {
        let first = date("2027-03-04");
        let expiry = date("2027-09-04");
        let product = AmericanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            1_000_000.0,
            OptionSide::Put,
            vec![first, expiry],
        )
        .expect("American contract");
        assert_eq!(product.exercise_dates(), [first, expiry]);
        assert!(product.strike().get() > 0.0);

        assert!(matches!(
            AmericanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                1.0,
                OptionSide::Call,
                Vec::new(),
            ),
            Err(CoreError::EmptyInput {
                field: "american_exercise_dates"
            })
        ));
        for invalid in [vec![first, first, expiry], vec![expiry, first]] {
            assert!(matches!(
                AmericanVanillaSpec::new(
                    UnderlyingId::new(1),
                    CurrencyId::new(2),
                    expiry,
                    100.0,
                    1.0,
                    OptionSide::Call,
                    invalid,
                ),
                Err(CoreError::InvalidOrdering {
                    field: "american_exercise_dates"
                })
            ));
        }
        assert!(matches!(
            AmericanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                1.0,
                OptionSide::Call,
                vec![first],
            ),
            Err(CoreError::InvalidOrdering {
                field: "american_exercise_dates_must_end_at_expiry"
            })
        ));
    }

    #[test]
    fn business_day_schedule_materializes_adjusted_boundaries_and_holidays() {
        let calendar = Calendar::new([date("2026-09-07")]);
        let schedule = every_business_day_exercise_schedule(
            date("2026-09-05"),
            date("2026-09-11"),
            &calendar,
            BusinessDayAdjustment::Following,
            16,
        )
        .expect("schedule");
        assert_eq!(
            schedule.as_ref(),
            [
                date("2026-09-08"),
                date("2026-09-09"),
                date("2026-09-10"),
                date("2026-09-11"),
            ]
        );
        assert!(matches!(
            every_business_day_exercise_schedule(
                date("2026-09-05"),
                date("2026-09-06"),
                &Calendar::weekend_only(),
                BusinessDayAdjustment::Following,
                16,
            ),
            Err(CoreError::InvalidOrdering {
                field: "american_exercise_schedule_adjustment_collision"
            })
        ));
        assert!(matches!(
            every_business_day_exercise_schedule(
                date("2026-09-07"),
                date("2026-09-11"),
                &Calendar::weekend_only(),
                BusinessDayAdjustment::Unadjusted,
                4,
            ),
            Err(CoreError::InvalidOrdering {
                field: "american_exercise_schedule_limit"
            })
        ));
    }

    #[test]
    fn exercise_events_record_post_dividend_collision_order() {
        let first = date("2027-03-04");
        let expiry = date("2027-09-04");
        let product = AmericanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            1.0,
            OptionSide::Put,
            vec![first, expiry],
        )
        .expect("American contract");
        let events = product.normalized_exercise_events(&[first]);
        assert_eq!(events[0].date(), first);
        assert_eq!(events[0].exercise_index(), 0);
        assert!(events[0].dividend_collision());
        assert!(!events[0].terminal());
        assert_eq!(
            events[0].observation_timing(),
            ExerciseObservationTiming::PostDividendSpot
        );
        assert!(events[1].terminal());
        assert!(!events[1].dividend_collision());
    }

    #[test]
    fn european_contract_requires_positive_strike_and_notional() {
        let expiry = "2027-09-04".parse().expect("valid date");
        let valid = EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            1_000_000.0,
            OptionSide::Call,
        )
        .expect("valid contract");
        assert_eq!(valid.strike().get(), 100.0);
        assert_eq!(valid.side(), OptionSide::Call);
        assert!(
            EuropeanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                0.0,
                1.0,
                OptionSide::Put,
            )
            .is_err()
        );
    }

    #[test]
    fn digital_contract_requires_positive_strike_and_payout() {
        let expiry = "2027-09-04".parse().expect("valid date");
        let valid = DigitalSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            10.0,
            OptionSide::Put,
            DigitalPayout::Cash,
        )
        .expect("valid digital");
        assert_eq!(valid.payout().get(), 10.0);
        assert_eq!(valid.payout_kind(), DigitalPayout::Cash);
        assert_eq!(valid.payment_date(), expiry);
        assert!(
            DigitalSpec::with_payment_date(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                10.0,
                OptionSide::Call,
                DigitalPayout::Cash,
                "2027-09-05".parse().expect("payment"),
            )
            .is_ok()
        );
        assert!(
            DigitalSpec::with_payment_date(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                10.0,
                OptionSide::Call,
                DigitalPayout::Cash,
                "2027-09-03".parse().expect("payment"),
            )
            .is_err()
        );
        assert!(
            DigitalSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                0.0,
                OptionSide::Call,
                DigitalPayout::Asset,
            )
            .is_err()
        );
    }

    #[test]
    fn barrier_contract_validates_monitoring_and_payment_dates() {
        let first = "2027-03-04".parse().expect("first");
        let expiry = "2027-09-04".parse().expect("expiry");
        let product = BarrierSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            120.0,
            1.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            BarrierMonitoring::Continuous,
            vec![first, expiry],
            Some(3.0),
            expiry,
        )
        .expect("barrier");
        assert_eq!(product.expiry(), expiry);
        assert_eq!(product.monitoring_dates(), [first, expiry]);
        assert_eq!(product.monitoring(), BarrierMonitoring::Continuous);
        assert_eq!(product.rebate().expect("rebate").get(), 3.0);
        assert!(
            BarrierSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                first,
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Discrete,
                vec![expiry],
                None,
                first,
            )
            .is_err()
        );
        assert!(
            BarrierSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry,
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Discrete,
                vec![expiry, first],
                None,
                expiry,
            )
            .is_err()
        );
    }

    #[test]
    fn arithmetic_asian_contract_validates_schedule_and_weights() {
        let first = "2027-03-04".parse().expect("date");
        let second = "2027-09-04".parse().expect("date");
        let product = ArithmeticAsianSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            100.0,
            1.0,
            OptionSide::Call,
            vec![
                AsianObservation::unknown(first, 0.25).expect("first"),
                AsianObservation::known(second, 0.75, 105.0).expect("second"),
            ],
            second,
        )
        .expect("asian");
        assert_eq!(product.expiry(), second);
        assert_eq!(product.observations().len(), 2);
        assert!(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown(second, 0.5).expect("first"),
                    AsianObservation::unknown(first, 0.5).expect("second"),
                ],
                second,
            )
            .is_err()
        );
        assert!(matches!(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                Vec::new(),
                second,
            ),
            Err(CoreError::EmptyInput {
                field: "asian_observations"
            })
        ));
        assert!(matches!(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown(first, 0.5).expect("first"),
                    AsianObservation::known(first, 0.5, 105.0).expect("duplicate"),
                ],
                second,
            ),
            Err(CoreError::InvalidOrdering {
                field: "asian_observations"
            })
        ));
        assert!(matches!(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown(first, 0.2).expect("first"),
                    AsianObservation::unknown(second, 0.7).expect("second"),
                ],
                second,
            ),
            Err(CoreError::InvalidWeights {
                field: "asian_observation_weights"
            })
        ));
        assert!(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown(first, 0.5).expect("first"),
                    AsianObservation::unknown(second, 0.5).expect("second"),
                ],
                first,
            )
            .is_err()
        );
        assert!(AsianObservation::unknown(first, f64::NAN).is_err());
        assert!(AsianObservation::known(first, 1.0, 0.0).is_err());
    }

    #[test]
    fn fixed_lookback_contract_validates_monitoring_and_payment_dates() {
        let first = "2027-03-04".parse().expect("date");
        let second = "2027-09-04".parse().expect("date");
        let product = FixedLookbackSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            100.0,
            1.0,
            OptionSide::Put,
            vec![first, second],
            None,
            second,
        )
        .expect("lookback");
        assert_eq!(product.expiry(), second);
        assert_eq!(product.monitoring_dates(), [first, second]);
        assert!(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![second, first],
                None,
                second,
            )
            .is_err()
        );
        assert!(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![first, second],
                None,
                first,
            )
            .is_err()
        );
        assert!(matches!(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                Vec::new(),
                None,
                second,
            ),
            Err(CoreError::EmptyInput {
                field: "lookback_monitoring_dates"
            })
        ));
        assert!(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![first, first],
                None,
                second,
            )
            .is_err()
        );
        assert!(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![first, second],
                Some(0.0),
                second,
            )
            .is_err()
        );
    }
}
