use crate::core::{CoreError, Date};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum BusinessDayAdjustment {
    #[default]
    Unadjusted,
    Following,
    ModifiedFollowing,
    Preceding,
}

/// An immutable Saturday/Sunday calendar with a sorted custom-holiday set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Calendar {
    holidays: Box<[Date]>,
}

impl Calendar {
    #[must_use]
    pub fn weekend_only() -> Self {
        Self {
            holidays: Box::default(),
        }
    }

    #[must_use]
    pub fn new<I>(holidays: I) -> Self
    where
        I: IntoIterator<Item = Date>,
    {
        let mut holidays: Vec<_> = holidays.into_iter().collect();
        holidays.sort_unstable();
        holidays.dedup();
        Self {
            holidays: holidays.into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn holidays(&self) -> &[Date] {
        &self.holidays
    }

    #[must_use]
    pub fn is_holiday(&self, date: Date) -> bool {
        self.holidays.binary_search(&date).is_ok()
    }

    #[must_use]
    pub fn is_business_day(&self, date: Date) -> bool {
        !date.weekday().is_weekend() && !self.is_holiday(date)
    }

    pub fn adjust(&self, date: Date, convention: BusinessDayAdjustment) -> Result<Date, CoreError> {
        match convention {
            BusinessDayAdjustment::Unadjusted => Ok(date),
            BusinessDayAdjustment::Following => self.following(date),
            BusinessDayAdjustment::ModifiedFollowing => {
                let adjusted = self.following(date)?;
                if (adjusted.year(), adjusted.month()) == (date.year(), date.month()) {
                    Ok(adjusted)
                } else {
                    self.preceding(date)
                }
            }
            BusinessDayAdjustment::Preceding => self.preceding(date),
        }
    }

    fn following(&self, mut date: Date) -> Result<Date, CoreError> {
        while !self.is_business_day(date) {
            date = date.next_day()?;
        }
        Ok(date)
    }

    fn preceding(&self, mut date: Date) -> Result<Date, CoreError> {
        while !self.is_business_day(date) {
            date = date.previous_day()?;
        }
        Ok(date)
    }
}

impl Default for Calendar {
    fn default() -> Self {
        Self::weekend_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(value: &str) -> Date {
        value.parse().expect("valid test date")
    }

    #[test]
    fn weekend_calendars_preserve_business_days_and_unadjusted_dates() {
        for calendar in [
            Calendar::weekend_only(),
            Calendar::default(),
            Calendar::new([]),
        ] {
            assert!(calendar.holidays().is_empty());
            for (text, business_day) in [
                ("2026-09-05", false),
                ("2026-09-06", false),
                ("2026-09-07", true),
            ] {
                let value = date(text);
                assert_eq!(calendar.is_business_day(value), business_day);
                assert!(!calendar.is_holiday(value));
                assert_eq!(
                    calendar.adjust(value, BusinessDayAdjustment::Unadjusted),
                    Ok(value)
                );
            }
            for convention in [
                BusinessDayAdjustment::Following,
                BusinessDayAdjustment::ModifiedFollowing,
                BusinessDayAdjustment::Preceding,
            ] {
                assert_eq!(
                    calendar.adjust(date("2026-09-07"), convention),
                    Ok(date("2026-09-07"))
                );
            }
        }
    }

    #[test]
    fn holiday_membership_is_independent_of_weekends() {
        let calendar = Calendar::new([date("2026-09-07"), date("2026-09-05"), date("2026-09-07")]);
        for (text, holiday, business_day) in [
            ("2026-09-04", false, true),
            ("2026-09-05", true, false),
            ("2026-09-06", false, false),
            ("2026-09-07", true, false),
            ("2026-09-08", false, true),
        ] {
            assert_eq!(calendar.is_holiday(date(text)), holiday);
            assert_eq!(calendar.is_business_day(date(text)), business_day);
        }
    }

    #[test]
    fn adjustments_skip_holiday_runs_and_are_idempotent() {
        let calendar = Calendar::new(
            [
                "2026-12-24",
                "2026-12-25",
                "2026-12-28",
                "2026-12-29",
                "2026-12-31",
                "2027-01-01",
            ]
            .map(date),
        );
        for (original, following, modified, preceding) in [
            ("2026-12-23", "2026-12-23", "2026-12-23", "2026-12-23"),
            ("2026-12-24", "2026-12-30", "2026-12-30", "2026-12-23"),
            ("2026-12-26", "2026-12-30", "2026-12-30", "2026-12-23"),
            ("2026-12-31", "2027-01-04", "2026-12-30", "2026-12-30"),
        ] {
            for (convention, expected) in [
                (BusinessDayAdjustment::Following, following),
                (BusinessDayAdjustment::ModifiedFollowing, modified),
                (BusinessDayAdjustment::Preceding, preceding),
            ] {
                let adjusted = calendar.adjust(date(original), convention).unwrap();
                assert_eq!(adjusted, date(expected));
                assert!(calendar.is_business_day(adjusted));
                assert_eq!(calendar.adjust(adjusted, convention), Ok(adjusted));
            }
        }
    }

    #[test]
    fn adjustment_propagates_date_range_errors() {
        let calendar = Calendar::new([Date::MIN, Date::MAX]);
        assert_eq!(
            calendar.adjust(Date::MIN, BusinessDayAdjustment::Preceding),
            Err(CoreError::DateOutOfRange {
                date: Date::MIN,
                direction: "preceding",
            })
        );
        assert_eq!(
            calendar.adjust(Date::MAX, BusinessDayAdjustment::Following),
            Err(CoreError::DateOutOfRange {
                date: Date::MAX,
                direction: "following",
            })
        );
        assert_eq!(
            calendar.adjust(Date::MIN, BusinessDayAdjustment::Unadjusted),
            Ok(Date::MIN)
        );
        assert_eq!(
            calendar.adjust(Date::MAX, BusinessDayAdjustment::Unadjusted),
            Ok(Date::MAX)
        );
    }

    #[test]
    fn modified_following_does_not_accept_the_same_month_in_another_year() {
        // An explicit calendar may close every date in a year. The first
        // following business day is in January again, but a different month
        // in the calendar: modified following must move back to December.
        let holidays = (1..=12)
            .zip([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            .flat_map(|(month, last)| {
                (1..=last).map(move |day| Date::new(2026, month, day).unwrap())
            });
        let calendar = Calendar::new(holidays);
        let value = date("2026-01-01");
        assert_eq!(
            calendar.adjust(value, BusinessDayAdjustment::Following),
            Ok(date("2027-01-01"))
        );
        assert_eq!(
            calendar.adjust(value, BusinessDayAdjustment::ModifiedFollowing),
            Ok(date("2025-12-31"))
        );
    }

    #[test]
    fn custom_holidays_are_sorted_and_deduplicated() {
        let calendar = Calendar::new([date("2026-12-28"), date("2026-12-25"), date("2026-12-25")]);
        assert_eq!(
            calendar.holidays(),
            &[date("2026-12-25"), date("2026-12-28")]
        );
    }

    #[test]
    fn adjustments_follow_weekends_and_holidays() {
        let calendar = Calendar::new([date("2026-08-31")]);
        let saturday = date("2026-08-29");
        assert_eq!(
            calendar.adjust(saturday, BusinessDayAdjustment::Following),
            Ok(date("2026-09-01"))
        );
        assert_eq!(
            calendar.adjust(saturday, BusinessDayAdjustment::ModifiedFollowing),
            Ok(date("2026-08-28"))
        );
        assert_eq!(
            calendar.adjust(saturday, BusinessDayAdjustment::Preceding),
            Ok(date("2026-08-28"))
        );
        assert_eq!(
            calendar.adjust(saturday, BusinessDayAdjustment::Unadjusted),
            Ok(saturday)
        );
    }
}
