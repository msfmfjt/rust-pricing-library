use crate::{CoreError, Date};

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
                if adjusted.month() == date.month() {
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
