use std::fmt;
use std::str::FromStr;

use crate::CoreError;

/// A date in the proleptic Gregorian calendar, restricted to years 1..=9999.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Date {
    year: u16,
    month: u8,
    day: u8,
}

impl Date {
    pub const MIN: Self = Self {
        year: 1,
        month: 1,
        day: 1,
    };
    pub const MAX: Self = Self {
        year: 9999,
        month: 12,
        day: 31,
    };

    pub fn new(year: u16, month: u8, day: u8) -> Result<Self, CoreError> {
        if year == 0 || year > 9999 {
            return Err(CoreError::InvalidDate {
                year,
                month,
                day,
                reason: "year must be in 1..=9999",
            });
        }
        let maximum = days_in_month(year, month).ok_or(CoreError::InvalidDate {
            year,
            month,
            day,
            reason: "month must be in 1..=12",
        })?;
        if day == 0 || day > maximum {
            return Err(CoreError::InvalidDate {
                year,
                month,
                day,
                reason: "day is outside the selected month",
            });
        }
        Ok(Self { year, month, day })
    }

    #[must_use]
    pub const fn year(self) -> u16 {
        self.year
    }

    #[must_use]
    pub const fn month(self) -> u8 {
        self.month
    }

    #[must_use]
    pub const fn day(self) -> u8 {
        self.day
    }

    #[must_use]
    pub fn days_until(self, end: Self) -> i32 {
        end.ordinal() - self.ordinal()
    }

    #[must_use]
    pub fn weekday(self) -> Weekday {
        match (self.ordinal() - 1).rem_euclid(7) {
            0 => Weekday::Monday,
            1 => Weekday::Tuesday,
            2 => Weekday::Wednesday,
            3 => Weekday::Thursday,
            4 => Weekday::Friday,
            5 => Weekday::Saturday,
            _ => Weekday::Sunday,
        }
    }

    pub fn next_day(self) -> Result<Self, CoreError> {
        if self == Self::MAX {
            return Err(CoreError::DateOutOfRange {
                date: self,
                direction: "following",
            });
        }
        let maximum = days_in_month(self.year, self.month).expect("validated date");
        if self.day < maximum {
            return Self::new(self.year, self.month, self.day + 1);
        }
        if self.month < 12 {
            return Self::new(self.year, self.month + 1, 1);
        }
        Self::new(self.year + 1, 1, 1)
    }

    pub fn previous_day(self) -> Result<Self, CoreError> {
        if self == Self::MIN {
            return Err(CoreError::DateOutOfRange {
                date: self,
                direction: "preceding",
            });
        }
        if self.day > 1 {
            return Self::new(self.year, self.month, self.day - 1);
        }
        if self.month > 1 {
            let month = self.month - 1;
            let day = days_in_month(self.year, month).expect("valid previous month");
            return Self::new(self.year, month, day);
        }
        Self::new(self.year - 1, 12, 31)
    }

    fn ordinal(self) -> i32 {
        let previous_year = i32::from(self.year) - 1;
        let days_before_year = 365 * previous_year + previous_year / 4
            - previous_year / 100
            + previous_year / 400;
        let cumulative = [0_i32, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let leap_day = i32::from(self.month > 2 && is_leap_year(self.year));
        days_before_year + cumulative[usize::from(self.month - 1)] + leap_day + i32::from(self.day)
    }
}

impl fmt::Display for Date {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl FromStr for Date {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
            return Err(CoreError::InvalidIsoDate {
                value: value.to_owned(),
            });
        }
        if !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
        {
            return Err(CoreError::InvalidIsoDate {
                value: value.to_owned(),
            });
        }
        let year = parse_digits(&bytes[0..4]) as u16;
        let month = parse_digits(&bytes[5..7]) as u8;
        let day = parse_digits(&bytes[8..10]) as u8;
        Self::new(year, month, day)
    }
}

fn parse_digits(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0_u32, |value, byte| value * 10 + u32::from(byte - b'0'))
}

#[must_use]
pub const fn is_leap_year(year: u16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[must_use]
pub const fn days_in_month(year: u16, month: u8) -> Option<u8> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 if is_leap_year(year) => Some(29),
        2 => Some(28),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Weekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl Weekday {
    #[must_use]
    pub const fn is_weekend(self) -> bool {
        matches!(self, Self::Saturday | Self::Sunday)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DayCountConvention {
    #[default]
    Act365F,
    Act360,
}

impl DayCountConvention {
    #[must_use]
    pub fn year_fraction(self, start: Date, end: Date) -> f64 {
        let denominator = match self {
            Self::Act365F => 365.0,
            Self::Act360 => 360.0,
        };
        f64::from(start.days_until(end)) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(value: &str) -> Date {
        value.parse().expect("valid test date")
    }

    #[test]
    fn parser_is_strict_and_validates_calendar_dates() {
        assert_eq!(date("2024-02-29").to_string(), "2024-02-29");
        assert!("2023-02-29".parse::<Date>().is_err());
        assert!("2024-2-29".parse::<Date>().is_err());
        assert!("0000-01-01".parse::<Date>().is_err());
        assert!("2024/02/29".parse::<Date>().is_err());
    }

    #[test]
    fn ordinal_difference_handles_leap_years_and_direction() {
        assert_eq!(date("2024-02-28").days_until(date("2024-03-01")), 2);
        assert_eq!(date("2023-02-28").days_until(date("2023-03-01")), 1);
        assert_eq!(date("2024-03-01").days_until(date("2024-02-28")), -2);
    }

    #[test]
    fn weekdays_have_a_fixed_gregorian_anchor() {
        assert_eq!(date("0001-01-01").weekday(), Weekday::Monday);
        assert_eq!(date("2026-09-04").weekday(), Weekday::Friday);
        assert_eq!(date("2026-09-05").weekday(), Weekday::Saturday);
    }

    #[test]
    fn next_and_previous_day_cross_boundaries() {
        assert_eq!(date("2024-02-29").next_day(), Ok(date("2024-03-01")));
        assert_eq!(date("2024-01-01").previous_day(), Ok(date("2023-12-31")));
        assert!(Date::MAX.next_day().is_err());
        assert!(Date::MIN.previous_day().is_err());
    }

    #[test]
    fn day_counts_use_actual_calendar_days() {
        let start = date("2024-01-01");
        let end = date("2025-01-01");
        assert_eq!(DayCountConvention::Act365F.year_fraction(start, end), 366.0 / 365.0);
        assert_eq!(DayCountConvention::Act360.year_fraction(start, end), 366.0 / 360.0);
    }
}
