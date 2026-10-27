use std::fmt;
use std::str::FromStr;

use crate::core::CoreError;

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
        let maximum = days_in_month(self.year, self.month).ok_or(CoreError::InvalidDate {
            year: self.year,
            month: self.month,
            day: self.day,
            reason: "month must be in 1..=12",
        })?;
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
            let day = days_in_month(self.year, month).ok_or(CoreError::InvalidDate {
                year: self.year,
                month,
                day: self.day,
                reason: "month must be in 1..=12",
            })?;
            return Self::new(self.year, month, day);
        }
        Self::new(self.year - 1, 12, 31)
    }

    fn ordinal(self) -> i32 {
        let previous_year = i32::from(self.year) - 1;
        let days_before_year =
            365 * previous_year + previous_year / 4 - previous_year / 100 + previous_year / 400;
        let cumulative = [0_i32, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let leap_day = i32::from(self.month > 2 && is_leap_year(self.year));
        days_before_year + cumulative[usize::from(self.month - 1)] + leap_day + i32::from(self.day)
    }
}

impl fmt::Display for Date {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04}-{:02}-{:02}",
            self.year, self.month, self.day
        )
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
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
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
    fn components_and_iso_format_preserve_the_full_date() {
        for (year, month, day, text) in [
            (1, 1, 1, "0001-01-01"),
            (9, 2, 3, "0009-02-03"),
            (87, 6, 9, "0087-06-09"),
            (1900, 12, 31, "1900-12-31"),
            (2000, 2, 29, "2000-02-29"),
            (9999, 12, 31, "9999-12-31"),
        ] {
            let value = Date::new(year, month, day).unwrap();
            assert_eq!(
                (value.year(), value.month(), value.day()),
                (year, month, day)
            );
            assert_eq!(value.to_string(), text);
            assert_eq!(text.parse::<Date>(), Ok(value));
        }
    }

    #[test]
    fn constructor_rejects_out_of_domain_components() {
        for (year, month, day) in [
            (0, 1, 1),
            (10_000, 1, 1),
            (u16::MAX, 1, 1),
            (2024, 0, 1),
            (2024, 13, 1),
            (2024, u8::MAX, 1),
            (2024, 1, 0),
            (2024, 1, 32),
            (2024, 1, u8::MAX),
            (2024, 4, 31),
            (1900, 2, 29),
            (2100, 2, 29),
            (2000, 2, 30),
        ] {
            assert!(matches!(
                Date::new(year, month, day),
                Err(CoreError::InvalidDate { year: y, month: m, day: d, .. })
                    if (y, m, d) == (year, month, day)
            ));
        }
    }

    #[test]
    fn parser_rejects_noncanonical_ascii_and_utf8_without_panicking() {
        for text in [
            "",
            "2024",
            "2024-2-29",
            "2024-02-29x",
            "2024/02-29",
            "2024-02/29",
            "+024-02-29",
            "2024-0/-29",
            "2024-0x-29",
            "2024-é-29",
            "２０２４-02-29",
            "2024-02-\0x",
            " 2024-02-29",
            "2024-02-29 ",
        ] {
            assert_eq!(
                text.parse::<Date>(),
                Err(CoreError::InvalidIsoDate {
                    value: text.to_owned()
                }),
                "{text:?}"
            );
        }
    }

    #[test]
    fn gregorian_century_rules_and_all_month_lengths() {
        for (year, leap) in [
            (1, false),
            (4, true),
            (100, false),
            (400, true),
            (1600, true),
            (1700, false),
            (1800, false),
            (1900, false),
            (2000, true),
            (2024, true),
            (2100, false),
            (2400, true),
            (9999, false),
        ] {
            assert_eq!(is_leap_year(year), leap, "year {year}");
        }
        for (year, lengths) in [
            (1900, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
            (2000, [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
        ] {
            for (month, last) in (1..=12).zip(lengths) {
                assert_eq!(days_in_month(year, month), Some(last));
                assert!(Date::new(year, month, last).is_ok());
                assert!(Date::new(year, month, last + 1).is_err());
            }
        }
        for month in [0, 13, u8::MAX] {
            assert_eq!(days_in_month(2024, month), None);
        }
    }

    #[test]
    fn daily_successors_and_predecessors_cross_each_boundary() {
        for (left, right) in [
            ("0001-01-01", "0001-01-02"),
            ("1900-02-28", "1900-03-01"),
            ("2000-02-28", "2000-02-29"),
            ("2000-02-29", "2000-03-01"),
            ("2023-02-28", "2023-03-01"),
            ("2024-04-30", "2024-05-01"),
            ("2026-09-05", "2026-09-06"),
            ("2026-12-31", "2027-01-01"),
            ("9999-12-30", "9999-12-31"),
        ] {
            let (left, right) = (date(left), date(right));
            assert_eq!(left.next_day(), Ok(right));
            assert_eq!(right.previous_day(), Ok(left));
            assert_eq!(left.days_until(right), 1);
        }
    }

    #[test]
    fn date_limits_report_the_direction_and_original_date() {
        assert_eq!(Date::MIN, date("0001-01-01"));
        assert_eq!(Date::MAX, date("9999-12-31"));
        assert_eq!(
            Date::MIN.previous_day(),
            Err(CoreError::DateOutOfRange {
                date: Date::MIN,
                direction: "preceding",
            })
        );
        assert_eq!(
            Date::MAX.next_day(),
            Err(CoreError::DateOutOfRange {
                date: Date::MAX,
                direction: "following",
            })
        );
    }

    #[test]
    fn day_offsets_and_weekdays_match_independent_gregorian_anchors() {
        // Ordinals and weekdays independently checked with Python datetime.date.
        // Gregorian dates are proleptic: no omitted days at the 1582 transition.
        for (year, month, day, ordinal, weekday) in [
            (1, 1, 1, 1, Weekday::Monday),
            (1, 12, 31, 365, Weekday::Monday),
            (4, 2, 29, 1155, Weekday::Sunday),
            (100, 3, 1, 36_219, Weekday::Monday),
            (400, 2, 29, 145_791, Weekday::Tuesday),
            (1582, 10, 15, 577_736, Weekday::Friday),
            (1600, 2, 29, 584_082, Weekday::Tuesday),
            (1700, 3, 1, 620_607, Weekday::Monday),
            (1900, 3, 1, 693_655, Weekday::Thursday),
            (1970, 1, 1, 719_163, Weekday::Thursday),
            (2000, 1, 1, 730_120, Weekday::Saturday),
            (2000, 2, 29, 730_179, Weekday::Tuesday),
            (2024, 2, 29, 738_945, Weekday::Thursday),
            (2026, 9, 9, 739_868, Weekday::Wednesday),
            (9999, 12, 31, 3_652_059, Weekday::Friday),
        ] {
            let value = Date::new(year, month, day).unwrap();
            assert_eq!(Date::MIN.days_until(value), ordinal - 1, "{value}");
            assert_eq!(value.days_until(Date::MIN), 1 - ordinal, "{value}");
            assert_eq!(value.weekday(), weekday, "{value}");
        }
        for (day, weekday, weekend) in [
            (7, Weekday::Monday, false),
            (8, Weekday::Tuesday, false),
            (9, Weekday::Wednesday, false),
            (10, Weekday::Thursday, false),
            (11, Weekday::Friday, false),
            (12, Weekday::Saturday, true),
            (13, Weekday::Sunday, true),
        ] {
            assert_eq!(Date::new(2026, 9, day).unwrap().weekday(), weekday);
            assert_eq!(weekday.is_weekend(), weekend);
        }
    }

    #[test]
    fn day_count_handles_short_periods_zero_and_reverse_dates() {
        for (start, end, days) in [
            ("1900-02-28", "1900-03-01", 1.0),
            ("2000-02-28", "2000-03-01", 2.0),
            ("2026-09-04", "2026-09-09", 5.0),
            ("2026-09-09", "2026-09-09", 0.0),
            ("0001-01-01", "9999-12-31", 3_652_058.0),
        ] {
            for (convention, denominator) in [
                (DayCountConvention::Act365F, 365.0),
                (DayCountConvention::Act360, 360.0),
            ] {
                assert_eq!(
                    convention.year_fraction(date(start), date(end)),
                    days / denominator
                );
                assert_eq!(
                    convention.year_fraction(date(end), date(start)),
                    -days / denominator
                );
            }
        }
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
        assert_eq!(
            DayCountConvention::Act365F.year_fraction(start, end),
            366.0 / 365.0
        );
        assert_eq!(
            DayCountConvention::Act360.year_fraction(start, end),
            366.0 / 360.0
        );
    }
}
