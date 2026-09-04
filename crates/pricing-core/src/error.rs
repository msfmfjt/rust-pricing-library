use std::error::Error;
use std::fmt;

use crate::{Date, JsonPointer, SchemaVersion};

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CoreError {
    InvalidDate {
        year: u16,
        month: u8,
        day: u8,
        reason: &'static str,
    },
    InvalidIsoDate {
        value: String,
    },
    DateOutOfRange {
        date: Date,
        direction: &'static str,
    },
    NonFiniteNumber {
        field: &'static str,
        bits: u64,
    },
    NumberNotPositive {
        field: &'static str,
        bits: u64,
    },
    NumberNegative {
        field: &'static str,
        bits: u64,
    },
    IdExhausted {
        kind: &'static str,
    },
    InvalidSchemaVersion {
        value: u32,
    },
    InvalidJsonPointer {
        value: String,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDate {
                year,
                month,
                day,
                reason,
            } => write!(
                formatter,
                "invalid date {year:04}-{month:02}-{day:02}: {reason}"
            ),
            Self::InvalidIsoDate { value } => {
                write!(formatter, "invalid ISO date {value:?}; expected YYYY-MM-DD")
            }
            Self::DateOutOfRange { date, direction } => {
                write!(formatter, "cannot move {direction} from date {date}")
            }
            Self::NonFiniteNumber { field, bits } => {
                write!(
                    formatter,
                    "{field} must be finite; received bits 0x{bits:016x}"
                )
            }
            Self::NumberNotPositive { field, bits } => write!(
                formatter,
                "{field} must be greater than zero; received bits 0x{bits:016x}"
            ),
            Self::NumberNegative { field, bits } => write!(
                formatter,
                "{field} must be non-negative; received bits 0x{bits:016x}"
            ),
            Self::IdExhausted { kind } => write!(formatter, "{kind} capacity is exhausted"),
            Self::InvalidSchemaVersion { value } => {
                write!(
                    formatter,
                    "schema version must be non-zero; received {value}"
                )
            }
            Self::InvalidJsonPointer { value } => {
                write!(formatter, "invalid RFC 6901 JSON Pointer {value:?}")
            }
        }
    }
}

impl Error for CoreError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DocumentKind {
    PricingRequest,
    PricingResult,
}

impl DocumentKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PricingRequest => "pricing_request",
            Self::PricingResult => "pricing_result",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ValidationPhase {
    SyntaxAndLimits,
    DeclaredSchema,
    Migration,
    CurrentSchema,
    Domain,
}

impl ValidationPhase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SyntaxAndLimits => "syntax_and_limits",
            Self::DeclaredSchema => "declared_schema",
            Self::Migration => "migration",
            Self::CurrentSchema => "current_schema",
            Self::Domain => "domain",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationIssue {
    code: &'static str,
    instance_path: JsonPointer,
    phase: ValidationPhase,
    schema_version: SchemaVersion,
    document_kind: DocumentKind,
    message: String,
    migration_source_version: Option<SchemaVersion>,
    migration_target_version: Option<SchemaVersion>,
}

impl ValidationIssue {
    #[must_use]
    pub fn new(
        code: &'static str,
        instance_path: JsonPointer,
        phase: ValidationPhase,
        schema_version: SchemaVersion,
        document_kind: DocumentKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            instance_path,
            phase,
            schema_version,
            document_kind,
            message: message.into(),
            migration_source_version: None,
            migration_target_version: None,
        }
    }

    #[must_use]
    pub fn with_migration_versions(mut self, source: SchemaVersion, target: SchemaVersion) -> Self {
        self.migration_source_version = Some(source);
        self.migration_target_version = Some(target);
        self
    }

    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    #[must_use]
    pub fn instance_path(&self) -> &str {
        self.instance_path.as_str()
    }

    #[must_use]
    pub const fn phase(&self) -> ValidationPhase {
        self.phase
    }

    #[must_use]
    pub const fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }

    #[must_use]
    pub const fn document_kind(&self) -> DocumentKind {
        self.document_kind
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    #[must_use]
    pub const fn migration_source_version(&self) -> Option<SchemaVersion> {
        self.migration_source_version
    }

    #[must_use]
    pub const fn migration_target_version(&self) -> Option<SchemaVersion> {
        self.migration_target_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationErrors {
    issues: Box<[ValidationIssue]>,
}

impl ValidationErrors {
    pub const DEFAULT_MAX_ISSUES: usize = 64;

    #[must_use]
    pub fn collect<I>(issues: I) -> Option<Self>
    where
        I: IntoIterator<Item = ValidationIssue>,
    {
        let issues: Vec<_> = issues.into_iter().take(Self::DEFAULT_MAX_ISSUES).collect();
        (!issues.is_empty()).then(|| Self {
            issues: issues.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let first = &self.issues[0];
        write!(
            formatter,
            "{} validation issue(s); first: {} at {}: {}",
            self.issues.len(),
            first.code(),
            first.instance_path(),
            first.message()
        )
    }
}

impl Error for ValidationErrors {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_collection_is_bounded_and_ordered() {
        let issues = (0..70).map(|index| {
            ValidationIssue::new(
                "invalid_value",
                format!("/values/{index}").parse().expect("valid pointer"),
                ValidationPhase::Domain,
                SchemaVersion::CURRENT,
                DocumentKind::PricingRequest,
                format!("value {index} is invalid"),
            )
        });
        let errors = ValidationErrors::collect(issues).expect("non-empty");
        assert_eq!(errors.issues().len(), 64);
        assert_eq!(errors.issues()[0].instance_path(), "/values/0");
        assert_eq!(errors.issues()[63].instance_path(), "/values/63");
    }

    #[test]
    fn empty_issue_collection_has_no_error() {
        assert!(ValidationErrors::collect([]).is_none());
    }
}
