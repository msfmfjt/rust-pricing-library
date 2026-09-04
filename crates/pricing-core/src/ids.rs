use std::fmt;
use std::num::NonZeroU32;

use crate::CoreError;

macro_rules! define_id {
    ($name:ident, $raw:ty, $kind:literal) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name($raw);

        impl $name {
            #[must_use]
            pub const fn new(value: $raw) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn get(self) -> $raw {
                self.0
            }

            pub fn checked_next(self) -> Result<Self, CoreError> {
                self.0
                    .checked_add(1)
                    .map(Self)
                    .ok_or(CoreError::IdExhausted { kind: $kind })
            }
        }

        impl From<$raw> for $name {
            fn from(value: $raw) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> Self {
                value.get()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

define_id!(UnderlyingId, u32, "underlying_id");
define_id!(CurveId, u32, "curve_id");
define_id!(CurrencyId, u16, "currency_id");
define_id!(EventId, u32, "event_id");
define_id!(NodeId, u32, "node_id");
define_id!(TimeIndex, u32, "time_index");
define_id!(PathIndex, u64, "path_index");

/// Version of a public request/result wire schema.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SchemaVersion(NonZeroU32);

impl SchemaVersion {
    pub const CURRENT: Self = Self(NonZeroU32::MIN);

    pub fn new(value: u32) -> Result<Self, CoreError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(CoreError::InvalidSchemaVersion { value })
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl Default for SchemaVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_preserve_values_and_stop_at_capacity() {
        assert_eq!(NodeId::new(7).get(), 7);
        assert_eq!(NodeId::new(7).checked_next(), Ok(NodeId::new(8)));
        assert_eq!(
            NodeId::new(u32::MAX).checked_next(),
            Err(CoreError::IdExhausted { kind: "node_id" })
        );
        assert_eq!(
            PathIndex::new(u64::MAX).checked_next(),
            Err(CoreError::IdExhausted { kind: "path_index" })
        );
    }

    #[test]
    fn schema_version_is_strictly_positive() {
        assert_eq!(SchemaVersion::CURRENT.get(), 1);
        assert_eq!(SchemaVersion::new(9).expect("valid version").get(), 9);
        assert_eq!(
            SchemaVersion::new(0),
            Err(CoreError::InvalidSchemaVersion { value: 0 })
        );
    }
}
