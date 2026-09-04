use std::num::{NonZeroU32, NonZeroU64};

use crate::EngineConfigError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VarianceReduction {
    antithetic: bool,
    brownian_bridge: bool,
}

impl VarianceReduction {
    #[must_use]
    pub const fn new(antithetic: bool, brownian_bridge: bool) -> Self {
        Self {
            antithetic,
            brownian_bridge,
        }
    }

    #[must_use]
    pub const fn antithetic(self) -> bool {
        self.antithetic
    }

    #[must_use]
    pub const fn brownian_bridge(self) -> bool {
        self.brownian_bridge
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PseudoMcConfig {
    master_seed: u64,
    independent_sampling_units: NonZeroU64,
    variance_reduction: VarianceReduction,
}

impl PseudoMcConfig {
    pub fn new(
        master_seed: u64,
        independent_sampling_units: u64,
        variance_reduction: VarianceReduction,
    ) -> Result<Self, EngineConfigError> {
        Ok(Self {
            master_seed,
            independent_sampling_units: NonZeroU64::new(independent_sampling_units)
                .ok_or(EngineConfigError::ZeroSamplingUnits)?,
            variance_reduction,
        })
    }

    #[must_use]
    pub const fn master_seed(self) -> u64 {
        self.master_seed
    }

    #[must_use]
    pub const fn independent_sampling_units(self) -> NonZeroU64 {
        self.independent_sampling_units
    }

    #[must_use]
    pub const fn evaluated_paths(self) -> u128 {
        let multiplier = if self.variance_reduction.antithetic() {
            2
        } else {
            1
        };
        self.independent_sampling_units.get() as u128 * multiplier
    }

    #[must_use]
    pub const fn variance_reduction(self) -> VarianceReduction {
        self.variance_reduction
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RqmcConfig {
    points_per_scramble: NonZeroU64,
    scramble_count: NonZeroU32,
    master_scramble_seed: u64,
    variance_reduction: VarianceReduction,
}

impl RqmcConfig {
    pub const DEFAULT_SCRAMBLE_COUNT: u32 = 16;
    pub const MAX_SOBOL_DIMENSION: u32 = 21_201;

    pub fn new(
        points_per_scramble: u64,
        scramble_count: u32,
        master_scramble_seed: u64,
        variance_reduction: VarianceReduction,
    ) -> Result<Self, EngineConfigError> {
        let points_per_scramble = NonZeroU64::new(points_per_scramble).ok_or(
            EngineConfigError::InvalidSobolPointCount {
                value: points_per_scramble,
            },
        )?;
        if !points_per_scramble.get().is_power_of_two() {
            return Err(EngineConfigError::InvalidSobolPointCount {
                value: points_per_scramble.get(),
            });
        }
        Ok(Self {
            points_per_scramble,
            scramble_count: NonZeroU32::new(scramble_count)
                .ok_or(EngineConfigError::ZeroScrambleCount)?,
            master_scramble_seed,
            variance_reduction,
        })
    }

    pub fn with_default_scrambles(
        points_per_scramble: u64,
        master_scramble_seed: u64,
        variance_reduction: VarianceReduction,
    ) -> Result<Self, EngineConfigError> {
        Self::new(
            points_per_scramble,
            Self::DEFAULT_SCRAMBLE_COUNT,
            master_scramble_seed,
            variance_reduction,
        )
    }

    pub fn rounded_point_count(requested: u64) -> Result<u64, EngineConfigError> {
        requested
            .max(1)
            .checked_next_power_of_two()
            .ok_or(EngineConfigError::SobolPointCountOverflow { requested })
    }

    #[must_use]
    pub const fn points_per_scramble(self) -> NonZeroU64 {
        self.points_per_scramble
    }

    #[must_use]
    pub const fn scramble_count(self) -> NonZeroU32 {
        self.scramble_count
    }

    #[must_use]
    pub const fn master_scramble_seed(self) -> u64 {
        self.master_scramble_seed
    }

    #[must_use]
    pub const fn variance_reduction(self) -> VarianceReduction {
        self.variance_reduction
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineConfig {
    PseudoMonteCarlo(PseudoMcConfig),
    RandomizedQuasiMonteCarlo(RqmcConfig),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_path_count_means_independent_units() {
        let config =
            PseudoMcConfig::new(7, 100, VarianceReduction::new(true, false)).expect("valid config");
        assert_eq!(config.independent_sampling_units().get(), 100);
        assert_eq!(config.evaluated_paths(), 200);
        assert!(PseudoMcConfig::new(7, 0, VarianceReduction::new(false, false)).is_err());
    }

    #[test]
    fn rqmc_requires_power_of_two_and_defaults_to_sixteen_scrambles() {
        let config =
            RqmcConfig::with_default_scrambles(1 << 12, 99, VarianceReduction::new(true, true))
                .expect("valid config");
        assert_eq!(config.scramble_count().get(), 16);
        assert!(RqmcConfig::new(1000, 16, 99, VarianceReduction::new(false, true)).is_err());
        assert!(RqmcConfig::new(1024, 0, 99, VarianceReduction::new(false, true)).is_err());
    }

    #[test]
    fn rounding_helper_does_not_mutate_explicit_input() {
        assert_eq!(RqmcConfig::rounded_point_count(0), Ok(1));
        assert_eq!(RqmcConfig::rounded_point_count(1024), Ok(1024));
        assert_eq!(RqmcConfig::rounded_point_count(1025), Ok(2048));
        assert!(RqmcConfig::rounded_point_count(u64::MAX).is_err());
    }
}
