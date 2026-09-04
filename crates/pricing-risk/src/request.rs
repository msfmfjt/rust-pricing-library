use std::num::NonZeroU32;

use pricing_core::{Date, FiniteF64, PositiveF64};

use crate::RiskConfigError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmileDynamics {
    StickyLogMoneyness,
    StickyStrike,
    StickyDelta,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpotBump {
    Absolute(PositiveF64),
    Relative(PositiveF64),
}

impl SpotBump {
    pub fn absolute(value: f64) -> Result<Self, RiskConfigError> {
        Ok(Self::Absolute(PositiveF64::new(
            value,
            "gamma_absolute_bump",
        )?))
    }

    pub fn relative(value: f64) -> Result<Self, RiskConfigError> {
        Ok(Self::Relative(PositiveF64::new(
            value,
            "gamma_relative_bump",
        )?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GammaConfig {
    bump: SpotBump,
}

impl GammaConfig {
    #[must_use]
    pub const fn new(bump: SpotBump) -> Self {
        Self { bump }
    }

    #[must_use]
    pub const fn bump(self) -> SpotBump {
        self.bump
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VegaKtConfig {
    maturity_nodes: Box<[Date]>,
    log_forward_moneyness_nodes: Box<[FiniteF64]>,
    relative_density_threshold: PositiveF64,
    full_bucket_covariance: bool,
}

impl VegaKtConfig {
    pub fn new(
        maturity_nodes: Vec<Date>,
        log_forward_moneyness_nodes: Vec<f64>,
        relative_density_threshold: f64,
        full_bucket_covariance: bool,
    ) -> Result<Self, RiskConfigError> {
        if maturity_nodes.len() < 2 {
            return Err(RiskConfigError::TooFewVegaKtMaturities {
                count: maturity_nodes.len(),
            });
        }
        if log_forward_moneyness_nodes.len() < 2 {
            return Err(RiskConfigError::TooFewVegaKtStrikes {
                count: log_forward_moneyness_nodes.len(),
            });
        }
        for (left_index, pair) in maturity_nodes.windows(2).enumerate() {
            if pair[1] <= pair[0] {
                return Err(RiskConfigError::UnsortedVegaKtMaturities { left_index });
            }
        }
        let log_forward_moneyness_nodes = log_forward_moneyness_nodes
            .into_iter()
            .map(|value| FiniteF64::new(value, "vega_kt_log_forward_moneyness"))
            .collect::<Result<Vec<_>, _>>()?;
        for (left_index, pair) in log_forward_moneyness_nodes.windows(2).enumerate() {
            if pair[1].get() <= pair[0].get() {
                return Err(RiskConfigError::UnsortedVegaKtStrikes { left_index });
            }
        }
        let relative_density_threshold = PositiveF64::new(
            relative_density_threshold,
            "vega_kt_relative_density_threshold",
        )?;
        if relative_density_threshold.get() > 1.0 {
            return Err(RiskConfigError::InvalidDensityThreshold {
                bits: relative_density_threshold.get().to_bits(),
            });
        }
        Ok(Self {
            maturity_nodes: maturity_nodes.into_boxed_slice(),
            log_forward_moneyness_nodes: log_forward_moneyness_nodes.into_boxed_slice(),
            relative_density_threshold,
            full_bucket_covariance,
        })
    }

    #[must_use]
    pub fn maturity_nodes(&self) -> &[Date] {
        &self.maturity_nodes
    }

    #[must_use]
    pub fn log_forward_moneyness_nodes(&self) -> &[FiniteF64] {
        &self.log_forward_moneyness_nodes
    }

    #[must_use]
    pub const fn relative_density_threshold(&self) -> PositiveF64 {
        self.relative_density_threshold
    }

    #[must_use]
    pub const fn full_bucket_covariance(&self) -> bool {
        self.full_bucket_covariance
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RiskRequest {
    delta: bool,
    gamma: Option<GammaConfig>,
    vega: bool,
    vega_kt: Option<VegaKtConfig>,
    smile_dynamics: SmileDynamics,
    checkpoint_interval: Option<NonZeroU32>,
    aad_tile_capacity: Option<NonZeroU32>,
}

impl RiskRequest {
    pub fn new(
        delta: bool,
        gamma: Option<GammaConfig>,
        vega: bool,
        vega_kt: Option<VegaKtConfig>,
        smile_dynamics: SmileDynamics,
        checkpoint_interval: Option<u32>,
        aad_tile_capacity: Option<u32>,
    ) -> Result<Self, RiskConfigError> {
        let checkpoint_interval = checkpoint_interval
            .map(|value| NonZeroU32::new(value).ok_or(RiskConfigError::ZeroCheckpointInterval))
            .transpose()?;
        let aad_tile_capacity = aad_tile_capacity
            .map(|value| NonZeroU32::new(value).ok_or(RiskConfigError::ZeroAadTileCapacity))
            .transpose()?;
        Ok(Self {
            delta,
            gamma,
            vega,
            vega_kt,
            smile_dynamics,
            checkpoint_interval,
            aad_tile_capacity,
        })
    }

    #[must_use]
    pub fn price_only(smile_dynamics: SmileDynamics) -> Self {
        Self {
            delta: false,
            gamma: None,
            vega: false,
            vega_kt: None,
            smile_dynamics,
            checkpoint_interval: None,
            aad_tile_capacity: None,
        }
    }

    #[must_use]
    pub const fn delta(&self) -> bool {
        self.delta
    }

    #[must_use]
    pub const fn gamma(&self) -> Option<GammaConfig> {
        self.gamma
    }

    #[must_use]
    pub const fn vega(&self) -> bool {
        self.vega
    }

    #[must_use]
    pub const fn vega_kt(&self) -> Option<&VegaKtConfig> {
        self.vega_kt.as_ref()
    }

    #[must_use]
    pub const fn smile_dynamics(&self) -> SmileDynamics {
        self.smile_dynamics
    }

    #[must_use]
    pub const fn checkpoint_interval(&self) -> Option<NonZeroU32> {
        self.checkpoint_interval
    }

    #[must_use]
    pub const fn aad_tile_capacity(&self) -> Option<NonZeroU32> {
        self.aad_tile_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_bump_and_aad_overrides_are_positive() {
        let gamma = GammaConfig::new(SpotBump::relative(0.01).expect("positive bump"));
        let request = RiskRequest::new(
            true,
            Some(gamma),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(256),
        )
        .expect("valid request");
        assert_eq!(request.checkpoint_interval().map(NonZeroU32::get), Some(16));
        assert_eq!(request.aad_tile_capacity().map(NonZeroU32::get), Some(256));
        assert!(
            RiskRequest::new(
                true,
                Some(gamma),
                true,
                None,
                SmileDynamics::StickyLogMoneyness,
                Some(0),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn vega_kt_grid_is_explicit_and_strictly_increasing() {
        let maturities = vec![
            "2027-01-01".parse().expect("date"),
            "2028-01-01".parse().expect("date"),
        ];
        let config = VegaKtConfig::new(maturities.clone(), vec![-0.2, 0.0, 0.2], 1.0e-8, false)
            .expect("valid grid");
        assert_eq!(config.maturity_nodes(), maturities);
        assert!(VegaKtConfig::new(maturities.clone(), vec![0.0, 0.0], 1.0e-8, false).is_err());
        assert!(VegaKtConfig::new(maturities, vec![-0.2, 0.2], 1.1, false).is_err());
    }
}
