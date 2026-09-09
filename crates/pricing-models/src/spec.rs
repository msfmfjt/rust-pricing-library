use pricing_core::{CoreError, NonNegativeF64};
use pricing_market::{LocalVarianceGrid, MarketError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlackScholesSpec {
    volatility: NonNegativeF64,
}

impl BlackScholesSpec {
    pub fn new(volatility: f64) -> Result<Self, CoreError> {
        Ok(Self {
            volatility: NonNegativeF64::new(volatility, "volatility")?,
        })
    }

    #[must_use]
    pub const fn volatility(self) -> NonNegativeF64 {
        self.volatility
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolatilitySpec {
    local_variance_grid: LocalVarianceGrid,
}

impl LocalVolatilitySpec {
    pub fn from_explicit_grid(
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        values: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, MarketError> {
        Ok(Self {
            local_variance_grid: LocalVarianceGrid::new(
                time_nodes,
                log_moneyness_nodes,
                values,
                floor,
                cap,
            )?,
        })
    }

    #[must_use]
    pub const fn local_variance_grid(&self) -> &LocalVarianceGrid {
        &self.local_variance_grid
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelSpec {
    BlackScholes(BlackScholesSpec),
    LocalVolatility(LocalVolatilitySpec),
}

impl ModelSpec {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BlackScholes(_) => "black_scholes",
            Self::LocalVolatility(_) => "local_volatility",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_scholes_allows_zero_volatility_limit() {
        assert_eq!(
            BlackScholesSpec::new(0.0)
                .expect("zero-volatility limit")
                .volatility()
                .get(),
            0.0
        );
        assert!(BlackScholesSpec::new(-0.01).is_err());
        assert!(BlackScholesSpec::new(f64::NAN).is_err());
    }

    #[test]
    fn local_volatility_explicit_grid_preserves_row_major_shape() {
        let spec = LocalVolatilitySpec::from_explicit_grid(
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
            1.0e-8,
            4.0,
        )
        .expect("local vol");
        assert_eq!(spec.local_variance_grid().time_nodes(), [0.25, 1.0]);
        assert_eq!(
            spec.local_variance_grid().log_moneyness_nodes(),
            [-0.1, 0.0, 0.2]
        );
        assert_eq!(spec.local_variance_grid().values()[4], 0.045);
    }
}
