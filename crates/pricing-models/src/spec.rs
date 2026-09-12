use pricing_core::{CoreError, NonNegativeF64, PositiveF64};
use pricing_market::{ImpliedVarianceSurface, LocalVarianceGrid, MarketError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlackScholesSpec {
    volatility: NonNegativeF64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Black76Spec {
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

impl Black76Spec {
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
    reporting_iv_basis: Option<LocalVolatilityReportingBasis>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolatilityReportingBasis {
    maturity_nodes: Box<[f64]>,
    log_forward_moneyness_nodes: Box<[f64]>,
    implied_volatilities: Box<[f64]>,
}

impl LocalVolatilitySpec {
    pub fn from_surface(
        surface: &dyn ImpliedVarianceSurface,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, MarketError> {
        Ok(Self {
            local_variance_grid: LocalVarianceGrid::from_surface(
                surface,
                time_nodes,
                log_moneyness_nodes,
                floor,
                cap,
            )?,
            reporting_iv_basis: None,
        })
    }

    pub fn from_surface_with_reporting_basis(
        surface: &dyn ImpliedVarianceSurface,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
        reporting_maturity_nodes: Vec<f64>,
        reporting_log_moneyness_nodes: Vec<f64>,
    ) -> Result<Self, MarketError> {
        let reporting_iv_basis = LocalVolatilityReportingBasis::from_surface(
            surface,
            reporting_maturity_nodes,
            reporting_log_moneyness_nodes,
        )?;
        Ok(Self {
            local_variance_grid: LocalVarianceGrid::from_surface(
                surface,
                time_nodes,
                log_moneyness_nodes,
                floor,
                cap,
            )?,
            reporting_iv_basis: Some(reporting_iv_basis),
        })
    }

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
            reporting_iv_basis: None,
        })
    }

    #[must_use]
    pub const fn local_variance_grid(&self) -> &LocalVarianceGrid {
        &self.local_variance_grid
    }

    #[must_use]
    pub const fn reporting_iv_basis(&self) -> Option<&LocalVolatilityReportingBasis> {
        self.reporting_iv_basis.as_ref()
    }

    #[must_use]
    pub fn with_reporting_iv_basis(mut self, basis: LocalVolatilityReportingBasis) -> Self {
        self.reporting_iv_basis = Some(basis);
        self
    }
}

impl LocalVolatilityReportingBasis {
    pub fn from_surface(
        surface: &dyn ImpliedVarianceSurface,
        maturity_nodes: Vec<f64>,
        log_forward_moneyness_nodes: Vec<f64>,
    ) -> Result<Self, MarketError> {
        let capacity =
            reporting_iv_value_count(maturity_nodes.len(), log_forward_moneyness_nodes.len())
                .map_err(core_error_to_market)?;
        let mut implied_volatilities = Vec::with_capacity(capacity);
        for maturity in maturity_nodes.iter().copied() {
            let maturity = PositiveF64::new(maturity, "reporting_iv_maturity")
                .map_err(core_error_to_market)?
                .get();
            for log_moneyness in log_forward_moneyness_nodes.iter().copied() {
                let derivatives = surface.total_variance_derivatives(maturity, log_moneyness)?;
                let variance = PositiveF64::new(
                    derivatives.total_variance / maturity,
                    "reporting_iv_variance",
                )
                .map_err(core_error_to_market)?;
                implied_volatilities.push(variance.get().sqrt());
            }
        }
        Self::new(
            maturity_nodes,
            log_forward_moneyness_nodes,
            implied_volatilities,
        )
        .map_err(core_error_to_market)
    }

    pub fn new(
        maturity_nodes: Vec<f64>,
        log_forward_moneyness_nodes: Vec<f64>,
        implied_volatilities: Vec<f64>,
    ) -> Result<Self, CoreError> {
        validate_reporting_nodes(&maturity_nodes, "reporting_iv_maturity")?;
        validate_reporting_nodes(
            &log_forward_moneyness_nodes,
            "reporting_iv_log_forward_moneyness",
        )?;
        let expected =
            reporting_iv_value_count(maturity_nodes.len(), log_forward_moneyness_nodes.len())?;
        if implied_volatilities.len() != expected {
            return Err(CoreError::NumberNotPositive {
                field: "reporting_iv_implied_volatility_count",
                bits: (implied_volatilities.len() as f64).to_bits(),
            });
        }
        for value in implied_volatilities.iter().copied() {
            PositiveF64::new(value, "reporting_iv_implied_volatility")?;
        }
        Ok(Self {
            maturity_nodes: maturity_nodes.into_boxed_slice(),
            log_forward_moneyness_nodes: log_forward_moneyness_nodes.into_boxed_slice(),
            implied_volatilities: implied_volatilities.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn maturity_nodes(&self) -> &[f64] {
        &self.maturity_nodes
    }

    #[must_use]
    pub fn log_forward_moneyness_nodes(&self) -> &[f64] {
        &self.log_forward_moneyness_nodes
    }

    #[must_use]
    pub fn implied_volatilities(&self) -> &[f64] {
        &self.implied_volatilities
    }
}

fn validate_reporting_nodes(values: &[f64], name: &'static str) -> Result<(), CoreError> {
    if values.len() < 2 {
        return Err(CoreError::NumberNotPositive {
            field: name,
            bits: (values.len() as f64).to_bits(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(CoreError::NonFiniteNumber {
                field: name,
                bits: value.to_bits(),
            });
        }
        if index > 0 && value <= values[index - 1] {
            return Err(CoreError::NumberNotPositive {
                field: name,
                bits: (value - values[index - 1]).to_bits(),
            });
        }
    }
    Ok(())
}

fn reporting_iv_value_count(
    maturity_count: usize,
    log_forward_moneyness_count: usize,
) -> Result<usize, CoreError> {
    maturity_count
        .checked_mul(log_forward_moneyness_count)
        .ok_or(CoreError::NumberNotPositive {
            field: "reporting_iv_implied_volatility_count",
            bits: (usize::MAX as f64).to_bits(),
        })
}

fn core_error_to_market(error: CoreError) -> MarketError {
    match error {
        CoreError::NonFiniteNumber { field, bits }
        | CoreError::NumberNotPositive { field, bits } => MarketError::InvalidSurfaceParameter {
            parameter: field,
            bits,
        },
        other => MarketError::InvalidSurfaceParameter {
            parameter: "reporting_iv_basis",
            bits: other.to_string().len() as u64,
        },
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelSpec {
    BlackScholes(BlackScholesSpec),
    Black76(Black76Spec),
    LocalVolatility(LocalVolatilitySpec),
}

impl ModelSpec {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BlackScholes(_) => "black_scholes",
            Self::Black76(_) => "black_76",
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
    fn black_76_uses_same_constant_volatility_validation() {
        assert_eq!(
            Black76Spec::new(0.0)
                .expect("zero-volatility limit")
                .volatility()
                .get(),
            0.0
        );
        assert!(Black76Spec::new(-0.01).is_err());
        assert!(Black76Spec::new(f64::NAN).is_err());
        assert_eq!(
            ModelSpec::Black76(Black76Spec::new(0.2).expect("model")).name(),
            "black_76"
        );
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

    #[test]
    fn local_volatility_can_retain_reporting_iv_basis() {
        let basis = LocalVolatilityReportingBasis::new(
            vec![0.25, 1.0],
            vec![-0.2, 0.0, 0.2],
            vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
        )
        .expect("basis");
        let spec = LocalVolatilitySpec::from_explicit_grid(
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
            1.0e-8,
            4.0,
        )
        .expect("local vol")
        .with_reporting_iv_basis(basis);
        let basis = spec.reporting_iv_basis().expect("basis");
        assert_eq!(basis.maturity_nodes(), [0.25, 1.0]);
        assert_eq!(basis.log_forward_moneyness_nodes(), [-0.2, 0.0, 0.2]);
        assert_eq!(basis.implied_volatilities()[4], 0.22);
    }

    #[test]
    fn reporting_iv_value_count_rejects_overflow() {
        assert_eq!(reporting_iv_value_count(2, 3).expect("count"), 6);
        assert!(matches!(
            reporting_iv_value_count(usize::MAX, 2),
            Err(CoreError::NumberNotPositive {
                field: "reporting_iv_implied_volatility_count",
                ..
            })
        ));
    }

    #[test]
    fn local_volatility_can_materialize_from_implied_variance_surface() {
        use pricing_market::{EssviSlice, EssviSurface, SurfaceValidationTolerance};

        let surface = EssviSurface::new(
            vec![
                EssviSlice::new(0.25, 0.02, 0.1, -0.03).expect("first"),
                EssviSlice::new(1.0, 0.04, 0.2, -0.06).expect("second"),
            ],
            0.02,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("surface");
        let spec = LocalVolatilitySpec::from_surface(
            &surface,
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            1.0e-8,
            4.0,
        )
        .expect("local vol");
        assert_eq!(spec.local_variance_grid().values().len(), 6);
    }

    #[test]
    fn local_volatility_can_materialize_reporting_basis_from_surface() {
        use pricing_market::{EssviSlice, EssviSurface, SurfaceValidationTolerance};

        let surface = EssviSurface::new(
            vec![
                EssviSlice::new(0.25, 0.02, 0.1, -0.03).expect("first"),
                EssviSlice::new(1.0, 0.04, 0.2, -0.06).expect("second"),
            ],
            0.02,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("surface");
        let spec = LocalVolatilitySpec::from_surface_with_reporting_basis(
            &surface,
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            1.0e-8,
            4.0,
            vec![0.25, 1.0],
            vec![-0.2, 0.0, 0.2],
        )
        .expect("local vol");
        assert_eq!(
            spec.reporting_iv_basis()
                .expect("basis")
                .implied_volatilities()
                .len(),
            6
        );
    }
}
