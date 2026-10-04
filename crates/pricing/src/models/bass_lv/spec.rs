use super::{BassError, BassLvConfig, BassMarginal, BassSurfaceProjectionConfig};
use crate::market::MarketIvSurface;

/// Bass target IVs on the normalized residual-equity martingale `X` (`E[X]=1`).
/// Quotes use log(K_f/F_f), not untransformed physical spot strikes with cash dividends.
#[derive(Clone, Debug, PartialEq)]
pub struct BassLvSpec {
    surface: MarketIvSurface,
    projection_nodes: Vec<f64>,
    config: BassLvConfig,
    projection_config: BassSurfaceProjectionConfig,
    iv_bump: f64,
}
impl BassLvSpec {
    pub fn new(
        surface: MarketIvSurface,
        projection_nodes: Vec<f64>,
        config: BassLvConfig,
        projection_config: BassSurfaceProjectionConfig,
        iv_bump: f64,
    ) -> Result<Self, BassError> {
        config.validate()?;
        if !iv_bump.is_finite()
            || iv_bump <= 0.0
            || surface
                .implied_volatilities()
                .iter()
                .any(|v| *v <= iv_bump || v + iv_bump == *v || v - iv_bump == *v)
        {
            return Err(BassError::InvalidInput(
                "Bass IV bump must be positive, representable and smaller than every quote".into(),
            ));
        }
        // Validate projection shape/domain/tolerances without running calibration.
        BassMarginal::from_surface(
            surface.maturity_nodes()[0],
            1.0,
            &surface,
            &projection_nodes,
            projection_config,
        )?;
        Ok(Self {
            surface,
            projection_nodes,
            config,
            projection_config,
            iv_bump,
        })
    }
    pub fn surface(&self) -> &MarketIvSurface {
        &self.surface
    }
    pub fn projection_nodes(&self) -> &[f64] {
        &self.projection_nodes
    }
    pub fn config(&self) -> BassLvConfig {
        self.config
    }
    pub fn projection_config(&self) -> BassSurfaceProjectionConfig {
        self.projection_config
    }
    pub fn iv_bump(&self) -> f64 {
        self.iv_bump
    }
}
