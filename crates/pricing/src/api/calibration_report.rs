//! Owned reporting data, independent of calibration plans and workspaces.
use crate::{ReportAxis, WireError};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CalibrationKind {
    ParticleLsv,
    HullWhiteLsv,
    LocalCorrelation,
    BassFixedPoint,
    BassSurfaceProjection,
}
impl CalibrationKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ParticleLsv => "particle_lsv",
            Self::HullWhiteLsv => "hull_white_lsv",
            Self::LocalCorrelation => "local_correlation",
            Self::BassFixedPoint => "bass_fixed_point",
            Self::BassSurfaceProjection => "bass_surface_projection",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DiagnosticUnit {
    Dimensionless,
    Count,
    Indicator,
    Time,
    Variance,
    EquityCoordinate,
    BrownianCoordinate,
}
impl DiagnosticUnit {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dimensionless => "dimensionless",
            Self::Count => "count",
            Self::Indicator => "indicator",
            Self::Time => "time",
            Self::Variance => "variance",
            Self::EquityCoordinate => "equity_coordinate",
            Self::BrownianCoordinate => "brownian_coordinate",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationMetric {
    /// Stable metric identifier within the calibration kind.
    pub name: String,
    /// Row-major values on the enclosing block's Cartesian axes.
    pub values: Vec<f64>,
    pub unit: DiagnosticUnit,
    /// Native meaning, including any aggregation or coordinate convention.
    pub convention: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReportedCalibration {
    pub kind: CalibrationKind,
    /// Native asset order; a basket block contains all constituent IDs.
    pub underlying_ids: Vec<u32>,
    /// Absent for deterministic calibration. Never the pricing seed.
    pub seed: Option<u64>,
    pub axes: Vec<ReportAxis>,
    /// Absent metrics are unavailable, never implicit zeros.
    pub metrics: Vec<CalibrationMetric>,
}

pub(crate) fn validate(blocks: &[ReportedCalibration]) -> Result<(), WireError> {
    let mut identities = std::collections::HashSet::new();
    for (i, b) in blocks.iter().enumerate() {
        let invalid = || WireError::DomainAt {
            pointer: format!("/calibration_diagnostics/{i}"),
            message: "invalid calibration identity, axes, metric shape, unit or value".into(),
        };
        let mut ids = std::collections::HashSet::new();
        let mut axes = std::collections::HashSet::new();
        let count = b.axes.iter().try_fold(1_usize, |n, a| {
            if a.name.is_empty()
                || !axes.insert(&a.name)
                || a.values.is_empty()
                || a.values.iter().any(|v| !v.is_finite())
                || a.values.windows(2).any(|v| v[0] >= v[1])
            {
                None
            } else {
                n.checked_mul(a.values.len())
            }
        });
        if b.underlying_ids.is_empty()
            || b.underlying_ids.iter().any(|id| !ids.insert(id))
            || !identities.insert((b.kind, &b.underlying_ids))
            || count.is_none()
            || b.metrics.is_empty()
        {
            return Err(invalid());
        }
        let mut names = std::collections::HashSet::new();
        for m in &b.metrics {
            if m.name.is_empty()
                || m.convention.is_empty()
                || !names.insert(&m.name)
                || count != Some(m.values.len())
                || m.values.iter().any(|v| {
                    !v.is_finite()
                        || match m.unit {
                            DiagnosticUnit::Count => {
                                *v < 0.0 || *v > 9_007_199_254_740_992.0 || v.fract() != 0.0
                            }
                            DiagnosticUnit::Indicator => *v != 0.0 && *v != 1.0,
                            _ => false,
                        }
                })
            {
                return Err(invalid());
            }
        }
    }
    Ok(())
}
