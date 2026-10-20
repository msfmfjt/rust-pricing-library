//! Lossless source forms for the existing six rough families.
use super::*;
use crate::models::rough_volatility::ForwardVarianceKind;
use crate::models::{
    ForwardVarianceCurve, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, Rfsv, RoughHeston,
    RoughSabr, RoughVolatilityModel,
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum RoughModelV1 {
    RoughHeston {
        hurst: f64,
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
    },
    LiftedHeston {
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
        weights: Vec<f64>,
        rates: Vec<f64>,
    },
    QuadraticRoughHeston {
        hurst: f64,
        initial_state: f64,
        mean_reversion: f64,
        vol_of_vol: f64,
        quadratic: f64,
        shift: f64,
        variance_floor: f64,
    },
    MixedRoughBergomi {
        hurst: f64,
        correlation: f64,
        weights: Vec<f64>,
        vol_of_vols: Vec<f64>,
        forward_variance: ForwardVarianceV1,
    },
    RoughSabr {
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
        beta: f64,
        forward_variance: ForwardVarianceV1,
    },
    Rfsv {
        hurst: f64,
        mean_reversion: f64,
        vol_of_log_vol: f64,
        mean_log_vol: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        initial_log_vol: Option<f64>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ForwardVarianceV1 {
    Knots { times: Vec<f64>, values: Vec<f64> },
    Exponential { initial: f64, growth: f64 },
}
impl From<&ForwardVarianceCurve> for ForwardVarianceV1 {
    fn from(value: &ForwardVarianceCurve) -> Self {
        match &value.kind {
            ForwardVarianceKind::Knots { times, values } => Self::Knots {
                times: times.clone(),
                values: values.clone(),
            },
            ForwardVarianceKind::Exponential { initial, growth } => Self::Exponential {
                initial: *initial,
                growth: *growth,
            },
        }
    }
}
impl ForwardVarianceV1 {
    fn into_domain(self) -> Result<ForwardVarianceCurve, WireError> {
        match self {
            Self::Knots { times, values } => ForwardVarianceCurve::piecewise_linear(times, values),
            Self::Exponential { initial, growth } => {
                ForwardVarianceCurve::exponential(initial, growth)
            }
        }
        .map_err(|e| domain_at("/model/equity/model/forward_variance", e))
    }
}
impl From<&RoughVolatilityModel> for RoughModelV1 {
    fn from(model: &RoughVolatilityModel) -> Self {
        match model {
            RoughVolatilityModel::RoughHeston(m) => Self::RoughHeston {
                hurst: m.hurst,
                initial_variance: m.initial_variance,
                mean_reversion: m.mean_reversion,
                long_run_variance: m.long_run_variance,
                vol_of_vol: m.vol_of_vol,
                correlation: m.correlation,
            },
            RoughVolatilityModel::LiftedHeston(m) => Self::LiftedHeston {
                initial_variance: m.heston.initial_variance,
                mean_reversion: m.heston.mean_reversion,
                long_run_variance: m.heston.long_run_variance,
                vol_of_vol: m.heston.vol_of_vol,
                correlation: m.heston.correlation,
                weights: m.weights.clone(),
                rates: m.rates.clone(),
            },
            RoughVolatilityModel::QuadraticRoughHeston(m) => Self::QuadraticRoughHeston {
                hurst: m.hurst,
                initial_state: m.initial_state,
                mean_reversion: m.mean_reversion,
                vol_of_vol: m.vol_of_vol,
                quadratic: m.quadratic,
                shift: m.shift,
                variance_floor: m.variance_floor,
            },
            RoughVolatilityModel::MixedRoughBergomi(m) => Self::MixedRoughBergomi {
                hurst: m.hurst,
                correlation: m.correlation,
                weights: m.weights.clone(),
                vol_of_vols: m.vol_of_vols.clone(),
                forward_variance: (&m.forward_variance).into(),
            },
            RoughVolatilityModel::RoughSabr(m) => Self::RoughSabr {
                hurst: m.hurst,
                vol_of_vol: m.vol_of_vol,
                correlation: m.correlation,
                beta: m.beta,
                forward_variance: (&m.forward_variance).into(),
            },
            RoughVolatilityModel::Rfsv(m) => Self::Rfsv {
                hurst: m.hurst,
                mean_reversion: m.mean_reversion,
                vol_of_log_vol: m.vol_of_log_vol,
                mean_log_vol: m.mean_log_vol,
                initial_log_vol: m.initial_log_vol,
            },
        }
    }
}
impl RoughModelV1 {
    pub(super) fn into_domain(self) -> Result<RoughVolatilityModel, WireError> {
        Ok(match self {
            Self::RoughHeston {
                hurst,
                initial_variance,
                mean_reversion,
                long_run_variance,
                vol_of_vol,
                correlation,
            } => {
                let model = RoughHeston::new(
                    hurst,
                    initial_variance,
                    mean_reversion,
                    long_run_variance,
                    vol_of_vol,
                    correlation,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                RoughVolatilityModel::RoughHeston(model)
            }
            Self::LiftedHeston {
                initial_variance,
                mean_reversion,
                long_run_variance,
                vol_of_vol,
                correlation,
                weights,
                rates,
            } => {
                let model = LiftedHeston::new(
                    initial_variance,
                    mean_reversion,
                    long_run_variance,
                    vol_of_vol,
                    correlation,
                    weights,
                    rates,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                RoughVolatilityModel::LiftedHeston(model)
            }
            Self::QuadraticRoughHeston {
                hurst,
                initial_state,
                mean_reversion,
                vol_of_vol,
                quadratic,
                shift,
                variance_floor,
            } => {
                let model = QuadraticRoughHeston::new(
                    hurst,
                    initial_state,
                    mean_reversion,
                    vol_of_vol,
                    quadratic,
                    shift,
                    variance_floor,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                RoughVolatilityModel::QuadraticRoughHeston(model)
            }
            Self::MixedRoughBergomi {
                hurst,
                correlation,
                weights,
                vol_of_vols,
                forward_variance,
            } => {
                let encoded_weights = weights.clone();
                let mut model = MixedRoughBergomi::new(
                    hurst,
                    correlation,
                    weights,
                    vol_of_vols,
                    forward_variance.into_domain()?,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                // The constructor validates the convex-combination tolerance. Keep
                // the encoded, already normalized bits to avoid replay drift caused
                // by renormalizing a mixture on every JSON load.
                model.weights = encoded_weights;
                RoughVolatilityModel::MixedRoughBergomi(model)
            }
            Self::RoughSabr {
                hurst,
                vol_of_vol,
                correlation,
                beta,
                forward_variance,
            } => {
                let model = RoughSabr::new(
                    hurst,
                    vol_of_vol,
                    correlation,
                    beta,
                    forward_variance.into_domain()?,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                RoughVolatilityModel::RoughSabr(model)
            }
            Self::Rfsv {
                hurst,
                mean_reversion,
                vol_of_log_vol,
                mean_log_vol,
                initial_log_vol,
            } => {
                let model = Rfsv::new(
                    hurst,
                    mean_reversion,
                    vol_of_log_vol,
                    mean_log_vol,
                    initial_log_vol,
                )
                .map_err(|e| domain_at("/model/equity/model", e))?;
                RoughVolatilityModel::Rfsv(model)
            }
        })
    }
}
