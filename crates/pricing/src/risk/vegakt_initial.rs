use crate::market::LocalVarianceGrid;
use crate::mc::LocalVolLogEulerPlan;
use crate::risk::{VegaKtStepMoment, VegaKtStepSamplingUnit};
use std::sync::Arc;

pub const INITIAL_GAMMA_CONDITIONAL_FD_LABEL: &str = "initial_gamma_conditional_fd_diagnostic_v1";

#[derive(Clone, Debug)]
pub(crate) struct InitialGammaLayout {
    pub simulation: LocalVolLogEulerPlan,
    pub grid: LocalVarianceGrid,
    pub strikes: Box<[f64]>,
    pub stencils: Box<[[f64; 7]]>,
    pub bumps: Box<[f64]>,
    pub relative_bump: f64,
    pub first_variance: f64,
    pub vega_scale: f64,
    pub kernel_tail_masses: [f64; 2],
    pub components: usize,
}
/// Additional conditional repricing, not inversion of the scalar initial Vega.
/// Diagnostic: spatially flat variance, constant forward and state-only payoff.
#[derive(Clone, Debug)]
pub struct VegaKtInitialGammaPlan {
    pub(crate) layout: Arc<InitialGammaLayout>,
}
#[derive(Clone, Debug)]
pub struct VegaKtInitialGammaSample {
    pub(crate) layout: Arc<InitialGammaLayout>,
    pub(crate) values: Box<[f64]>,
    pub(crate) sampling_unit: VegaKtStepSamplingUnit,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtGammaBumpEstimate {
    pub(crate) coarse: VegaKtStepMoment,
    pub(crate) fine: VegaKtStepMoment,
    pub(crate) difference: VegaKtStepMoment,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtInitialGammaEstimate {
    pub(crate) maturity: f64,
    pub(crate) strikes: Box<[f64]>,
    pub(crate) relative_bump: f64,
    pub(crate) nodes: Box<[VegaKtGammaBumpEstimate]>,
    pub(crate) first_step_vega: VegaKtGammaBumpEstimate,
    pub(crate) kernel_tail_masses: [f64; 2],
    pub(crate) tail_vegas: [VegaKtStepMoment; 2],
    pub(crate) count: u64,
    pub(crate) sampling_unit: VegaKtStepSamplingUnit,
}
