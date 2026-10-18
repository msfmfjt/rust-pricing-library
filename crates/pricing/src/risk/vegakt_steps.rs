use std::sync::Arc;

use crate::market::LocalVarianceGrid;
use crate::mc::LocalVolLogEulerPlan;
use crate::risk::{AnalyticCallDensityRow, Equation11GammaRow};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtStepSamplingUnit {
    IndependentPath,
    AntitheticPair,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VegaKtStepLayout {
    pub simulation: LocalVolLogEulerPlan,
    pub grid: LocalVarianceGrid,
    pub densities: Box<[AnalyticCallDensityRow]>,
    pub variances: Box<[f64]>,
    pub node_count: usize,
    pub direct_offset: usize,
    pub direct_totals_offset: usize,
    pub combined_total_index: usize,
    pub component_count: usize,
}

/// Positive-time equation-(11) recovery on actual simulation intervals.
/// The initial row remains observable local Vega without a Gamma estimate.
#[derive(Clone, Debug)]
pub struct VegaKtStepRecoveryPlan {
    pub(crate) layout: Arc<VegaKtStepLayout>,
}

/// One path, or one antithetic pair averaged before statistics are computed.
/// Samples can only be combined with their originating compiled plan (or clone).
#[derive(Clone, Debug)]
pub struct VegaKtStepSample {
    pub(crate) layout: Arc<VegaKtStepLayout>,
    pub(crate) values: Box<[f64]>,
    pub(crate) sampling_unit: VegaKtStepSamplingUnit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtStepMoment {
    pub(crate) mean: f64,
    pub(crate) sample_variance: Option<f64>,
    pub(crate) standard_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtStepRowEstimate {
    pub(crate) source_maturity: f64,
    pub(crate) recovered_maturity: f64,
    pub(crate) forward: f64,
    pub(crate) nodes: Box<[VegaKtStepMoment]>,
    pub(crate) total: VegaKtStepMoment,
    pub(crate) retained: Option<VegaKtStepMoment>,
    pub(crate) excluded: Option<VegaKtStepMoment>,
    pub(crate) gamma: Option<Equation11GammaRow>,
    pub(crate) gamma_standard_errors: Option<Box<[f64]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtStepEstimate {
    pub(crate) independent_sampling_units: u64,
    pub(crate) sampling_unit: VegaKtStepSamplingUnit,
    pub(crate) log_moneyness_nodes: Box<[f64]>,
    pub(crate) rows: Box<[VegaKtStepRowEstimate]>,
    pub(crate) total: VegaKtStepMoment,
    pub(crate) direct_rows: Box<[VegaKtDirectVarianceRowEstimate]>,
    pub(crate) combined_total: VegaKtStepMoment,
    pub(crate) has_unrecovered_direct_variance: bool,
}

/// Direct effective-variance dependence at one actual node, including maturity.
/// No interval length or Gamma interpretation is assigned to this contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtDirectVarianceRowEstimate {
    pub(crate) maturity: f64,
    pub(crate) nodes: Box<[VegaKtStepMoment]>,
    pub(crate) total: VegaKtStepMoment,
}
