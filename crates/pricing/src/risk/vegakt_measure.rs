use crate::risk::{ReportingIvBasis, VegaKtVanillaProjection};

/// A signed distributional Gamma at a strictly positive physical strike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtGammaAtom {
    pub(crate) strike: f64,
    pub(crate) mass: f64,
}

/// Finite signed measure; sorted, duplicate strikes combined, zero masses removed.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtGammaAtoms {
    pub(crate) atoms: Box<[VegaKtGammaAtom]>,
}

/// Exact representation on S > 0: cash + slope*S + sum mass*(S-strike)+.
/// Affine terms have zero Gamma but are retained to audit payoff reconstruction.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtTerminalPayoff {
    pub(crate) cash: f64,
    pub(crate) slope: f64,
    pub(crate) gamma: VegaKtGammaAtoms,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtTerminalLocalRow {
    pub source_maturity: f64,
    pub destination_maturity: f64,
    pub source_total_variance: f64,
    pub interval_volatility: f64,
    /// Exact population expectations of native source-log-hat deposition,
    /// in local-volatility units (before grid-time interpolation).
    pub expected_source_hat_vegas: Vec<f64>,
}

pub const VEGA_KT_TERMINAL_MEASURE_LABEL: &str = "flat_variance_terminal_measure_diagnostic_v1";

/// Deterministic terminal-state diagnostic, separate from MC risk acceptance.
/// No estimated moments, synthetic samples or fitted reconciliation residuals.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtTerminalDiagnostic {
    pub policy_label: &'static str,
    pub payoff: VegaKtTerminalPayoff,
    pub reporting_basis: ReportingIvBasis,
    pub projection: VegaKtVanillaProjection,
    pub source_log_moneyness_nodes: Vec<f64>,
    pub local_rows: Vec<VegaKtTerminalLocalRow>,
    pub total_variance: f64,
    pub expected_local_parallel_vega: f64,
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
}
