//! Deterministic terminal Gamma measure; deliberately separate from MC reports.
use pricing::risk::VegaKtTerminalDiagnostic;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(frozen, name = "VegaKtTerminalDiagnostic")]
#[derive(Clone, Debug)]
pub struct PyVegaKtTerminalDiagnostic {
    pub(crate) inner: VegaKtTerminalDiagnostic,
}

#[pymethods]
impl PyVegaKtTerminalDiagnostic {
    #[getter]
    fn policy_label(&self) -> &'static str {
        self.inner.policy_label
    }
    #[getter]
    fn gamma_atoms(&self) -> Vec<(f64, f64)> {
        self.inner
            .payoff
            .gamma()
            .atoms()
            .iter()
            .map(|a| (a.strike(), a.mass()))
            .collect()
    }
    #[getter]
    fn raw_buckets(&self) -> Vec<f64> {
        self.inner.projection.position_risk().raw_buckets().to_vec()
    }
    #[getter]
    fn hedge_quantities(&self) -> Vec<f64> {
        self.inner.projection.hedge_quantities().to_vec()
    }
    #[getter]
    fn scalar_vega(&self) -> f64 {
        self.inner.projection.position_risk().scalar_vega()
    }
    #[getter]
    fn expected_local_parallel_vega(&self) -> f64 {
        self.inner.expected_local_parallel_vega
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner.reporting_basis.maturity_nodes().to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.reporting_basis.log_moneyness_nodes().to_vec()
    }

    /// JSON-compatible diagnostic. Sampling settings do not enter this method;
    /// no standard errors, fabricated paths, covariance or acceptance flag.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let r = &self.inner;
        d.set_item("document_kind", "vega_kt_terminal_measure_diagnostic")?;
        d.set_item("schema_version", 1)?;
        d.set_item("diagnostic_only", true)?;
        d.set_item("calculation_kind", "deterministic")?;
        d.set_item("policy_label", self.policy_label())?;
        d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
        d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
        d.set_item("affine_cash", r.payoff.cash())?;
        d.set_item("affine_slope", r.payoff.slope())?;
        d.set_item("gamma_atoms", self.gamma_atoms())?;
        d.set_item("maturity_nodes", self.maturity_nodes())?;
        d.set_item("log_moneyness_nodes", self.log_moneyness_nodes())?;
        d.set_item(
            "implied_volatilities",
            r.reporting_basis.implied_volatilities().to_vec(),
        )?;
        d.set_item("position_iv_buckets", self.raw_buckets())?;
        d.set_item("offsetting_hedge_quantities", self.hedge_quantities())?;
        d.set_item("total_position_iv_vega", self.scalar_vega())?;
        d.set_item("total_variance", r.total_variance)?;
        d.set_item(
            "expected_local_parallel_vega",
            self.expected_local_parallel_vega(),
        )?;
        d.set_item("source_log_moneyness_nodes", &r.source_log_moneyness_nodes)?;
        d.set_item(
            "terminal_measure_projection_residual",
            r.projection.position_risk().signed_residual(),
        )?;
        let stats = r.projection.position_risk().reporting_stats();
        d.set_item(
            "reporting_edge_iv_vegas",
            [
                stats.left_edge_sensitivity(),
                stats.right_edge_sensitivity(),
            ],
        )?;
        let rows = r
            .local_rows
            .iter()
            .map(|row| {
                let d = PyDict::new(py);
                d.set_item("source_maturity", row.source_maturity)?;
                d.set_item("destination_maturity", row.destination_maturity)?;
                d.set_item("source_total_variance", row.source_total_variance)?;
                d.set_item("interval_volatility", row.interval_volatility)?;
                d.set_item("expected_source_hat_vegas", &row.expected_source_hat_vegas)?;
                Ok(d)
            })
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("local_rows", rows)?;
        Ok(d)
    }
}
