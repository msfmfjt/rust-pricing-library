//! Signed Gamma atoms and exact flat-variance forward operators.
use crate::core::{FiniteF64, PositiveF64};
use crate::risk::{
    ReportingIvBasis, RiskConfigError, VegaKtGammaAtom, VegaKtGammaAtoms, VegaKtTerminalPayoff,
    VegaKtVanillaProjection, gaussian_log_hat_probabilities,
    project_vanilla_hedge_measure_to_reporting_iv,
};
use pricing_numerics::NeumaierSum;

pub(super) fn invalid(quantity: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity }
}

impl VegaKtGammaAtom {
    #[must_use]
    pub const fn strike(self) -> f64 {
        self.strike
    }
    #[must_use]
    pub const fn mass(self) -> f64 {
        self.mass
    }
}

impl VegaKtGammaAtoms {
    /// Accepts signed masses, including an empty (affine) measure. The 4096-atom
    /// budget bounds compilation and forward-operator work; no positivity fit.
    pub fn new(mut atoms: Vec<(f64, f64)>) -> Result<Self, RiskConfigError> {
        if atoms.len() > 4096 {
            return Err(invalid("Gamma measure exceeds 4096-atom budget"));
        }
        for &(strike, mass) in &atoms {
            PositiveF64::new(strike, "gamma_atom_strike")?;
            FiniteF64::new(mass, "gamma_atom_mass")?;
        }
        atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Vec::new();
        let mut i = 0;
        while i < atoms.len() {
            let strike = atoms[i].0;
            let mut sum = NeumaierSum::new();
            while i < atoms.len() && atoms[i].0 == strike {
                sum.add(atoms[i].1);
                i += 1;
            }
            let mass = FiniteF64::new(sum.total(), "gamma_atom_combined_mass")?.get();
            if mass != 0. {
                out.push(VegaKtGammaAtom { strike, mass });
            }
        }
        Ok(Self {
            atoms: out.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn atoms(&self) -> &[VegaKtGammaAtom] {
        &self.atoms
    }

    /// Backward propagation by a positive, spatially flat total variance.
    /// Zero variance is still a measure and cannot be evaluated as a density.
    pub fn gamma_density(&self, spot: f64, variance: f64) -> Result<f64, RiskConfigError> {
        PositiveF64::new(spot, "gamma_density_spot")?;
        PositiveF64::new(variance, "gamma_density_variance")?;
        let sd = variance.sqrt();
        let mut sum = NeumaierSum::new();
        for a in &self.atoms {
            let d1 = -log_ratio(a.strike, spot) / sd + 0.5 * sd;
            let kernel =
                (-0.5 * d1 * d1 - spot.ln() - sd.ln() - 0.5 * (2. * std::f64::consts::PI).ln())
                    .exp();
            sum.add(FiniteF64::new(a.mass * kernel, "propagated_gamma_atom")?.get());
        }
        Ok(FiniteF64::new(sum.total(), "propagated_gamma_density")?.get())
    }

    /// Exact expectation of local-vega deposition on SOURCE log hats for a
    /// terminal Gamma measure. source_variance is cumulative from time zero;
    /// remaining_variance runs from that source through the terminal boundary.
    /// sigma_dt = interval volatility * interval length, NOT total stddev.
    /// The conditional bridge is X_source | X_terminal=log(K/S0).
    pub fn expected_source_hat_vegas(
        &self,
        spot: f64,
        source_variance: f64,
        remaining_variance: f64,
        sigma_dt: f64,
        nodes: &[f64],
    ) -> Result<Vec<f64>, RiskConfigError> {
        PositiveF64::new(spot, "gamma_forward_spot")?;
        FiniteF64::new(source_variance, "gamma_source_variance")?;
        PositiveF64::new(remaining_variance, "gamma_remaining_variance")?;
        PositiveF64::new(sigma_dt, "gamma_sigma_dt")?;
        if source_variance < 0. {
            return Err(invalid("negative source variance"));
        }
        crate::risk::mass_lumped_hat_areas(nodes)?;
        let total =
            PositiveF64::new(source_variance + remaining_variance, "gamma_total_variance")?.get();
        let fraction = source_variance / total;
        let bridge_variance = fraction * remaining_variance;
        let sd = total.sqrt();
        let mut sums = vec![NeumaierSum::new(); nodes.len()];
        for a in &self.atoms {
            let x = log_ratio(a.strike, spot);
            let hats = gaussian_log_hat_probabilities(fraction * x, bridge_variance, nodes)?;
            let z = (x + 0.5 * total) / sd;
            // K^2 times terminal density; integrate in log space before
            // multiplying the signed mass. No node-density division at t=0.
            let k2_density =
                (a.strike.ln() - 0.5 * z * z - sd.ln() - 0.5 * (2. * std::f64::consts::PI).ln())
                    .exp();
            let scale =
                FiniteF64::new(a.mass * sigma_dt * k2_density, "gamma_forward_scale")?.get();
            for (sum, h) in sums.iter_mut().zip(hats) {
                sum.add(scale * h);
            }
        }
        sums.into_iter()
            .map(|s| Ok(FiniteF64::new(s.total(), "gamma_source_hat_vega")?.get()))
            .collect()
    }

    /// Equation (4) propagation followed by equation (9) continuum IV Vega,
    /// then reporting-log-hat projection. Exact Gaussian product integration
    /// over all positive strikes, including clamped reporting tails. Returns
    /// positive Gamma contribution; position risk subtracts this from current
    /// Gamma. Only spatially flat IV matching source cumulative variance applies.
    pub fn project_propagated(
        &self,
        basis: &ReportingIvBasis,
        maturity: f64,
        spot: f64,
        source_variance: f64,
        next_variance: f64,
    ) -> Result<Vec<f64>, RiskConfigError> {
        PositiveF64::new(source_variance, "propagation_source_variance")?;
        let i = basis
            .maturity_nodes()
            .iter()
            .position(|&t| t == maturity)
            .ok_or_else(|| invalid("propagation maturity must be a reporting node"))?;
        let width = basis.log_moneyness_nodes().len();
        let sigma = (source_variance / maturity).sqrt();
        if basis.implied_volatilities()[i * width..(i + 1) * width]
            .iter()
            .any(|&v| (v - sigma).abs() > 1e-10 * v.abs().max(sigma.abs()))
        {
            return Err(invalid(
                "propagation IV differs from flat cumulative variance",
            ));
        }
        let row = self.expected_source_hat_vegas(
            spot,
            source_variance,
            next_variance,
            sigma * maturity,
            basis.log_moneyness_nodes(),
        )?;
        let mut result = vec![0.; basis.bucket_count()];
        result[i * width..(i + 1) * width].copy_from_slice(&row);
        Ok(result)
    }

    /// Terminal hedge measure is minus payoff Gamma. Equation (9) applies
    /// continuum Vega at each atom before reporting-IV hat interpolation.
    pub fn project_terminal(
        &self,
        basis: &ReportingIvBasis,
        maturity: f64,
        spot: f64,
    ) -> Result<VegaKtVanillaProjection, RiskConfigError> {
        project_vanilla_hedge_measure_to_reporting_iv(
            basis,
            maturity,
            spot,
            &self.atoms.iter().map(|a| a.strike).collect::<Vec<_>>(),
            &self.atoms.iter().map(|a| -a.mass).collect::<Vec<_>>(),
        )
    }
}

pub(super) fn log_ratio(strike: f64, spot: f64) -> f64 {
    let rel = (strike - spot) / spot;
    if rel.abs() < 0.5 {
        rel.ln_1p()
    } else {
        strike.ln() - spot.ln()
    }
}

impl VegaKtTerminalPayoff {
    #[must_use]
    pub const fn cash(&self) -> f64 {
        self.cash
    }
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }
    #[must_use]
    pub const fn gamma(&self) -> &VegaKtGammaAtoms {
        &self.gamma
    }
    pub fn value(&self, spot: f64) -> Result<f64, RiskConfigError> {
        PositiveF64::new(spot, "terminal_payoff_spot")?;
        let mut sum = NeumaierSum::new();
        sum.add(self.cash);
        sum.add(self.slope * spot);
        for a in self.gamma.atoms() {
            sum.add(a.mass * (spot - a.strike).max(0.));
        }
        Ok(FiniteF64::new(sum.total(), "terminal_payoff_value")?.get())
    }
}
