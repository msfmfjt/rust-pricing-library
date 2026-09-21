//! One shared HW rate factor, paired marginal calibration and affine payouts.
use super::*;
use crate::core::DayCountConvention;
use crate::market::{DiscountCurve, LocalVarianceGrid};
use crate::mc::LocalVolTimeGrid;
use crate::mc::hull_white::{
    CalibratedHullWhiteLsv, HullWhiteEquityPlan, HullWhiteLsvTarget, HybridEquityVolatility,
    HybridState, HybridVolatilityFactor, calibrate_hybrid_lsv_with_dividends,
};
use crate::models::hull_white_dividends::{HullWhiteDividendNodeAdjoints, HullWhiteDividendPlan};
use crate::models::{HullWhite1Factor, HybridCorrelation};
use crate::multi_asset::{MultiAssetError as E, MultiAssetProduct};
use crate::product::{GraphLimitPolicy, SourceGraph};

mod drivers;
mod risk;
pub(super) use drivers::compile_drivers;

/// Rate correlations follow the Brownian order: all prices, then each asset's
/// volatility factors. Full matrices append the common rate Brownian last.
/// Paired targets use the continuous normalized equity coordinate, per asset.
#[derive(Clone, Debug)]
pub struct MultiAssetHullWhiteConfig {
    pub rate_model: HullWhite1Factor,
    pub rate_correlations: Vec<f64>,
    pub lsv_targets: Vec<Option<HullWhiteLsvTarget>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetHullWhiteLsvRisk {
    pub forward_log_density_adjoints: Vec<f64>,
    pub density_standard_errors: Option<Vec<f64>>,
    pub vega_kt_raw: Option<Vec<f64>>,
    pub vega_kt_market_scaled: Option<Vec<f64>>,
    pub vega_kt_standard_errors: Option<Vec<f64>>,
    pub vega_kt_maturity_nodes: Vec<f64>,
    pub vega_kt_log_moneyness_nodes: Vec<f64>,
    pub parallel_vega: Option<f64>,
    pub parallel_vega_standard_error: Option<f64>,
    pub method: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetHullWhiteCurveRisk {
    pub discount_time_nodes: Vec<f64>,
    pub discount_log_df_adjoints: Vec<Estimate>,
    pub discount_node_dv01: Vec<Estimate>,
    pub dividend_time_nodes: Vec<Vec<f64>>,
    pub dividend_log_df_adjoints: Vec<Vec<Estimate>>,
}

#[derive(Clone, Debug)]
pub(super) struct HwContext {
    pub config: MultiAssetHullWhiteConfig,
    pub cashflows: Vec<Cashflow>,
}
#[derive(Clone, Debug)]
pub(super) struct Cashflow {
    pub payoff: CompiledPayoff,
    pub observation_node: usize,
    pub payment_time: f64,
    pub discount: f64,
    pub rate_loading: f64,
    pub log_discount_constant: f64,
}

impl HwContext {
    pub fn new(
        config: MultiAssetHullWhiteConfig,
        product: &MultiAssetProduct,
        valuation: Date,
        times: &[f64],
        market: &EquityForward,
    ) -> Result<Self, E> {
        let fraction = |d| DayCountConvention::Act365F.year_fraction(valuation, d);
        let mut cashflows = Vec::new();
        for (&output, &payment) in product.graph.outputs().iter().zip(&product.payment_dates) {
            let payoff = SourceGraph::new(product.graph.nodes().to_vec(), vec![output])
                .compile(GraphLimitPolicy::default())?;
            let time = payoff
                .terminal_observations()
                .iter()
                .map(|(_, d)| fraction(*d))
                .fold(0.0, f64::max);
            let observation_node = times
                .binary_search_by(|t| t.total_cmp(&time))
                .map_err(|_| E::Invalid("missing HW cashflow observation"))?;
            let payment_time = fraction(payment);
            let transition = config
                .rate_model
                .transition(
                    time,
                    payment_time,
                    0.0,
                    HybridCorrelation::new(0.0, 0.0, 0.0).map_err(E::numerical)?,
                )
                .map_err(E::numerical)?;
            cashflows.push(Cashflow {
                payoff,
                observation_node,
                payment_time,
                discount: market.discount_curve().evaluate(payment_time)?.discount,
                rate_loading: transition.integral_loading,
                log_discount_constant: -0.5
                    * config
                        .rate_model
                        .integrated_variance(payment_time)
                        .map_err(E::numerical)?
                    + 0.5 * transition.covariance[3][3],
            });
        }
        Ok(Self { config, cashflows })
    }
    pub fn discount(&self, cashflow: &Cashflow, states: &[HybridState]) -> Result<f64, E> {
        let state = &states[cashflow.observation_node];
        let discount = cashflow.discount
            * (cashflow.log_discount_constant
                - state.integrated_rate_factor
                - cashflow.rate_loading * state.rate_factor)
                .exp();
        if !discount.is_finite() || discount <= 0.0 {
            return Err(E::Invalid("nonpositive or nonfinite HW cashflow discount"));
        }
        Ok(discount)
    }
}

#[derive(Clone, Debug)]
pub(super) struct HwAsset {
    pub process: HullWhiteEquityPlan,
    pub dividends: HullWhiteDividendPlan,
    pub calibration: Option<CalibratedHullWhiteLsv>,
    pub target: Option<HullWhiteLsvTarget>,
    pub config: Option<MultiAssetBergomiLsvConfig>,
    pub driver_offset: usize,
    parameter_offsets: Vec<usize>,
}

pub(super) struct HwAssetPath {
    pub states: Vec<HybridState>,
    pub equity_increments: Vec<f64>,
}

impl HwAsset {
    pub fn volatility_parameter_count(&self) -> usize {
        self.calibration
            .as_ref()
            .map_or(1, |c| c.surface.squared_leverage().len())
    }
    pub fn parameter_count(&self) -> usize {
        *self.parameter_offsets.last().unwrap()
    }
    pub fn add_node_parameters(
        &self,
        row: usize,
        node: &HullWhiteDividendNodeAdjoints,
        out: &mut [f64],
    ) {
        let offset = self.parameter_offsets[row];
        out[offset] += node.scale;
        out[offset + 1] += node.deterministic_reserve;
        for (target, value) in out[offset + 2..].iter_mut().zip(&node.bond_amounts) {
            *target += value;
        }
    }
    pub fn add_dividend_parameters(
        &self,
        nodes: &[HullWhiteDividendNodeAdjoints],
        out: &mut [f64],
    ) {
        let mut offset = self.volatility_parameter_count() + 1;
        for n in nodes {
            out[offset] += n.scale;
            out[offset + 1] += n.deterministic_reserve;
            offset += 2;
            for &v in &n.bond_amounts {
                out[offset] += v;
                offset += 1;
            }
        }
    }
    pub fn market_reverse(
        &self,
        parameters: &[f64],
    ) -> Result<crate::models::hull_white_dividends::HullWhiteDividendMarketAdjoints, E> {
        let count = self.volatility_parameter_count();
        let mut coefficients = self.dividends.zero_adjoints();
        let mut offset = count + 1;
        for n in &mut coefficients {
            n.scale = parameters[offset];
            n.deterministic_reserve = parameters[offset + 1];
            offset += 2;
            let len = n.bond_amounts.len();
            n.bond_amounts
                .copy_from_slice(&parameters[offset..offset + len]);
            offset += len;
        }
        let mut spot = parameters[count];
        if let Some(c) = &self.calibration
            && parameters[..count].iter().any(|v| *v != 0.0)
        {
            let adj = c
                .reverse_leverage(&parameters[..count])
                .map_err(E::numerical)?;
            spot += adj.initial_spot;
            if let Some(nodes) = adj.dividends {
                for (a, b) in coefficients.iter_mut().zip(nodes) {
                    a.scale += b.scale;
                    a.deterministic_reserve += b.deterministic_reserve;
                    for (a, b) in a.bond_amounts.iter_mut().zip(b.bond_amounts) {
                        *a += b;
                    }
                }
            }
        }
        let mut result = self
            .dividends
            .reverse_market(&coefficients)
            .map_err(E::numerical)?;
        result.spot += spot;
        Ok(result)
    }
    #[allow(clippy::type_complexity)]
    pub fn observations(
        &self,
        states: &[HybridState],
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), E> {
        let mut post = Vec::with_capacity(states.len());
        let mut pre = Vec::with_capacity(states.len());
        let mut delta = Vec::with_capacity(states.len());
        let mut pre_delta = Vec::with_capacity(states.len());
        for (s, n) in states.iter().zip(self.dividends.nodes()) {
            let (s1, s0) = n
                .spots(s.normalized_equity, s.rate_factor)
                .map_err(E::numerical)?;
            post.push(s1);
            pre.push(s0.unwrap_or(s1));
            delta.push(n.scale() * s.normalized_equity / self.dividends.risky_spot());
            pre_delta.push(n.pre_scale() * s.normalized_equity / self.dividends.risky_spot());
        }
        Ok((post, pre, delta, pre_delta))
    }
    pub fn observation_reverse(
        &self,
        states: &[HybridState],
        seeds: &[(f64, f64)],
        parameters: &mut [f64],
    ) -> Result<(Vec<f64>, Vec<f64>), E> {
        let mut equity = Vec::with_capacity(states.len());
        let mut rates = Vec::with_capacity(states.len());
        for (row, ((s, n), &(post, pre))) in states
            .iter()
            .zip(self.dividends.nodes())
            .zip(seeds)
            .enumerate()
        {
            let mut node = n.zero_adjoints();
            equity.push(
                n.reverse_spots(s.normalized_equity, s.rate_factor, post, pre, &mut node)
                    .map_err(E::numerical)?,
            );
            rates.push(
                (post + pre * n.pre_scale() / n.scale()) * n.reserve_rate_derivative(s.rate_factor),
            );
            self.add_node_parameters(row, &node, parameters);
        }
        Ok((equity, rates))
    }
    pub fn compile(
        market: &EquityForward,
        model: &ModelSpec,
        grid: &LocalVolTimeGrid,
        config: Option<MultiAssetBergomiLsvConfig>,
        hw: &MultiAssetHullWhiteConfig,
        asset: usize,
        driver_offset: usize,
    ) -> Result<Self, E> {
        let rates = &hw.rate_model;
        let dividends =
            HullWhiteDividendPlan::new(rates, market, grid.nodes()).map_err(E::numerical)?;
        let rho_s = hw.rate_correlations[asset];
        let target = &hw.lsv_targets[asset];
        let mut risk_target = target.clone();
        let (volatility, calibration, correlation) = if let Some(config) = &config {
            let target = target.as_ref().ok_or(E::Invalid(
                "every HW LSV asset requires a paired variance/density target",
            ))?;
            let ModelSpec::LocalVolatility(model) = model else {
                return Err(E::Invalid("HW LSV requires a LocalVolatility target model"));
            };
            if model.local_variance_grid() != target.grid() {
                return Err(E::Invalid("HW paired target and model grid must match"));
            }
            let (factor, particles, rho_v) = match config {
                MultiAssetBergomiLsvConfig::OneFactor(c) => (
                    HybridVolatilityFactor::Bergomi(c.factor),
                    &c.particles,
                    c.factor.correlation(),
                ),
                MultiAssetBergomiLsvConfig::TwoFactor(c) => (
                    HybridVolatilityFactor::BergomiTwoFactor {
                        factor: c.factor,
                        second_vol_rate_correlation: hw.rate_correlations[driver_offset + 1],
                    },
                    &c.particles,
                    c.factor.spot_correlations()[0],
                ),
                MultiAssetBergomiLsvConfig::Rough(c) => (
                    HybridVolatilityFactor::Rough(c.factor),
                    &c.particles,
                    c.factor.correlation(),
                ),
            };
            let corr = HybridCorrelation::new(rho_v, rho_s, hw.rate_correlations[driver_offset])
                .map_err(E::numerical)?;
            let refined = refine_target(target, grid.nodes())?;
            if target.market_iv_surface().is_some() {
                risk_target = Some(refined.clone());
            }
            let mut calibration = calibrate_hybrid_lsv_with_dividends(
                &refined,
                factor,
                rates,
                corr,
                market.spot().get(),
                particles,
                Some(&dividends),
            )
            .map_err(E::numerical)?;
            // Keep the public multi-asset diagnostic in unit-forward coordinates.
            // The path/calibration kernel now stores F in initial-Spot units.
            for row in &mut calibration.diagnostics {
                row.mean_discounted_normalized_equity /= market.spot().get();
            }
            let leverage = calibration.surface.clone();
            let volatility = match factor {
                HybridVolatilityFactor::Bergomi(factor) => {
                    HybridEquityVolatility::BergomiLsv { factor, leverage }
                }
                HybridVolatilityFactor::BergomiTwoFactor {
                    factor,
                    second_vol_rate_correlation,
                } => HybridEquityVolatility::Bergomi2FactorLsv {
                    factor,
                    second_vol_rate_correlation,
                    leverage,
                },
                HybridVolatilityFactor::Rough(factor) => {
                    HybridEquityVolatility::RoughBergomiLsv { factor, leverage }
                }
            };
            (volatility, Some(calibration), corr)
        } else {
            if target.is_some() {
                return Err(E::Invalid("HW target requires an LSV configuration"));
            }
            let ModelSpec::BlackScholes(model) = model else {
                return Err(E::Invalid(
                    "LocalVolatility under HW requires an LSV configuration and paired target; use zero vol-of-vol for the local-vol limit",
                ));
            };
            (
                HybridEquityVolatility::BlackScholes(model.volatility().get()),
                None,
                HybridCorrelation::new(0.0, rho_s, 0.0).map_err(E::numerical)?,
            )
        };
        let mut parameter_offsets = vec![
            calibration
                .as_ref()
                .map_or(1, |c| c.surface.squared_leverage().len())
                + 1,
        ];
        for node in dividends.nodes() {
            parameter_offsets.push(parameter_offsets.last().unwrap() + node.coefficient_count());
        }
        Ok(Self {
            parameter_offsets,
            process: HullWhiteEquityPlan::new(rates.clone(), volatility, correlation, grid)
                .and_then(|p| p.with_dividends(dividends.clone()))
                .map_err(E::numerical)?,
            dividends,
            calibration,
            target: risk_target,
            config,
            driver_offset,
        })
    }
    pub fn factor_count(&self) -> usize {
        self.config
            .as_ref()
            .map_or(0, MultiAssetBergomiLsvConfig::factor_count)
    }
    pub fn target_reverse(&self, seeds: &[f64]) -> Result<Vec<f64>, E> {
        let c = self.calibration.as_ref().expect("HW LSV");
        let target = self.target.as_ref().expect("HW target");
        let adj = c
            .reverse_leverage(&seeds[..self.volatility_parameter_count()])
            .map_err(E::numerical)?;
        let g = target.grid();
        let mut variance = vec![0.0; g.values().len()];
        let mut density = vec![0.0; variance.len()];
        for (r, &t) in c.surface.times().iter().enumerate() {
            for (j, &x) in g.log_moneyness_nodes().iter().enumerate() {
                let k = r * g.log_moneyness_nodes().len() + j;
                let w = g.interpolate(t, x)?;
                w.transpose_accumulate(
                    adj.local_variance[k],
                    &mut variance,
                    g.log_moneyness_nodes().len(),
                );
                w.transpose_accumulate(
                    adj.forward_log_density[k],
                    &mut density,
                    g.log_moneyness_nodes().len(),
                );
            }
        }
        let quotes = target
            .market_iv_surface()
            .map(|_| target.reverse_market_iv(&variance, &density))
            .transpose()
            .map_err(E::numerical)?;
        variance.extend(density);
        if let Some(quotes) = quotes {
            let parallel = quotes.iter().sum();
            variance.extend(quotes);
            variance.push(parallel);
        }
        Ok(variance)
    }
    pub fn reverse(
        &self,
        path: &HwAssetPath,
        state_seeds: &[f64],
    ) -> Result<crate::mc::hull_white::HullWhitePathAdjoints, E> {
        crate::engine::processes::hull_white::reverse::reverse_hybrid_states(
            &self.process,
            self.dividends.initial_spot(),
            &path.states,
            &path.equity_increments,
            state_seeds,
        )
        .map_err(E::numerical)
    }
}

impl HwContext {
    pub fn fingerprint(&self, h: &mut blake3::Hasher, assets: &[Asset]) {
        use super::compile::floats;
        h.update(b"multi-asset-bergomi-hw-escrowed-v2");
        floats(h, &[self.config.rate_model.mean_reversion()]);
        floats(h, self.config.rate_model.volatility_times());
        floats(h, self.config.rate_model.volatilities());
        floats(h, &self.config.rate_correlations);
        for asset in assets {
            let a = asset.hw.as_ref().expect("HW asset");
            h.update(&a.process.parameter_fingerprint_bytes());
            h.update(&a.dividends.fingerprint_bytes());
            if let Some(c) = &a.config {
                let p = match c {
                    MultiAssetBergomiLsvConfig::OneFactor(c) => &c.particles,
                    MultiAssetBergomiLsvConfig::TwoFactor(c) => &c.particles,
                    MultiAssetBergomiLsvConfig::Rough(c) => &c.particles,
                };
                floats(h, &[p.log_bandwidth(), p.minimum_effective_samples()]);
                h.update(&(p.particle_count() as u64).to_be_bytes());
                h.update(&p.seed().to_be_bytes());
                h.update(&[u8::from(p.retain_reverse_trace())]);
                floats(
                    h,
                    a.calibration
                        .as_ref()
                        .expect("HW LSV")
                        .surface
                        .squared_leverage(),
                );
            }
            if let Some(t) = &a.target {
                floats(h, t.log_densities());
                if let Some(iv) = t.market_iv_surface() {
                    h.update(b"paired-market-iv-source");
                    floats(h, iv.maturity_nodes());
                    floats(h, iv.log_moneyness_nodes());
                    floats(h, iv.implied_volatilities());
                }
            }
        }
    }
}

impl MultiAssetPricingPlan {
    pub fn hull_white_model(&self) -> Option<&HullWhite1Factor> {
        self.hull_white.as_ref().map(|h| &h.config.rate_model)
    }
    pub fn hull_white_lsv_calibrations(&self) -> Vec<Option<&CalibratedHullWhiteLsv>> {
        self.assets
            .iter()
            .map(|a| a.hw.as_ref().and_then(|h| h.calibration.as_ref()))
            .collect()
    }
    pub(super) fn hw_paths(&self, shocks: &[Vec<f64>]) -> Result<Vec<super::path::AssetPath>, E> {
        let drivers = self.lsv_drivers.as_ref().expect("HW drivers");
        let rate_index = self.assets.len() + drivers.asset_indices.len();
        self.assets
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let h = a.hw.as_ref().expect("HW asset");
                let near_index = drivers
                    .rough_asset_indices
                    .iter()
                    .position(|a| *a == i)
                    .map(|j| rate_index + 2 + j);
                let n = self.times.len() - 1;
                let innovations: Vec<_> = (0..n)
                    .map(|s| {
                        [
                            shocks[i][s],
                            if h.factor_count() > 0 {
                                shocks[h.driver_offset][s]
                            } else {
                                0.0
                            },
                            shocks[rate_index][s],
                            shocks[rate_index + 1][s],
                            if let Some(near) = near_index {
                                shocks[near][s]
                            } else if h.factor_count() == 2 {
                                shocks[h.driver_offset + 1][s]
                            } else {
                                0.0
                            },
                        ]
                    })
                    .collect();
                let states = h
                    .process
                    .evolve_with_innovations(a.forward.spot().get(), &innovations)
                    .map_err(E::numerical)?;
                let (spots, pre_spots, spot_derivatives, pre_spot_derivatives) =
                    h.observations(&states)?;
                Ok(super::path::AssetPath {
                    spots,
                    pre_spots,
                    spot_derivatives,
                    pre_spot_derivatives,
                    bs_vega: Vec::new(),
                    local: None,
                    lsv: None,
                    local_correlation: None,
                    hw: Some(HwAssetPath {
                        states,
                        equity_increments: shocks[i].clone(),
                    }),
                })
            })
            .collect()
    }
    pub(super) fn payoff_value(&self, paths: &[super::path::AssetPath]) -> Result<f64, E> {
        if let Some(hw) = &self.hull_white {
            let states = &paths[0].hw.as_ref().expect("HW path").states;
            let mut value = 0.0;
            for c in &hw.cashflows {
                let cash = c.payoff.evaluate_with_pre_dividend_spots(
                    |u, d| self.observe(paths, u, d, false, None),
                    |u, d| self.observe(paths, u, d, true, None),
                )?[0];
                value += cash * hw.discount(c, states)?;
            }
            Ok(value)
        } else {
            Ok(self.payoff.evaluate_with_pre_dividend_spots(
                |u, d| self.observe(paths, u, d, false, None),
                |u, d| self.observe(paths, u, d, true, None),
            )?[0])
        }
    }
    pub(super) fn hw_payoff_adjoints(
        &self,
        paths: &[super::path::AssetPath],
        bump: Option<(usize, f64)>,
    ) -> Result<crate::product::PayoffEvaluation, E> {
        let hw = self.hull_white.as_ref().expect("HW");
        let states = &paths[0].hw.as_ref().expect("HW path").states;
        let mut value = 0.0;
        let mut terminal = Vec::new();
        let mut pre = Vec::new();
        for c in &hw.cashflows {
            let p = c.payoff.evaluate_single_with_observation_adjoints(
                |u, d| self.observe(paths, u, d, false, bump),
                |u, d| self.observe(paths, u, d, true, bump),
            )?;
            let discount = hw.discount(c, states)?;
            value += discount * p.value;
            terminal.extend(
                p.terminal_adjoints
                    .iter()
                    .map(|v| crate::product::TerminalAdjoint {
                        value: discount * v.value,
                        ..*v
                    }),
            );
            pre.extend(p.pre_dividend_adjoints.iter().map(|v| {
                crate::product::PreDividendAdjoint {
                    value: discount * v.value,
                    ..*v
                }
            }));
        }
        Ok(crate::product::PayoffEvaluation {
            value,
            terminal_adjoints: terminal.into(),
            pre_dividend_adjoints: pre.into(),
        })
    }
}

/// Regenerate both paired inputs from retained IV quotes on the common grid.
/// Explicit paired samples must already contain every time node: interpolating
/// a time-zero Dirac density would change the calibration identity.
pub(super) fn refine_target(
    target: &HullWhiteLsvTarget,
    times: &[f64],
) -> Result<HullWhiteLsvTarget, E> {
    let g = target.grid();
    if times[0] != 0.0 || times[times.len() - 1] > g.time_nodes()[g.time_nodes().len() - 1] {
        return Err(E::Invalid("HW target must cover the pricing grid"));
    }
    if let Some(surface) = target.market_iv_surface() {
        return HullWhiteLsvTarget::from_market_iv(
            surface.clone(),
            times.to_vec(),
            g.log_moneyness_nodes().to_vec(),
            g.floor(),
            g.cap(),
        )
        .map_err(E::numerical);
    }
    let nx = g.log_moneyness_nodes().len();
    let mut variance = Vec::new();
    let mut density = Vec::new();
    for &t in times {
        let row = g.time_nodes().binary_search_by(|v| v.total_cmp(&t)).map_err(|_| E::Invalid("explicit HW paired target must contain every common time node; retain market-IV quotes to regenerate at new times"))?;
        variance.extend_from_slice(&g.values()[row * nx..(row + 1) * nx]);
        density.extend_from_slice(&target.log_densities()[row * nx..(row + 1) * nx]);
    }
    let grid = LocalVarianceGrid::new(
        times.to_vec(),
        g.log_moneyness_nodes().to_vec(),
        variance,
        g.floor(),
        g.cap(),
    )?;
    HullWhiteLsvTarget::new(grid, density).map_err(E::numerical)
}
