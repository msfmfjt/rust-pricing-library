//! Hybrid path/payoff reverse followed by one particle-calibration VJP per
//! pricing mean (per scramble for RQMC). No calibration is run per pricing path.

use super::*;
use crate::mc::hull_white::HULL_WHITE_AAD_METHOD;
use crate::mc::lsv::LsvError;
use crate::models::hull_white_dividends::{
    HullWhiteDividendMarketAdjoints, HullWhiteDividendNodeAdjoints, transpose_log_curve,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HullWhiteAadRisk {
    pub price: HullWhitePrice,
    /// Order: Spot, optional BS sigma / rough sigma0, discount log-DF pillars, dividend
    /// log-DF pillars, row-major local variance, row-major forward log density,
    /// optional row-major market IV and parallel market IV.
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    /// Same order as derivatives. Only independent RQMC scrambles; conditional
    /// on the realized calibration and its fixed discrete branch decisions.
    pub standard_errors: Option<Box<[f64]>>,
    pub discount_times: Box<[f64]>,
    pub dividend_times: Box<[f64]>,
    pub target_time_nodes: Box<[f64]>,
    pub target_log_moneyness_nodes: Box<[f64]>,
    pub vega_kt_maturity_nodes: Box<[f64]>,
    pub vega_kt_log_moneyness_nodes: Box<[f64]>,
    pub vega_kt_implied_volatilities: Box<[f64]>,
    pub method: &'static str,
    is_bs: bool,
}
impl HullWhiteAadRisk {
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.derivatives[0]
    }
    #[must_use]
    pub fn vega(&self) -> Option<f64> {
        if self.is_bs {
            Some(self.derivatives[1])
        } else if !self.vega_kt_implied_volatilities.is_empty() {
            self.derivatives.last().copied()
        } else {
            None
        }
    }
    fn curve_offset(&self) -> usize {
        1 + usize::from(self.is_bs)
    }
    #[must_use]
    pub fn discount_log_df_adjoints(&self) -> &[f64] {
        let start = self.curve_offset();
        &self.derivatives[start..start + self.discount_times.len()]
    }
    #[must_use]
    pub fn dividend_log_df_adjoints(&self) -> &[f64] {
        let start = self.curve_offset() + self.discount_times.len();
        &self.derivatives[start..start + self.dividend_times.len()]
    }
    /// Signed dPrice for +1 bp in each continuously compounded zero-rate pillar.
    #[must_use]
    pub fn discount_node_dv01(&self) -> Vec<f64> {
        self.discount_log_df_adjoints()
            .iter()
            .zip(&self.discount_times)
            .map(|(bar, t)| -1e-4 * t * bar)
            .collect()
    }
    #[must_use]
    pub fn parallel_discount_dv01(&self) -> f64 {
        let mut sum = pricing_numerics::NeumaierSum::new();
        for v in self.discount_node_dv01() {
            sum.add(v);
        }
        sum.total()
    }
    #[must_use]
    pub fn local_variance_adjoints(&self) -> &[f64] {
        let start = self.curve_offset() + self.discount_times.len() + self.dividend_times.len();
        let count = self.target_time_nodes.len() * self.target_log_moneyness_nodes.len();
        &self.derivatives[start..start + count]
    }
    #[must_use]
    pub fn forward_log_density_adjoints(&self) -> &[f64] {
        let start = self.curve_offset()
            + self.discount_times.len()
            + self.dividend_times.len()
            + self.target_time_nodes.len() * self.target_log_moneyness_nodes.len();
        let count = self.target_time_nodes.len() * self.target_log_moneyness_nodes.len();
        &self.derivatives[start..start + count]
    }
    fn quote_range(&self) -> Option<std::ops::Range<usize>> {
        let n = self.vega_kt_implied_volatilities.len();
        if n == 0 {
            None
        } else {
            let end = self.derivatives.len() - 1;
            Some(end - n..end)
        }
    }
    #[must_use]
    pub fn vega_kt_raw(&self) -> Option<&[f64]> {
        self.quote_range().map(|r| &self.derivatives[r])
    }
    /// Currency per +1 absolute volatility point (0.01).
    #[must_use]
    pub fn vega_kt_market_scaled(&self) -> Option<Vec<f64>> {
        self.vega_kt_raw()
            .map(|v| v.iter().map(|v| 0.01 * v).collect())
    }
    #[must_use]
    pub fn vega_kt_standard_errors(&self) -> Option<&[f64]> {
        Some(&self.standard_errors.as_ref()?[self.quote_range()?])
    }
    /// SE of the scramble-wise sum, including cross-bucket covariance.
    #[must_use]
    pub fn parallel_vega_standard_error(&self) -> Option<f64> {
        let errors = self.standard_errors.as_ref()?;
        if self.is_bs {
            Some(errors[1])
        } else {
            self.quote_range().and_then(|_| errors.last().copied())
        }
    }
    #[must_use]
    pub fn vega_kt_method(&self) -> Option<&'static str> {
        self.quote_range()
            .map(|_| crate::market::MARKET_IV_INTERPOLATION)
    }
}

struct Context {
    dividends: Option<HullWhiteDividendPlan>,
    node_indices: Vec<usize>,
    coefficient_count: usize,
    leverage_count: usize,
    is_bs: bool,
}
impl Context {
    fn new(plan: &HullWhiteEquityPricingPlan) -> Result<Self, MonteCarloError> {
        if plan.affine_dividends.is_some() {
            return Ok(Self {
                dividends: None,
                node_indices: Vec::new(),
                coefficient_count: plan.market.discount_curve().times().len()
                    + plan.market.dividend_curve().times().len(),
                leverage_count: plan
                    .calibration
                    .as_ref()
                    .map_or(0, |c| c.surface.squared_leverage().len()),
                is_bs: plan.calibration.is_none(),
            });
        }
        let dividends = if let Some(d) = plan.path.dividends() {
            d.clone()
        } else {
            // The default path need not stop at a proportional payout. Its
            // observation map can still be differentiated on this augmented grid.
            let mut times = plan.time_nodes().to_vec();
            if let Some(d) = plan.market.discrete_dividends() {
                times.extend(
                    d.events()
                        .iter()
                        .filter(|e| e.ex_time() <= *plan.time_nodes().last().unwrap())
                        .map(|e| e.ex_time()),
                );
            }
            times.sort_by(f64::total_cmp);
            times.dedup();
            HullWhiteDividendPlan::new(plan.path.rates(), &plan.market, &times)?
        };
        let node_indices = plan
            .time_nodes()
            .iter()
            .map(|t| {
                dividends
                    .nodes()
                    .binary_search_by(|n| n.time().total_cmp(t))
                    .expect("included pricing node")
            })
            .collect();
        let coefficient_count = dividends
            .zero_adjoints()
            .iter()
            .map(|n| 2 + n.bond_amounts.len())
            .sum();
        Ok(Self {
            dividends: Some(dividends),
            node_indices,
            coefficient_count,
            leverage_count: plan
                .calibration
                .as_ref()
                .map_or(0, |c| c.surface.squared_leverage().len()),
            is_bs: plan.calibration.is_none(),
        })
    }
    fn width(&self) -> usize {
        3 + self.coefficient_count + self.leverage_count
    }

    fn reverse_mean(
        &self,
        plan: &HullWhiteEquityPricingPlan,
        mean: &[f64],
    ) -> Result<Vec<f64>, MonteCarloError> {
        let mut coefficients = self
            .dividends
            .as_ref()
            .map_or_else(Vec::new, HullWhiteDividendPlan::zero_adjoints);
        let mut offset = 3;
        for n in &mut coefficients {
            n.scale = mean[offset];
            n.deterministic_reserve = mean[offset + 1];
            offset += 2;
            let count = n.bond_amounts.len();
            n.bond_amounts
                .copy_from_slice(&mean[offset..offset + count]);
            offset += count;
        }
        // Affine observations accumulate curve pillars directly; legacy escrow
        // observations accumulate deterministic reserve/scale coefficients.
        let offset = 3 + self.coefficient_count;
        let mut spot = mean[1];
        let mut target = Vec::new();
        let mut quotes = Vec::new();
        if let Some(c) = &plan.calibration {
            let adj = c.reverse_leverage(&mean[offset..])?;
            spot += adj.initial_spot;
            if let Some(nodes) = &adj.dividends {
                add_nodes(&mut coefficients, nodes);
            }
            if let Some(source) = &plan.market_iv_target {
                quotes = source.reverse_market_iv(&adj.local_variance, &adj.forward_log_density)?;
                let mut sum = pricing_numerics::NeumaierSum::new();
                for &q in &quotes {
                    sum.add(q);
                }
                quotes.push(sum.total());
            }
            target.extend(adj.local_variance);
            target.extend(adj.forward_log_density);
        }
        let mut market = if let Some(dividends) = &self.dividends {
            dividends.reverse_market(&coefficients)?
        } else {
            let split = 3 + plan.market.discount_curve().times().len();
            HullWhiteDividendMarketAdjoints {
                spot: 0.0,
                discount_log_df: mean[3..split].to_vec(),
                dividend_log_df: mean[split..offset].to_vec(),
            }
        };
        // The exact relative HW discount/bond is curve-independent. Re-fitting
        // to a changed initial curve differentiates its P0(payment) prefactor.
        transpose_log_curve(
            plan.market.discount_curve(),
            plan.payment_time,
            mean[0],
            &mut market.discount_log_df,
        )?;
        let mut out = vec![spot + market.spot];
        if self.is_bs {
            out.push(mean[2]);
        }
        out.extend(market.discount_log_df);
        out.extend(market.dividend_log_df);
        out.extend(target);
        out.extend(quotes);
        if out.iter().any(|v| !v.is_finite()) {
            return Err(HullWhiteError::InvalidInput {
                field: "hybrid_risk_estimator",
                index: 0,
            }
            .into());
        }
        Ok(out)
    }
}
fn add_nodes(
    target: &mut [HullWhiteDividendNodeAdjoints],
    source: &[HullWhiteDividendNodeAdjoints],
) {
    assert_eq!(target.len(), source.len());
    for (a, b) in target.iter_mut().zip(source) {
        a.scale += b.scale;
        a.deterministic_reserve += b.deterministic_reserve;
        assert_eq!(a.bond_amounts.len(), b.bond_amounts.len());
        for (a, b) in a.bond_amounts.iter_mut().zip(&b.bond_amounts) {
            *a += b;
        }
    }
}

impl HullWhiteEquityPricingPlan {
    /// First-order reverse at fixed HW/Bergomi parameters, correlations, payout
    /// amounts, grid axes and paired target coordinates. LSV includes the finite
    /// particle recalibration and requires retain_reverse_trace=true at compile.
    pub fn evaluate_aad(&self) -> Result<HullWhiteAadRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "HW discontinuous payoff requires explicit smoothing",
            });
        }
        if self
            .calibration
            .as_ref()
            .is_some_and(|c| !c.retains_reverse_trace())
        {
            return Err(LsvError::ReverseTraceNotRetained.into());
        }
        let context = Context::new(self)?;
        let executor = DeterministicExecutor::new(self.policy)?;
        let width = context.width();
        let (price_stats, units, paths, derivatives, standard_errors) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                    let z = self.path.pseudo_shocks(
                        config.master_seed(),
                        p,
                        RandomDomain::Valuation,
                    )?;
                    self.sample_aad(
                        &context,
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                let mean = stats
                    .iter()
                    .map(|s| s.sum().total() / count as f64)
                    .collect::<Vec<_>>();
                (
                    stats[0],
                    count,
                    config.evaluated_paths(),
                    context.reverse_mean(self, &mean)?,
                    None,
                )
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let dimension =
                    (self.path.random_factor_count() * (self.time_nodes().len() - 1)) as u32;
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut prices = Vec::new();
                let mut risks = Vec::new();
                for scramble in 0..config.scramble_count().get() {
                    let stats =
                        executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc.uniform(scramble, p, d).map_err(|_| {
                                        HullWhiteError::InvalidInput {
                                            field: "rqmc_uniform",
                                            index: d as usize,
                                        }
                                    })?;
                                    inverse_standard_normal(u).map_err(|_| {
                                        HullWhiteError::InvalidInput {
                                            field: "rqmc_normal",
                                            index: d as usize,
                                        }
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            self.sample_aad(
                                &context,
                                z,
                                bridge.as_ref(),
                                config.variance_reduction().antithetic(),
                                out,
                            )
                        })?;
                    let mean = stats
                        .iter()
                        .map(|s| s.sum().total() / count as f64)
                        .collect::<Vec<_>>();
                    prices.push(mean[0]);
                    risks.push(context.reverse_mean(self, &mean)?);
                }
                let scrambles = u64::from(config.scramble_count().get());
                let mut derivatives = Vec::new();
                let mut errors = Vec::new();
                for j in 0..risks[0].len() {
                    let values = risks.iter().map(|v| v[j]).collect::<Vec<_>>();
                    let stats = DeterministicStatistics::from_ordered_values_two_pass(&values);
                    derivatives.push(stats.sum().total() / scrambles as f64);
                    errors.push(standard_error(stats, scrambles)?);
                }
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                    derivatives,
                    Some(errors.into_boxed_slice()),
                )
            }
        };
        let mut labels = vec!["spot".to_owned()];
        if context.is_bs {
            labels.push(
                if self.path.is_direct_rough() {
                    "initial_volatility"
                } else {
                    "bs_volatility"
                }
                .to_owned(),
            );
        }
        for (name, count) in [
            (
                "discount_log_df",
                self.market.discount_curve().times().len(),
            ),
            (
                "dividend_log_df",
                self.market.dividend_curve().times().len(),
            ),
            ("local_variance", context.leverage_count),
            ("forward_log_density", context.leverage_count),
        ] {
            labels.extend((0..count).map(|i| format!("{name}[{i}]")));
        }
        let market_iv = self
            .market_iv_target
            .as_ref()
            .and_then(HullWhiteLsvTarget::market_iv_surface);
        if let Some(surface) = market_iv {
            labels.extend(
                (0..surface.implied_volatilities().len()).map(|i| format!("market_iv[{i}]")),
            );
            labels.push("parallel_market_iv".to_owned());
        }
        let price = HullWhitePrice {
            value: price_stats.sum().total() / units as f64,
            standard_error: standard_error(price_stats, units)?,
            independent_sampling_units: units,
            evaluated_paths: paths,
            plan_fingerprint: self.fingerprint,
            scheme: self.path.scheme(),
            calibration_method: self.path.calibration_method(),
            calibration_seed: self.calibration_seed,
            cash_dividend_model: self.cash_dividend_model(),
        };
        Ok(HullWhiteAadRisk {
            price,
            parameter_labels: labels.into(),
            derivatives: derivatives.into(),
            standard_errors,
            discount_times: self.market.discount_curve().times().into(),
            dividend_times: self.market.dividend_curve().times().into(),
            target_time_nodes: self
                .calibration
                .as_ref()
                .map_or_else(Vec::new, |c| c.surface.times().to_vec())
                .into(),
            target_log_moneyness_nodes: self
                .calibration
                .as_ref()
                .map_or_else(Vec::new, |c| c.surface.log_nodes().to_vec())
                .into(),
            method: HULL_WHITE_AAD_METHOD,
            is_bs: context.is_bs,
            vega_kt_maturity_nodes: market_iv.map_or(&[][..], |s| s.maturity_nodes()).into(),
            vega_kt_log_moneyness_nodes: market_iv
                .map_or(&[][..], |s| s.log_moneyness_nodes())
                .into(),
            vega_kt_implied_volatilities: market_iv
                .map_or(&[][..], |s| s.implied_volatilities())
                .into(),
        })
    }

    fn sample_aad(
        &self,
        context: &Context,
        z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let steps = self.time_nodes().len() - 1;
        let z = if let Some(bridge) = bridge {
            let mut transformed = Vec::with_capacity(z.len());
            for block in z.chunks_exact(steps) {
                transformed.extend(
                    bridge
                        .apply_one_factor(block)
                        .map_err(|e| MonteCarloError::LocalVol(e.into()))?,
                );
            }
            transformed
        } else {
            z
        };
        out.fill(0.0);
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
            let path = self.path.record_path(self.spot, &shocks)?;
            let states = path.states();
            let last = states[steps];
            let time = self.time_nodes()[steps];
            let relative_discount = self
                .path
                .rates()
                .relative_discount(time, last.integrated_rate_factor)?
                * self
                    .path
                    .rates()
                    .relative_bond(time, self.payment_time, last.rate_factor)?;
            if let Some(dividends) = &self.affine_dividends {
                let observations = dividends.record(states)?;
                let (payoff, seeds) = self
                    .base
                    .hybrid_spot_payoff_adjoints(self.time_nodes(), &observations.spots)?;
                let adjoints =
                    dividends.reverse(&observations, states, &seeds, relative_discount)?;
                let reverse = path.reverse(&adjoints.equity)?;
                out[0] += payoff * relative_discount;
                out[1] += reverse.initial_spot;
                out[2] += reverse.bs_volatility.unwrap_or(0.0);
                for (target, source) in out[3..].iter_mut().zip(
                    adjoints
                        .discount_log_df
                        .iter()
                        .chain(&adjoints.dividend_log_df),
                ) {
                    *target += source;
                }
                let offset = 3 + context.coefficient_count;
                for (target, source) in out[offset..].iter_mut().zip(reverse.squared_leverage) {
                    *target += source;
                }
                continue;
            }
            let dividends = context.dividends.as_ref().expect("legacy dividend context");
            let spots = if self.path.dividends().is_some() {
                states
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        dividends.nodes()[context.node_indices[i]]
                            .spots(s.normalized_equity, s.rate_factor)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                self.base.hybrid_proportional_spots(
                    self.time_nodes(),
                    &states
                        .iter()
                        .map(|s| s.normalized_equity)
                        .collect::<Vec<_>>(),
                )?
            };
            let (payoff, payoff_seeds) = self
                .base
                .hybrid_spot_payoff_adjoints(self.time_nodes(), &spots)?;
            out[0] += payoff * relative_discount;
            let mut coefficients = dividends.zero_adjoints();
            let mut state_seeds = Vec::with_capacity(states.len());
            for (i, (s, &(post, pre))) in states.iter().zip(&payoff_seeds).enumerate() {
                let node = context.node_indices[i];
                state_seeds.push(dividends.nodes()[node].reverse_spots(
                    s.normalized_equity,
                    s.rate_factor,
                    post * relative_discount,
                    pre * relative_discount,
                    &mut coefficients[node],
                )?);
            }
            let reverse = path.reverse(&state_seeds)?;
            out[1] += reverse.initial_spot;
            out[2] += reverse.bs_volatility.unwrap_or(0.0);
            if let Some(nodes) = &reverse.dividends {
                add_nodes(&mut coefficients, nodes);
            }
            let mut offset = 3;
            for n in coefficients {
                out[offset] += n.scale;
                out[offset + 1] += n.deterministic_reserve;
                offset += 2;
                for v in n.bond_amounts {
                    out[offset] += v;
                    offset += 1;
                }
            }
            for (target, source) in out[offset..].iter_mut().zip(reverse.squared_leverage) {
                *target += source;
            }
        }
        if antithetic {
            for v in out {
                *v *= 0.5;
            }
        }
        Ok(())
    }
}
fn standard_error(stats: DeterministicStatistics, count: u64) -> Result<f64, MonteCarloError> {
    let variance = stats
        .moments()
        .sample_variance()
        .ok_or(MonteCarloError::InsufficientSamplingUnits { count })?;
    let error = (variance / count as f64).sqrt();
    if !error.is_finite() {
        return Err(HullWhiteError::InvalidInput {
            field: "risk_standard_error",
            index: 0,
        }
        .into());
    }
    Ok(error)
}
