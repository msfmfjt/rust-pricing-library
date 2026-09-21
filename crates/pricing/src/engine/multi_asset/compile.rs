use super::*;
use crate::core::DayCountConvention;
use crate::market::{DiscountCurve, EquityMarket};
use crate::mc::LocalVolTimeGrid;
use crate::multi_asset::{MultiAssetError as E, MultiAssetProduct};
use crate::product::{GraphLimitPolicy, SourceGraph, SourceGraphBuilder, SourceOpcode};

impl MultiAssetPricingPlan {
    /// Markets, models, product components and correlation rows use one explicit order.
    /// All markets must share currency and exactly the same discount curve.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
    ) -> Result<Self, E> {
        let lsv = vec![None; models.len()];
        Self::compile_with_lsv(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            lsv,
            None,
        )
    }

    /// An optional LSV configuration per asset turns its LV model into a calibration
    /// target. Full Brownian matrices are optional and align with the correlation
    /// dates: all spot drivers first, then the configured LSV volatility drivers.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_lsv(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        lsv_configs: Vec<Option<MultiAssetLsvConfig>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    ) -> Result<Self, E> {
        Self::compile_with_bergomi_lsv(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            lsv_configs.into_iter().map(|c| c.map(Into::into)).collect(),
            driver_correlations,
        )
    }

    /// Mixed one- and two-factor LSV. Driver order is all spots, followed by each
    /// asset's volatility factors in configured order. Each marginal block is fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_bergomi_lsv(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        lsv_configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    ) -> Result<Self, E> {
        Self::compile_impl(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            lsv_configs,
            driver_correlations,
            None,
            None,
        )
    }
    /// Shared one-factor HW with mixed BS and one-/two-factor Bergomi LSV.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_hull_white(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        lsv_configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        hull_white: MultiAssetHullWhiteConfig,
    ) -> Result<Self, E> {
        Self::compile_impl(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            lsv_configs,
            driver_correlations,
            Some(hull_white),
            None,
        )
    }
    /// Calibrate one positive basket of normalized BS/LV equities by particle
    /// projection between two PSD correlation schedules, with deterministic rates.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_local_correlation(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        local_correlation: LocalCorrelationConfig,
    ) -> Result<Self, E> {
        let configs = vec![None; models.len()];
        Self::compile_impl(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            configs,
            None,
            None,
            Some((local_correlation, LocalCorrelationExtensions::default())),
        )
    }
    /// Particle local correlation with fixed marginal LSV/HW driver blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_joint_local_correlation(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        lsv_configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        hull_white: Option<MultiAssetHullWhiteConfig>,
        local_correlation: LocalCorrelationConfig,
        extensions: LocalCorrelationExtensions,
    ) -> Result<Self, E> {
        Self::compile_impl(
            valuation_date,
            product,
            markets,
            models,
            correlation,
            engine,
            execution,
            maximum_step,
            lsv_configs,
            driver_correlations,
            hull_white,
            Some((local_correlation, extensions)),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_impl(
        valuation_date: Date,
        product: MultiAssetProduct,
        markets: Vec<EquityMarket>,
        models: Vec<ModelSpec>,
        correlation: CorrelationTermStructure,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
        lsv_configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        hull_white: Option<MultiAssetHullWhiteConfig>,
        local_correlation: Option<(LocalCorrelationConfig, LocalCorrelationExtensions)>,
    ) -> Result<Self, E> {
        if markets.is_empty() || markets.len() != models.len() {
            return Err(E::Invalid("market/model dimensions differ"));
        }
        if lsv_configs.len() != models.len() {
            return Err(E::Invalid("LSV configuration count must equal asset count"));
        }
        let rough_count = lsv_configs
            .iter()
            .flatten()
            .filter(|c| c.rough().is_some())
            .count();
        if hull_white.is_none() && rough_count > 0 {
            return Err(E::Invalid(
                "rough-LSV requires the shared HW adapter and paired targets; use a zero-volatility HW model for deterministic rates",
            ));
        }
        for (config, model) in lsv_configs.iter().zip(&models) {
            if config.is_some() && !matches!(model, ModelSpec::LocalVolatility(_)) {
                return Err(E::Invalid(
                    "each LSV asset requires a LocalVolatility target model",
                ));
            }
        }
        let order: Vec<_> = markets.iter().map(|m| m.forward().underlying()).collect();
        if order != product.underlyings || order != correlation.underlyings() {
            return Err(E::Invalid(
                "product, market and correlation underlying order must match",
            ));
        }
        let discount = markets[0].forward().discount_curve();
        product.graph.compile(GraphLimitPolicy::default())?;
        for node in product.graph.nodes() {
            if let SourceOpcode::TerminalSpot {
                underlying,
                observation_date,
            }
            | SourceOpcode::PreDividendSpot {
                underlying,
                observation_date,
            } = node.opcode()
                && (!order.contains(&underlying) || observation_date < valuation_date)
            {
                return Err(E::Invalid(
                    "unknown underlying or past observation in source graph",
                ));
            }
        }
        if markets
            .iter()
            .any(|m| m.currency() != product.currency || m.forward().discount_curve() != discount)
        {
            return Err(E::Invalid(
                "all assets must share the product currency and discount curve",
            ));
        }
        if correlation.entries()[0].0 > valuation_date {
            return Err(E::Invalid("correlation must cover the valuation date"));
        }
        let fraction = |date| DayCountConvention::Act365F.year_fraction(valuation_date, date);
        // Compile each output separately to inspect only its live observation dependencies.
        let mut pv = SourceGraphBuilder::from_graph(&product.graph);
        let mut sum = pv.literal(0.0)?;
        let mut dates = Vec::new();
        for (&output, &payment) in product.graph.outputs().iter().zip(&product.payment_dates) {
            if payment < valuation_date {
                return Err(E::Invalid("payment before valuation"));
            }
            let tape = SourceGraph::new(product.graph.nodes().to_vec(), vec![output])
                .compile(GraphLimitPolicy::default())?;
            for (underlying, date) in tape.terminal_observations() {
                if !order.contains(&underlying) {
                    return Err(E::Invalid("graph contains an unknown underlying"));
                }
                if date < valuation_date {
                    return Err(E::Invalid(
                        "past observations require a seasoned contract and are unsupported",
                    ));
                }
                if date > payment {
                    return Err(E::Invalid(
                        "cash flow depends on an observation after its payment",
                    ));
                }
                dates.push(date);
            }
            let df = discount.evaluate(fraction(payment))?.discount;
            let df = pv.literal(df)?;
            let value = pv.push(SourceOpcode::Multiply {
                left: output,
                right: df,
            })?;
            sum = pv.push(SourceOpcode::Add {
                left: sum,
                right: value,
            })?;
        }
        let payoff = pv.finish(vec![sum]).compile(GraphLimitPolicy::default())?;
        let horizon = dates.iter().map(|&d| fraction(d)).fold(0.0, f64::max);
        if horizon <= 0.0 {
            return Err(E::Invalid(
                "multi-asset plan requires at least one future observation",
            ));
        }
        let mut events: Vec<_> = dates.iter().map(|&d| fraction(d)).collect();
        if let Some((c, _)) = &local_correlation {
            events.extend(
                c.target
                    .time_nodes()
                    .iter()
                    .copied()
                    .filter(|t| *t <= horizon),
            );
        }
        for (date, _) in correlation.entries() {
            let t = fraction(*date);
            if t > 0.0 && t < horizon {
                events.push(t);
            }
        }
        for (market, model) in markets.iter().zip(&models) {
            match model {
                ModelSpec::BlackScholes(_) => {}
                ModelSpec::LocalVolatility(lv) => {
                    let times = lv.local_variance_grid().time_nodes();
                    if times[0] != 0.0 || times[times.len() - 1] < horizon {
                        return Err(E::Invalid(
                            "local variance grid must cover [0, last observation]",
                        ));
                    }
                    events.extend(times.iter().copied().filter(|t| *t <= horizon));
                }
                _ => {
                    return Err(E::Invalid(
                        "multi-asset models support BlackScholes and LocalVolatility",
                    ));
                }
            }
            if let Some(dividends) = market.forward().discrete_dividends() {
                events.extend(
                    dividends
                        .events()
                        .iter()
                        .map(|d| d.ex_time())
                        .filter(|t| *t <= horizon),
                );
            }
        }
        let grid = LocalVolTimeGrid::compile(events, maximum_step).map_err(E::numerical)?;
        let times = grid.nodes().to_vec();
        let driver_layout = driver_layout::DriverLayout::compile(
            lsv_configs
                .iter()
                .map(|c| c.as_ref().map_or(0, MultiAssetBergomiLsvConfig::factor_count)),
            usize::from(hull_white.is_some()) * 2,
            rough_count,
        )?;
        let dimension = driver_layout.dimension(
            grid.step_count(),
            1 + usize::from(local_correlation.is_some()),
        )?;
        let variance_reduction = match engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                if c.independent_sampling_units().get() < 2 {
                    return Err(E::Invalid("at least two independent MC units required"));
                }
                c.variance_reduction()
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                if c.scramble_count().get() < 2 {
                    return Err(E::Invalid(
                        "at least two independent QMC scrambles required",
                    ));
                }
                c.variance_reduction()
            }
        };
        let qmc = match engine {
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                Some(RqmcPlan::compile(c, dimension).map_err(E::numerical)?)
            }
            _ => None,
        };
        // Independent factors are bridged before correlation. RQMC assigns
        // bridge-rank-major dimensions; the other sampling modes keep their layout.
        let bridge = if variance_reduction.brownian_bridge() {
            Some(BrownianBridgePlan::compile(times.clone(), 1).map_err(E::numerical)?)
        } else {
            None
        };
        let interval_correlations: Vec<_> = times[..times.len() - 1]
            .iter()
            .map(|t| {
                correlation
                    .entries()
                    .partition_point(|(d, _)| fraction(*d) <= *t)
                    - 1
            })
            .collect();
        let lsv_drivers = if let Some(hw) = &hull_white {
            Some(hull_white::compile_drivers(
                &correlation,
                &times,
                &interval_correlations,
                &lsv_configs,
                driver_correlations,
                hw,
            )?)
        } else {
            lsv::LsvDrivers::compile(
                &correlation,
                &times,
                &interval_correlations,
                &lsv_configs,
                driver_correlations,
            )?
        };
        let joint_configs = lsv_configs.clone();
        let mut assets = Vec::new();
        for (asset_index, ((market, model), lsv_config)) in
            markets.into_iter().zip(models).zip(lsv_configs).enumerate()
        {
            let forward = market.forward().clone();
            let timeline = forward
                .discrete_dividends()
                .map(|d| d.event_timeline())
                .transpose()?
                .unwrap_or_default();
            let mut forwards = Vec::new();
            let mut coordinates = Vec::new();
            let mut pre_coordinates = Vec::new();
            for &t in &times {
                let f = forward.evaluate(t)?;
                forwards.push(f.forward);
                coordinates.push(f.affine_coordinate);
                pre_coordinates.push(
                    timeline
                        .iter()
                        .find(|e| e.ex_time() == t)
                        .map_or(f.affine_coordinate, |e| e.before()),
                );
            }
            let process =
                LocalVolLogEulerPlan::new(grid.clone(), forwards).map_err(E::numerical)?;
            let hw = hull_white
                .as_ref()
                .map(|config| {
                    hull_white::HwAsset::compile(
                        &forward,
                        &model,
                        &grid,
                        lsv_config.clone(),
                        config,
                        asset_index,
                        driver_layout.volatility(asset_index).start,
                    )
                })
                .transpose()?
                .map(std::sync::Arc::new);
            let lsv = if hull_white.is_some() {
                None
            } else if let Some(config) = lsv_config {
                let ModelSpec::LocalVolatility(lv) = &model else {
                    unreachable!("validated LSV target")
                };
                Some(std::sync::Arc::new(lsv::LsvAsset::compile(
                    lv.local_variance_grid(),
                    &grid,
                    config,
                )?))
            } else {
                None
            };
            assets.push(Asset {
                forward,
                model,
                process,
                coordinates,
                pre_coordinates,
                lsv,
                hw,
            });
        }
        let hull_white = hull_white
            .map(|config| {
                hull_white::HwContext::new(
                    config,
                    &product,
                    valuation_date,
                    &times,
                    &assets[0].forward,
                )
            })
            .transpose()?;
        let mut plan = Self {
            valuation_date,
            assets,
            correlation,
            interval_correlations,
            times,
            payoff,
            engine,
            execution,
            bridge,
            qmc,
            fingerprint: String::new(),
            market_weights: std::sync::Arc::new(std::sync::OnceLock::new()),
            lsv_drivers,
            driver_layout,
            hull_white,
            local_correlation: None,
        };
        plan.local_correlation = local_correlation
            .map(|(config, extensions)| {
                LocalCorrelationCalibration::compile(&plan, config, extensions, &joint_configs)
                    .map(std::sync::Arc::new)
            })
            .transpose()?;
        plan.fingerprint = plan.make_fingerprint(&product, maximum_step);
        Ok(plan)
    }
    pub fn time_nodes(&self) -> &[f64] {
        &self.times
    }
    pub fn correlation_entry_indices(&self) -> &[usize] {
        &self.interval_correlations
    }
    pub fn correlation(&self) -> &CorrelationTermStructure {
        &self.correlation
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn underlyings(&self) -> Vec<UnderlyingId> {
        self.assets.iter().map(|a| a.forward.underlying()).collect()
    }

    fn make_fingerprint(&self, product: &MultiAssetProduct, maximum_step: f64) -> String {
        let mut h = blake3::Hasher::new();
        if self.qmc.is_some() && self.bridge.is_some() {
            h.update(b"multi-asset-bs-lv-affine-v2-bridge-rank-major-before-correlation");
        } else {
            h.update(b"multi-asset-bs-lv-affine-v1-factor-major-bridge-before-correlation");
        }
        if self.assets.iter().any(|a| {
            a.forward
                .discrete_dividends()
                .is_some_and(|d| d.initial_reserve() > 0.0)
        }) {
            h.update(b"escrowed-dividend-reserve-v1");
        }
        h.update(self.payoff.source_fingerprint().as_bytes());
        h.update(self.payoff.tape_fingerprint().as_bytes());
        h.update(self.valuation_date.to_string().as_bytes());
        h.update(&product.currency.get().to_be_bytes());
        h.update(&maximum_step.to_bits().to_be_bytes());
        h.update(&self.execution.reduction_block_size().get().to_be_bytes());
        h.update(&self.execution.version().to_be_bytes());
        for date in &product.payment_dates {
            h.update(date.to_string().as_bytes());
        }
        for a in &self.assets {
            h.update(&a.forward.underlying().get().to_be_bytes());
            floats(&mut h, &[a.forward.spot().get()]);
            for curve in [a.forward.discount_curve(), a.forward.dividend_curve()] {
                h.update(&curve.id().get().to_be_bytes());
                floats(&mut h, curve.times());
                floats(&mut h, curve.discount_factors());
            }
            let dividends = a
                .forward
                .discrete_dividends()
                .map(|d| d.events())
                .unwrap_or(&[]);
            h.update(&(dividends.len() as u64).to_be_bytes());
            for d in dividends {
                h.update(&d.event().get().to_be_bytes());
                floats(&mut h, &[d.ex_time(), d.fixed_cash(), d.beta()]);
            }
            match &a.model {
                ModelSpec::BlackScholes(v) => {
                    h.update(&[0]);
                    floats(&mut h, &[v.volatility().get()]);
                }
                ModelSpec::LocalVolatility(v) => {
                    h.update(&[1]);
                    let g = v.local_variance_grid();
                    floats(&mut h, g.time_nodes());
                    floats(&mut h, g.log_moneyness_nodes());
                    floats(&mut h, g.values());
                    floats(&mut h, &[g.floor(), g.cap()]);
                }
                _ => unreachable!("models validated"),
            }
            if let Some(lsv) = &a.lsv {
                h.update(if lsv.calibration.factor_count() == 1 {
                    b"multi-asset-bergomi-lsv-unit-martingale-joint-ou-v1"
                } else {
                    b"multi-asset-bergomi-two-factor-lsv-unit-martingale-joint-ou-v1"
                });
                let c = &lsv.calibration;
                let mut parameters = c.parameters();
                let p = c.config();
                parameters.extend([p.log_bandwidth(), p.minimum_effective_samples()]);
                floats(&mut h, &parameters);
                h.update(&(p.particle_count() as u64).to_be_bytes());
                h.update(&p.seed().to_be_bytes());
                h.update(&[u8::from(p.retain_reverse_trace())]);
                floats(&mut h, c.surface().squared_leverage());
            }
        }
        let t = self.correlation.tolerances();
        floats(
            &mut h,
            &[
                t.symmetry_abs_tol,
                t.diagonal_abs_tol,
                t.psd_abs_tol,
                t.psd_rel_tol,
                t.zero_pivot_abs_tol,
                t.zero_pivot_rel_tol,
            ],
        );
        for (d, c) in self.correlation.entries() {
            h.update(d.to_string().as_bytes());
            floats(&mut h, c.raw());
            floats(&mut h, c.canonical());
            floats(&mut h, c.lower());
        }
        if let Some(d) = &self.lsv_drivers {
            h.update(b"joint-spot-ou-inputs-v1");
            if !d.rough_asset_indices.is_empty() {
                h.update(b"multi-asset-rough-hw-kappa1-joint-power-ou-v1");
                for &asset in &d.rough_asset_indices {
                    h.update(&(asset as u64).to_be_bytes());
                }
            }
            for c in &d.entries {
                floats(&mut h, c.raw());
                floats(&mut h, c.canonical());
                floats(&mut h, c.lower());
            }
            for (c, scale) in d.intervals.iter().zip(&d.scales) {
                floats(&mut h, c.canonical());
                floats(&mut h, c.lower());
                floats(&mut h, scale);
            }
            if let Some(permutations) = &d.permutations {
                h.update(b"hw-spot-prefix-pivoted-ou-rate-v1");
                for order in permutations {
                    for &index in order {
                        h.update(&(index as u64).to_be_bytes());
                    }
                }
            }
        }
        if let Some(hw) = &self.hull_white {
            hw.fingerprint(&mut h, &self.assets);
        }
        if let Some(c) = &self.local_correlation {
            c.fingerprint(&mut h);
        }
        floats(&mut h, &self.times);
        match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                h.update(&[0]);
                h.update(&c.master_seed().to_be_bytes());
                h.update(&c.independent_sampling_units().get().to_be_bytes());
                h.update(&[
                    u8::from(c.variance_reduction().antithetic()),
                    u8::from(c.variance_reduction().brownian_bridge()),
                ]);
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                h.update(&[1]);
                h.update(&c.master_scramble_seed().to_be_bytes());
                h.update(&c.points_per_scramble().get().to_be_bytes());
                h.update(&c.scramble_count().get().to_be_bytes());
                h.update(&[
                    u8::from(c.variance_reduction().antithetic()),
                    u8::from(c.variance_reduction().brownian_bridge()),
                ]);
                let q = self.qmc.as_ref().expect("QMC");
                h.update(&q.direction_checksum());
                h.update(&q.scramble_checksum());
            }
        }
        h.finalize().to_hex().to_string()
    }
}
pub(super) fn floats(h: &mut blake3::Hasher, values: &[f64]) {
    h.update(&(values.len() as u64).to_be_bytes());
    for v in values {
        h.update(&v.to_bits().to_be_bytes());
    }
}
