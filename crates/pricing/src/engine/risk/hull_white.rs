//! Experimental one-currency equity/Hull–White pricing. Calibration input is
//! explicit and the stable JSON schema is unchanged. Compile price-only requests;
//! use evaluate_aad for the separate first-order hybrid sensitivity contract.
//! The default cash-dividend model uses paid-cash affine carry. The explicitly
//! named cash-dividend constructors retain the legacy escrowed model.

use crate::core::DayCountConvention;
use crate::mc::hull_white::{
    CalibratedHullWhiteLsv, HullWhiteEquityPlan, HullWhiteLsvTarget, HybridEquityVolatility,
    HybridVolatilityFactor, calibrate_hybrid_lsv_with_dividends,
};
use crate::mc::lsv::LsvParticleConfig;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, RandomDomain, RqmcPlan, VarianceReduction,
    inverse_standard_normal,
};
use crate::models::hull_white_dividends::{HULL_WHITE_CASH_DIVIDEND_MODEL, HullWhiteDividendPlan};
use crate::models::{
    Bergomi1Factor, HullWhite1Factor, HullWhiteError, HybridCorrelation, ModelSpec, RoughBergomi,
};
use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};

mod aad;
mod affine;
pub use aad::HullWhiteAadRisk;
use affine::{AffineDividendPlan, HULL_WHITE_AFFINE_DIVIDEND_MODEL};

#[derive(Clone, Debug, PartialEq)]
pub struct HullWhitePrice {
    pub value: f64,
    pub standard_error: f64,
    pub independent_sampling_units: u64,
    pub evaluated_paths: u128,
    pub plan_fingerprint: Fingerprint,
    pub scheme: &'static str,
    pub calibration_method: Option<&'static str>,
    pub calibration_seed: Option<u64>,
    pub cash_dividend_model: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct HullWhiteEquityPricingPlan {
    base: SimulationPlan,
    path: HullWhiteEquityPlan,
    affine_dividends: Option<AffineDividendPlan>,
    calibration: Option<CalibratedHullWhiteLsv>,
    calibration_seed: Option<u64>,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    spot: f64,
    payment_time: f64,
    fingerprint: Fingerprint,
    market: crate::market::EquityForward,
    risk_supported: bool,
    market_iv_target: Option<HullWhiteLsvTarget>,
}

impl HullWhiteEquityPricingPlan {
    /// Volatility applies to the continuous equity coordinate. Fixed cash is
    /// subtracted at ex-dates and its affine offset carries at realized r-q.
    pub fn compile_bs(
        request: &PricingRequest,
        rates: HullWhite1Factor,
        equity_rate_correlation: f64,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_bs_impl(
            request,
            rates,
            HybridCorrelation::new(0.0, equity_rate_correlation, 0.0)?,
            maximum_step,
            policy,
            false,
            None,
        )
    }
    /// Explicit legacy escrowed model: volatility applies to residual equity.
    /// Prefer compile_bs for the common paid-cash affine convention.
    pub fn compile_bs_with_cash_dividends(
        request: &PricingRequest,
        rates: HullWhite1Factor,
        equity_rate_correlation: f64,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_bs_impl(
            request,
            rates,
            HybridCorrelation::new(0.0, equity_rate_correlation, 0.0)?,
            maximum_step,
            policy,
            true,
            None,
        )
    }
    /// Pure rough Bergomi with flat initial forward variance from the request's
    /// BlackScholes volatility; H/eta/correlations remain fixed during AAD.
    pub fn compile_rough_bergomi(
        request: &PricingRequest,
        factor: RoughBergomi,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_bs_impl(
            request,
            rates,
            correlation,
            maximum_step,
            policy,
            false,
            Some(factor),
        )
    }
    /// Explicit legacy escrowed cash model.
    pub fn compile_rough_bergomi_with_cash_dividends(
        request: &PricingRequest,
        factor: RoughBergomi,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_bs_impl(
            request,
            rates,
            correlation,
            maximum_step,
            policy,
            true,
            Some(factor),
        )
    }
    fn compile_bs_impl(
        request: &PricingRequest,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        maximum_step: f64,
        policy: ExecutionPolicy,
        cash: bool,
        rough: Option<RoughBergomi>,
    ) -> Result<Self, MonteCarloError> {
        let ModelSpec::BlackScholes(bs) = request.model() else {
            return Err(HullWhiteError::Unsupported {
                feature: "BS-HW compile requires a BlackScholes request",
            }
            .into());
        };
        let base = compile_base(request, policy)?;
        let expiry = expiry_time(request);
        let mut events = base.hybrid_observation_times().to_vec();
        events.push(expiry);
        let market = request.market().equity().forward();
        if (cash || has_fixed_cash(market, expiry))
            && let Some(d) = market.discrete_dividends()
        {
            events.extend(
                d.events()
                    .iter()
                    .filter(|e| e.ex_time() <= expiry)
                    .map(|e| e.ex_time()),
            );
        }
        let grid = LocalVolTimeGrid::compile(events, maximum_step)
            .map_err(|e| MonteCarloError::HullWhite(e.into()))?;
        let dividends = if cash {
            Some(HullWhiteDividendPlan::new(&rates, market, grid.nodes())?)
        } else {
            None
        };
        let volatility = rough.map_or(
            HybridEquityVolatility::BlackScholes(bs.volatility().get()),
            |factor| HybridEquityVolatility::RoughBergomi {
                factor,
                initial_volatility: bs.volatility().get(),
            },
        );
        let mut path = HullWhiteEquityPlan::new(rates, volatility, correlation, &grid)?;
        if let Some(d) = dividends {
            path = path.with_dividends(d)?;
        }
        Self::finish(request, base, path, None, None, None, policy)
    }

    /// The density/variance target must match the request model and contain all
    /// contractual observation times, including ex-dates and expiry. For the
    /// default affine model it describes the continuous equity coordinate, not
    /// an unconverted physical-spot smile or an escrowed residual-equity smile.
    pub fn compile_lsv(
        request: &PricingRequest,
        target: &HullWhiteLsvTarget,
        factor: Bergomi1Factor,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_lsv_impl(
            request,
            target,
            factor.into(),
            rates,
            correlation,
            particles,
            policy,
            false,
        )
    }
    /// Explicit legacy escrow-coordinate smile with bond covariance.
    pub fn compile_lsv_with_cash_dividends(
        request: &PricingRequest,
        target: &HullWhiteLsvTarget,
        factor: Bergomi1Factor,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_lsv_impl(
            request,
            target,
            factor.into(),
            rates,
            correlation,
            particles,
            policy,
            true,
        )
    }
    pub fn compile_rough_lsv(
        request: &PricingRequest,
        target: &HullWhiteLsvTarget,
        factor: RoughBergomi,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_lsv_impl(
            request,
            target,
            factor.into(),
            rates,
            correlation,
            particles,
            policy,
            false,
        )
    }
    /// Explicit legacy escrowed cash model.
    pub fn compile_rough_lsv_with_cash_dividends(
        request: &PricingRequest,
        target: &HullWhiteLsvTarget,
        factor: RoughBergomi,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Self::compile_lsv_impl(
            request,
            target,
            factor.into(),
            rates,
            correlation,
            particles,
            policy,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_lsv_impl(
        request: &PricingRequest,
        target: &HullWhiteLsvTarget,
        factor: HybridVolatilityFactor,
        rates: HullWhite1Factor,
        correlation: HybridCorrelation,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
        cash: bool,
    ) -> Result<Self, MonteCarloError> {
        let ModelSpec::LocalVolatility(lv) = request.model() else {
            return Err(HullWhiteError::Unsupported {
                feature: "LSV-HW compile requires a LocalVolatility target request",
            }
            .into());
        };
        if !matching_target_grid(lv.local_variance_grid(), target.grid()) {
            return Err(HullWhiteError::InvalidInput {
                field: "density_target_model_mismatch",
                index: 0,
            }
            .into());
        }
        let base = compile_base(request, policy)?;
        let expiry = expiry_time(request);
        for &t in base
            .hybrid_observation_times()
            .iter()
            .chain(std::iter::once(&expiry))
        {
            if !target.grid().time_nodes().contains(&t) {
                return Err(HullWhiteError::InvalidInput {
                    field: "target_missing_contractual_time",
                    index: 0,
                }
                .into());
            }
        }
        // Validate the affine event grid before starting particle calibration.
        if !cash && has_fixed_cash(request.market().equity().forward(), expiry) {
            AffineDividendPlan::new(
                request.market().equity().forward(),
                &rates,
                &target
                    .grid()
                    .time_nodes()
                    .iter()
                    .copied()
                    .filter(|t| *t <= expiry)
                    .collect::<Vec<_>>(),
            )?;
        }
        let dividends = if cash {
            Some(HullWhiteDividendPlan::new(
                &rates,
                request.market().equity().forward(),
                target.grid().time_nodes(),
            )?)
        } else {
            None
        };
        // Affine cash affects physical observations, not U or its leverage.
        // Therefore use the continuous-coordinate discounted-particle equation
        // (no reserve loading, cross term or escrowed quadratic calibration).
        let calibration = calibrate_hybrid_lsv_with_dividends(
            target,
            factor,
            &rates,
            correlation,
            request.market().equity().forward().spot().get(),
            &particles,
            dividends.as_ref(),
        )?;
        let events = target
            .grid()
            .time_nodes()
            .iter()
            .copied()
            .filter(|t| *t <= expiry)
            .collect();
        let grid = LocalVolTimeGrid::compile(events, expiry)
            .map_err(|e| MonteCarloError::HullWhite(e.into()))?;
        let volatility = match factor {
            HybridVolatilityFactor::Bergomi(factor) => HybridEquityVolatility::BergomiLsv {
                factor,
                leverage: calibration.surface.clone(),
            },
            HybridVolatilityFactor::Rough(factor) => HybridEquityVolatility::RoughBergomiLsv {
                factor,
                leverage: calibration.surface.clone(),
            },
        };
        let mut path = HullWhiteEquityPlan::new(rates, volatility, correlation, &grid)?;
        if let Some(d) = dividends {
            path = path.with_dividends(d)?;
        }
        Self::finish(
            request,
            base,
            path,
            Some(calibration),
            Some(&particles),
            Some(target),
            policy,
        )
    }

    fn finish(
        request: &PricingRequest,
        base: SimulationPlan,
        path: HullWhiteEquityPlan,
        calibration: Option<CalibratedHullWhiteLsv>,
        particles: Option<&LsvParticleConfig>,
        target: Option<&HullWhiteLsvTarget>,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let dimension = (path.times().len() - 1)
            .checked_mul(path.random_factor_count())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(HullWhiteError::InvalidInput {
                field: "random_dimension",
                index: 0,
            })?;
        if let EngineConfig::RandomizedQuasiMonteCarlo(config) = request.engine() {
            RqmcPlan::compile(config, dimension)?;
        }
        let market = request.market().equity().forward();
        let has_cash = has_fixed_cash(market, *path.times().last().unwrap());
        let affine_dividends = if path.dividends().is_none() && has_cash {
            Some(AffineDividendPlan::new(market, path.rates(), path.times())?)
        } else {
            None
        };
        let mut hash = blake3::Hasher::new();
        hash.update(b"pricing/equity-hull-white-plan/v1\0");
        hash.update(base.plan_fingerprint().as_bytes());
        if affine_dividends.is_some() {
            hash.update(HULL_WHITE_AFFINE_DIVIDEND_MODEL.as_bytes());
        }
        hash.update(&path.rates().mean_reversion().to_bits().to_be_bytes());
        for values in [
            path.rates().volatility_times(),
            path.rates().volatilities(),
            path.times(),
        ] {
            hash.update(&(values.len() as u64).to_be_bytes());
            for &v in values {
                hash.update(&v.to_bits().to_be_bytes());
            }
        }
        let corr = path.correlation();
        for v in [corr.equity_vol, corr.equity_rate, corr.vol_rate] {
            hash.update(&v.to_bits().to_be_bytes());
        }
        if let Some(particles) = particles {
            hash.update(&particles.seed().to_be_bytes());
            hash.update(&(particles.particle_count() as u64).to_be_bytes());
            for v in [
                particles.log_bandwidth(),
                particles.minimum_effective_samples(),
            ] {
                hash.update(&v.to_bits().to_be_bytes());
            }
            if particles.retain_reverse_trace() {
                hash.update(b"hybrid-aad-trace-v1\0");
            }
        }
        if let Some(calibration) = &calibration {
            for &v in calibration.surface.squared_leverage() {
                hash.update(&v.to_bits().to_be_bytes());
            }
        }
        if let Some(target) = target {
            let g = target.grid();
            for n in [g.time_nodes().len(), g.log_moneyness_nodes().len()] {
                hash.update(&(n as u64).to_be_bytes());
            }
            for &v in g
                .time_nodes()
                .iter()
                .chain(g.log_moneyness_nodes())
                .chain(g.values())
                .chain(&[g.floor(), g.cap()])
                .chain(target.log_densities())
            {
                hash.update(&v.to_bits().to_be_bytes());
            }
            if let Some(surface) = target.market_iv_surface() {
                hash.update(crate::market::MARKET_IV_INTERPOLATION.as_bytes());
                for n in [
                    surface.maturity_nodes().len(),
                    surface.log_moneyness_nodes().len(),
                ] {
                    hash.update(&(n as u64).to_be_bytes());
                }
                for &v in surface
                    .maturity_nodes()
                    .iter()
                    .chain(surface.log_moneyness_nodes())
                    .chain(surface.implied_volatilities())
                {
                    hash.update(&v.to_bits().to_be_bytes());
                }
            }
        }
        // The stochastic-volatility parameters are also encoded by the path.
        hash.update(&path.parameter_fingerprint_bytes());
        if let Some(d) = path.dividends() {
            hash.update(&d.fingerprint_bytes());
        }
        Ok(Self {
            base,
            path,
            affine_dividends,
            calibration,
            calibration_seed: particles.map(LsvParticleConfig::seed),
            engine: request.engine(),
            policy,
            spot: market.spot().get(),
            payment_time: DayCountConvention::Act365F
                .year_fraction(request.valuation_date(), request.product().payment_date()),
            fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            market: market.clone(),
            risk_supported: request.product().supports_pathwise_risk()
                || request.risk().payoff_smoothing().is_some(),
            market_iv_target: target.filter(|t| t.market_iv_surface().is_some()).cloned(),
        })
    }
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn random_factor_count(&self) -> usize {
        self.path.random_factor_count()
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        self.path.times()
    }
    #[must_use]
    pub fn calibration(&self) -> Option<&CalibratedHullWhiteLsv> {
        self.calibration.as_ref()
    }
    #[must_use]
    pub fn cash_dividend_model(&self) -> Option<&'static str> {
        if self.affine_dividends.is_some() {
            Some(HULL_WHITE_AFFINE_DIVIDEND_MODEL)
        } else {
            self.path
                .dividends()
                .map(|_| HULL_WHITE_CASH_DIVIDEND_MODEL)
        }
    }
    /// The affine model starts U at the full Spot; only explicit escrowed mode
    /// subtracts a future-dividend reserve from the risky initial state.
    #[must_use]
    pub fn risky_spot(&self) -> f64 {
        self.path
            .dividends()
            .map_or(self.spot, HullWhiteDividendPlan::risky_spot)
    }

    pub fn evaluate(&self) -> Result<HullWhitePrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.policy)?;
        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics(count, |p| {
                    let z = self.path.pseudo_shocks(
                        config.master_seed(),
                        p,
                        RandomDomain::Valuation,
                    )?;
                    self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                })?;
                (stats, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let dimension =
                    (self.path.random_factor_count() * (self.path.times().len() - 1)) as u32;
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut prices = Vec::new();
                for scramble in 0..config.scramble_count().get() {
                    let stats = executor.try_map_reduce_statistics(count, |p| {
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
                        self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                    })?;
                    prices.push(stats.sum().total() / count as f64);
                }
                let scrambles = u64::from(config.scramble_count().get());
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
                )
            }
        };
        let variance = statistics
            .moments()
            .sample_variance()
            .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
        let value = statistics.sum().total() / units as f64;
        let standard_error = (variance / units as f64).sqrt();
        if !value.is_finite() || !standard_error.is_finite() {
            return Err(HullWhiteError::InvalidInput {
                field: "price_estimator",
                index: 0,
            }
            .into());
        }
        Ok(HullWhitePrice {
            value,
            standard_error,
            independent_sampling_units: units,
            evaluated_paths: paths,
            plan_fingerprint: self.fingerprint,
            scheme: self.path.scheme(),
            calibration_method: self.path.calibration_method(),
            calibration_seed: self.calibration_seed,
            cash_dividend_model: self.cash_dividend_model(),
        })
    }

    fn bridge(&self, vr: VarianceReduction) -> Result<Option<BrownianBridgePlan>, MonteCarloError> {
        if !vr.brownian_bridge() {
            return Ok(None);
        }
        BrownianBridgePlan::compile(self.path.times().to_vec(), 1)
            .map(Some)
            .map_err(|e| MonteCarloError::LocalVol(e.into()))
    }
    fn sample(
        &self,
        z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
    ) -> Result<f64, MonteCarloError> {
        let n = self.path.times().len() - 1;
        let z = if let Some(bridge) = bridge {
            let mut transformed = Vec::with_capacity(z.len());
            for block in z.chunks_exact(n) {
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
        let mut value = 0.0;
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
            let states = self.path.evolve_path(self.spot, &shocks)?;
            let last = states[n];
            let terminal_time = self.path.times()[n];
            let relative_discount = self
                .path
                .rates()
                .relative_discount(terminal_time, last.integrated_rate_factor)?
                * self.path.rates().relative_bond(
                    terminal_time,
                    self.payment_time,
                    last.rate_factor,
                )?;
            let payoff = if let Some(d) = &self.affine_dividends {
                let observations = d.record(&states)?;
                self.base
                    .hybrid_spot_payoff(self.path.times(), &observations.spots)?
            } else if let Some(d) = self.path.dividends() {
                let spots = d
                    .nodes()
                    .iter()
                    .zip(&states)
                    .map(|(n, s)| n.spots(s.normalized_equity, s.rate_factor))
                    .collect::<Result<Vec<_>, _>>()?;
                self.base.hybrid_spot_payoff(self.path.times(), &spots)?
            } else {
                let equity = states
                    .iter()
                    .map(|s| s.normalized_equity)
                    .collect::<Vec<_>>();
                self.base.hybrid_payoff(self.path.times(), &equity)?
            };
            value += payoff * relative_discount;
        }
        Ok(value / if antithetic { 2.0 } else { 1.0 })
    }
}

fn has_fixed_cash(market: &crate::market::EquityForward, horizon: f64) -> bool {
    market.discrete_dividends().is_some_and(|d| {
        d.events()
            .iter()
            .any(|e| e.ex_time() <= horizon && e.fixed_cash() != 0.0)
    })
}

fn expiry_time(request: &PricingRequest) -> f64 {
    DayCountConvention::Act365F.year_fraction(request.valuation_date(), request.product().expiry())
}

// JSON parsing can move a decimal value by a few ulps. Accept only round-off
// differences; calibration always uses the target's own paired density/grid.
fn matching_target_grid(
    a: &crate::market::LocalVarianceGrid,
    b: &crate::market::LocalVarianceGrid,
) -> bool {
    fn close(x: f64, y: f64) -> bool {
        x == y || (x - y).abs() <= 8.0 * f64::EPSILON * x.abs().max(y.abs())
    }
    close(a.floor(), b.floor())
        && close(a.cap(), b.cap())
        && [
            (a.time_nodes(), b.time_nodes()),
            (a.log_moneyness_nodes(), b.log_moneyness_nodes()),
            (a.values(), b.values()),
        ]
        .iter()
        .all(|(x, y)| x.len() == y.len() && x.iter().zip(*y).all(|(&x, &y)| close(x, y)))
}
fn compile_base(
    request: &PricingRequest,
    policy: ExecutionPolicy,
) -> Result<SimulationPlan, MonteCarloError> {
    let risk = request.risk();
    if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
        return Err(HullWhiteError::Unsupported {
            feature: "hybrid Greeks; submit a price-only request",
        }
        .into());
    }
    let expiry = expiry_time(request);
    if expiry <= 0.0 {
        return Err(HullWhiteError::InvalidInput {
            field: "positive_horizon",
            index: 0,
        }
        .into());
    }
    SimulationPlan::compile_hybrid_base(request, policy)
}
