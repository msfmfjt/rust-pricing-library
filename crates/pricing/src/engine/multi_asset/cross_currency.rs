//! Foreign-equity options paid in a different currency with correlated FX and optional domestic and foreign short rates.
mod dividend_path;
mod exotic;
mod model;
mod model_path;
mod rate_pricing;
mod rates;
mod sampling;
use crate::analytical::{BlackForwardOracleInputs, evaluate_black_forward};
use crate::core::{Date, DayCountConvention, PositiveF64, UnderlyingId};
use crate::market::{
    CorrelationTermStructure, CorrelationToleranceConfig, DiscountCurve, EquityForward,
    EquityMarket, LogLinearDiscountCurve,
};
use crate::mc::{EngineConfig, ExecutionPolicy, LsmConfig};
use crate::models::{BlackScholesSpec, BuehlerDividendModel, ModelSpec};
use crate::multi_asset::{MultiAssetError as Error, MultiAssetPricingPlan, MultiAssetProduct};
use crate::product::{
    EuropeanVanillaSpec, ProductSpec, SourceGraphBuilder, SourceOpcode as Op, multi_asset::option,
};
use crate::{EquityModelSpec, Estimate, EstimatorKind};
use rates::CrossCurrencyHullWhite;
use std::sync::Arc;

/// All exchange rates are units of settlement currency per unit of equity currency.
#[derive(Clone, Copy, Debug)]
pub enum EquityFxSettlement {
    /// Foreign strike, fixed contractual conversion after exercise.
    Quanto { fixed_fx_rate: PositiveF64 },
    /// Foreign strike, conversion at the expiry FX fixing after exercise.
    FloatingFx,
    /// Strike in settlement currency, conversion of the equity before exercise.
    Composite,
}

/// Constant diffusion coefficients. Correlation is between the foreign equity
/// return and the return of FX in the stated quote direction.
#[derive(Clone, Copy, Debug)]
pub struct EquityFxDynamics {
    equity: BlackScholesSpec,
    fx: BlackScholesSpec,
    correlation: f64,
}

impl EquityFxDynamics {
    pub fn new(
        equity_volatility: f64,
        fx_volatility: f64,
        correlation: f64,
    ) -> Result<Self, Error> {
        if !(-1.0..=1.0).contains(&correlation) {
            return Err(Error::Invalid("equity/FX correlation must lie in [-1, 1]"));
        }
        Ok(Self {
            equity: BlackScholesSpec::new(equity_volatility)?,
            fx: BlackScholesSpec::new(fx_volatility)?,
            correlation,
        })
    }

    fn composite_volatility(self) -> f64 {
        let rho = self.correlation;
        let fx = self.fx.volatility().get();
        (self.equity.volatility().get() + rho * fx).hypot(fx * ((1.0 - rho) * (1.0 + rho)).sqrt())
    }
}

#[derive(Clone, Debug)]
pub struct CrossCurrencyEquityOption {
    valuation_date: Date,
    product: ProductSpec,
    foreign_fixing_product: Option<ProductSpec>,
    foreign_market: EquityMarket,
    fx_spot: PositiveF64,
    payment_date: Date,
    settlement: EquityFxSettlement,
    domestic_curve: Arc<LogLinearDiscountCurve>,
    time: f64,
    foreign_equity_forward: f64,
    fx_forward: PositiveF64,
    dynamics: EquityFxDynamics,
    equity_model: Option<EquityModelSpec>,
    maximum_step: f64,
    rates: Option<CrossCurrencyHullWhite>,
    dividends: Option<BuehlerDividendModel>,
    payment_fx_fixings: std::collections::BTreeMap<Date, Date>,
}

impl CrossCurrencyEquityOption {
    /// The product currency identifies the settlement curve. FX is a cash
    /// quote for value on the valuation date. Fix both assets at expiry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valuation_date: Date,
        product: EuropeanVanillaSpec,
        foreign_market: &EquityMarket,
        settlement_curve: Arc<LogLinearDiscountCurve>,
        fx_spot: f64,
        dynamics: EquityFxDynamics,
        settlement: EquityFxSettlement,
        payment_date: Date,
    ) -> Result<Self, Error> {
        Self::from_product(
            valuation_date,
            ProductSpec::EuropeanVanilla(product),
            foreign_market,
            settlement_curve,
            fx_spot,
            dynamics,
            settlement,
            Some(payment_date),
        )
    }

    /// Exotic products carry their contractual payment dates. A European
    /// vanilla can optionally specify a later payment date.
    #[allow(clippy::too_many_arguments)]
    pub fn from_product(
        valuation_date: Date,
        product: ProductSpec,
        foreign_market: &EquityMarket,
        settlement_curve: Arc<LogLinearDiscountCurve>,
        fx_spot: f64,
        dynamics: EquityFxDynamics,
        settlement: EquityFxSettlement,
        payment_date: Option<Date>,
    ) -> Result<Self, Error> {
        Self::build(
            valuation_date,
            product,
            foreign_market,
            settlement_curve,
            fx_spot,
            dynamics,
            settlement,
            payment_date,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        valuation_date: Date,
        product: ProductSpec,
        foreign_market: &EquityMarket,
        settlement_curve: Arc<LogLinearDiscountCurve>,
        fx_spot: f64,
        dynamics: EquityFxDynamics,
        settlement: EquityFxSettlement,
        payment_date: Option<Date>,
        dividends: Option<BuehlerDividendModel>,
    ) -> Result<Self, Error> {
        let payment_date = payment_date.unwrap_or_else(|| product.payment_date());
        if !matches!(product, ProductSpec::EuropeanVanilla(_))
            && payment_date != product.payment_date()
        {
            return Err(Error::Invalid(
                "use the payment dates specified by the exotic product",
            ));
        }
        if matches!(&product, ProductSpec::Graph(graph) if graph.underlyings().len() != 1) {
            return Err(Error::Invalid(
                "cross-currency equity graphs require one equity underlying",
            ));
        }
        let equity = foreign_market.forward();
        if product.underlying() != equity.underlying() {
            return Err(Error::Invalid(
                "option and equity market underlyings must match",
            ));
        }
        if product.currency() == foreign_market.currency() {
            return Err(Error::Invalid(
                "equity and settlement currencies must differ",
            ));
        }
        if product.expiry() < valuation_date || payment_date < product.expiry() {
            return Err(Error::Invalid(
                "require valuation date <= expiry <= payment date",
            ));
        }
        if dividends.is_none()
            && equity
                .discrete_dividends()
                .is_some_and(|d| !d.events().is_empty())
        {
            return Err(Error::Invalid(
                "discrete cash dividends require a dividend_model",
            ));
        }
        let fx_spot = PositiveF64::new(fx_spot, "fx_spot")?;
        let time = DayCountConvention::Act365F.year_fraction(valuation_date, product.expiry());
        let log_carry =
            equity.discount_curve().log_discount(time)? - settlement_curve.log_discount(time)?;
        let fx_forward = PositiveF64::new(fx_spot.get() * log_carry.exp(), "fx_forward")?;
        Ok(Self {
            valuation_date,
            product,
            foreign_fixing_product: None,
            foreign_market: foreign_market.clone(),
            fx_spot,
            payment_date,
            settlement,
            domestic_curve: settlement_curve,
            time,
            foreign_equity_forward: equity.forward(time)?,
            fx_forward,
            dynamics,
            equity_model: None,
            maximum_step: 0.02,
            rates: None,
            dividends,
            payment_fx_fixings: Default::default(),
        })
    }

    /// Use the foreign-market equity dynamics, with a lognormal FX rate.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model(
        valuation_date: Date,
        product: ProductSpec,
        foreign_market: &EquityMarket,
        settlement_curve: Arc<LogLinearDiscountCurve>,
        fx_spot: f64,
        equity_model: EquityModelSpec,
        fx_volatility: f64,
        correlation: f64,
        settlement: EquityFxSettlement,
        payment_date: Option<Date>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        Self::from_model_with_dividends(
            valuation_date,
            product,
            foreign_market,
            settlement_curve,
            fx_spot,
            equity_model,
            fx_volatility,
            correlation,
            settlement,
            payment_date,
            maximum_step,
            None,
        )
    }

    /// Supply a Buehler factor for foreign cash dividend means.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_with_dividends(
        valuation_date: Date,
        product: ProductSpec,
        foreign_market: &EquityMarket,
        settlement_curve: Arc<LogLinearDiscountCurve>,
        fx_spot: f64,
        equity_model: EquityModelSpec,
        fx_volatility: f64,
        correlation: f64,
        settlement: EquityFxSettlement,
        payment_date: Option<Date>,
        maximum_step: f64,
        dividends: Option<BuehlerDividendModel>,
    ) -> Result<Self, Error> {
        PositiveF64::new(maximum_step, "maximum_step")?;
        let sigma = match &equity_model {
            EquityModelSpec::Standard(ModelSpec::BlackScholes(m)) => Some(m.volatility().get()),
            EquityModelSpec::Standard(ModelSpec::Black76(m)) => Some(m.volatility().get()),
            _ => None,
        };
        let mut option = Self::build(
            valuation_date,
            product,
            foreign_market,
            settlement_curve,
            fx_spot,
            EquityFxDynamics::new(sigma.unwrap_or(0.0), fx_volatility, correlation)?,
            settlement,
            payment_date,
            dividends,
        )?;
        if dividends.is_some() {
            dividend_path::validate(&equity_model, foreign_market.forward())?;
        }
        if sigma.is_none() {
            option.equity_model = Some(equity_model);
        }
        option.maximum_step = maximum_step;
        Ok(option)
    }

    /// Add domestic and foreign short-rate dynamics under the settlement measure.
    pub fn with_hull_white_rates(
        mut self,
        domestic: crate::models::HullWhite1Factor,
        foreign: crate::models::HullWhite1Factor,
        correlations: Option<[[f64; 4]; 4]>,
    ) -> Result<Self, Error> {
        let rho = self.dynamics.correlation;
        let rates = match correlations {
            Some(c) => CrossCurrencyHullWhite::new(domestic, foreign, c)?,
            None => CrossCurrencyHullWhite::independent(domestic, foreign, rho)?,
        };
        if self.dividends.is_some()
            && !rates.foreign.is_deterministic()
            && self.equity_model.is_some()
        {
            return Err(Error::Invalid(
                "stochastic cash dividends with a stochastic foreign rate require constant equity volatility",
            ));
        }
        if (rates.correlations[0][1] - self.dynamics.correlation).abs() > 1e-12 {
            return Err(Error::Invalid(
                "stock/FX correlation must agree with the rate correlation matrix",
            ));
        }
        self.rates = if rates.domestic.is_deterministic() && rates.foreign.is_deterministic() {
            None
        } else {
            Some(rates)
        };
        Ok(self)
    }

    fn domestic_equity_forward(&self) -> Result<PositiveF64, Error> {
        let covariance = self.dynamics.correlation
            * self.dynamics.equity.volatility().get()
            * self.dynamics.fx.volatility().get()
            * self.time;
        Ok(PositiveF64::new(
            self.foreign_equity_forward * (-covariance).exp(),
            "quanto_adjusted_forward",
        )?)
    }

    /// Present value in the product's settlement currency.
    pub fn analytical_price(&self) -> Result<f64, Error> {
        if self.equity_model.is_some() || self.rates.is_some() || self.dividends.is_some() {
            return Err(Error::Invalid("this equity model requires price_mc"));
        }
        let ProductSpec::EuropeanVanilla(product) = &self.product else {
            return Err(Error::Invalid(
                "analytical_price supports European vanilla only; use price_mc for exotics",
            ));
        };
        let equity_vol = self.dynamics.equity.volatility().get();
        let (forward, volatility, conversion) = match self.settlement {
            EquityFxSettlement::Quanto { fixed_fx_rate } => (
                self.domestic_equity_forward()?.get(),
                equity_vol,
                fixed_fx_rate.get(),
            ),
            EquityFxSettlement::FloatingFx => (
                self.foreign_equity_forward,
                equity_vol,
                self.fx_forward.get(),
            ),
            EquityFxSettlement::Composite => (
                self.foreign_equity_forward * self.fx_forward.get(),
                self.dynamics.composite_volatility(),
                1.0,
            ),
        };
        let payment_time =
            DayCountConvention::Act365F.year_fraction(self.valuation_date, self.payment_date);
        let result = evaluate_black_forward(BlackForwardOracleInputs {
            side: product.side(),
            strike: product.strike().get(),
            notional: product.notional().get() * conversion,
            discount: self.domestic_curve.discount(payment_time)?,
            forward,
            volatility,
            time: self.time,
        })
        .map_err(Error::numerical)?;
        Ok(result.price)
    }

    /// Price in settlement currency using the supplied MC or RQMC engine.
    pub fn price_mc(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        self.price_mc_with_lsm(engine, execution, None)
    }

    /// Scheduled early exercise uses the native LSM training configuration.
    pub fn price_mc_with_lsm(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<LsmConfig>,
    ) -> Result<Estimate, Error> {
        if let Some(rates) = &self.rates {
            return self.price_with_rates(rates, engine, execution, lsm);
        }
        if let Some(model) = &self.equity_model {
            return self.price_model(model, self.maximum_step, engine, execution, lsm);
        }
        if self.dividends.is_some() {
            let model = EquityModelSpec::Standard(ModelSpec::BlackScholes(self.dynamics.equity));
            return self.price_model(&model, self.maximum_step, engine, execution, lsm);
        }
        let ProductSpec::EuropeanVanilla(product) = &self.product else {
            return self.price_exotic(engine, execution, lsm);
        };
        if lsm.is_some() {
            return Err(Error::Invalid("LSM settings apply only to early exercise"));
        }
        if self.time == 0.0 {
            let value = self.analytical_price()?;
            return Estimate::new(value, 0.0, value, value, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        let equity_id = self.product.underlying();
        // The FX coordinate is private to this two-factor price calculation.
        let fx_id = UnderlyingId::new(if equity_id.get() == 0 { 1 } else { 0 });
        let order = vec![equity_id, fx_id];
        let mut graph = SourceGraphBuilder::new();
        let terminal = |underlying| Op::TerminalSpot {
            underlying,
            observation_date: self.product.expiry(),
        };
        let s = graph.push(terminal(equity_id))?;
        let x = graph.push(terminal(fx_id))?;
        let underlier = match self.settlement {
            EquityFxSettlement::Composite => graph.push(Op::Multiply { left: x, right: s })?,
            _ => s,
        };
        let intrinsic = option(
            &mut graph,
            underlier,
            product.side(),
            product.strike().get(),
            product.notional().get(),
            None,
        )?;
        let payoff = match self.settlement {
            EquityFxSettlement::Quanto { fixed_fx_rate } => {
                let fixed = graph.literal(fixed_fx_rate.get())?;
                graph.push(Op::Multiply {
                    left: fixed,
                    right: intrinsic,
                })?
            }
            EquityFxSettlement::FloatingFx => graph.push(Op::Multiply {
                left: x,
                right: intrinsic,
            })?,
            EquityFxSettlement::Composite => intrinsic,
        };
        let product = MultiAssetProduct::from_graph(
            self.product.currency(),
            order.clone(),
            graph.finish(vec![payoff]),
            vec![self.payment_date],
        )?;
        // Under deterministic rates the two expiry forwards are martingales
        // under the domestic measure. Equal carry curves implement zero drift
        // for these forward coordinates and keep the actual domestic payment DF.
        let marginal = |id, forward| {
            EquityMarket::new(
                self.product.currency(),
                EquityForward::new(
                    id,
                    forward,
                    self.domestic_curve.clone(),
                    self.domestic_curve.clone(),
                ),
            )
        };
        let rho = self.dynamics.correlation;
        let correlation = CorrelationTermStructure::new(
            order,
            vec![(self.valuation_date, vec![vec![1.0, rho], vec![rho, 1.0]])],
            CorrelationToleranceConfig {
                symmetry_abs_tol: 0.0,
                diagonal_abs_tol: 0.0,
                psd_abs_tol: 0.0,
                psd_rel_tol: 0.0,
                zero_pivot_abs_tol: 0.0,
                zero_pivot_rel_tol: 0.0,
            },
        )?;
        let plan = MultiAssetPricingPlan::compile(
            self.valuation_date,
            product,
            vec![
                marginal(equity_id, self.domestic_equity_forward()?),
                marginal(fx_id, self.fx_forward),
            ],
            vec![
                ModelSpec::BlackScholes(self.dynamics.equity),
                ModelSpec::BlackScholes(self.dynamics.fx),
            ],
            correlation,
            engine,
            execution,
            self.time,
        )?;
        Ok(plan.evaluate()?.price)
    }
}
