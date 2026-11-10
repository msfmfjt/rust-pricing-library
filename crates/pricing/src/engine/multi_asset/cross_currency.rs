//! European foreign-equity options paid in a different currency under joint GBM.
use crate::analytical::{BlackForwardOracleInputs, evaluate_black_forward};
use crate::core::{Date, DayCountConvention, PositiveF64, UnderlyingId};
use crate::market::{
    CorrelationTermStructure, CorrelationToleranceConfig, DiscountCurve, EquityForward,
    EquityMarket, LogLinearDiscountCurve,
};
use crate::mc::{EngineConfig, ExecutionPolicy};
use crate::models::{BlackScholesSpec, ModelSpec};
use crate::multi_asset::{MultiAssetError as Error, MultiAssetPricingPlan, MultiAssetProduct};
use crate::product::{
    EuropeanVanillaSpec, SourceGraphBuilder, SourceOpcode as Op, multi_asset::option,
};
use crate::{Estimate, EstimatorKind};
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
}

#[derive(Clone, Debug)]
pub struct CrossCurrencyEquityOption {
    valuation_date: Date,
    product: EuropeanVanillaSpec,
    payment_date: Date,
    settlement: EquityFxSettlement,
    domestic_curve: Arc<LogLinearDiscountCurve>,
    time: f64,
    foreign_equity_forward: f64,
    fx_forward: PositiveF64,
    dynamics: EquityFxDynamics,
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
        if equity
            .discrete_dividends()
            .is_some_and(|d| !d.events().is_empty())
        {
            return Err(Error::Invalid(
                "joint lognormal equity/FX requires continuous dividend carry",
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
            payment_date,
            settlement,
            domestic_curve: settlement_curve,
            time,
            foreign_equity_forward: equity.forward(time)?,
            fx_forward,
            dynamics,
        })
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
        let equity_vol = self.dynamics.equity.volatility().get();
        let fx_vol = self.dynamics.fx.volatility().get();
        let rho = self.dynamics.correlation;
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
            EquityFxSettlement::Composite => {
                // Orthogonal Brownian loadings also handle rho=-1 and equal vols.
                let joint_vol =
                    (equity_vol + rho * fx_vol).hypot(fx_vol * ((1.0 - rho) * (1.0 + rho)).sqrt());
                (
                    self.foreign_equity_forward * self.fx_forward.get(),
                    joint_vol,
                    1.0,
                )
            }
        };
        let payment_time =
            DayCountConvention::Act365F.year_fraction(self.valuation_date, self.payment_date);
        let result = evaluate_black_forward(BlackForwardOracleInputs {
            side: self.product.side(),
            strike: self.product.strike().get(),
            notional: self.product.notional().get() * conversion,
            discount: self.domestic_curve.discount(payment_time)?,
            forward,
            volatility,
            time: self.time,
        })
        .map_err(Error::numerical)?;
        Ok(result.price)
    }

    /// Price the contractual payoff on correlated terminal equity and FX draws.
    /// The native engine supplies pseudo-MC or RQMC errors and discounting.
    pub fn price_mc(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
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
            self.product.side(),
            self.product.strike().get(),
            self.product.notional().get(),
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
