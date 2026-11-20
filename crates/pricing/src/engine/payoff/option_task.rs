//! Lower holder exercise and continuous monitoring to native MC products.
use crate::contract::{
    AdvancedMcSettings, BarrierDirection, Contract, ContractError, OptionFeature, Term,
};

impl Contract {
    /// Build a native MC task. Dates in an American grid are an explicit
    /// numerical approximation and never restrict legal lifecycle notices.
    pub fn pricing_task(
        &self,
        template: &crate::PricingSpecification,
        operation: crate::PricingOperation,
        settings: &AdvancedMcSettings,
    ) -> Result<crate::PricingTask, ContractError> {
        use crate::product::{
            AmericanVanillaSpec, BarrierDirection as NativeDirection, BarrierMonitoring,
            BarrierSpec, ProductSpec,
        };
        let as_of = template.valuation_date();
        let Term::VanillaOption { option } = self.root() else {
            if settings.lsm.is_some()
                || settings.american_grid.is_some()
                || settings.brownian_bridge
            {
                return Err(ContractError::Invalid(
                    "advanced settings require a standalone exercise or continuous barrier option"
                        .into(),
                ));
            }
            return self
                .compile_mc(as_of, vec![template.product().underlying()])?
                .pricing_task(template, operation);
        };
        let expiry = self
            .observations()
            .iter()
            .find(|o| o.id == option.expiry_observation)
            .ok_or_else(|| ContractError::Invalid("unknown expiry fixing".into()))?;
        if expiry.underlying != template.product().underlying()
            || option.currency != template.product().currency()
        {
            return Err(ContractError::Invalid(
                "template underlying or currency differs from option".into(),
            ));
        }
        if !matches!(template.model().rates, crate::RateModelSpec::Deterministic)
            || !matches!(
                template.model().equity,
                crate::EquityModelSpec::Standard(
                    crate::models::ModelSpec::BlackScholes(_)
                        | crate::models::ModelSpec::Black76(_)
                        | crate::models::ModelSpec::LocalVolatility(_)
                )
            )
        {
            return Err(ContractError::Unsupported("native option policy requires deterministic rates and a constant-volatility or local-volatility model".into()));
        }
        let mut numerics = template.numerics().clone();
        let unsupported = |e: crate::core::CoreError| ContractError::Invalid(e.to_string());
        let product = match &option.feature {
            OptionFeature::ContinuousBarrier(b) => {
                if !settings.brownian_bridge
                    || settings.american_grid.is_some()
                    || settings.lsm.is_some()
                {
                    return Err(ContractError::Invalid(
                        "select the native Brownian bridge explicitly for continuous monitoring"
                            .into(),
                    ));
                }
                if as_of < b.start {
                    return Err(ContractError::Unsupported(
                        "native continuous bridge starts on the valuation date".into(),
                    ));
                }
                if as_of > b.start {
                    return Err(ContractError::MissingFixing(
                        "advance with confirmed continuous barrier status through valuation".into(),
                    ));
                }
                let direction = match b.direction {
                    BarrierDirection::Up => NativeDirection::Up,
                    BarrierDirection::Down => NativeDirection::Down,
                };
                ProductSpec::Barrier(
                    BarrierSpec::new(
                        expiry.underlying,
                        option.currency,
                        expiry.date,
                        option.strike,
                        b.level,
                        option.notional,
                        option.side,
                        direction,
                        b.style,
                        BarrierMonitoring::Continuous,
                        vec![b.end],
                        (b.rebate > 0.0).then_some(b.rebate),
                        b.payment_date,
                    )
                    .map_err(unsupported)?,
                )
            }
            exercise => {
                if settings.brownian_bridge {
                    return Err(ContractError::Invalid(
                        "bridge setting on exercise option".into(),
                    ));
                }
                let dates = match exercise {
                    OptionFeature::Bermudan { dates } => {
                        if settings.american_grid.is_some() {
                            return Err(ContractError::Invalid(
                                "Bermudan dates are contractual".into(),
                            ));
                        }
                        dates
                    }
                    OptionFeature::American { start } => {
                        let grid = settings.american_grid.as_ref().ok_or_else(|| {
                            ContractError::Invalid(
                                "American option requires an explicit discrete exercise grid"
                                    .into(),
                            )
                        })?;
                        if grid.first().is_some_and(|date| date < start) {
                            return Err(ContractError::Invalid(
                                "exercise grid starts before the legal window".into(),
                            ));
                        }
                        grid
                    }
                    OptionFeature::ContinuousBarrier(_) => unreachable!(),
                };
                if let Some(lsm) = &settings.lsm {
                    numerics.lsm = Some(lsm.clone());
                }
                ProductSpec::AmericanVanilla(
                    AmericanVanillaSpec::new(
                        expiry.underlying,
                        option.currency,
                        expiry.date,
                        option.strike,
                        option.notional,
                        option.side,
                        dates.clone(),
                    )
                    .map_err(unsupported)?,
                )
            }
        };
        let spec = crate::PricingSpecification::new(
            as_of,
            product,
            template.market().clone(),
            template.model().clone(),
            template.engine(),
            template.risk().clone(),
            numerics,
        )
        .map_err(|e| ContractError::Unsupported(e.to_string()))?;
        crate::PricingTask::new(crate::PricingSource::SingleAsset(spec), operation)
            .map_err(|e| ContractError::Unsupported(e.to_string()))
    }
}
