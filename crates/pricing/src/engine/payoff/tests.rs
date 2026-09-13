use crate::core::{Date, FiniteF64, NodeId, UnderlyingId};
use crate::engine::payoff::tape::CompiledOpcode;
use crate::engine::payoff::tape::CompiledPayoff;
use crate::product::graph::GraphError;
use crate::product::graph::GraphLimitPolicy;
use crate::product::graph::SourceGraph;
use crate::product::graph::SourceGraphBuilder;
use crate::product::graph::SourceNode;
use crate::product::graph::SourceOpcode;
use crate::product::{
    AmericanVanillaSpec, ArithmeticAsianSpec, BarrierDirection, BarrierSpec, BarrierStyle,
    CompactC2Smoothing, DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec,
    OptionSide,
};

#[cfg(test)]
mod tests {
    use crate::core::CurrencyId;

    use super::*;
    use crate::product::BarrierMonitoring;

    fn option(side: OptionSide) -> EuropeanVanillaSpec {
        EuropeanVanillaSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("date"),
            100.0,
            2.0,
            side,
        )
        .expect("option")
    }

    #[test]
    fn european_builder_executes_exact_call_and_put_payoffs() {
        for (side, terminal, expected) in [
            (OptionSide::Call, 120.0, 40.0),
            (OptionSide::Call, 80.0, 0.0),
            (OptionSide::Put, 80.0, 40.0),
            (OptionSide::Put, 120.0, 0.0),
        ] {
            let compiled = option(side)
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.evaluate(|_, _| Some(terminal)).expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn american_builder_emits_intrinsic_values_in_exercise_order() {
        let first: Date = "2027-03-04".parse().expect("first");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = AmericanVanillaSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            expiry,
            100.0,
            2.0,
            OptionSide::Put,
            vec![first, expiry],
        )
        .expect("American option");
        let compiled = product
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        let values = compiled
            .evaluate(|_, observation_date| match observation_date {
                value if value == first => Some(90.0),
                value if value == expiry => Some(80.0),
                _ => None,
            })
            .expect("execute");
        assert_eq!(values, [20.0, 40.0]);

        let terminal_only = AmericanVanillaSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            expiry,
            100.0,
            2.0,
            OptionSide::Put,
            vec![expiry],
        )
        .expect("terminal-only option")
        .source_graph()
        .expect("terminal graph")
        .compile(GraphLimitPolicy::DEFAULT)
        .expect("compile terminal graph");
        assert_ne!(
            compiled.source_fingerprint(),
            terminal_only.source_fingerprint()
        );
    }

    #[test]
    fn american_reverse_selects_one_exercise_output() {
        let dates = [
            "2027-03-04".parse().expect("first"),
            "2027-09-04".parse().expect("expiry"),
        ];
        let compiled = AmericanVanillaSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            dates[1],
            100.0,
            2.0,
            OptionSide::Put,
            dates.to_vec(),
        )
        .expect("American option")
        .source_graph()
        .expect("graph")
        .compile(GraphLimitPolicy::DEFAULT)
        .expect("compile");
        let spots = [90.0, 80.0];
        for (output_index, expected_value) in [20.0, 40.0].into_iter().enumerate() {
            let evaluation = compiled
                .evaluate_output_with_observation_adjoints(
                    output_index,
                    |_, date| {
                        dates
                            .iter()
                            .position(|candidate| *candidate == date)
                            .map(|index| spots[index])
                    },
                    |_, _| None,
                )
                .expect("selected reverse");
            assert_eq!(evaluation.value, expected_value);
            assert_eq!(evaluation.terminal_adjoints.len(), 2);
            for adjoint in &evaluation.terminal_adjoints {
                let date_index = dates
                    .iter()
                    .position(|candidate| *candidate == adjoint.observation_date)
                    .expect("American observation date");
                assert_eq!(
                    adjoint.value,
                    if date_index == output_index {
                        -2.0
                    } else {
                        0.0
                    }
                );
            }
        }
        assert_eq!(
            compiled.evaluate_output_with_observation_adjoints(2, |_, _| Some(90.0), |_, _| None),
            Err(GraphError::InvalidOutputIndex { index: 2, count: 2 })
        );
        assert_eq!(
            compiled.evaluate_single_with_terminal_adjoint(|_, _| Some(90.0)),
            Err(GraphError::ReverseRequiresSingleOutput { count: 2 })
        );
    }

    #[test]
    fn european_reverse_returns_exact_terminal_adjoint() {
        for (side, terminal, expected_value, expected_adjoint) in [
            (OptionSide::Call, 120.0, 40.0, 2.0),
            (OptionSide::Call, 80.0, 0.0, 0.0),
            (OptionSide::Put, 80.0, 40.0, -2.0),
            (OptionSide::Put, 120.0, 0.0, 0.0),
            (OptionSide::Call, 100.0, 0.0, 2.0),
            (OptionSide::Put, 100.0, 0.0, -2.0),
        ] {
            let compiled = option(side)
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let result = compiled
                .evaluate_single_with_terminal_adjoint(|_, _| Some(terminal))
                .expect("reverse");
            assert_eq!(result.value, expected_value);
            assert_eq!(result.terminal_adjoints.len(), 1);
            assert_eq!(result.terminal_adjoints[0].value, expected_adjoint);
            assert_eq!(result.terminal_adjoints[0].underlying, UnderlyingId::new(4));
            assert_eq!(
                result.terminal_adjoints[0].observation_date,
                "2027-09-04".parse().expect("date")
            );
        }
    }

    #[test]
    fn digital_builder_executes_exact_cash_and_asset_payoffs() {
        for (side, payout_kind, terminal, expected) in [
            (OptionSide::Call, DigitalPayout::Cash, 120.0, 10.0),
            (OptionSide::Call, DigitalPayout::Cash, 99.0, 0.0),
            (OptionSide::Call, DigitalPayout::Cash, 100.0, 10.0),
            (OptionSide::Put, DigitalPayout::Cash, 80.0, 10.0),
            (OptionSide::Put, DigitalPayout::Cash, 101.0, 0.0),
            (OptionSide::Put, DigitalPayout::Cash, 100.0, 10.0),
            (OptionSide::Call, DigitalPayout::Asset, 120.0, 1200.0),
            (OptionSide::Put, DigitalPayout::Asset, 80.0, 800.0),
        ] {
            let product = DigitalSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("date"),
                100.0,
                10.0,
                side,
                payout_kind,
            )
            .expect("digital");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.evaluate(|_, _| Some(terminal)).expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn digital_builder_executes_smoothed_cash_and_asset_payoffs_with_adjoints() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        for (side, payout_kind, expected_value, expected_adjoint) in [
            (OptionSide::Call, DigitalPayout::Cash, 5.0, 4.6875),
            (OptionSide::Put, DigitalPayout::Cash, 5.0, -4.6875),
            (OptionSide::Call, DigitalPayout::Asset, 500.0, 473.75),
            (OptionSide::Put, DigitalPayout::Asset, 500.0, -463.75),
        ] {
            let product = DigitalSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("date"),
                100.0,
                10.0,
                side,
                payout_kind,
            )
            .expect("digital");
            let compiled = product
                .smoothed_source_graph(smoothing)
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let result = compiled
                .evaluate_single_with_terminal_adjoint(|_, _| Some(100.0))
                .expect("evaluate");
            assert_eq!(result.value, expected_value);
            assert_eq!(result.terminal_adjoints.len(), 1);
            assert_eq!(result.terminal_adjoints[0].value, expected_adjoint);
        }
    }

    #[test]
    fn digital_exact_and_smoothed_graphs_remain_distinct() {
        let product = DigitalSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("date"),
            100.0,
            10.0,
            OptionSide::Call,
            DigitalPayout::Cash,
        )
        .expect("digital");
        let exact = product
            .source_graph()
            .expect("exact graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("exact compile");
        let smoothed = product
            .smoothed_source_graph(CompactC2Smoothing::new(2.0).expect("smoothing"))
            .expect("smoothed graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("smoothed compile");
        assert_ne!(exact.source_fingerprint(), smoothed.source_fingerprint());
        assert_ne!(exact.tape_fingerprint(), smoothed.tape_fingerprint());
        assert_eq!(
            exact.evaluate(|_, _| Some(100.0)).expect("exact"),
            vec![10.0]
        );
        assert_eq!(
            smoothed.evaluate(|_, _| Some(100.0)).expect("smoothed"),
            vec![5.0]
        );
    }

    #[test]
    fn barrier_builder_executes_discrete_knock_out_and_knock_in_payoffs() {
        for (style, march_spot, expiry_spot, expected) in [
            (BarrierStyle::KnockOut, 110.0, 115.0, 30.0),
            (BarrierStyle::KnockOut, 125.0, 115.0, 0.0),
            (BarrierStyle::KnockIn, 110.0, 115.0, 0.0),
            (BarrierStyle::KnockIn, 125.0, 115.0, 30.0),
        ] {
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                style,
                BarrierMonitoring::Discrete,
                vec![
                    "2027-03-04".parse().expect("monitoring"),
                    "2027-09-04".parse().expect("expiry"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
            )
            .expect("barrier");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled
                    .evaluate(|_, date| match date.to_string().as_str() {
                        "2027-03-04" => Some(march_spot),
                        "2027-09-04" => Some(expiry_spot),
                        _ => None,
                    })
                    .expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn barrier_builder_pays_rebate_when_vanilla_branch_is_inactive() {
        for (style, march_spot, expected) in [
            (BarrierStyle::KnockOut, 125.0, 7.0),
            (BarrierStyle::KnockIn, 110.0, 7.0),
        ] {
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                style,
                BarrierMonitoring::Discrete,
                vec![
                    "2027-03-04".parse().expect("monitoring"),
                    "2027-09-04".parse().expect("expiry"),
                ],
                Some(7.0),
                "2027-09-04".parse().expect("payment"),
            )
            .expect("barrier");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled
                    .evaluate(|_, date| match date.to_string().as_str() {
                        "2027-03-04" => Some(march_spot),
                        "2027-09-04" => Some(115.0),
                        _ => None,
                    })
                    .expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn barrier_hit_state_is_monotone_inclusive_and_limit_checked() {
        let product = BarrierSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            120.0,
            2.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockIn,
            BarrierMonitoring::Discrete,
            vec![
                "2027-03-04".parse().expect("monitoring"),
                "2027-09-04".parse().expect("expiry"),
            ],
            None,
            "2027-09-04".parse().expect("payment"),
        )
        .expect("barrier");
        let graph = product.source_graph().expect("graph");
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.opcode(), SourceOpcode::BarrierHitState { .. }))
                .count(),
            2
        );
        let compiled = graph.compile(GraphLimitPolicy::DEFAULT).expect("compile");
        let monitoring = "2027-03-04".parse().expect("monitoring");
        let expiry = "2027-09-04".parse().expect("expiry");
        assert_eq!(
            compiled
                .evaluate(|_, date| (date == monitoring)
                    .then_some(120.0)
                    .or_else(|| { (date == expiry).then_some(110.0) }))
                .expect("inclusive touch"),
            vec![20.0]
        );
        let mut limits = GraphLimitPolicy::DEFAULT;
        limits.state_slots = 1;
        assert!(matches!(
            graph.compile(limits),
            Err(GraphError::SoftLimitExceeded {
                field: "state_slots",
                observed: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn smoothed_barrier_knock_parity_and_reverse_hold_for_up_and_down() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        let monitoring = "2027-03-04".parse().expect("monitoring");
        let expiry = "2027-09-04".parse().expect("expiry");
        for (direction, first_spot, terminal_spot) in [
            (BarrierDirection::Up, 119.5, 121.0),
            (BarrierDirection::Down, 80.5, 79.0),
        ] {
            let barrier = match direction {
                BarrierDirection::Up => 120.0,
                BarrierDirection::Down => 80.0,
            };
            let compile = |style, rebate| {
                BarrierSpec::new(
                    UnderlyingId::new(4),
                    CurrencyId::new(1),
                    expiry,
                    70.0,
                    barrier,
                    2.0,
                    OptionSide::Call,
                    direction,
                    style,
                    BarrierMonitoring::Discrete,
                    vec![monitoring, expiry],
                    rebate,
                    expiry,
                )
                .expect("barrier")
                .smoothed_source_graph(smoothing)
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
            };
            let knock_in = compile(BarrierStyle::KnockIn, Some(7.0));
            let knock_out = compile(BarrierStyle::KnockOut, Some(7.0));
            let observe = |date| {
                if date == monitoring {
                    Some(first_spot)
                } else if date == expiry {
                    Some(terminal_spot)
                } else {
                    None
                }
            };
            let knock_in_value = knock_in
                .evaluate(|_, date| observe(date))
                .expect("knock in")[0];
            let knock_out_value = knock_out
                .evaluate(|_, date| observe(date))
                .expect("knock out")[0];
            let vanilla = (terminal_spot - 70.0) * 2.0;
            assert!((knock_in_value + knock_out_value - (vanilla + 7.0)).abs() < 1.0e-12);

            let evaluated = knock_in
                .evaluate_single_with_terminal_adjoint(|_, date| observe(date))
                .expect("reverse");
            for date in [monitoring, expiry] {
                let epsilon = 1.0e-5;
                let bumped = |shift| {
                    knock_in
                        .evaluate(|_, query| {
                            if query == monitoring {
                                Some(first_spot + if date == monitoring { shift } else { 0.0 })
                            } else if query == expiry {
                                Some(terminal_spot + if date == expiry { shift } else { 0.0 })
                            } else {
                                None
                            }
                        })
                        .expect("bump")[0]
                };
                let finite_difference = (bumped(epsilon) - bumped(-epsilon)) / (2.0 * epsilon);
                let adjoint = evaluated
                    .terminal_adjoints
                    .iter()
                    .filter(|item| item.observation_date == date)
                    .map(|item| item.value)
                    .sum::<f64>();
                assert!((adjoint - finite_difference).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn smoothed_barrier_dividend_jump_matches_p0_fixture_and_reverse() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/path-dependence/reference-cases-v0.1.json"
        ))
        .expect("fixture");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        for case in fixture["jump_cases"].as_array().expect("jump cases") {
            let parse = |field: &str| {
                case[field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let expected = |field: &str| {
                case["expected"][field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let pre = parse("pre_jump_spot");
            let post = parse("post_jump_spot");
            let smoothing = CompactC2Smoothing::new(parse("half_width")).expect("smoothing");
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                expiry,
                1.0,
                parse("barrier"),
                1.0,
                OptionSide::Call,
                BarrierDirection::Down,
                BarrierStyle::KnockIn,
                BarrierMonitoring::Discrete,
                vec![expiry],
                None,
                expiry,
            )
            .expect("barrier");
            let compiled = product
                .smoothed_source_graph_with_dividend_jumps(smoothing, &[expiry])
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.pre_dividend_observations(),
                vec![(UnderlyingId::new(4), expiry)]
            );
            let value = compiled
                .evaluate_with_pre_dividend_spots(|_, _| Some(post), |_, _| Some(pre))
                .expect("evaluate")[0];
            assert!((value / (post - 1.0) - expected("hit_weight")).abs() < 2.0e-15);

            let evaluation = compiled
                .evaluate_single_with_observation_adjoints(|_, _| Some(post), |_, _| Some(pre))
                .expect("reverse");
            let epsilon = 1.0e-5;
            let finite_difference = |bump_pre: bool| {
                let bumped = |shift| {
                    compiled
                        .evaluate_with_pre_dividend_spots(
                            |_, _| Some(post + if bump_pre { 0.0 } else { shift }),
                            |_, _| Some(pre + if bump_pre { shift } else { 0.0 }),
                        )
                        .expect("bump")[0]
                };
                (bumped(epsilon) - bumped(-epsilon)) / (2.0 * epsilon)
            };
            let pre_adjoint = evaluation
                .pre_dividend_adjoints
                .iter()
                .map(|adjoint| adjoint.value)
                .sum::<f64>();
            let post_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .map(|adjoint| adjoint.value)
                .sum::<f64>();
            assert!((pre_adjoint - finite_difference(true)).abs() < 1.0e-7);
            assert!((post_adjoint - finite_difference(false)).abs() < 1.0e-7);
        }
    }

    #[test]
    fn arithmetic_asian_builder_executes_weighted_average_payoff() {
        let product = ArithmeticAsianSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            100.0,
            2.0,
            OptionSide::Call,
            vec![
                crate::product::AsianObservation::known(
                    "2027-03-04".parse().expect("date"),
                    0.25,
                    90.0,
                )
                .expect("known"),
                crate::product::AsianObservation::unknown(
                    "2027-09-04".parse().expect("date"),
                    0.75,
                )
                .expect("unknown"),
            ],
            "2027-09-04".parse().expect("payment"),
        )
        .expect("asian");
        let compiled = product
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled
                .evaluate(|_, date| (date.to_string() == "2027-09-04").then_some(130.0))
                .expect("execute"),
            vec![40.0]
        );
        assert_eq!(
            compiled.terminal_observations(),
            vec![(UnderlyingId::new(4), "2027-09-04".parse().expect("date"))]
        );
    }

    #[test]
    fn arithmetic_asian_is_invariant_to_weighted_observation_pair_permutation() {
        let dates = [
            "2027-03-04".parse().expect("first"),
            "2027-06-04".parse().expect("second"),
            "2027-09-04".parse().expect("third"),
        ];
        let compile = |weights: [f64; 3]| {
            ArithmeticAsianSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                100.0,
                2.0,
                OptionSide::Call,
                dates
                    .into_iter()
                    .zip(weights)
                    .map(|(date, weight)| {
                        crate::product::AsianObservation::unknown(date, weight)
                            .expect("observation")
                    })
                    .collect(),
                dates[2],
            )
            .expect("Asian")
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile")
        };
        let first = compile([0.2, 0.3, 0.5]);
        let permuted = compile([0.5, 0.2, 0.3]);
        let first_spots = [80.0, 110.0, 130.0];
        let permuted_spots = [130.0, 80.0, 110.0];
        let evaluate = |compiled: &CompiledPayoff, spots: [f64; 3]| {
            compiled
                .evaluate(|_, date| {
                    dates
                        .iter()
                        .position(|candidate| *candidate == date)
                        .map(|index| spots[index])
                })
                .expect("evaluate")[0]
        };
        assert_eq!(
            evaluate(&first, first_spots),
            evaluate(&permuted, permuted_spots)
        );
    }

    #[test]
    fn fixed_lookback_builder_executes_running_extremum_payoff() {
        let product = FixedLookbackSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            100.0,
            2.0,
            OptionSide::Call,
            vec![
                "2027-03-04".parse().expect("date"),
                "2027-06-04".parse().expect("date"),
                "2027-09-04".parse().expect("date"),
            ],
            Some(112.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("lookback");
        let compiled = product
            .source_graph("2027-05-04".parse().expect("valuation"))
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled
                .evaluate(|_, date| match date.to_string().as_str() {
                    "2027-06-04" => Some(95.0),
                    "2027-09-04" => Some(118.0),
                    _ => None,
                })
                .expect("execute"),
            vec![36.0]
        );
        assert_eq!(
            compiled.terminal_observations(),
            vec![
                (UnderlyingId::new(4), "2027-06-04".parse().expect("date")),
                (UnderlyingId::new(4), "2027-09-04".parse().expect("date")),
            ]
        );
    }

    #[test]
    fn fixed_lookback_payoff_is_monotone_under_monitoring_refinement() {
        let dates = [
            "2027-03-04".parse().expect("first"),
            "2027-06-04".parse().expect("second"),
            "2027-09-04".parse().expect("third"),
        ];
        for (side, middle_spot) in [(OptionSide::Call, 150.0), (OptionSide::Put, 70.0)] {
            let compile = |monitoring_dates: Vec<Date>| {
                FixedLookbackSpec::new(
                    UnderlyingId::new(4),
                    CurrencyId::new(1),
                    100.0,
                    2.0,
                    side,
                    monitoring_dates,
                    None,
                    dates[2],
                )
                .expect("Lookback")
                .source_graph("2026-09-04".parse().expect("valuation"))
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
            };
            let coarse = compile(vec![dates[0], dates[2]]);
            let refined = compile(dates.to_vec());
            let observe = |date| {
                if date == dates[0] {
                    Some(90.0)
                } else if date == dates[1] {
                    Some(middle_spot)
                } else if date == dates[2] {
                    Some(130.0)
                } else {
                    None
                }
            };
            let coarse_value = coarse.evaluate(|_, date| observe(date)).expect("coarse")[0];
            let refined_value = refined.evaluate(|_, date| observe(date)).expect("refined")[0];
            assert!(refined_value >= coarse_value);
        }
    }

    #[test]
    fn manual_and_standard_graphs_have_identical_fingerprints() {
        let contract = option(OptionSide::Call);
        let standard = contract.source_graph().expect("standard graph");
        let nodes = vec![
            SourceNode::new(
                NodeId::new(0),
                SourceOpcode::TerminalSpot {
                    underlying: contract.underlying(),
                    observation_date: contract.expiry(),
                },
            ),
            SourceNode::new(
                NodeId::new(1),
                SourceOpcode::Literal(
                    FiniteF64::new(contract.strike().get(), "strike").expect("strike"),
                ),
            ),
            SourceNode::new(
                NodeId::new(2),
                SourceOpcode::Subtract {
                    left: NodeId::new(0),
                    right: NodeId::new(1),
                },
            ),
            SourceNode::new(
                NodeId::new(3),
                SourceOpcode::Literal(FiniteF64::new(0.0, "zero").expect("zero")),
            ),
            SourceNode::new(
                NodeId::new(4),
                SourceOpcode::Maximum {
                    left: NodeId::new(2),
                    right: NodeId::new(3),
                },
            ),
            SourceNode::new(
                NodeId::new(5),
                SourceOpcode::Literal(
                    FiniteF64::new(contract.notional().get(), "notional").expect("notional"),
                ),
            ),
            SourceNode::new(
                NodeId::new(6),
                SourceOpcode::Multiply {
                    left: NodeId::new(4),
                    right: NodeId::new(5),
                },
            ),
        ];
        let manual = SourceGraph::new(nodes, vec![NodeId::new(6)]);
        let standard = standard
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("standard");
        let manual = manual.compile(GraphLimitPolicy::DEFAULT).expect("manual");
        assert_eq!(standard.source_fingerprint(), manual.source_fingerprint());
        assert_eq!(standard.tape_fingerprint(), manual.tape_fingerprint());
    }

    #[test]
    fn input_order_does_not_change_kahn_order_or_fingerprints() {
        let graph = option(OptionSide::Put).source_graph().expect("graph");
        let mut reversed = graph.nodes().to_vec();
        reversed.reverse();
        let reordered = SourceGraph::new(reversed, graph.outputs().to_vec());
        let left = graph.compile(GraphLimitPolicy::DEFAULT).expect("left");
        let right = reordered.compile(GraphLimitPolicy::DEFAULT).expect("right");
        assert_eq!(left.source_fingerprint(), right.source_fingerprint());
        assert_eq!(left.tape_fingerprint(), right.tape_fingerprint());
        assert_eq!(left.opcodes(), right.opcodes());
    }

    #[test]
    fn dead_nodes_are_validated_before_removal() {
        let one = FiniteF64::new(1.0, "one").expect("one");
        let zero = FiniteF64::new(0.0, "zero").expect("zero");
        let invalid_dead = SourceGraph::new(
            vec![
                SourceNode::new(NodeId::new(0), SourceOpcode::Literal(one)),
                SourceNode::new(NodeId::new(1), SourceOpcode::Literal(zero)),
                SourceNode::new(
                    NodeId::new(2),
                    SourceOpcode::Divide {
                        numerator: NodeId::new(0),
                        denominator: NodeId::new(1),
                    },
                ),
            ],
            vec![NodeId::new(0)],
        );
        assert!(matches!(
            invalid_dead.compile(GraphLimitPolicy::DEFAULT),
            Err(GraphError::NonFiniteConstantFold {
                node,
                opcode: "divide",
                ..
            }) if node == NodeId::new(2)
        ));

        let valid_dead = SourceGraph::new(
            vec![
                SourceNode::new(NodeId::new(0), SourceOpcode::Literal(one)),
                SourceNode::new(NodeId::new(9), SourceOpcode::Literal(zero)),
            ],
            vec![NodeId::new(0)],
        )
        .compile(GraphLimitPolicy::DEFAULT)
        .expect("compile");
        assert_eq!(valid_dead.removed_source_nodes(), &[NodeId::new(9)]);
    }

    #[test]
    fn cycles_duplicate_ids_unknown_references_and_limits_are_errors() {
        let cycle = SourceGraph::new(
            vec![
                SourceNode::new(
                    NodeId::new(0),
                    SourceOpcode::Negate {
                        input: NodeId::new(1),
                    },
                ),
                SourceNode::new(
                    NodeId::new(1),
                    SourceOpcode::Negate {
                        input: NodeId::new(0),
                    },
                ),
            ],
            vec![NodeId::new(0)],
        );
        assert!(matches!(
            cycle.compile(GraphLimitPolicy::DEFAULT),
            Err(GraphError::Cycle { .. })
        ));

        let mut limits = GraphLimitPolicy::DEFAULT;
        limits.source_nodes = 1;
        assert!(matches!(
            option(OptionSide::Call)
                .source_graph()
                .expect("graph")
                .compile(limits),
            Err(GraphError::SoftLimitExceeded {
                field: "source_nodes",
                ..
            })
        ));
    }

    #[test]
    fn constant_fold_preserves_node_identity_and_signed_zero() {
        let mut builder = SourceGraphBuilder::new();
        let positive = builder.literal(0.0).expect("positive zero");
        let negative = builder.literal(-0.0).expect("negative zero");
        let sum = builder
            .push(SourceOpcode::Add {
                left: positive,
                right: negative,
            })
            .expect("sum");
        let compiled = builder
            .finish(vec![sum])
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(compiled.opcodes().len(), 3);
        assert_eq!(compiled.source_to_slot().len(), 3);

        let positive_only = SourceGraph::new(
            vec![SourceNode::new(
                NodeId::new(0),
                SourceOpcode::Literal(FiniteF64::new(0.0, "value").expect("value")),
            )],
            vec![NodeId::new(0)],
        );
        let negative_only = SourceGraph::new(
            vec![SourceNode::new(
                NodeId::new(0),
                SourceOpcode::Literal(FiniteF64::new(-0.0, "value").expect("value")),
            )],
            vec![NodeId::new(0)],
        );
        assert_ne!(
            positive_only
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
                .source_fingerprint(),
            negative_only
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
                .source_fingerprint()
        );
    }

    #[test]
    fn smooth_indicator_executes_and_reverses_the_p0_kernel() {
        let mut builder = SourceGraphBuilder::new();
        let date = "2027-09-04".parse().expect("date");
        let input = builder
            .push(SourceOpcode::TerminalSpot {
                underlying: UnderlyingId::new(4),
                observation_date: date,
            })
            .expect("input");
        let output = builder
            .push(SourceOpcode::SmoothIndicator {
                input,
                smoothing: CompactC2Smoothing::new(2.0).expect("smoothing"),
            })
            .expect("indicator");
        let compiled = builder
            .finish(vec![output])
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");

        let evaluation = compiled
            .evaluate_single_with_terminal_adjoint(|_, _| Some(0.0))
            .expect("evaluate");
        assert_eq!(evaluation.value, 0.5);
        assert_eq!(evaluation.terminal_adjoints.len(), 1);
        assert_eq!(evaluation.terminal_adjoints[0].value, 0.468_75);

        for (input, expected) in [
            (-3.0, 0.0),
            (-1.0, 0.103_515_625),
            (1.0, 0.896_484_375),
            (3.0, 1.0),
        ] {
            assert_eq!(
                compiled.evaluate(|_, _| Some(input)).expect("evaluate"),
                vec![expected]
            );
        }
    }

    #[test]
    fn smooth_extrema_execute_and_reverse_as_a_paired_partition() {
        let first_date = "2027-03-04".parse().expect("first date");
        let second_date = "2027-09-04".parse().expect("second date");
        for maximum in [true, false] {
            let mut builder = SourceGraphBuilder::new();
            let left = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: first_date,
                })
                .expect("left");
            let right = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: second_date,
                })
                .expect("right");
            let smoothing = CompactC2Smoothing::new(4.0).expect("smoothing");
            let output = if maximum {
                builder.push(SourceOpcode::SmoothMaximum {
                    left,
                    right,
                    smoothing,
                })
            } else {
                builder.push(SourceOpcode::SmoothMinimum {
                    left,
                    right,
                    smoothing,
                })
            }
            .expect("extremum");
            let compiled = builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let evaluation = compiled
                .evaluate_single_with_terminal_adjoint(|_, date| {
                    Some(if date == first_date { 101.0 } else { 99.0 })
                })
                .expect("evaluate");
            assert_eq!(
                evaluation.value,
                if maximum {
                    101.056_640_625
                } else {
                    98.943_359_375
                }
            );
            let left_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .find(|adjoint| adjoint.observation_date == first_date)
                .expect("left adjoint")
                .value;
            let right_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .find(|adjoint| adjoint.observation_date == second_date)
                .expect("right adjoint")
                .value;
            let expected_left = if maximum {
                0.896_484_375
            } else {
                0.103_515_625
            };
            assert_eq!(left_adjoint, expected_left);
            assert_eq!(right_adjoint, 1.0 - expected_left);
        }
    }

    #[test]
    fn smooth_constant_folding_and_fingerprints_include_half_width() {
        let compile = |half_width| {
            let mut builder = SourceGraphBuilder::new();
            let input = builder.literal(0.0).expect("input");
            let output = builder
                .push(SourceOpcode::SmoothIndicator {
                    input,
                    smoothing: CompactC2Smoothing::new(half_width).expect("smoothing"),
                })
                .expect("indicator");
            builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
        };
        let narrow = compile(1.0);
        let wide = compile(2.0);
        assert!(matches!(
            narrow.opcodes()[1],
            CompiledOpcode::Literal { value: 0.5, .. }
        ));
        assert_ne!(narrow.source_fingerprint(), wide.source_fingerprint());

        let compile_dynamic = |half_width| {
            let mut builder = SourceGraphBuilder::new();
            let input = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: "2027-09-04".parse().expect("date"),
                })
                .expect("input");
            let output = builder
                .push(SourceOpcode::SmoothIndicator {
                    input,
                    smoothing: CompactC2Smoothing::new(half_width).expect("smoothing"),
                })
                .expect("indicator");
            builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
        };
        assert_ne!(
            compile_dynamic(1.0).tape_fingerprint(),
            compile_dynamic(2.0).tape_fingerprint()
        );
    }

    #[test]
    fn existing_exact_payoff_fingerprint_is_unchanged() {
        let compiled = option(OptionSide::Call)
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled.tape_fingerprint().to_string(),
            "blake3-256:32f30625f81f30987a54c18c1b62e20549a86959873c2e33f2b68c84e97aabf3"
        );
    }
}
