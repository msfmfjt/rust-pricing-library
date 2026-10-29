//! Bounded microbenchmark of the conditional-payoff compiler's public entry point.
//! Each case preserves its graph and checks the entire signed Gamma measure.
use pricing::core::{Date, UnderlyingId};
use pricing::product::{CompiledPayoff, GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op};
use pricing::risk::terminal_payoff_gamma_measure;
use std::hint::black_box;
use std::time::Instant;

fn graph(intervals: usize, lookback: bool) -> CompiledPayoff {
    let mut b = SourceGraphBuilder::new();
    let spot = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: expiry(),
        })
        .unwrap();
    let mut total = b.literal(0.).unwrap();
    for i in 0..intervals {
        let scale = b.literal(0.7 + 0.1 * i as f64).unwrap();
        let cash = b.literal(3. * i as f64).unwrap();
        let scaled = b
            .push(Op::Multiply {
                left: spot,
                right: scale,
            })
            .unwrap();
        let observation = b
            .push(Op::Add {
                left: scaled,
                right: cash,
            })
            .unwrap();
        total = b
            .push(if lookback {
                Op::Maximum {
                    left: total,
                    right: observation,
                }
            } else {
                Op::Add {
                    left: total,
                    right: observation,
                }
            })
            .unwrap();
    }
    if !lookback {
        let count = b.literal(intervals as f64).unwrap();
        total = b
            .push(Op::Divide {
                numerator: total,
                denominator: count,
            })
            .unwrap();
    }
    let strike = b.literal(100.).unwrap();
    let zero = b.literal(0.).unwrap();
    let intrinsic = b
        .push(Op::Subtract {
            left: total,
            right: strike,
        })
        .unwrap();
    let call = b
        .push(Op::Maximum {
            left: intrinsic,
            right: zero,
        })
        .unwrap();
    b.finish(vec![call])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap()
}

fn expiry() -> Date {
    "2027-01-01".parse().unwrap()
}

fn main() {
    let repetitions = std::env::args()
        .nth(1)
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(20_000);
    assert!(repetitions > 0);
    let u = UnderlyingId::new(1);
    let date = expiry();
    for intervals in [4, 6] {
        for lookback in [false, true] {
            let graph = graph(intervals, lookback);
            let expected = terminal_payoff_gamma_measure(&graph, u, date).unwrap();
            let started = Instant::now();
            for _ in 0..repetitions {
                let result = terminal_payoff_gamma_measure(black_box(&graph), u, date).unwrap();
                assert_eq!(result, expected);
                black_box(result);
            }
            println!(
                "{}",
                serde_json::json!({
                    "intervals": intervals, "lookback": lookback,
                    "opcodes": graph.opcodes().len(), "repetitions": repetitions,
                    "elapsed_seconds": started.elapsed().as_secs_f64(),
                    "payoff": format!("{expected:?}"),
                })
            );
        }
    }
}
