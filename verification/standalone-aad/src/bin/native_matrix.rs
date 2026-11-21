//! Compile the shared numeric program using the original Pricing payoff compiler.
use pricing::core::{Date, UnderlyingId};
use pricing::product::{
    CompiledPayoff, GraphComparison, GraphLimitPolicy, PayoffWorkspace, SourceGraphBuilder,
    SourceOpcode as Native,
};
use rust_aad_bench::{
    Backend,
    fixture::{Op, Program},
    run,
};

struct Pricing;
impl Backend for Pricing {
    const NAME: &'static str = "pricing-native";
    type Graph = (CompiledPayoff, usize);
    type Work = (PayoffWorkspace, Vec<f64>);

    fn compile(p: &Program) -> Self::Graph {
        let mut b = SourceGraphBuilder::new();
        let mut slots = Vec::with_capacity(p.nodes.len());
        let mut inputs = 0;
        let date: Date = "2027-01-04".parse().unwrap();
        for op in &p.nodes {
            let id = match *op {
                Op::Constant(v) => b.literal(v),
                Op::Input(i) => {
                    assert_eq!(i, inputs);
                    inputs += 1;
                    b.push(Native::TerminalSpot {
                        underlying: UnderlyingId::new(i as u32),
                        observation_date: date,
                    })
                }
                Op::Add(i, j) => b.push(Native::Add {
                    left: slots[i],
                    right: slots[j],
                }),
                Op::Sub(i, j) => b.push(Native::Subtract {
                    left: slots[i],
                    right: slots[j],
                }),
                Op::Mul(i, j) => b.push(Native::Multiply {
                    left: slots[i],
                    right: slots[j],
                }),
                Op::Max(i, j) => b.push(Native::Maximum {
                    left: slots[i],
                    right: slots[j],
                }),
                Op::Greater(i, j) => b.push(Native::Compare {
                    left: slots[i],
                    right: slots[j],
                    operation: GraphComparison::Greater,
                }),
                Op::Neg(i) => b.push(Native::Negate { input: slots[i] }),
                Op::Log(i) => b.push(Native::Log { input: slots[i] }),
                Op::Select(i, j, k) => b.push(Native::Select {
                    condition: slots[i],
                    then_value: slots[j],
                    else_value: slots[k],
                }),
            }
            .unwrap();
            slots.push(id);
        }
        (
            b.finish(p.outputs.iter().map(|&i| slots[i]).collect())
                .compile(GraphLimitPolicy::DEFAULT)
                .unwrap(),
            inputs,
        )
    }
    fn prepare((graph, n): &Self::Graph) -> Self::Work {
        (graph.workspace(), vec![0.; *n])
    }
    fn forward<'a>((g, _): &Self::Graph, w: &'a mut Self::Work, x: &[f64]) -> &'a [f64] {
        g.evaluate_reusing(&mut w.0, |u, _| Some(x[u.get() as usize]), |_, _| None)
            .unwrap()
    }
    fn vjp<'a>((g, _): &Self::Graph, w: &'a mut Self::Work, x: &[f64], seeds: &[f64]) -> &'a [f64] {
        w.1.fill(0.);
        // Public native API has one reverse sweep per output and returns owned adjoints.
        for (i, seed) in seeds.iter().enumerate() {
            let adj = g
                .evaluate_output_with_observation_adjoints(
                    i,
                    |u, _| Some(x[u.get() as usize]),
                    |_, _| None,
                )
                .unwrap();
            for a in adj.terminal_adjoints {
                w.1[a.underlying.get() as usize] += seed * a.value;
            }
        }
        &w.1
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [backend, id, paths] = args.as_slice() else {
        panic!("usage: native_matrix pricing-native WORKLOAD PATHS")
    };
    assert_eq!(backend, Pricing::NAME);
    println!("{}", run::<Pricing>(id, paths.parse().unwrap()));
}
