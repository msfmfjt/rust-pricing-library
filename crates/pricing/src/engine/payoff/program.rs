//! Bounded lazy execution schedule derived from the native SSA tape.
//!
//! Unique dependencies are inlined; shared dependencies become memoized calls.
//! This avoids both per-path graph searches and exponential code expansion for
//! DAGs such as x[n + 1] = x[n] + x[n]. Construction itself is iterative.
use super::tape::CompiledOpcode;
use crate::product::GraphError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Instruction {
    Evaluate(u32),
    Copy {
        input: u32,
        output: u32,
    },
    ShortcutZero {
        condition: u32,
        value: u32,
        output: u32,
        target: u32,
    },
    ShortcutNonZero {
        condition: u32,
        value: u32,
        output: u32,
        target: u32,
    },
    Call {
        slot: u32,
        target: u32,
    },
    BranchZero {
        condition: u32,
        target: u32,
    },
    Jump(u32),
    Return(u32),
    Stop,
}

enum Build {
    Ensure(u32),
    Body(u32),
    Evaluate(u32),
    Choice(u32),
    Else {
        slot: u32,
        branch: usize,
    },
    EndChoice {
        slot: u32,
        jump: usize,
    },
    EndShortcut {
        slot: u32,
        input: u32,
        branch: usize,
    },
}

pub(super) fn compile(
    opcodes: &[CompiledOpcode],
    outputs: &[u32],
) -> Result<(Box<[Instruction]>, usize), GraphError> {
    let mut uses = vec![0_usize; opcodes.len()];
    for &output in outputs {
        uses[output as usize] += 1;
    }
    for &opcode in opcodes {
        let (operands, count) = opcode.operands();
        for &operand in &operands[..count] {
            uses[operand as usize] += 1;
        }
    }
    let mut builder = Builder {
        opcodes,
        uses,
        code: Vec::new(),
        calls: Vec::new(),
        work: Vec::new(),
    };
    for &output in outputs {
        builder.emit(Build::Ensure(output))?;
    }
    builder.code.push(Instruction::Stop);
    let mut functions = vec![(0, 0); opcodes.len()];
    let mut memo_count = 0;
    for (i, target) in functions.iter_mut().enumerate() {
        if builder.uses[i] > 1 && !matches!(opcodes[i], CompiledOpcode::Literal { .. }) {
            let memo = offset(memo_count)?;
            *target = (memo, offset(builder.code.len())?);
            memo_count += 1;
            builder.emit(Build::Body(i as u32))?;
            builder.code.push(Instruction::Return(memo));
        }
    }
    for (index, slot) in builder.calls {
        let (memo, target) = functions[slot as usize];
        builder.code[index] = Instruction::Call { slot: memo, target };
    }
    offset(builder.code.len())?;
    Ok((builder.code.into_boxed_slice(), memo_count))
}
fn offset(value: usize) -> Result<u32, GraphError> {
    u32::try_from(value).map_err(|_| GraphError::HardCapacity {
        field: "evaluation_instructions",
        observed: value,
    })
}
struct Builder<'a> {
    opcodes: &'a [CompiledOpcode],
    uses: Vec<usize>,
    code: Vec<Instruction>,
    calls: Vec<(usize, u32)>,
    work: Vec<Build>,
}
impl Builder<'_> {
    fn emit(&mut self, root: Build) -> Result<(), GraphError> {
        self.work.push(root);
        while let Some(work) = self.work.pop() {
            match work {
                Build::Ensure(slot) => {
                    if matches!(self.opcodes[slot as usize], CompiledOpcode::Literal { .. }) {
                        // Constants are initialized once per workspace/tape.
                    } else if self.uses[slot as usize] > 1 {
                        self.calls.push((self.code.len(), slot));
                        self.code.push(Instruction::Call { slot, target: 0 });
                    } else {
                        self.work.push(Build::Body(slot));
                    }
                }
                Build::Body(slot) => {
                    let opcode = self.opcodes[slot as usize];
                    if let CompiledOpcode::Select { condition, .. } = opcode {
                        self.work.push(Build::Choice(slot));
                        self.work.push(Build::Ensure(condition));
                    } else {
                        self.work.push(Build::Evaluate(slot));
                        let (operands, count) = opcode.operands();
                        for &operand in operands[..count].iter().rev() {
                            self.work.push(Build::Ensure(operand));
                        }
                    }
                }
                Build::Evaluate(slot) => self.code.push(Instruction::Evaluate(slot)),
                Build::Choice(slot) => {
                    let CompiledOpcode::Select {
                        condition,
                        then_value,
                        else_value,
                        ..
                    } = self.opcodes[slot as usize]
                    else {
                        unreachable!()
                    };
                    let then_constant = matches!(
                        self.opcodes[then_value as usize],
                        CompiledOpcode::Literal { .. }
                    );
                    let else_constant = matches!(
                        self.opcodes[else_value as usize],
                        CompiledOpcode::Literal { .. }
                    );
                    if then_constant && else_constant {
                        // Both alternatives are already prepared; no control-flow walk.
                        self.code.push(Instruction::Evaluate(slot));
                        continue;
                    }
                    let branch = self.code.len();
                    if then_constant || else_constant {
                        // A constant arm can assign its value and skip the other arm
                        // in one instruction, especially for short-circuit Booleans.
                        self.code.push(if then_constant {
                            Instruction::ShortcutNonZero {
                                condition,
                                value: then_value,
                                output: slot,
                                target: 0,
                            }
                        } else {
                            Instruction::ShortcutZero {
                                condition,
                                value: else_value,
                                output: slot,
                                target: 0,
                            }
                        });
                        let input = if then_constant {
                            else_value
                        } else {
                            then_value
                        };
                        self.work.push(Build::EndShortcut {
                            slot,
                            input,
                            branch,
                        });
                        self.work.push(Build::Ensure(input));
                        continue;
                    }
                    self.code.push(Instruction::BranchZero {
                        condition,
                        target: 0,
                    });
                    self.work.push(Build::Else { slot, branch });
                    self.work.push(Build::Ensure(then_value));
                }
                Build::Else { slot, branch } => {
                    let CompiledOpcode::Select {
                        condition,
                        else_value,
                        ..
                    } = self.opcodes[slot as usize]
                    else {
                        unreachable!()
                    };
                    let jump = self.code.len();
                    self.code.push(Instruction::Jump(0));
                    self.code[branch] = Instruction::BranchZero {
                        condition,
                        target: offset(self.code.len())?,
                    };
                    self.work.push(Build::EndChoice { slot, jump });
                    self.work.push(Build::Ensure(else_value));
                }
                Build::EndShortcut {
                    slot,
                    input,
                    branch,
                } => {
                    self.code.push(Instruction::Copy {
                        input,
                        output: slot,
                    });
                    let end = offset(self.code.len())?;
                    match &mut self.code[branch] {
                        Instruction::ShortcutZero { target, .. }
                        | Instruction::ShortcutNonZero { target, .. } => *target = end,
                        _ => unreachable!(),
                    }
                }
                Build::EndChoice { slot, jump } => {
                    self.code[jump] = Instruction::Jump(offset(self.code.len())?);
                    self.code.push(Instruction::Evaluate(slot));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{Date, UnderlyingId},
        product::{GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op},
    };

    #[test]
    fn generated_shared_dags_match_independent_path_values_across_workspace_reuse() {
        #[derive(Clone, Copy)]
        enum Reference {
            Observation(usize),
            Add(usize, usize),
            Subtract(usize, usize),
            Maximum(usize, usize),
            Select(usize, usize, usize),
        }
        let dates: [Date; 3] =
            ["2027-01-01", "2027-02-01", "2027-03-01"].map(|s| s.parse().unwrap());
        let mut seed = 1847_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        for _ in 0..100 {
            let mut graph = SourceGraphBuilder::new();
            let mut nodes = Vec::new();
            let mut reference = Vec::new();
            for (index, &observation_date) in dates.iter().enumerate() {
                nodes.push(
                    graph
                        .push(Op::TerminalSpot {
                            underlying: UnderlyingId::new(1),
                            observation_date,
                        })
                        .unwrap(),
                );
                reference.push(Reference::Observation(index));
            }
            for _ in 0..50 {
                let left = next() % nodes.len();
                let right = next() % nodes.len();
                let condition = next() % 3;
                let (op, expected) = match next() % 4 {
                    0 => (
                        Op::Add {
                            left: nodes[left],
                            right: nodes[right],
                        },
                        Reference::Add(left, right),
                    ),
                    1 => (
                        Op::Subtract {
                            left: nodes[left],
                            right: nodes[right],
                        },
                        Reference::Subtract(left, right),
                    ),
                    2 => (
                        Op::Maximum {
                            left: nodes[left],
                            right: nodes[right],
                        },
                        Reference::Maximum(left, right),
                    ),
                    _ => (
                        Op::Select {
                            condition: nodes[condition],
                            then_value: nodes[left],
                            else_value: nodes[right],
                        },
                        Reference::Select(condition, left, right),
                    ),
                };
                nodes.push(graph.push(op).unwrap());
                reference.push(expected);
            }
            let outputs = [nodes.len() - 1, nodes.len() - 3, nodes.len() - 1];
            let payoff = graph
                .finish(outputs.map(|index| nodes[index]).to_vec())
                .compile(GraphLimitPolicy::DEFAULT)
                .unwrap();
            let mut workspace = payoff.workspace();
            for scenario in 0..20 {
                let observed = [
                    if scenario % 2 == 0 { 0.0 } else { 1.0 },
                    (scenario % 3) as f64 - 1.0,
                    scenario as f64 - 10.0,
                ];
                let mut values: Vec<f64> = Vec::new();
                for &op in &reference {
                    values.push(match op {
                        Reference::Observation(index) => observed[index],
                        Reference::Add(a, b) => values[a] + values[b],
                        Reference::Subtract(a, b) => values[a] - values[b],
                        Reference::Maximum(a, b) => values[a].max(values[b]),
                        Reference::Select(c, a, b) => values[if values[c] != 0.0 { a } else { b }],
                    });
                }
                let actual = payoff
                    .evaluate_reusing(
                        &mut workspace,
                        |_, date| Some(observed[dates.iter().position(|d| *d == date).unwrap()]),
                        |_, _| None,
                    )
                    .unwrap();
                assert_eq!(actual, outputs.map(|index| values[index]));
            }
        }
    }

    #[test]
    fn shared_and_deep_dependencies_have_linear_program_size_and_lazy_observations() {
        let mut graph = SourceGraphBuilder::new();
        let date: Date = "2027-01-01".parse().unwrap();
        let observed = graph
            .push(Op::TerminalSpot {
                underlying: UnderlyingId::new(1),
                observation_date: date,
            })
            .unwrap();
        let zero = graph.literal(0.0).unwrap();
        let mut shared = observed;
        // An inlined expression tree here would contain 2^5000 leaves.
        for _ in 0..5000 {
            shared = graph
                .push(Op::Add {
                    left: shared,
                    right: shared,
                })
                .unwrap();
        }
        let selected = graph
            .push(Op::Select {
                condition: observed,
                then_value: zero,
                else_value: shared,
            })
            .unwrap();
        let payoff = graph
            .finish(vec![selected])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        assert!(payoff.program.len() < payoff.opcodes().len() * 6);
        let mut reads = 0;
        assert_eq!(
            payoff
                .evaluate(|_, _| {
                    reads += 1;
                    Some(1.0)
                })
                .unwrap(),
            [0.0]
        );
        assert_eq!(reads, 1);
        // Zero stays finite through the deep selected branch; no Rust recursion.
        reads = 0;
        assert_eq!(
            payoff
                .evaluate(|_, _| {
                    reads += 1;
                    Some(0.0)
                })
                .unwrap(),
            [0.0]
        );
        assert_eq!(reads, 1);
    }
}
