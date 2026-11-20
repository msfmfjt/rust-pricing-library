//! Last-use allocation for the forward-only linear tape. The original SSA
//! stays available to reverse mode, wire fingerprints and diagnostics.
use super::tape::CompiledOpcode;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RegisterProgram {
    pub ops: Box<[CompiledOpcode]>,
    pub outputs: Box<[u32]>,
    pub registers: usize,
}
impl RegisterProgram {
    pub fn compile(ops: &[CompiledOpcode], outputs: &[u32], spare_bytes: usize) -> Option<Self> {
        let bytes = std::mem::size_of_val(ops).checked_add(std::mem::size_of_val(outputs))?;
        if ops.len() < 2 || bytes > spare_bytes {
            return None;
        }
        let mut uses = vec![0usize; ops.len()];
        for op in ops {
            let (inputs, count) = op.operands();
            for &input in &inputs[..count] {
                uses[input as usize] += 1;
            }
        }
        for &output in outputs {
            uses[output as usize] += 1;
        }
        let mut mapping = vec![0; ops.len()];
        let mut free = Vec::new();
        let mut registers = 0u32;
        for &op in ops {
            let (inputs, count) = op.operands();
            for &input in &inputs[..count] {
                uses[input as usize] -= 1;
                if uses[input as usize] == 0 {
                    free.push(mapping[input as usize]);
                }
            }
            // Instructions consume all operands before assigning the destination.
            mapping[op.output() as usize] = free.pop().unwrap_or_else(|| {
                let r = registers;
                registers += 1;
                r
            });
        }
        if registers as usize == ops.len() {
            return None;
        }
        let remap = |slot: &mut u32| *slot = mapping[*slot as usize];
        let remapped = ops
            .iter()
            .copied()
            .map(|mut op| {
                use CompiledOpcode::*;
                match &mut op {
                    Select {
                        condition,
                        then_value,
                        else_value,
                        output,
                    } => {
                        for slot in [condition, then_value, else_value, output] {
                            remap(slot);
                        }
                    }
                    Log { input, output }
                    | Exp { input, output }
                    | Indicator { input, output }
                    | SmoothIndicator { input, output, .. }
                    | Negate { input, output } => {
                        remap(input);
                        remap(output);
                    }
                    Add {
                        left,
                        right,
                        output,
                    }
                    | Subtract {
                        left,
                        right,
                        output,
                    }
                    | Multiply {
                        left,
                        right,
                        output,
                    }
                    | Minimum {
                        left,
                        right,
                        output,
                    }
                    | Maximum {
                        left,
                        right,
                        output,
                    }
                    | Compare {
                        left,
                        right,
                        output,
                        ..
                    }
                    | SmoothMinimum {
                        left,
                        right,
                        output,
                        ..
                    }
                    | SmoothMaximum {
                        left,
                        right,
                        output,
                        ..
                    }
                    | Divide {
                        numerator: left,
                        denominator: right,
                        output,
                    }
                    | BarrierHitState {
                        previous: left,
                        hit_weight: right,
                        output,
                    } => {
                        remap(left);
                        remap(right);
                        remap(output);
                    }
                    Literal { output, .. }
                    | TerminalSpot { output, .. }
                    | PreDividendSpot { output, .. } => remap(output),
                }
                op
            })
            .collect();
        Some(Self {
            ops: remapped,
            outputs: outputs.iter().map(|&s| mapping[s as usize]).collect(),
            registers: registers as usize,
        })
    }
    pub fn bytes(&self) -> usize {
        std::mem::size_of_val(self.ops.as_ref()) + std::mem::size_of_val(self.outputs.as_ref())
    }
}
