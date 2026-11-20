//! Batched public-API probes. Hexadecimal input preserves every binary64 bit.
use pricing::analytical::{BlackForwardOracleInputs, evaluate_black_forward};
use pricing::product::OptionSide;
use pricing_numerics::Complex64;
use std::io::{self, BufRead};

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.expect("read probe");
        let (kind, values) = line.split_once(' ').expect("probe and values");
        let values: Vec<_> = values
            .split_whitespace()
            .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).expect("binary64 bits")))
            .collect();
        let pair = match (kind, values.as_slice()) {
            ("div", &[a, b, c, d]) => {
                let z = Complex64::new(a, b) / Complex64::new(c, d);
                (z.re, z.im)
            }
            ("black", &[forward, strike, volatility]) => {
                let price = |side| {
                    evaluate_black_forward(BlackForwardOracleInputs {
                        side,
                        forward,
                        strike,
                        volatility,
                        notional: 1.0,
                        discount: 1.0,
                        time: 1.0,
                    })
                    .expect("finite price and Greeks")
                    .price
                };
                (price(OptionSide::Call), price(OptionSide::Put))
            }
            _ => panic!("unknown numerical probe"),
        };
        println!("{:016x} {:016x}", pair.0.to_bits(), pair.1.to_bits());
    }
}
