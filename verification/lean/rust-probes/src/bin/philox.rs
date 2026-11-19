//! Print public-API coordinates and check every word and open-unit midpoint on the grid.
use pricing::mc::{Philox4x32, RandomCoordinate, RandomDomain};

fn main() {
    let domains = [
        RandomDomain::Valuation,
        RandomDomain::LsmTrain,
        RandomDomain::RqmcScramble,
        RandomDomain::Diagnostics,
        RandomDomain::LsvCalibration,
    ];
    for seed in [0, 1, u64::MAX] {
        let generator = Philox4x32::from_seed(seed);
        for path in [0, u64::from(u32::MAX), 1_u64 << 32, u64::MAX] {
            for base in [0, 4, u32::MAX - 3] {
                for domain in domains {
                    let address = RandomCoordinate::new(path, base, domain);
                    let words = generator.generate(address.counter());
                    for (lane, &word) in (0..4_u32).zip(words.iter()) {
                        let coordinate = RandomCoordinate::new(path, base + lane, domain);
                        assert_eq!(generator.word(coordinate), word);
                        let numerator = 2 * u64::from(word) + 1;
                        let midpoint = numerator as f64 / 8_589_934_592.0;
                        assert_eq!(generator.uniform(coordinate).to_bits(), midpoint.to_bits());
                    }
                    println!(
                        "{seed} {path} {base} {} {} {} {} {}",
                        domain.id(),
                        words[0],
                        words[1],
                        words[2],
                        words[3]
                    );
                }
            }
        }
    }
}
