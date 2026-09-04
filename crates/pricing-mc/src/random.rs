use std::error::Error;
use std::fmt;

const PHILOX_M0: u32 = 0xd251_1f53;
const PHILOX_M1: u32 = 0xcd9e_8d57;
const PHILOX_W0: u32 = 0x9e37_79b9;
const PHILOX_W1: u32 = 0xbb67_ae85;
const U32_SCALE: f64 = 1.0 / 4_294_967_296.0;

/// Stable namespaces for counter-based random coordinates.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum RandomDomain {
    Valuation = 0,
    LsmTrain = 1,
    RqmcScramble = 2,
    Diagnostics = 3,
}

impl RandomDomain {
    #[must_use]
    pub const fn id(self) -> u32 {
        self as u32
    }
}

/// A random coordinate whose value is independent of execution order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RandomCoordinate {
    path: u64,
    dimension: u32,
    domain: RandomDomain,
}

impl RandomCoordinate {
    #[must_use]
    pub const fn new(path: u64, dimension: u32, domain: RandomDomain) -> Self {
        Self {
            path,
            dimension,
            domain,
        }
    }

    #[must_use]
    pub const fn path(self) -> u64 {
        self.path
    }

    #[must_use]
    pub const fn dimension(self) -> u32 {
        self.dimension
    }

    #[must_use]
    pub const fn domain(self) -> RandomDomain {
        self.domain
    }

    #[must_use]
    pub const fn counter(self) -> [u32; 4] {
        [
            self.path as u32,
            (self.path >> 32) as u32,
            self.dimension / 4,
            self.domain.id(),
        ]
    }

    #[must_use]
    pub const fn lane(self) -> usize {
        (self.dimension % 4) as usize
    }
}

/// Philox4x32-10 with the Random123 constants and round schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Philox4x32 {
    key: [u32; 2],
}

impl Philox4x32 {
    pub const ROUNDS: u32 = 10;

    #[must_use]
    pub const fn from_seed(master_seed: u64) -> Self {
        Self {
            key: [master_seed as u32, (master_seed >> 32) as u32],
        }
    }

    #[must_use]
    pub const fn key(self) -> [u32; 2] {
        self.key
    }

    #[must_use]
    pub fn generate(self, mut counter: [u32; 4]) -> [u32; 4] {
        let mut key = self.key;
        let mut round = 0;
        while round < Self::ROUNDS {
            counter = philox_round(counter, key);
            if round + 1 < Self::ROUNDS {
                key[0] = key[0].wrapping_add(PHILOX_W0);
                key[1] = key[1].wrapping_add(PHILOX_W1);
            }
            round += 1;
        }
        counter
    }

    #[must_use]
    pub fn word(self, coordinate: RandomCoordinate) -> u32 {
        self.generate(coordinate.counter())[coordinate.lane()]
    }

    #[must_use]
    pub fn uniform(self, coordinate: RandomCoordinate) -> f64 {
        open_unit_interval(self.word(coordinate))
    }

    #[must_use]
    pub fn standard_normal(self, coordinate: RandomCoordinate) -> f64 {
        inverse_standard_normal(self.uniform(coordinate))
            .expect("a u32 midpoint mapping is strictly inside the unit interval")
    }
}

#[inline]
fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let product_0 = u64::from(PHILOX_M0) * u64::from(counter[0]);
    let product_1 = u64::from(PHILOX_M1) * u64::from(counter[2]);
    let high_0 = (product_0 >> 32) as u32;
    let high_1 = (product_1 >> 32) as u32;
    [
        high_1 ^ counter[1] ^ key[0],
        product_1 as u32,
        high_0 ^ counter[3] ^ key[1],
        product_0 as u32,
    ]
}

/// Maps one `u32` to its bin midpoint, hence always strictly inside `(0, 1)`.
#[must_use]
pub fn open_unit_interval(word: u32) -> f64 {
    (f64::from(word) + 0.5) * U32_SCALE
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalQuantileError {
    probability: f64,
}

impl NormalQuantileError {
    #[must_use]
    pub const fn probability(self) -> f64 {
        self.probability
    }
}

impl fmt::Display for NormalQuantileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "normal quantile probability must be finite and strictly between zero and one; received {}",
            self.probability
        )
    }
}

impl Error for NormalQuantileError {}

/// Wichura's AS241 inverse standard-normal CDF using fixed binary64 operations.
pub fn inverse_standard_normal(probability: f64) -> Result<f64, NormalQuantileError> {
    if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
        return Err(NormalQuantileError { probability });
    }

    const A: [f64; 8] = [
        3.387_132_872_796_366_5,
        133.141_667_891_784_38,
        1_971.590_950_306_551_3,
        13_731.693_765_509_46,
        45_921.953_931_549_87,
        67_265.770_927_008_7,
        33_430.575_583_588_36,
        2_509.080_928_730_122_7,
    ];
    const B: [f64; 8] = [
        1.0,
        42.313_330_701_600_91,
        687.187_007_492_057_9,
        5_394.196_021_424_751,
        21_213.794_301_586_597,
        39_307.895_800_092_71,
        28_729.085_735_721_943,
        5_226.495_278_852_855,
    ];
    const C: [f64; 8] = [
        1.423_437_110_749_683_5,
        4.630_337_846_156_546,
        5.769_497_221_460_691,
        3.647_848_324_763_204_5,
        1.270_458_252_452_368_4,
        0.241_780_725_177_450_6,
        0.022_723_844_989_269_185,
        0.000_774_545_014_278_341_4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.053_191_626_637_759,
        1.676_384_830_183_803_8,
        0.689_767_334_985_1,
        0.148_103_976_427_480_08,
        0.015_198_666_563_616_457,
        0.000_547_593_808_499_534_5,
        1.050_750_071_644_416_8e-9,
    ];
    const E: [f64; 8] = [
        6.657_904_643_501_104,
        5.463_784_911_164_1145,
        1.784_826_539_917_291_3,
        0.296_560_571_828_504_9,
        0.026_532_189_526_576_123,
        0.001_242_660_947_388_078_4,
        0.000_027_115_555_687_434_876,
        0.000_000_201_033_439_929_228_82,
    ];
    const F: [f64; 8] = [
        1.0,
        0.599_832_206_555_887_9,
        0.136_929_880_922_735_8,
        0.014_875_361_290_850_615,
        0.000_786_869_131_145_613_3,
        0.000_018_463_183_170_105_468,
        0.000_000_142_151_175_831_684_48,
        2.044_263_103_389_939_7e-15,
    ];

    let centered = probability - 0.5;
    if centered.abs() <= 0.425 {
        let argument = 0.180_625 - centered * centered;
        return Ok(centered * polynomial(argument, &A) / polynomial(argument, &B));
    }

    let tail_probability = if centered < 0.0 {
        probability
    } else {
        1.0 - probability
    };
    let radius = (-tail_probability.ln()).sqrt();
    let magnitude = if radius <= 5.0 {
        let argument = radius - 1.6;
        polynomial(argument, &C) / polynomial(argument, &D)
    } else {
        let argument = radius - 5.0;
        polynomial(argument, &E) / polynomial(argument, &F)
    };
    Ok(if centered < 0.0 {
        -magnitude
    } else {
        magnitude
    })
}

#[inline]
fn polynomial(argument: f64, coefficients: &[f64; 8]) -> f64 {
    let mut value = coefficients[7];
    let mut index = 7;
    while index > 0 {
        index -= 1;
        value = value * argument + coefficients[index];
    }
    value
}

/// Produces the antithetic normal without consuming another random coordinate.
#[must_use]
pub fn antithetic_normal(normal: f64) -> f64 {
    -normal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_matches_random123_zero_vector() {
        assert_eq!(
            Philox4x32::from_seed(0).generate([0; 4]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn seed_counter_and_lane_layout_are_explicit() {
        let generator = Philox4x32::from_seed(0x0123_4567_89ab_cdef);
        assert_eq!(generator.key(), [0x89ab_cdef, 0x0123_4567]);

        for dimension in 0..4 {
            let coordinate = RandomCoordinate::new(
                0x1122_3344_5566_7788,
                dimension,
                RandomDomain::LsmTrain,
            );
            assert_eq!(coordinate.counter(), [0x5566_7788, 0x1122_3344, 0, 1]);
            assert_eq!(coordinate.lane(), dimension as usize);
        }
        let next_block = RandomCoordinate::new(9, 4, RandomDomain::Diagnostics);
        assert_eq!(next_block.counter(), [9, 0, 1, 3]);
        assert_eq!(next_block.lane(), 0);
    }

    #[test]
    fn adjacent_dimensions_use_the_four_output_lanes() {
        let generator = Philox4x32::from_seed(17);
        let block = generator.generate([29, 0, 0, RandomDomain::Valuation.id()]);
        for (dimension, expected) in block.into_iter().enumerate() {
            let coordinate = RandomCoordinate::new(
                29,
                u32::try_from(dimension).expect("four lanes fit u32"),
                RandomDomain::Valuation,
            );
            assert_eq!(generator.word(coordinate), expected);
        }
    }

    #[test]
    fn domain_separates_streams_and_repeated_coordinates_are_stable() {
        let generator = Philox4x32::from_seed(42);
        let valuation = RandomCoordinate::new(5, 7, RandomDomain::Valuation);
        let training = RandomCoordinate::new(5, 7, RandomDomain::LsmTrain);
        assert_eq!(generator.word(valuation), generator.word(valuation));
        assert_ne!(generator.word(valuation), generator.word(training));
    }

    #[test]
    fn midpoint_mapping_is_open_at_both_ends() {
        assert_eq!(open_unit_interval(0), 2.0_f64.powi(-33));
        assert_eq!(open_unit_interval(u32::MAX), 1.0 - 2.0_f64.powi(-33));
        assert!(open_unit_interval(0) > 0.0);
        assert!(open_unit_interval(u32::MAX) < 1.0);
    }

    #[test]
    fn as241_matches_reference_quantiles_and_symmetry() {
        assert_eq!(inverse_standard_normal(0.5), Ok(0.0));
        let upper = inverse_standard_normal(0.975).expect("interior probability");
        assert!((upper - 1.959_963_984_540_054).abs() < 2.0e-15);
        let far_left = inverse_standard_normal(1.0e-10).expect("interior probability");
        assert!((far_left - -6.361_340_902_404_056).abs() < 2.0e-15);
        let lower = inverse_standard_normal(0.025).expect("interior probability");
        assert!((upper + lower).abs() < 2.0e-15);
        assert!(inverse_standard_normal(0.0).is_err());
        assert!(inverse_standard_normal(1.0).is_err());
        assert!(inverse_standard_normal(f64::NAN).is_err());
    }

    #[test]
    fn antithetic_is_an_exact_sign_change() {
        let normal = Philox4x32::from_seed(8).standard_normal(RandomCoordinate::new(
            12,
            3,
            RandomDomain::Valuation,
        ));
        assert_eq!(antithetic_normal(normal).to_bits(), (-normal).to_bits());
    }
}
