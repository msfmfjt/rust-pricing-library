use std::error::Error;
use std::fmt;

use crate::{Philox4x32, RandomCoordinate, RandomDomain, RqmcConfig, open_unit_interval};

const SOBOL_BITS: usize = 32;
const DIRECTION_HEADER_BYTES: usize = 20;
const DIRECTION_MAGIC: &[u8; 8] = b"JK621201";
const DIRECTION_FORMAT_VERSION: u32 = 1;
const DIRECTION_DATA: &[u8] = include_bytes!("../data/joe-kuo-6.21201-u32be.bin");

pub const JOE_KUO_DIRECTION_SET: &str = "joe-kuo-6.21201-u32-v1";
pub const RQMC_SCRAMBLE_ABI: &str = "rqmc-lms32-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SobolDimensionError {
    dimension: u32,
}

impl SobolDimensionError {
    #[must_use]
    pub const fn dimension(self) -> u32 {
        self.dimension
    }
}

impl fmt::Display for SobolDimensionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "zero-based Sobol dimension {} is outside 0..{}",
            self.dimension,
            RqmcConfig::MAX_SOBOL_DIMENSION
        )
    }
}

impl Error for SobolDimensionError {}

/// One dimension of the 32-bit Joe--Kuo 6.21201 Sobol sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sobol32 {
    directions: [u32; SOBOL_BITS],
}

impl Sobol32 {
    pub fn for_dimension(zero_based_dimension: u32) -> Result<Self, SobolDimensionError> {
        if zero_based_dimension >= RqmcConfig::MAX_SOBOL_DIMENSION {
            return Err(SobolDimensionError {
                dimension: zero_based_dimension,
            });
        }
        debug_assert_direction_asset();
        let dimension = usize::try_from(zero_based_dimension).expect("u32 fits usize");
        let start = DIRECTION_HEADER_BYTES + dimension * SOBOL_BITS * size_of::<u32>();
        let mut directions = [0_u32; SOBOL_BITS];
        for (index, direction) in directions.iter_mut().enumerate() {
            let offset = start + index * size_of::<u32>();
            *direction = u32::from_be_bytes(
                DIRECTION_DATA[offset..offset + size_of::<u32>()]
                    .try_into()
                    .expect("validated fixed-width direction word"),
            );
        }
        Ok(Self { directions })
    }

    /// Returns the raw Sobol word. Point zero is intentionally included.
    #[must_use]
    pub fn word(self, point_index: u32) -> u32 {
        let mut gray_code = point_index ^ (point_index >> 1);
        let mut word = 0_u32;
        let mut bit = 0_usize;
        while gray_code != 0 {
            if gray_code & 1 != 0 {
                word ^= self.directions[bit];
            }
            gray_code >>= 1;
            bit += 1;
        }
        word
    }
}

fn debug_assert_direction_asset() {
    debug_assert_eq!(&DIRECTION_DATA[..8], DIRECTION_MAGIC);
    debug_assert_eq!(read_header_u32(8), DIRECTION_FORMAT_VERSION);
    debug_assert_eq!(read_header_u32(12), RqmcConfig::MAX_SOBOL_DIMENSION);
    debug_assert_eq!(read_header_u32(16), SOBOL_BITS as u32);
    debug_assert_eq!(
        DIRECTION_DATA.len(),
        DIRECTION_HEADER_BYTES
            + usize::try_from(RqmcConfig::MAX_SOBOL_DIMENSION).expect("dimension fits usize")
                * SOBOL_BITS
                * size_of::<u32>()
    );
}

fn read_header_u32(offset: usize) -> u32 {
    u32::from_be_bytes(
        DIRECTION_DATA[offset..offset + size_of::<u32>()]
            .try_into()
            .expect("direction asset header is fixed width"),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scramble32 {
    shift: u32,
    rows: [u32; SOBOL_BITS],
}

impl Scramble32 {
    #[must_use]
    pub const fn shift(self) -> u32 {
        self.shift
    }

    #[must_use]
    pub const fn rows(self) -> [u32; SOBOL_BITS] {
        self.rows
    }

    #[must_use]
    pub fn apply(self, input: u32) -> u32 {
        let mut output = 0_u32;
        for (row_index, row) in self.rows.into_iter().enumerate() {
            let parity = (input & row).count_ones() & 1;
            output |= parity << (31 - row_index);
        }
        output ^ self.shift
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RqmcPlanError {
    ZeroEffectiveDimension,
    DimensionLimitExceeded { requested: u32, maximum: u32 },
    TableSizeOverflow,
    AllocationFailed,
}

impl fmt::Display for RqmcPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEffectiveDimension => {
                write!(formatter, "effective Sobol dimension must be positive")
            }
            Self::DimensionLimitExceeded { requested, maximum } => write!(
                formatter,
                "effective Sobol dimension {requested} exceeds the Joe--Kuo limit {maximum}"
            ),
            Self::TableSizeOverflow => write!(formatter, "RQMC plan table size overflowed usize"),
            Self::AllocationFailed => write!(formatter, "RQMC plan table allocation failed"),
        }
    }
}

impl Error for RqmcPlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RqmcPointError {
    ScrambleOutOfRange { requested: u32, count: u32 },
    DimensionOutOfRange { requested: u32, count: u32 },
    PointOutOfRange { requested: u64, count: u64 },
}

impl fmt::Display for RqmcPointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScrambleOutOfRange { requested, count } => write!(
                formatter,
                "scramble index {requested} is outside 0..{count}"
            ),
            Self::DimensionOutOfRange { requested, count } => write!(
                formatter,
                "Sobol dimension {requested} is outside 0..{count}"
            ),
            Self::PointOutOfRange { requested, count } => {
                write!(formatter, "Sobol point {requested} is outside 0..{count}")
            }
        }
    }
}

impl Error for RqmcPointError {}

/// Immutable, eagerly compiled direction and scramble data for an RQMC request.
#[derive(Clone, Debug)]
pub struct RqmcPlan {
    points_per_scramble: u64,
    scramble_count: u32,
    master_scramble_seed: u64,
    effective_dimension: u32,
    directions: Box<[Sobol32]>,
    scrambles: Box<[Scramble32]>,
    direction_checksum: [u8; 32],
    scramble_checksum: [u8; 32],
}

impl RqmcPlan {
    pub fn compile(config: RqmcConfig, effective_dimension: u32) -> Result<Self, RqmcPlanError> {
        if effective_dimension == 0 {
            return Err(RqmcPlanError::ZeroEffectiveDimension);
        }
        if effective_dimension > RqmcConfig::MAX_SOBOL_DIMENSION {
            return Err(RqmcPlanError::DimensionLimitExceeded {
                requested: effective_dimension,
                maximum: RqmcConfig::MAX_SOBOL_DIMENSION,
            });
        }

        let dimension_count =
            usize::try_from(effective_dimension).map_err(|_| RqmcPlanError::TableSizeOverflow)?;
        let scramble_count = usize::try_from(config.scramble_count().get())
            .map_err(|_| RqmcPlanError::TableSizeOverflow)?;
        let table_length = dimension_count
            .checked_mul(scramble_count)
            .ok_or(RqmcPlanError::TableSizeOverflow)?;

        let mut directions = Vec::new();
        directions
            .try_reserve_exact(dimension_count)
            .map_err(|_| RqmcPlanError::AllocationFailed)?;
        for dimension in 0..effective_dimension {
            directions.push(
                Sobol32::for_dimension(dimension)
                    .expect("effective dimension was validated against the same limit"),
            );
        }

        let mut scrambles = Vec::new();
        scrambles
            .try_reserve_exact(table_length)
            .map_err(|_| RqmcPlanError::AllocationFailed)?;
        let philox = Philox4x32::from_seed(config.master_scramble_seed());
        let mut checksum = blake3::Hasher::new();
        for scramble in 0..config.scramble_count().get() {
            for dimension in 0..effective_dimension {
                let value = compile_scramble(philox, u64::from(scramble), dimension);
                checksum.update(&value.shift.to_be_bytes());
                for row in value.rows {
                    checksum.update(&row.to_be_bytes());
                }
                scrambles.push(value);
            }
        }

        Ok(Self {
            points_per_scramble: config.points_per_scramble().get(),
            scramble_count: config.scramble_count().get(),
            master_scramble_seed: config.master_scramble_seed(),
            effective_dimension,
            directions: directions.into_boxed_slice(),
            scrambles: scrambles.into_boxed_slice(),
            direction_checksum: *blake3::hash(DIRECTION_DATA).as_bytes(),
            scramble_checksum: *checksum.finalize().as_bytes(),
        })
    }

    #[must_use]
    pub const fn points_per_scramble(&self) -> u64 {
        self.points_per_scramble
    }

    #[must_use]
    pub const fn scramble_count(&self) -> u32 {
        self.scramble_count
    }

    #[must_use]
    pub const fn master_scramble_seed(&self) -> u64 {
        self.master_scramble_seed
    }

    #[must_use]
    pub const fn effective_dimension(&self) -> u32 {
        self.effective_dimension
    }

    #[must_use]
    pub const fn direction_checksum(&self) -> [u8; 32] {
        self.direction_checksum
    }

    #[must_use]
    pub const fn scramble_checksum(&self) -> [u8; 32] {
        self.scramble_checksum
    }

    pub fn scramble(&self, scramble: u32, dimension: u32) -> Result<Scramble32, RqmcPointError> {
        let index = self.table_index(scramble, dimension)?;
        Ok(self.scrambles[index])
    }

    pub fn word(
        &self,
        scramble: u32,
        point_index: u64,
        dimension: u32,
    ) -> Result<u32, RqmcPointError> {
        if point_index >= self.points_per_scramble {
            return Err(RqmcPointError::PointOutOfRange {
                requested: point_index,
                count: self.points_per_scramble,
            });
        }
        let index = self.table_index(scramble, dimension)?;
        let point = u32::try_from(point_index).expect("RQMC configuration limits points to 2^32");
        Ok(self.scrambles[index].apply(
            self.directions[usize::try_from(dimension).expect("u32 fits usize")].word(point),
        ))
    }

    pub fn uniform(
        &self,
        scramble: u32,
        point_index: u64,
        dimension: u32,
    ) -> Result<f64, RqmcPointError> {
        self.word(scramble, point_index, dimension)
            .map(open_unit_interval)
    }

    fn table_index(&self, scramble: u32, dimension: u32) -> Result<usize, RqmcPointError> {
        if scramble >= self.scramble_count {
            return Err(RqmcPointError::ScrambleOutOfRange {
                requested: scramble,
                count: self.scramble_count,
            });
        }
        if dimension >= self.effective_dimension {
            return Err(RqmcPointError::DimensionOutOfRange {
                requested: dimension,
                count: self.effective_dimension,
            });
        }
        Ok(usize::try_from(scramble).expect("u32 fits usize")
            * usize::try_from(self.effective_dimension).expect("u32 fits usize")
            + usize::try_from(dimension).expect("u32 fits usize"))
    }
}

fn compile_scramble(philox: Philox4x32, scramble: u64, dimension: u32) -> Scramble32 {
    let block = dimension
        .checked_mul(SOBOL_BITS as u32)
        .expect("maximum supported dimension fits the Philox coordinate");
    let shift = philox.word(RandomCoordinate::new(
        scramble,
        block,
        RandomDomain::RqmcScramble,
    ));
    let mut rows = [0_u32; SOBOL_BITS];
    rows[0] = 0x8000_0000;
    for row in 1_u32..SOBOL_BITS as u32 {
        let raw = philox.word(RandomCoordinate::new(
            scramble,
            block + row,
            RandomDomain::RqmcScramble,
        ));
        let columns_before_diagonal = u32::MAX << (SOBOL_BITS as u32 - row);
        let diagonal = 1_u32 << (31 - row);
        rows[usize::try_from(row).expect("row fits usize")] =
            (raw & columns_before_diagonal) | diagonal;
    }
    Scramble32 { shift, rows }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarianceReduction;

    #[test]
    fn joe_kuo_words_match_known_first_three_dimensions() {
        let expected = [
            [0, 0, 0],
            [0x8000_0000, 0x8000_0000, 0x8000_0000],
            [0xc000_0000, 0x4000_0000, 0x4000_0000],
            [0x4000_0000, 0xc000_0000, 0xc000_0000],
            [0x6000_0000, 0x6000_0000, 0xa000_0000],
            [0xe000_0000, 0xe000_0000, 0x2000_0000],
            [0xa000_0000, 0x2000_0000, 0xe000_0000],
            [0x2000_0000, 0xa000_0000, 0x6000_0000],
        ];
        let dimensions = [
            Sobol32::for_dimension(0).expect("dimension 1"),
            Sobol32::for_dimension(1).expect("dimension 2"),
            Sobol32::for_dimension(2).expect("dimension 3"),
        ];
        for (point, expected_words) in expected.into_iter().enumerate() {
            for (dimension, expected_word) in expected_words.into_iter().enumerate() {
                assert_eq!(
                    dimensions[dimension].word(u32::try_from(point).expect("small point")),
                    expected_word
                );
            }
        }
    }

    #[test]
    fn full_direction_range_is_available_and_strict() {
        assert!(Sobol32::for_dimension(0).is_ok());
        assert!(Sobol32::for_dimension(RqmcConfig::MAX_SOBOL_DIMENSION - 1).is_ok());
        assert_eq!(
            Sobol32::for_dimension(RqmcConfig::MAX_SOBOL_DIMENSION),
            Err(SobolDimensionError {
                dimension: RqmcConfig::MAX_SOBOL_DIMENSION
            })
        );
    }

    #[test]
    fn lms_rows_are_msb_first_unit_lower_triangular() {
        let scramble = compile_scramble(Philox4x32::from_seed(123), 4, 7);
        assert_eq!(scramble.rows[0], 0x8000_0000);
        for row in 0..SOBOL_BITS {
            let diagonal = 1_u32 << (31 - row);
            assert_ne!(scramble.rows[row] & diagonal, 0);
            assert_eq!(scramble.rows[row] & diagonal.wrapping_sub(1), 0);
        }
    }

    #[test]
    fn compile_materializes_exact_scramble_dimension_product() {
        let config = RqmcConfig::new(
            8,
            3,
            0x0123_4567_89ab_cdef,
            VarianceReduction::new(false, false),
        )
        .expect("valid RQMC config");
        let plan = RqmcPlan::compile(config, 2).expect("valid plan");
        assert_eq!(plan.directions.len(), 2);
        assert_eq!(plan.scrambles.len(), 6);
        assert_eq!(plan.points_per_scramble(), 8);
        assert_eq!(plan.scramble_count(), 3);
        assert_eq!(plan.effective_dimension(), 2);
        assert_ne!(plan.direction_checksum(), [0; 32]);
        assert_ne!(plan.scramble_checksum(), [0; 32]);
    }

    #[test]
    fn point_zero_is_shift_and_midpoint_mapping_stays_open() {
        let config = RqmcConfig::new(2, 1, 99, VarianceReduction::new(false, false))
            .expect("valid RQMC config");
        let plan = RqmcPlan::compile(config, 1).expect("valid plan");
        assert_eq!(plan.word(0, 0, 0), Ok(plan.scramble(0, 0).unwrap().shift()));
        let uniform = plan.uniform(0, 0, 0).expect("valid coordinate");
        assert!(uniform > 0.0 && uniform < 1.0);
    }

    #[test]
    fn replay_is_stable_and_coordinates_are_independent() {
        let config = RqmcConfig::new(8, 2, 42, VarianceReduction::new(false, false))
            .expect("valid RQMC config");
        let first = RqmcPlan::compile(config, 2).expect("valid plan");
        let replay = RqmcPlan::compile(config, 2).expect("valid plan");
        assert_eq!(first.scramble_checksum(), replay.scramble_checksum());
        assert_eq!(first.word(1, 5, 1), replay.word(1, 5, 1));
        assert_ne!(first.scramble(0, 0), first.scramble(0, 1));
        assert_ne!(first.scramble(0, 0), first.scramble(1, 0));
    }

    #[test]
    fn scramble_abi_matches_known_answer_words() {
        let config = RqmcConfig::new(
            8,
            3,
            0x0123_4567_89ab_cdef,
            VarianceReduction::new(false, false),
        )
        .expect("valid RQMC config");
        let plan = RqmcPlan::compile(config, 2).expect("valid plan");
        assert_eq!(plan.word(0, 0, 0), Ok(0x46c5_fd35));
        assert_eq!(plan.word(0, 1, 0), Ok(0xe684_a5d7));
        assert_eq!(plan.word(0, 2, 1), Ok(0x7d21_cc82));
        assert_eq!(plan.word(1, 2, 0), Ok(0xaa93_e083));
        assert_eq!(plan.word(2, 7, 1), Ok(0xc2b4_0c80));
    }

    #[test]
    fn plan_rejects_invalid_coordinates_and_dimensions() {
        let config = RqmcConfig::new(4, 1, 0, VarianceReduction::new(false, false))
            .expect("valid RQMC config");
        assert_eq!(
            RqmcPlan::compile(config, 0).unwrap_err(),
            RqmcPlanError::ZeroEffectiveDimension
        );
        let plan = RqmcPlan::compile(config, 1).expect("valid plan");
        assert!(matches!(
            plan.word(1, 0, 0),
            Err(RqmcPointError::ScrambleOutOfRange { .. })
        ));
        assert!(matches!(
            plan.word(0, 4, 0),
            Err(RqmcPointError::PointOutOfRange { .. })
        ));
        assert!(matches!(
            plan.word(0, 0, 1),
            Err(RqmcPointError::DimensionOutOfRange { .. })
        ));
    }
}
