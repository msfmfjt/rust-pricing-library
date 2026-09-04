use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::num::NonZeroU32;

const LANES_PER_ALIGNMENT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AadConfigError {
    ZeroReductionBlockSize,
    TileCapacityExceedsReductionBlock {
        tile_capacity: u32,
        block_size: u64,
    },
    WorkspaceSizeOverflow,
    SlotOutOfRange {
        slot: usize,
        slot_count: usize,
    },
    LaneOutOfRange {
        lane: usize,
        logical_capacity: usize,
    },
}

impl fmt::Display for AadConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroReductionBlockSize => {
                write!(formatter, "reduction-block size must be positive")
            }
            Self::TileCapacityExceedsReductionBlock {
                tile_capacity,
                block_size,
            } => write!(
                formatter,
                "AAD tile capacity {tile_capacity} exceeds reduction-block size {block_size}"
            ),
            Self::WorkspaceSizeOverflow => write!(formatter, "AAD workspace size overflowed"),
            Self::SlotOutOfRange { slot, slot_count } => {
                write!(
                    formatter,
                    "AAD slot {slot} is outside slot count {slot_count}"
                )
            }
            Self::LaneOutOfRange {
                lane,
                logical_capacity,
            } => write!(
                formatter,
                "AAD lane {lane} is outside logical capacity {logical_capacity}"
            ),
        }
    }
}

impl Error for AadConfigError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AadTilePolicy {
    override_capacity: Option<NonZeroU32>,
    resolved_capacity: NonZeroU32,
}

impl AadTilePolicy {
    pub const VERSION: u32 = 1;
    pub const DEFAULT_CAPACITY: u32 = 256;

    pub fn resolve(
        reduction_block_size: u64,
        override_capacity: Option<NonZeroU32>,
    ) -> Result<Self, AadConfigError> {
        if reduction_block_size == 0 {
            return Err(AadConfigError::ZeroReductionBlockSize);
        }
        let candidate = override_capacity.map_or(
            u64::from(Self::DEFAULT_CAPACITY).min(reduction_block_size),
            |value| u64::from(value.get()),
        );
        if candidate > reduction_block_size {
            return Err(AadConfigError::TileCapacityExceedsReductionBlock {
                tile_capacity: u32::try_from(candidate).unwrap_or(u32::MAX),
                block_size: reduction_block_size,
            });
        }
        let resolved_capacity = NonZeroU32::new(
            u32::try_from(candidate).map_err(|_| AadConfigError::WorkspaceSizeOverflow)?,
        )
        .ok_or(AadConfigError::ZeroReductionBlockSize)?;
        Ok(Self {
            override_capacity,
            resolved_capacity,
        })
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        Self::VERSION
    }

    #[must_use]
    pub const fn default_capacity(self) -> u32 {
        Self::DEFAULT_CAPACITY
    }

    #[must_use]
    pub const fn override_capacity(self) -> Option<NonZeroU32> {
        self.override_capacity
    }

    #[must_use]
    pub const fn resolved_capacity(self) -> NonZeroU32 {
        self.resolved_capacity
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointPolicy {
    override_interval: Option<NonZeroU32>,
    resolved_interval: NonZeroU32,
}

impl CheckpointPolicy {
    pub const VERSION: u32 = 1;
    pub const DEFAULT_INTERVAL: NonZeroU32 = NonZeroU32::new(64).expect("positive constant");

    #[must_use]
    pub const fn resolve(override_interval: Option<NonZeroU32>) -> Self {
        Self {
            override_interval,
            resolved_interval: match override_interval {
                Some(value) => value,
                None => Self::DEFAULT_INTERVAL,
            },
        }
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        Self::VERSION
    }

    #[must_use]
    pub const fn default_interval(self) -> NonZeroU32 {
        Self::DEFAULT_INTERVAL
    }

    #[must_use]
    pub const fn override_interval(self) -> Option<NonZeroU32> {
        self.override_interval
    }

    #[must_use]
    pub const fn resolved_interval(self) -> NonZeroU32 {
        self.resolved_interval
    }
}

#[repr(C, align(64))]
#[derive(Clone, Debug)]
struct AlignedBlock([f64; LANES_PER_ALIGNMENT]);

/// Safe 64-byte-aligned storage padded to complete groups of eight `f64` lanes.
#[derive(Clone, Debug)]
pub struct AlignedF64Buffer {
    blocks: Vec<AlignedBlock>,
    logical_capacity: usize,
}

impl AlignedF64Buffer {
    pub fn new(logical_capacity: usize) -> Result<Self, AadConfigError> {
        let block_count = logical_capacity
            .checked_add(LANES_PER_ALIGNMENT - 1)
            .ok_or(AadConfigError::WorkspaceSizeOverflow)?
            / LANES_PER_ALIGNMENT;
        Ok(Self {
            blocks: vec![AlignedBlock([0.0; LANES_PER_ALIGNMENT]); block_count],
            logical_capacity,
        })
    }

    #[must_use]
    pub const fn logical_capacity(&self) -> usize {
        self.logical_capacity
    }

    #[must_use]
    pub fn padded_capacity(&self) -> usize {
        self.blocks.len() * LANES_PER_ALIGNMENT
    }

    #[must_use]
    pub fn alignment_remainder(&self) -> usize {
        self.blocks.as_ptr().addr() % 64
    }

    pub fn get(&self, lane: usize) -> Result<f64, AadConfigError> {
        self.check_lane(lane)?;
        Ok(self.blocks[lane / LANES_PER_ALIGNMENT].0[lane % LANES_PER_ALIGNMENT])
    }

    pub fn set(&mut self, lane: usize, value: f64) -> Result<(), AadConfigError> {
        self.check_lane(lane)?;
        self.blocks[lane / LANES_PER_ALIGNMENT].0[lane % LANES_PER_ALIGNMENT] = value;
        Ok(())
    }

    pub fn reset_positive_zero(&mut self) {
        for block in &mut self.blocks {
            block.0.fill(0.0);
        }
    }

    pub fn padded_lane_bits(&self) -> impl Iterator<Item = u64> + '_ {
        let logical_capacity = self.logical_capacity;
        self.blocks
            .iter()
            .flat_map(|block| block.0)
            .enumerate()
            .filter_map(move |(index, value)| {
                (index >= logical_capacity).then_some(value.to_bits())
            })
    }

    fn check_lane(&self, lane: usize) -> Result<(), AadConfigError> {
        if lane < self.logical_capacity {
            Ok(())
        } else {
            Err(AadConfigError::LaneOutOfRange {
                lane,
                logical_capacity: self.logical_capacity,
            })
        }
    }
}

/// State-major primal and adjoint buffers with distinct aligned allocations per slot.
#[derive(Clone, Debug)]
pub struct SoaWorkspace {
    primal: Vec<AlignedF64Buffer>,
    adjoint: Vec<AlignedF64Buffer>,
    logical_capacity: usize,
    padded_stride: usize,
}

impl SoaWorkspace {
    pub fn new(slot_count: usize, logical_capacity: usize) -> Result<Self, AadConfigError> {
        let padded_stride = logical_capacity
            .checked_add(LANES_PER_ALIGNMENT - 1)
            .ok_or(AadConfigError::WorkspaceSizeOverflow)?
            / LANES_PER_ALIGNMENT
            * LANES_PER_ALIGNMENT;
        slot_count
            .checked_mul(padded_stride)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(size_of::<f64>()))
            .ok_or(AadConfigError::WorkspaceSizeOverflow)?;
        let primal = (0..slot_count)
            .map(|_| AlignedF64Buffer::new(logical_capacity))
            .collect::<Result<Vec<_>, _>>()?;
        let adjoint = (0..slot_count)
            .map(|_| AlignedF64Buffer::new(logical_capacity))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            primal,
            adjoint,
            logical_capacity,
            padded_stride,
        })
    }

    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.primal.len()
    }

    #[must_use]
    pub const fn logical_capacity(&self) -> usize {
        self.logical_capacity
    }

    #[must_use]
    pub const fn padded_stride(&self) -> usize {
        self.padded_stride
    }

    pub fn primal(&self, slot: usize) -> Result<&AlignedF64Buffer, AadConfigError> {
        self.primal.get(slot).ok_or(AadConfigError::SlotOutOfRange {
            slot,
            slot_count: self.primal.len(),
        })
    }

    pub fn primal_mut(&mut self, slot: usize) -> Result<&mut AlignedF64Buffer, AadConfigError> {
        let slot_count = self.primal.len();
        self.primal
            .get_mut(slot)
            .ok_or(AadConfigError::SlotOutOfRange { slot, slot_count })
    }

    pub fn adjoint(&self, slot: usize) -> Result<&AlignedF64Buffer, AadConfigError> {
        self.adjoint
            .get(slot)
            .ok_or(AadConfigError::SlotOutOfRange {
                slot,
                slot_count: self.adjoint.len(),
            })
    }

    pub fn adjoint_mut(&mut self, slot: usize) -> Result<&mut AlignedF64Buffer, AadConfigError> {
        let slot_count = self.adjoint.len();
        self.adjoint
            .get_mut(slot)
            .ok_or(AadConfigError::SlotOutOfRange { slot, slot_count })
    }

    pub fn reset_positive_zero(&mut self) {
        for buffer in self.primal.iter_mut().chain(&mut self.adjoint) {
            buffer.reset_positive_zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_and_checkpoint_policies_resolve_explicitly() {
        let tile = AadTilePolicy::resolve(4096, NonZeroU32::new(128)).expect("tile");
        assert_eq!(tile.version(), 1);
        assert_eq!(tile.default_capacity(), 256);
        assert_eq!(tile.resolved_capacity().get(), 128);
        assert!(AadTilePolicy::resolve(64, NonZeroU32::new(65)).is_err());
        assert_eq!(
            AadTilePolicy::resolve(100, None)
                .expect("small block")
                .resolved_capacity()
                .get(),
            100
        );

        let checkpoint = CheckpointPolicy::resolve(NonZeroU32::new(16));
        assert_eq!(checkpoint.version(), 1);
        assert_eq!(checkpoint.default_interval().get(), 64);
        assert_eq!(checkpoint.resolved_interval().get(), 16);
    }

    #[test]
    fn aligned_buffer_uses_eight_lane_padding_and_masks_access() {
        let mut buffer = AlignedF64Buffer::new(10).expect("buffer");
        assert_eq!(buffer.alignment_remainder(), 0);
        assert_eq!(buffer.logical_capacity(), 10);
        assert_eq!(buffer.padded_capacity(), 16);
        buffer.set(9, -0.0).expect("live lane");
        assert_eq!(
            buffer.get(9).expect("live lane").to_bits(),
            (-0.0_f64).to_bits()
        );
        assert!(buffer.set(10, 1.0).is_err());
        buffer.reset_positive_zero();
        assert!(
            buffer
                .padded_lane_bits()
                .all(|bits| bits == 0.0_f64.to_bits())
        );
    }

    #[test]
    fn soa_slots_have_distinct_aligned_primal_and_adjoint_rows() {
        let mut workspace = SoaWorkspace::new(3, 9).expect("workspace");
        assert_eq!(workspace.slot_count(), 3);
        assert_eq!(workspace.padded_stride(), 16);
        for slot in 0..3 {
            assert_eq!(
                workspace.primal(slot).expect("slot").alignment_remainder(),
                0
            );
            assert_eq!(
                workspace.adjoint(slot).expect("slot").alignment_remainder(),
                0
            );
        }
        workspace
            .primal_mut(1)
            .expect("slot")
            .set(2, 4.0)
            .expect("lane");
        workspace
            .adjoint_mut(1)
            .expect("slot")
            .set(2, 7.0)
            .expect("lane");
        assert_eq!(workspace.primal(1).expect("slot").get(2), Ok(4.0));
        assert_eq!(workspace.adjoint(1).expect("slot").get(2), Ok(7.0));
        workspace.reset_positive_zero();
        assert_eq!(workspace.primal(1).expect("slot").get(2), Ok(0.0));
        assert_eq!(workspace.adjoint(1).expect("slot").get(2), Ok(0.0));
    }
}
