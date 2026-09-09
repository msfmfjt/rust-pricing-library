use crate::{ImpliedVarianceSurface, MarketError, durrleman_density_factor};
use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};

const QUANTILE_EXPANSION_LIMIT: usize = 64;
const QUANTILE_BISECTION_STEPS: usize = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalVarianceRepairReason {
    Nan,
    NegativeInfinity,
    PositiveInfinity,
    NonPositive,
    BelowFloor,
    AboveCap,
}

impl LocalVarianceRepairReason {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Nan => "local_variance_nan",
            Self::NegativeInfinity => "local_variance_negative_infinity",
            Self::PositiveInfinity => "local_variance_positive_infinity",
            Self::NonPositive => "local_variance_non_positive",
            Self::BelowFloor => "local_variance_below_floor",
            Self::AboveCap => "local_variance_above_cap",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVarianceRepair {
    pub time_index: usize,
    pub log_moneyness_index: usize,
    pub time: f64,
    pub log_moneyness: f64,
    pub original_bits: u64,
    pub applied_value: f64,
    pub reason: LocalVarianceRepairReason,
    pub policy: &'static str,
    pub occurrence_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalVarianceBoundary {
    InRange,
    LeftFlat,
    RightFlat,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LocalVarianceBoundaryStats {
    pub left_flat_count: u64,
    pub right_flat_count: u64,
    pub max_left_excursion: f64,
    pub max_right_excursion: f64,
}

impl LocalVarianceBoundaryStats {
    pub fn record(&mut self, interpolation: LocalVarianceInterpolation) -> Result<(), MarketError> {
        match interpolation.boundary {
            LocalVarianceBoundary::InRange => {}
            LocalVarianceBoundary::LeftFlat => {
                self.left_flat_count = self.left_flat_count.checked_add(1).ok_or(
                    MarketError::LocalVarianceBoundaryCountOverflow {
                        boundary: "left_flat",
                    },
                )?;
                self.max_left_excursion = self
                    .max_left_excursion
                    .max(interpolation.boundary_excursion);
            }
            LocalVarianceBoundary::RightFlat => {
                self.right_flat_count = self.right_flat_count.checked_add(1).ok_or(
                    MarketError::LocalVarianceBoundaryCountOverflow {
                        boundary: "right_flat",
                    },
                )?;
                self.max_right_excursion = self
                    .max_right_excursion
                    .max(interpolation.boundary_excursion);
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn total_flat_count(self) -> u64 {
        self.left_flat_count + self.right_flat_count
    }

    #[must_use]
    pub fn max_excursion(self) -> f64 {
        self.max_left_excursion.max(self.max_right_excursion)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalVarianceInterpolation {
    pub value: f64,
    pub lower_time_index: usize,
    pub lower_log_moneyness_index: usize,
    pub time_weight: f64,
    pub log_moneyness_weight: f64,
    pub boundary: LocalVarianceBoundary,
    pub boundary_excursion: f64,
}

impl LocalVarianceInterpolation {
    pub fn transpose_accumulate(self, seed: f64, adjoints: &mut [f64], x_count: usize) {
        let row = self.lower_time_index * x_count;
        let next_row = row + x_count;
        let left = self.lower_log_moneyness_index;
        let right = (left + 1).min(x_count - 1);
        let time_left = 1.0 - self.time_weight;
        let time_right = self.time_weight;
        let x_left = 1.0 - self.log_moneyness_weight;
        let x_right = self.log_moneyness_weight;
        adjoints[row + left] += seed * time_left * x_left;
        adjoints[row + right] += seed * time_left * x_right;
        adjoints[next_row + left] += seed * time_right * x_left;
        adjoints[next_row + right] += seed * time_right * x_right;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVarianceGrid {
    time_nodes: Box<[f64]>,
    log_moneyness_nodes: Box<[f64]>,
    values: Box<[f64]>,
    floor: f64,
    cap: f64,
    repairs: Box<[LocalVarianceRepair]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalVarianceGridSuggestion {
    pub left_tail_probability: f64,
    pub right_tail_probability: f64,
    pub left_padding: f64,
    pub right_padding: f64,
    pub left_interval_count: usize,
    pub right_interval_count: usize,
    pub left_shape: f64,
    pub right_shape: f64,
}

impl LocalVarianceGrid {
    pub fn from_surface(
        surface: &dyn ImpliedVarianceSurface,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, MarketError> {
        validate_nodes("time", &time_nodes, true)?;
        validate_nodes("log_moneyness", &log_moneyness_nodes, false)?;
        validate_floor_cap(floor, cap)?;

        let mut values = Vec::with_capacity(time_nodes.len() * log_moneyness_nodes.len());
        let mut repairs = Vec::new();
        for (time_index, time) in time_nodes.iter().copied().enumerate() {
            for (log_moneyness_index, log_moneyness) in
                log_moneyness_nodes.iter().copied().enumerate()
            {
                let variance = surface.total_variance_derivatives(time, log_moneyness)?;
                if !variance.time_derivative.is_finite() {
                    return Err(MarketError::NonFiniteSurfaceValue {
                        field: "time_derivative",
                        time_bits: time.to_bits(),
                        log_moneyness_bits: log_moneyness.to_bits(),
                        value_bits: variance.time_derivative.to_bits(),
                    });
                }
                let density_factor = durrleman_density_factor(time, log_moneyness, variance)?;
                let raw = variance.time_derivative / density_factor;
                let (value, reason) = classify_local_variance(raw, floor, cap);
                if let Some(reason) = reason {
                    repairs.push(LocalVarianceRepair {
                        time_index,
                        log_moneyness_index,
                        time,
                        log_moneyness,
                        original_bits: raw.to_bits(),
                        applied_value: value,
                        reason,
                        policy: "mandatory_floor_cap",
                        occurrence_index: repairs.len(),
                    });
                }
                values.push(value);
            }
        }

        Ok(Self {
            time_nodes: time_nodes.into_boxed_slice(),
            log_moneyness_nodes: log_moneyness_nodes.into_boxed_slice(),
            values: values.into_boxed_slice(),
            floor,
            cap,
            repairs: repairs.into_boxed_slice(),
        })
    }

    pub fn new(
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        values: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, MarketError> {
        validate_nodes("time", &time_nodes, true)?;
        validate_nodes("log_moneyness", &log_moneyness_nodes, false)?;
        validate_floor_cap(floor, cap)?;
        let expected = time_nodes.len() * log_moneyness_nodes.len();
        if values.len() != expected {
            return Err(MarketError::LocalVarianceValueLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() || value < floor || value > cap {
                return Err(MarketError::InvalidLocalVarianceValue {
                    index,
                    bits: value.to_bits(),
                });
            }
        }
        Ok(Self {
            time_nodes: time_nodes.into_boxed_slice(),
            log_moneyness_nodes: log_moneyness_nodes.into_boxed_slice(),
            values: values.into_boxed_slice(),
            floor,
            cap,
            repairs: Box::default(),
        })
    }

    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        &self.time_nodes
    }

    #[must_use]
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_moneyness_nodes
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn floor(&self) -> f64 {
        self.floor
    }

    #[must_use]
    pub const fn cap(&self) -> f64 {
        self.cap
    }

    #[must_use]
    pub fn repairs(&self) -> &[LocalVarianceRepair] {
        &self.repairs
    }

    pub fn interpolate(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<LocalVarianceInterpolation, MarketError> {
        if !time.is_finite()
            || time < self.time_nodes[0]
            || time > self.time_nodes[self.time_nodes.len() - 1]
        {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        if !log_moneyness.is_finite() {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "log_moneyness",
                bits: log_moneyness.to_bits(),
            });
        }
        let time_cell = lower_cell(&self.time_nodes, time);
        let (x, boundary, boundary_excursion) = if log_moneyness < self.log_moneyness_nodes[0] {
            (
                self.log_moneyness_nodes[0],
                LocalVarianceBoundary::LeftFlat,
                self.log_moneyness_nodes[0] - log_moneyness,
            )
        } else if log_moneyness > self.log_moneyness_nodes[self.log_moneyness_nodes.len() - 1] {
            (
                self.log_moneyness_nodes[self.log_moneyness_nodes.len() - 1],
                LocalVarianceBoundary::RightFlat,
                log_moneyness - self.log_moneyness_nodes[self.log_moneyness_nodes.len() - 1],
            )
        } else {
            (log_moneyness, LocalVarianceBoundary::InRange, 0.0)
        };
        let x_cell = lower_cell(&self.log_moneyness_nodes, x);
        let time_weight = weight(
            self.time_nodes[time_cell],
            self.time_nodes[time_cell + 1],
            time,
        );
        let log_moneyness_weight = weight(
            self.log_moneyness_nodes[x_cell],
            self.log_moneyness_nodes[x_cell + 1],
            x,
        );
        let x_count = self.log_moneyness_nodes.len();
        let row = time_cell * x_count;
        let next_row = row + x_count;
        let v00 = self.values[row + x_cell];
        let v01 = self.values[row + x_cell + 1];
        let v10 = self.values[next_row + x_cell];
        let v11 = self.values[next_row + x_cell + 1];
        let lower = v00 * (1.0 - log_moneyness_weight) + v01 * log_moneyness_weight;
        let upper = v10 * (1.0 - log_moneyness_weight) + v11 * log_moneyness_weight;
        Ok(LocalVarianceInterpolation {
            value: lower * (1.0 - time_weight) + upper * time_weight,
            lower_time_index: time_cell,
            lower_log_moneyness_index: x_cell,
            time_weight,
            log_moneyness_weight,
            boundary,
            boundary_excursion,
        })
    }

    pub fn interpolate_and_record(
        &self,
        time: f64,
        log_moneyness: f64,
        stats: &mut LocalVarianceBoundaryStats,
    ) -> Result<LocalVarianceInterpolation, MarketError> {
        let interpolation = self.interpolate(time, log_moneyness)?;
        stats.record(interpolation)?;
        Ok(interpolation)
    }
}

pub fn piecewise_sinh_log_moneyness_nodes(
    left_boundary: f64,
    right_boundary: f64,
    left_interval_count: usize,
    right_interval_count: usize,
    left_shape: f64,
    right_shape: f64,
) -> Result<Vec<f64>, MarketError> {
    if !left_boundary.is_finite() || left_boundary >= 0.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter: "left_log_moneyness_boundary",
            bits: left_boundary.to_bits(),
        });
    }
    if !right_boundary.is_finite() || right_boundary <= 0.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter: "right_log_moneyness_boundary",
            bits: right_boundary.to_bits(),
        });
    }
    if left_interval_count == 0 {
        return Err(MarketError::InvalidLocalVarianceNodeCount {
            coordinate: "left_log_moneyness_intervals",
            count: left_interval_count,
        });
    }
    if right_interval_count == 0 {
        return Err(MarketError::InvalidLocalVarianceNodeCount {
            coordinate: "right_log_moneyness_intervals",
            count: right_interval_count,
        });
    }
    for (parameter, value) in [
        ("left_piecewise_sinh_shape", left_shape),
        ("right_piecewise_sinh_shape", right_shape),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MarketError::InvalidSurfaceParameter {
                parameter,
                bits: value.to_bits(),
            });
        }
    }

    let mut nodes = Vec::with_capacity(left_interval_count + right_interval_count + 1);
    for index in 0..=left_interval_count {
        let fraction = index as f64 / left_interval_count as f64;
        let ratio = sinh_ratio(left_shape, 1.0 - fraction);
        let node = if index == left_interval_count {
            0.0
        } else {
            left_boundary * ratio
        };
        nodes.push(node);
    }
    for index in 1..=right_interval_count {
        let fraction = index as f64 / right_interval_count as f64;
        nodes.push(right_boundary * sinh_ratio(right_shape, fraction));
    }
    validate_nodes("log_moneyness", &nodes, false)?;
    Ok(nodes)
}

pub fn suggest_log_moneyness_nodes_from_density(
    surface: &dyn ImpliedVarianceSurface,
    times: &[f64],
    config: LocalVarianceGridSuggestion,
) -> Result<Vec<f64>, MarketError> {
    validate_helper_times(times)?;
    validate_tail_probability("left_tail_probability", config.left_tail_probability)?;
    validate_tail_probability("right_tail_probability", config.right_tail_probability)?;
    validate_padding("left_log_moneyness_padding", config.left_padding)?;
    validate_padding("right_log_moneyness_padding", config.right_padding)?;

    let mut left_boundary = 0.0_f64;
    let mut right_boundary = 0.0_f64;
    for time in times.iter().copied() {
        let left = solve_left_tail_boundary(surface, time, config.left_tail_probability)?;
        let right = solve_right_tail_boundary(surface, time, config.right_tail_probability)?;
        left_boundary = left_boundary.min(left);
        right_boundary = right_boundary.max(right);
    }
    piecewise_sinh_log_moneyness_nodes(
        left_boundary - config.left_padding,
        right_boundary + config.right_padding,
        config.left_interval_count,
        config.right_interval_count,
        config.left_shape,
        config.right_shape,
    )
}

fn validate_nodes(
    coordinate: &'static str,
    values: &[f64],
    reject_negative: bool,
) -> Result<(), MarketError> {
    if values.len() < 2 {
        return Err(MarketError::InvalidLocalVarianceNodeCount {
            coordinate,
            count: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || (reject_negative && value < 0.0) {
            return Err(MarketError::InvalidLocalVarianceNode {
                coordinate,
                index,
                bits: value.to_bits(),
            });
        }
        if index > 0 && value <= values[index - 1] {
            return Err(MarketError::UnsortedLocalVarianceNodes {
                coordinate,
                left_index: index - 1,
                left_bits: values[index - 1].to_bits(),
                right_bits: value.to_bits(),
            });
        }
    }
    Ok(())
}

fn validate_helper_times(times: &[f64]) -> Result<(), MarketError> {
    if times.is_empty() {
        return Err(MarketError::InvalidLocalVarianceNodeCount {
            coordinate: "density_helper_times",
            count: times.len(),
        });
    }
    for (index, time) in times.iter().copied().enumerate() {
        if !time.is_finite() || time <= 0.0 {
            return Err(MarketError::InvalidLocalVarianceNode {
                coordinate: "density_helper_time",
                index,
                bits: time.to_bits(),
            });
        }
    }
    Ok(())
}

fn validate_floor_cap(floor: f64, cap: f64) -> Result<(), MarketError> {
    if !floor.is_finite() || floor <= 0.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter: "local_variance_floor",
            bits: floor.to_bits(),
        });
    }
    if !cap.is_finite() || cap < floor {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter: "local_variance_cap",
            bits: cap.to_bits(),
        });
    }
    Ok(())
}

fn validate_tail_probability(parameter: &'static str, value: f64) -> Result<(), MarketError> {
    if !value.is_finite() || value <= 0.0 || value >= 0.5 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

fn validate_padding(parameter: &'static str, value: f64) -> Result<(), MarketError> {
    if !value.is_finite() || value < 0.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

fn classify_local_variance(
    raw: f64,
    floor: f64,
    cap: f64,
) -> (f64, Option<LocalVarianceRepairReason>) {
    if raw.is_nan() {
        (floor, Some(LocalVarianceRepairReason::Nan))
    } else if raw == f64::NEG_INFINITY {
        (floor, Some(LocalVarianceRepairReason::NegativeInfinity))
    } else if raw == f64::INFINITY {
        (cap, Some(LocalVarianceRepairReason::PositiveInfinity))
    } else if raw <= 0.0 {
        (floor, Some(LocalVarianceRepairReason::NonPositive))
    } else if raw < floor {
        (floor, Some(LocalVarianceRepairReason::BelowFloor))
    } else if raw > cap {
        (cap, Some(LocalVarianceRepairReason::AboveCap))
    } else {
        (raw, None)
    }
}

fn lower_cell(nodes: &[f64], value: f64) -> usize {
    match nodes.binary_search_by(|node| node.total_cmp(&value)) {
        Ok(index) => index.min(nodes.len() - 2),
        Err(index) => index.saturating_sub(1).min(nodes.len() - 2),
    }
}

fn weight(left: f64, right: f64, value: f64) -> f64 {
    (value - left) / (right - left)
}

fn sinh_ratio(shape: f64, fraction: f64) -> f64 {
    if shape == 0.0 {
        fraction
    } else {
        (shape * fraction).sinh() / shape.sinh()
    }
}

fn solve_left_tail_boundary(
    surface: &dyn ImpliedVarianceSurface,
    time: f64,
    probability: f64,
) -> Result<f64, MarketError> {
    let mut left = -1.0;
    for _ in 0..QUANTILE_EXPANSION_LIMIT {
        if forward_log_moneyness_cdf(surface, time, left)? <= probability {
            let mut lower = left;
            let mut upper = 0.0;
            for _ in 0..QUANTILE_BISECTION_STEPS {
                let middle = 0.5 * (lower + upper);
                if forward_log_moneyness_cdf(surface, time, middle)? <= probability {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            return Ok(upper);
        }
        left *= 2.0;
    }
    Err(MarketError::SurfaceQuantileNotBracketed {
        side: "left",
        time_bits: time.to_bits(),
        probability_bits: probability.to_bits(),
    })
}

fn solve_right_tail_boundary(
    surface: &dyn ImpliedVarianceSurface,
    time: f64,
    probability: f64,
) -> Result<f64, MarketError> {
    let mut right = 1.0;
    for _ in 0..QUANTILE_EXPANSION_LIMIT {
        if forward_log_moneyness_survival(surface, time, right)? <= probability {
            let mut lower = 0.0;
            let mut upper = right;
            for _ in 0..QUANTILE_BISECTION_STEPS {
                let middle = 0.5 * (lower + upper);
                if forward_log_moneyness_survival(surface, time, middle)? <= probability {
                    upper = middle;
                } else {
                    lower = middle;
                }
            }
            return Ok(upper);
        }
        right *= 2.0;
    }
    Err(MarketError::SurfaceQuantileNotBracketed {
        side: "right",
        time_bits: time.to_bits(),
        probability_bits: probability.to_bits(),
    })
}

fn forward_log_moneyness_cdf(
    surface: &dyn ImpliedVarianceSurface,
    time: f64,
    log_moneyness: f64,
) -> Result<f64, MarketError> {
    let survival = forward_log_moneyness_survival(surface, time, log_moneyness)?;
    Ok(1.0 - survival)
}

fn forward_log_moneyness_survival(
    surface: &dyn ImpliedVarianceSurface,
    time: f64,
    log_moneyness: f64,
) -> Result<f64, MarketError> {
    let variance = surface.total_variance_derivatives(time, log_moneyness)?;
    let root_variance = variance.total_variance.sqrt();
    let d2 = -log_moneyness / root_variance - root_variance / 2.0;
    let survival = standard_normal_cdf(d2)
        - standard_normal_pdf(d2) * variance.log_moneyness_derivative / (2.0 * root_variance);
    if !survival.is_finite() || !(-1.0e-12..=1.0 + 1.0e-12).contains(&survival) {
        return Err(MarketError::NonFiniteSurfaceValue {
            field: "forward_log_moneyness_survival",
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: survival.to_bits(),
        });
    }
    Ok(survival.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ThetaRegion, TotalVarianceDerivatives};

    struct ConstantVarianceSurface {
        variance: f64,
    }

    impl ImpliedVarianceSurface for ConstantVarianceSurface {
        fn total_variance_derivatives(
            &self,
            time: f64,
            _log_moneyness: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            Ok(TotalVarianceDerivatives {
                total_variance: self.variance * time,
                log_moneyness_derivative: 0.0,
                log_moneyness_second_derivative: 0.0,
                time_derivative: self.variance,
                theta: self.variance * time,
                theta_derivative: self.variance,
                theta_region: ThetaRegion::Interpolated,
            })
        }
    }

    struct RawLocalVarianceSurface {
        raw: f64,
    }

    impl ImpliedVarianceSurface for RawLocalVarianceSurface {
        fn total_variance_derivatives(
            &self,
            _time: f64,
            _log_moneyness: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            Ok(TotalVarianceDerivatives {
                total_variance: 0.04,
                log_moneyness_derivative: 0.0,
                log_moneyness_second_derivative: 0.0,
                time_derivative: self.raw,
                theta: 0.04,
                theta_derivative: self.raw,
                theta_region: ThetaRegion::Interpolated,
            })
        }
    }

    #[test]
    fn dupire_grid_reproduces_constant_variance_surface() {
        let surface = ConstantVarianceSurface { variance: 0.09 };
        let grid = LocalVarianceGrid::from_surface(
            &surface,
            vec![0.5, 1.0, 2.0],
            vec![-0.2, 0.0, 0.3],
            0.0001,
            1.0,
        )
        .expect("grid");
        assert!(grid.repairs().is_empty());
        assert!(
            grid.values()
                .iter()
                .all(|value| (*value - 0.09).abs() < 1.0e-15)
        );
    }

    #[test]
    fn raw_local_variance_is_repaired_in_priority_order() {
        for (raw, expected_value, expected_reason) in [
            (f64::NAN, 0.01, LocalVarianceRepairReason::Nan),
            (
                f64::NEG_INFINITY,
                0.01,
                LocalVarianceRepairReason::NegativeInfinity,
            ),
            (
                f64::INFINITY,
                0.5,
                LocalVarianceRepairReason::PositiveInfinity,
            ),
            (0.0, 0.01, LocalVarianceRepairReason::NonPositive),
            (0.001, 0.01, LocalVarianceRepairReason::BelowFloor),
            (0.6, 0.5, LocalVarianceRepairReason::AboveCap),
        ] {
            let (value, reason) = classify_local_variance(raw, 0.01, 0.5);
            assert_eq!(value, expected_value);
            assert_eq!(reason, Some(expected_reason));
            assert!(expected_reason.code().starts_with("local_variance_"));
        }
    }

    #[test]
    fn finite_repairs_are_recorded_in_row_major_order() {
        let surface = RawLocalVarianceSurface { raw: 0.001 };
        let grid =
            LocalVarianceGrid::from_surface(&surface, vec![1.0, 2.0], vec![-0.1, 0.1], 0.01, 0.5)
                .expect("repaired grid");
        assert_eq!(grid.repairs().len(), 4);
        for (index, repair) in grid.repairs().iter().enumerate() {
            assert_eq!(repair.reason, LocalVarianceRepairReason::BelowFloor);
            assert_eq!(repair.occurrence_index, index);
        }
        assert_eq!(grid.repairs()[2].time_index, 1);
        assert_eq!(grid.repairs()[2].log_moneyness_index, 0);
        assert!(grid.values().iter().all(|value| *value == 0.01));
    }

    #[test]
    fn bilinear_interpolation_and_transpose_use_same_weights() {
        let grid = LocalVarianceGrid::new(
            vec![1.0, 2.0],
            vec![-1.0, 1.0],
            vec![1.0, 3.0, 5.0, 7.0],
            0.5,
            8.0,
        )
        .expect("grid");
        let interpolation = grid.interpolate(1.25, 0.0).expect("interpolate");
        assert_eq!(interpolation.boundary, LocalVarianceBoundary::InRange);
        assert!((interpolation.value - 3.0).abs() < 1.0e-15);

        let mut adjoints = vec![0.0; grid.values().len()];
        interpolation.transpose_accumulate(2.0, &mut adjoints, grid.log_moneyness_nodes().len());
        assert_eq!(adjoints, vec![0.75, 0.75, 0.25, 0.25]);
    }

    #[test]
    fn horizontal_extrapolation_is_flat_at_boundary() {
        let grid = LocalVarianceGrid::new(
            vec![1.0, 2.0],
            vec![-1.0, 1.0],
            vec![1.0, 3.0, 5.0, 7.0],
            0.5,
            8.0,
        )
        .expect("grid");
        let left = grid.interpolate(1.5, -2.5).expect("left flat");
        let right = grid.interpolate(1.5, 3.0).expect("right flat");
        assert_eq!(left.boundary, LocalVarianceBoundary::LeftFlat);
        assert_eq!(right.boundary, LocalVarianceBoundary::RightFlat);
        assert_eq!(left.boundary_excursion, 1.5);
        assert_eq!(right.boundary_excursion, 2.0);
        assert!((left.value - 3.0).abs() < 1.0e-15);
        assert!((right.value - 5.0).abs() < 1.0e-15);
    }

    #[test]
    fn boundary_stats_record_counts_and_maximum_excursions() {
        let grid = LocalVarianceGrid::new(
            vec![1.0, 2.0],
            vec![-1.0, 1.0],
            vec![1.0, 3.0, 5.0, 7.0],
            0.5,
            8.0,
        )
        .expect("grid");
        let mut stats = LocalVarianceBoundaryStats::default();
        grid.interpolate_and_record(1.5, 0.0, &mut stats)
            .expect("in range");
        grid.interpolate_and_record(1.5, -1.25, &mut stats)
            .expect("left one");
        grid.interpolate_and_record(1.5, -2.0, &mut stats)
            .expect("left two");
        grid.interpolate_and_record(1.5, 3.5, &mut stats)
            .expect("right");

        assert_eq!(stats.left_flat_count, 2);
        assert_eq!(stats.right_flat_count, 1);
        assert_eq!(stats.total_flat_count(), 3);
        assert_eq!(stats.max_left_excursion, 1.0);
        assert_eq!(stats.max_right_excursion, 2.5);
        assert_eq!(stats.max_excursion(), 2.5);
    }

    #[test]
    fn piecewise_sinh_nodes_are_strictly_ordered_with_exact_atm() {
        let nodes =
            piecewise_sinh_log_moneyness_nodes(-0.8, 0.5, 4, 3, 1.5, 0.0).expect("generated nodes");
        assert_eq!(nodes.len(), 8);
        assert_eq!(nodes[0], -0.8);
        assert_eq!(nodes[4].to_bits(), 0.0_f64.to_bits());
        assert_eq!(nodes[7], 0.5);
        assert!(nodes.windows(2).all(|pair| pair[1] > pair[0]));
        assert!((nodes[5] - 0.5 / 3.0).abs() < 1.0e-15);
    }

    #[test]
    fn piecewise_sinh_nodes_reject_invalid_boundaries_counts_and_shapes() {
        assert!(piecewise_sinh_log_moneyness_nodes(0.0, 0.5, 4, 3, 1.5, 0.0).is_err());
        assert!(piecewise_sinh_log_moneyness_nodes(-0.8, -0.5, 4, 3, 1.5, 0.0).is_err());
        assert!(piecewise_sinh_log_moneyness_nodes(-0.8, 0.5, 0, 3, 1.5, 0.0).is_err());
        assert!(piecewise_sinh_log_moneyness_nodes(-0.8, 0.5, 4, 0, 1.5, 0.0).is_err());
        assert!(piecewise_sinh_log_moneyness_nodes(-0.8, 0.5, 4, 3, f64::NAN, 0.0).is_err());
    }

    #[test]
    fn density_helper_materializes_rectangular_tail_covering_nodes() {
        let surface = ConstantVarianceSurface { variance: 0.04 };
        let nodes = suggest_log_moneyness_nodes_from_density(
            &surface,
            &[0.5, 1.0, 2.0],
            LocalVarianceGridSuggestion {
                left_tail_probability: 0.01,
                right_tail_probability: 0.02,
                left_padding: 0.05,
                right_padding: 0.07,
                left_interval_count: 5,
                right_interval_count: 6,
                left_shape: 1.0,
                right_shape: 1.0,
            },
        )
        .expect("nodes");
        assert_eq!(nodes.len(), 12);
        assert_eq!(nodes[5].to_bits(), 0.0_f64.to_bits());
        assert!(nodes.windows(2).all(|pair| pair[1] > pair[0]));
        for time in [0.5, 1.0, 2.0] {
            assert!(forward_log_moneyness_cdf(&surface, time, nodes[0]).expect("left cdf") < 0.01);
            assert!(
                forward_log_moneyness_survival(&surface, time, nodes[nodes.len() - 1])
                    .expect("right survival")
                    < 0.02
            );
        }
    }

    #[test]
    fn density_helper_rejects_invalid_tail_inputs() {
        let surface = ConstantVarianceSurface { variance: 0.04 };
        assert!(
            suggest_log_moneyness_nodes_from_density(
                &surface,
                &[],
                LocalVarianceGridSuggestion {
                    left_tail_probability: 0.01,
                    right_tail_probability: 0.02,
                    left_padding: 0.05,
                    right_padding: 0.07,
                    left_interval_count: 5,
                    right_interval_count: 6,
                    left_shape: 1.0,
                    right_shape: 1.0,
                },
            )
            .is_err()
        );
        assert!(
            suggest_log_moneyness_nodes_from_density(
                &surface,
                &[1.0],
                LocalVarianceGridSuggestion {
                    left_tail_probability: 0.5,
                    right_tail_probability: 0.02,
                    left_padding: 0.05,
                    right_padding: 0.07,
                    left_interval_count: 5,
                    right_interval_count: 6,
                    left_shape: 1.0,
                    right_shape: 1.0,
                },
            )
            .is_err()
        );
    }
}
