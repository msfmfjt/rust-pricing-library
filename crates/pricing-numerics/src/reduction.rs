/// A scalar Neumaier-compensated sum with a fixed update order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NeumaierSum {
    sum: f64,
    correction: f64,
}

impl NeumaierSum {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    pub fn add(&mut self, value: f64) {
        let updated = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - updated) + value;
        } else {
            self.correction += (value - updated) + self.sum;
        }
        self.sum = updated;
    }

    #[must_use]
    pub const fn sum(self) -> f64 {
        self.sum
    }

    #[must_use]
    pub const fn correction(self) -> f64 {
        self.correction
    }

    #[must_use]
    pub fn total(self) -> f64 {
        self.sum + self.correction
    }

    #[must_use]
    pub fn merged(left: Self, right: Self) -> Self {
        let mut merged = Self::new();
        merged.add(left.sum);
        merged.add(left.correction);
        merged.add(right.sum);
        merged.add(right.correction);
        merged
    }
}

impl Extend<f64> for NeumaierSum {
    fn extend<T: IntoIterator<Item = f64>>(&mut self, values: T) {
        for value in values {
            self.add(value);
        }
    }
}

impl FromIterator<f64> for NeumaierSum {
    fn from_iter<T: IntoIterator<Item = f64>>(values: T) -> Self {
        let mut accumulator = Self::new();
        accumulator.extend(values);
        accumulator
    }
}

/// Centered scalar moments combined by the Chan–Golub–LeVeque formula.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CenteredMoment {
    count: u64,
    mean: f64,
    second_moment: f64,
}

impl CenteredMoment {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            second_moment: 0.0,
        }
    }

    pub fn add(&mut self, value: f64) {
        let next_count = self.count + 1;
        let delta = value - self.mean;
        let next_mean = self.mean + delta / next_count as f64;
        self.second_moment += delta * (value - next_mean);
        self.mean = next_mean;
        self.count = next_count;
    }

    #[must_use]
    pub fn merged(left: Self, right: Self) -> Self {
        if left.count == 0 {
            return right;
        }
        if right.count == 0 {
            return left;
        }
        let count = left.count + right.count;
        let delta = right.mean - left.mean;
        let left_weight = left.count as f64;
        let right_weight = right.count as f64;
        let count_as_f64 = count as f64;
        let mean = left.mean + delta * (right_weight / count_as_f64);
        let between = delta * delta * (left_weight * right_weight / count_as_f64);
        let second_moment = (left.second_moment + right.second_moment) + between;
        Self {
            count,
            mean,
            second_moment,
        }
    }

    #[must_use]
    pub const fn count(self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }

    #[must_use]
    pub const fn second_moment(self) -> f64 {
        self.second_moment
    }

    #[must_use]
    pub fn sample_variance(self) -> Option<f64> {
        (self.count > 1).then(|| self.second_moment / (self.count - 1) as f64)
    }
}

/// Centered bivariate cross-moment for deterministic covariance diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CenteredCovariance {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    cross_moment: f64,
}

impl CenteredCovariance {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            cross_moment: 0.0,
        }
    }

    pub fn add(&mut self, x: f64, y: f64) {
        let next_count = self.count + 1;
        let delta_x = x - self.mean_x;
        let delta_y = y - self.mean_y;
        let next_mean_x = self.mean_x + delta_x / next_count as f64;
        let next_mean_y = self.mean_y + delta_y / next_count as f64;
        self.cross_moment += delta_x * (y - next_mean_y);
        self.mean_x = next_mean_x;
        self.mean_y = next_mean_y;
        self.count = next_count;
    }

    #[must_use]
    pub fn merged(left: Self, right: Self) -> Self {
        if left.count == 0 {
            return right;
        }
        if right.count == 0 {
            return left;
        }
        let count = left.count + right.count;
        let delta_x = right.mean_x - left.mean_x;
        let delta_y = right.mean_y - left.mean_y;
        let left_weight = left.count as f64;
        let right_weight = right.count as f64;
        let count_as_f64 = count as f64;
        let ratio = right_weight / count_as_f64;
        let mean_x = left.mean_x + delta_x * ratio;
        let mean_y = left.mean_y + delta_y * ratio;
        let between = delta_x * delta_y * (left_weight * right_weight / count_as_f64);
        let cross_moment = (left.cross_moment + right.cross_moment) + between;
        Self {
            count,
            mean_x,
            mean_y,
            cross_moment,
        }
    }

    #[must_use]
    pub const fn count(self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn means(self) -> (f64, f64) {
        (self.mean_x, self.mean_y)
    }

    #[must_use]
    pub const fn cross_moment(self) -> f64 {
        self.cross_moment
    }

    #[must_use]
    pub fn sample_covariance(self) -> Option<f64> {
        (self.count > 1).then(|| self.cross_moment / (self.count - 1) as f64)
    }
}

#[must_use]
pub fn reduce_sums(partials: Vec<NeumaierSum>) -> NeumaierSum {
    fixed_tree_reduce(partials, NeumaierSum::merged).unwrap_or_default()
}

#[must_use]
pub fn reduce_moments(partials: Vec<CenteredMoment>) -> CenteredMoment {
    fixed_tree_reduce(partials, CenteredMoment::merged).unwrap_or_default()
}

#[must_use]
pub fn reduce_covariances(partials: Vec<CenteredCovariance>) -> CenteredCovariance {
    fixed_tree_reduce(partials, CenteredCovariance::merged).unwrap_or_default()
}

fn fixed_tree_reduce<T: Copy>(mut level: Vec<T>, merge: fn(T, T) -> T) -> Option<T> {
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let (pairs, remainder) = level.as_chunks::<2>();
        for pair in pairs {
            next.push(merge(pair[0], pair[1]));
        }
        if let Some(last) = remainder.first() {
            next.push(*last);
        }
        level = next;
    }
    level.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_small_terms_and_has_fixed_merge_order() {
        let values = [1.0e16, 1.0, -1.0e16, 2.0];
        let total = values.into_iter().collect::<NeumaierSum>().total();
        assert_eq!(total, 3.0);

        let left = [1.0e16, 1.0].into_iter().collect();
        let right = [-1.0e16, 2.0].into_iter().collect();
        let expected = NeumaierSum::merged(left, right);
        assert_eq!(reduce_sums(vec![left, right]), expected);
    }

    #[test]
    fn balanced_tree_carries_an_unpaired_rightmost_partial() {
        let partials = [1.0, 2.0, 4.0]
            .into_iter()
            .map(|value| [value].into_iter().collect())
            .collect();
        assert_eq!(reduce_sums(partials).total(), 7.0);
    }

    #[test]
    fn centered_moment_merge_matches_known_sample() {
        let mut left = CenteredMoment::new();
        let mut right = CenteredMoment::new();
        for value in [1.0, 2.0] {
            left.add(value);
        }
        for value in [3.0, 4.0] {
            right.add(value);
        }
        let merged = reduce_moments(vec![left, right]);
        assert_eq!(merged.count(), 4);
        assert_eq!(merged.mean(), 2.5);
        assert_eq!(merged.sample_variance(), Some(5.0 / 3.0));
    }

    #[test]
    fn centered_covariance_merge_matches_known_sample() {
        let mut left = CenteredCovariance::new();
        let mut right = CenteredCovariance::new();
        left.add(1.0, 2.0);
        left.add(2.0, 4.0);
        right.add(3.0, 6.0);
        right.add(4.0, 8.0);
        let merged = reduce_covariances(vec![left, right]);
        assert_eq!(merged.count(), 4);
        assert_eq!(merged.means(), (2.5, 5.0));
        assert_eq!(merged.sample_covariance(), Some(10.0 / 3.0));
    }
}
