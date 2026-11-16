//! Two local option prices identify the two current OU coordinates.
//! All evaluations within a fit use the same random numbers.
use super::Error;
type Pair = [f64; 2];
type Matrix = [Pair; 2];

#[derive(Clone, Copy)]
pub(super) struct Fit {
    pub x: Pair,
    price: Pair,
    jacobian: Matrix,
}

fn max_abs(x: Pair) -> f64 {
    x[0].abs().max(x[1].abs())
}

fn inverse_step(j: Matrix, residual: Pair) -> Result<Pair, Error> {
    let [[a, b], [c, d]] = j;
    let det = a * d - b * c;
    let norm = (a.abs() + b.abs()).max(c.abs() + d.abs());
    let adj = (d.abs() + b.abs()).max(c.abs() + a.abs());
    // Reciprocal infinity-norm condition number, in Black-vol-scaled prices.
    if !det.is_finite() || det.abs() <= 1e-5 * norm * adj {
        return Err(Error::Invalid(
            "short and long option prices do not separately identify the two Bergomi states",
        ));
    }
    Ok([
        (d * residual[0] - b * residual[1]) / det,
        (a * residual[1] - c * residual[0]) / det,
    ])
}

impl Fit {
    pub fn start(x: Pair, price: &impl Fn(Pair) -> Result<Pair, Error>) -> Result<Self, Error> {
        Self {
            x,
            price: price(x)?,
            jacobian: [[0.; 2]; 2],
        }
        .linearized(price)
    }

    pub fn linearized(
        mut self,
        price: &impl Fn(Pair) -> Result<Pair, Error>,
    ) -> Result<Self, Error> {
        let mut jacobian = [[0.; 2]; 2];
        for j in 0..2 {
            let mut probe = self.x;
            probe[j] += 0.01;
            let other = price(probe)?;
            for i in 0..2 {
                jacobian[i][j] = (other[i] - self.price[i]) / 0.01;
            }
        }
        inverse_step(jacobian, [0.; 2])?;
        self.jacobian = jacobian;
        Ok(self)
    }

    pub fn solve(
        mut self,
        target: Pair,
        tolerance: f64,
        price: &impl Fn(Pair) -> Result<Pair, Error>,
    ) -> Result<Self, Error> {
        for _ in 0..24 {
            let residual = std::array::from_fn(|i| target[i] - self.price[i]);
            if max_abs(residual) <= tolerance {
                return Ok(self);
            }
            self = match self.advance(target, residual, price) {
                Ok(next) => next,
                // A secant Jacobian can become stale at a payoff kink.
                Err(_) => self.linearized(price)?.advance(target, residual, price)?,
            };
        }
        Err(Error::Invalid(
            "Bergomi state fit did not converge to the option prices",
        ))
    }

    fn advance(
        mut self,
        target: Pair,
        residual: Pair,
        price: &impl Fn(Pair) -> Result<Pair, Error>,
    ) -> Result<Self, Error> {
        let dx = inverse_step(self.jacobian, residual)?;
        // Backtracking controls the nonlinear fit without clipping the OU states.
        let mut scale = 1. / max_abs(dx).max(1.);
        let mut accepted = None;
        for _ in 0..10 {
            let step = dx.map(|v| scale * v);
            let x = std::array::from_fn(|i| self.x[i] + step[i]);
            if let Ok(p) = price(x)
                && max_abs(std::array::from_fn(|i| target[i] - p[i])) < max_abs(residual)
            {
                accepted = Some((x, p, step));
                break;
            }
            scale *= 0.5;
        }
        let (x, p, step) = accepted.ok_or(Error::Invalid(
            "Bergomi state fit could not reduce the option price error",
        ))?;
        let step_norm = step[0] * step[0] + step[1] * step[1];
        for (i, row) in self.jacobian.iter_mut().enumerate() {
            let correction =
                (p[i] - self.price[i] - row[0] * step[0] - row[1] * step[1]) / step_norm;
            row[0] += correction * step[0];
            row[1] += correction * step[1];
        }
        self.x = x;
        self.price = p;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_state_recovery_and_cross_tenor_refit() {
        let prices = |[a, b]: Pair| Ok([(a + 0.8 * b).exp(), (0.3 * a + b).exp()]);
        let truth = [0.24, -0.18];
        let target = prices(truth).unwrap();
        let initial = Fit::start([0., 0.], &prices).unwrap();
        let base = initial.solve(target, 1e-9, &prices).unwrap();
        assert!(max_abs([base.x[0] - truth[0], base.x[1] - truth[1]]) < 1e-8);
        let bumped = base
            .solve([target[0] + 0.03, target[1]], 1e-9, &prices)
            .unwrap();
        assert!(bumped.x[0] > base.x[0] && bumped.x[1] < base.x[1]);
        assert!((prices(bumped.x).unwrap()[1] - target[1]).abs() < 1e-9);
        assert!(Fit::start([0., 0.], &|[a, b]| Ok([a + b, 2. * (a + b)])).is_err());
    }
}
