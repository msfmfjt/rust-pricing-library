//! Optional reverse of a realized calibration. Price-only models need not
//! implement this capability. Associated outputs keep a variance-only target
//! distinct from HW variance/density/curve exposure and coupled basket targets.

pub(crate) trait CalibrationReverse {
    type Adjoints;
    type Error;

    /// Check the actual retained primal before starting an independent pricing
    /// simulation. Implementations retain their trace and active-set policies.
    fn validate_calibration_reverse(&self) -> Result<(), Self::Error>;
    fn calibration_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::LocalVarianceGrid;
    use crate::mc::hull_white::{HullWhiteLsvTarget, calibrate_hull_white_lsv};
    use crate::mc::lsv::{LsvError, LsvParticleConfig, calibrate_bergomi_lsv};
    use crate::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation};

    #[test]
    fn zero_vol_of_vol_still_requires_the_realized_calibration_trace() {
        let target = LocalVarianceGrid::new(
            vec![0.0, 0.5, 1.0],
            vec![-0.8, 0.0, 0.8],
            vec![0.04; 9],
            1e-8,
            4.0,
        )
        .unwrap();
        let c = calibrate_bergomi_lsv(
            &target,
            Bergomi1Factor::new(0.7, 0.0, -0.4).unwrap(),
            100.0,
            LsvParticleConfig::new(64, 71, 0.5, 2.0, false).unwrap(),
        )
        .unwrap();
        assert_eq!(
            c.validate_calibration_reverse(),
            Err(LsvError::ReverseTraceNotRetained)
        );
        assert_eq!(
            c.calibration_pullback(&[0.0; 9]),
            Err(LsvError::ReverseTraceNotRetained)
        );
        // Direct public reverse keeps its historical seed-validation order.
        assert_ne!(
            c.reverse_leverage(&[]).unwrap_err(),
            LsvError::ReverseTraceNotRetained
        );
    }

    #[test]
    fn hw_preflight_rejects_a_trace_whose_public_primal_changed() {
        let target =
            HullWhiteLsvTarget::flat(0.2, vec![0.0, 0.5, 1.0], vec![-0.8, 0.0, 0.8], 1e-8, 4.0)
                .unwrap();
        let mut c = calibrate_hull_white_lsv(
            &target,
            Bergomi1Factor::new(0.7, 0.3, -0.4).unwrap(),
            &HullWhite1Factor::new(0.1, vec![0.0], vec![0.005]).unwrap(),
            HybridCorrelation::new(-0.4, 0.1, -0.1).unwrap(),
            100.0,
            &LsvParticleConfig::new(64, 71, 0.5, 2.0, true).unwrap(),
        )
        .unwrap();
        assert!(c.validate_calibration_reverse().is_ok());
        c.conditional_second_moments[0] += 1e-4;
        let preflight = c.validate_calibration_reverse().unwrap_err().to_string();
        assert!(preflight.contains("calibration_changed_after_trace"));
        assert_eq!(
            c.reverse_leverage(&[0.0; 9]).unwrap_err().to_string(),
            preflight
        );
    }
}
