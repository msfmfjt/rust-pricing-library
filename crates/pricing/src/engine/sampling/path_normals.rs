//! sampling / path normals implementation.

use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{
    BrownianBridgePlan, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan,
    inverse_standard_normal,
};

impl SimulationPlan {
    pub(in crate::engine) fn normals(
        &self,
        generator: &Philox4x32,
        sampling_unit: u64,
        domain: RandomDomain,
    ) -> Vec<f64> {
        (0..self.observation_times.len())
            .map(|dimension| {
                if self.total_variance == 0.0 {
                    0.0
                } else {
                    generator.standard_normal(RandomCoordinate::new(
                        sampling_unit,
                        u32::try_from(dimension).expect("observation dimension fits u32"),
                        domain,
                    ))
                }
            })
            .collect()
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn rqmc_normals(
        &self,
        plan: &RqmcPlan,
        scramble: u32,
        point: u64,
    ) -> Result<Vec<f64>, MonteCarloError> {
        if self.total_variance == 0.0 {
            return Ok(vec![0.0; self.observation_times.len()]);
        }
        (0..self.observation_times.len())
            .map(|dimension| {
                let probability = plan
                    .uniform(
                        scramble,
                        point,
                        u32::try_from(dimension).expect("observation dimension fits u32"),
                    )
                    .expect("scramble, point, and dimension originate from the compiled plan");
                Ok(inverse_standard_normal(probability)
                    .expect("the Sobol midpoint mapping is strictly inside the unit interval"))
            })
            .collect()
    }
}

pub(in crate::engine) fn local_vol_rqmc_shocks(
    qmc: &RqmcPlan,
    bridge: Option<&BrownianBridgePlan>,
    scramble: u32,
    point: u64,
) -> Result<Vec<f64>, MonteCarloError> {
    let mut shocks =
        Vec::with_capacity(usize::try_from(qmc.effective_dimension()).expect("u32 fits usize"));
    for dimension in 0..qmc.effective_dimension() {
        let probability = qmc
            .uniform(scramble, point, dimension)
            .expect("scramble, point, and dimension originate from the compiled plan");
        shocks.push(
            inverse_standard_normal(probability)
                .expect("the Sobol midpoint mapping is strictly inside the unit interval"),
        );
    }
    if let Some(bridge) = bridge {
        bridge
            .apply_one_factor(&shocks)
            .map_err(|error| MonteCarloError::LocalVol(error.into()))
    } else {
        Ok(shocks)
    }
}
