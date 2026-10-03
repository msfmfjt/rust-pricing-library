//! Paired-build microbenchmark. Fixed paths, unchanged schemes, no wall-clock gate.
//! Build the same example against both source revisions and compare JSON output.
use pricing::mc::RandomDomain;
use pricing::rough_volatility::{
    QuadraticRoughHeston, RoughHeston, RoughVolatilityModel, RoughVolatilityPath,
    RoughVolatilityPathPlan,
};
use serde_json::json;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

fn digest_path(hash: &mut blake3::Hasher, path: &RoughVolatilityPath) {
    for field in [&path.forwards, &path.variances, &path.latent_states] {
        for x in field {
            hash.update(&x.to_le_bytes());
        }
    }
    hash.update(&path.negative_variance_nodes.to_le_bytes());
    hash.update(&path.absorbed_forward_steps.to_le_bytes());
}
fn positive_argument(
    index: usize,
    default: usize,
    maximum: usize,
) -> Result<usize, Box<dyn Error>> {
    let value = std::env::args()
        .nth(index)
        .map_or(Ok(default), |s| s.parse::<usize>())?;
    if value == 0 || value > maximum {
        return Err(format!("argument {index} must be in 1..={maximum}").into());
    }
    Ok(value)
}
fn main() -> Result<(), Box<dyn Error>> {
    // [paths per case] [timing repetitions]. Work remains explicitly bounded.
    let paths = positive_argument(1, 128, 4096)?;
    let repetitions = positive_argument(2, 5, 20)?;
    if std::env::args().len() > 3 {
        return Err("usage: benchmark_rough_diffusion_cache [paths] [repetitions]".into());
    }
    let mut rows = Vec::new();
    for family in ["rough_heston", "quadratic_rough_heston"] {
        for h in [0.1, 0.3, 0.5] {
            for steps in [64_usize, 256, 1024] {
                for nonuniform in [false, true] {
                    let model: RoughVolatilityModel = if family == "rough_heston" {
                        RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65)?.into()
                    } else {
                        QuadraticRoughHeston::new(h, 0.15, 1.1, 0.5, 0.8, 0.25, 0.02)?.into()
                    };
                    let times = (0..=steps)
                        .map(|i| {
                            let t = i as f64 / steps as f64;
                            if nonuniform { t * t } else { t }
                        })
                        .collect();
                    let start = Instant::now();
                    let plan = RoughVolatilityPathPlan::compile(model, times)?;
                    let compile_seconds = start.elapsed().as_secs_f64();
                    let shocks: Vec<_> = (0..paths)
                        .map(|i| plan.pseudo_shocks(91, i as u64, RandomDomain::Valuation))
                        .collect();
                    // Digest/warmup is outside the timed region. Normals are shared by all
                    // repetitions and builds. Hash every node and diagnostic, not only payoff.
                    let mut hash = blake3::Hasher::new();
                    let mut negative_nodes = 0;
                    for z in &shocks {
                        let path = plan.evolve_path(100.0, z)?;
                        negative_nodes += path.negative_variance_nodes;
                        digest_path(&mut hash, &path);
                    }
                    let mut seconds = Vec::new();
                    for _ in 0..repetitions {
                        let start = Instant::now();
                        for z in &shocks {
                            black_box(plan.evolve_path(black_box(100.0), black_box(z))?);
                        }
                        seconds.push(start.elapsed().as_secs_f64());
                    }
                    rows.push(json!({"family":family, "hurst":h, "steps":steps,
                        "grid":if nonuniform {"quadratic_time"} else {"uniform"},
                        "paths":paths, "seconds":seconds, "compile_seconds":compile_seconds,
                        "digest":hash.finalize().to_hex().to_string(),
                        "plan_fingerprint":plan.plan_fingerprint().to_string(),
                        "scheme":plan.scheme(), "negative_nodes":negative_nodes}));
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"rough-diffusion-cache-benchmark/v1", "seed":91,
            "scope":"path evolution including allocations; excludes RNG, compilation, payoff and reduction",
            "rows":rows
        }))?
    );
    Ok(())
}
