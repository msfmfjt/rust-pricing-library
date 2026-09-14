use pricing::market::{ImpliedVarianceSurface, MarketIvSurface};

fn surface(quotes: Vec<f64>) -> MarketIvSurface {
    MarketIvSurface::new(vec![0.3, 0.8, 1.4], vec![-0.7, -0.2, 0.0, 0.4, 0.9], quotes).unwrap()
}
fn vector(s: &MarketIvSurface, t: f64, x: f64) -> [f64; 4] {
    let v = s.total_variance_derivatives(t, x).unwrap();
    [
        v.total_variance,
        v.log_moneyness_derivative,
        v.log_moneyness_second_derivative,
        v.time_derivative,
    ]
}

#[test]
fn cubic_quote_transpose_matches_every_quote_bump_and_spatial_derivatives() {
    let quotes = (0..15)
        .map(|i| 0.22 + 0.004 * (i / 5) as f64 - 0.002 * (i % 5) as f64)
        .collect::<Vec<_>>();
    let s = surface(quotes.clone());
    let seeds = [0.7, -0.3, 0.4, -0.8];
    for (t, x) in [
        (0.1, -0.7),
        (0.3, -0.1),
        (0.6, 0.13),
        (0.8, 0.4),
        (1.4, 0.9),
        (2.0, 0.0),
    ] {
        let mut adj = vec![0.0; quotes.len()];
        s.transpose_accumulate(t, x, seeds, &mut adj).unwrap();
        for (i, &bar) in adj.iter().enumerate() {
            let mut up = quotes.clone();
            let mut down = quotes.clone();
            up[i] += 1e-6;
            down[i] -= 1e-6;
            let fd: f64 = vector(&surface(up), t, x)
                .iter()
                .zip(vector(&surface(down), t, x))
                .zip(seeds)
                .map(|((a, b), q)| (a - b) * q / 2e-6)
                .sum();
            assert!((fd - bar).abs() < 1e-8, "{t} {x} {i}: {fd} != {bar}");
        }
    }
    let t = 0.63;
    let x = 0.13;
    let h = 1e-4;
    let v = vector(&s, t, x);
    let up = vector(&s, t, x + h);
    let down = vector(&s, t, x - h);
    assert!((v[1] - (up[0] - down[0]) / (2.0 * h)).abs() < 1e-8);
    assert!((v[2] - (up[0] - 2.0 * v[0] + down[0]) / (h * h)).abs() < 1e-8);
    assert!((v[3] - (vector(&s, t + h, x)[0] - vector(&s, t - h, x)[0]) / (2.0 * h)).abs() < 1e-10);
}

#[test]
fn market_iv_interpolates_quotes_and_rejects_invalid_domains() {
    let s = surface(vec![0.2; 15]);
    for &t in s.maturity_nodes() {
        for &x in s.log_moneyness_nodes() {
            let v = vector(&s, t, x);
            assert!((v[0] - 0.04 * t).abs() < 1e-14);
            assert!((v[3] - 0.04).abs() < 1e-14);
            assert!(v[1].abs() < 1e-14 && v[2].abs() < 1e-14);
        }
    }
    for (times, xs, vols) in [
        (vec![0.0, 1.0], vec![-1.0, 1.0], vec![0.2; 4]),
        (vec![0.5, 0.5], vec![-1.0, 1.0], vec![0.2; 4]),
        (vec![0.5, 1.0], vec![0.0, 0.0], vec![0.2; 4]),
        (vec![0.5, 1.0], vec![-1.0, 1.0], vec![f64::NAN; 4]),
        (vec![0.5, 1.0], vec![-1.0, 1.0], vec![0.0; 4]),
        (vec![0.5, 1.0], vec![-1.0, 1.0], vec![0.2; 3]),
    ] {
        assert!(MarketIvSurface::new(times, xs, vols).is_err());
    }
    assert!(s.total_variance_derivatives(0.0, 0.0).is_err());
    assert!(s.total_variance_derivatives(0.5, 0.91).is_err());
    assert!(
        s.transpose_accumulate(0.5, 0.0, [1.0; 4], &mut [0.0; 1])
            .is_err()
    );
}
