// Empirical check: spectrum of TIG QKP V matrix.
//
// Generates challenges via tig-challenges and computes:
// - eigenvalue distribution (positive/negative split — tests PSD-ness empirically)
// - effective rank at 90/95/99% energy
// - stable rank
//
// Make-or-break for the Jaccard-PSD-low-rank Advance angle.

use nalgebra::DMatrix;
use tig_challenges::knapsack::{Challenge, Track};

fn analyze(ch: &Challenge, label: &str) {
    let n = ch.num_items;
    let mut v = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            v[(i, j)] = ch.interaction_values[i][j] as f64;
        }
    }

    let frob_sq: f64 = v.iter().map(|x| x * x).sum();
    let frob = frob_sq.sqrt();

    let eigs = v.symmetric_eigen().eigenvalues;
    let mut eig_vec: Vec<f64> = eigs.iter().copied().collect();

    let n_pos = eig_vec.iter().filter(|&&x| x > 1e-6).count();
    let n_neg = eig_vec.iter().filter(|&&x| x < -1e-6).count();
    let max_pos = eig_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_neg = eig_vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let nonzero = eig_vec.iter().filter(|&&x| x.abs() > 1e-6).count();

    eig_vec.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let top10: Vec<i64> = eig_vec.iter().take(10).map(|&x| x as i64).collect();

    let total_e: f64 = eig_vec.iter().map(|x| x * x).sum();
    let mut cum = 0.0;
    let (mut r90, mut r95, mut r99) = (0, 0, 0);
    for (i, &e) in eig_vec.iter().enumerate() {
        cum += e * e;
        let frac = cum / total_e;
        if r90 == 0 && frac >= 0.90 { r90 = i + 1; }
        if r95 == 0 && frac >= 0.95 { r95 = i + 1; }
        if r99 == 0 && frac >= 0.99 { r99 = i + 1; }
    }

    let stable_rank = total_e / (max_pos.max(min_neg.abs())).powi(2);

    println!("\n=== {} (n={}) ===", label, n);
    println!("  Frobenius norm:       {:.0}", frob);
    println!("  largest +eig:         {:.0}", max_pos);
    println!("  most -eig:            {:.0}", min_neg);
    println!("  +eigs / -eigs / nz:   {} / {} / {}", n_pos, n_neg, nonzero);
    println!("  stable rank:          {:.2}", stable_rank);
    println!("  effective rank @ 90%: {} ({:.1}% of n)", r90, 100.0 * r90 as f64 / n as f64);
    println!("  effective rank @ 95%: {} ({:.1}% of n)", r95, 100.0 * r95 as f64 / n as f64);
    println!("  effective rank @ 99%: {} ({:.1}% of n)", r99, 100.0 * r99 as f64 / n as f64);
    println!("  top 10 |eigs|:        {:?}", top10);
}

fn main() {
    for &n_items in &[500usize, 1000, 2000] {
        for &budget in &[10u32, 50, 90] {
            for nonce in 0..3u64 {
                // Construct seed analogous to tig-runtime BenchmarkSettings::calc_seed
                let mut seed = [0u8; 32];
                let s = format!("{}_{}_{}", n_items, budget, nonce);
                let bytes = s.as_bytes();
                for (i, b) in bytes.iter().take(32).enumerate() {
                    seed[i] = *b;
                }
                let track = Track { n_items, budget };
                match Challenge::generate_instance(&seed, &track) {
                    Ok(ch) => {
                        let label = format!("n={}, budget={}, nonce={}", n_items, budget, nonce);
                        analyze(&ch, &label);
                    }
                    Err(e) => println!("Failed n={} budget={} nonce={}: {}", n_items, budget, nonce, e),
                }
            }
        }
    }
}
