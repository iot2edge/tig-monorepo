// near_vector_v1 — fork of autovector_final3 (chain leader, 94.57% adoption).
//
// One targeted change: extend the FP16-input cuBLAS path (which the chain
// leader currently uses ONLY for n_queries=9000 via its track_t20) to the
// three larger tracks (n_queries=11000/13000/15000). Today those go through
// plain FP32 cuBLAS with a small TILE_DB=4096 — the obvious bottleneck.
//
// Expected effect on the larger tracks: ~2x throughput from halved GEMM
// memory bandwidth + a larger tile (32768 vs 4096), with FP32 accumulator
// keeping the quality identical to the FP32 path (the FP16 input precision
// is far better than the gap between 1-NN and 2-NN in clustered data).
//
// Tracks not changed: n_queries=7000 stays on track_t19 (already optimal:
// tensor cores via padded 256-dim FP16 with COMPUTE_32F_FAST_16F).

mod track_large;
mod track_shared;
mod track_t19;
mod track_t20;

use anyhow::Result;
use cudarc::driver::safe::{CudaModule, CudaStream};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Hyperparameters {}

pub fn help() {
    println!("near_vector_v1 — autovector_final3 fork. FP16-input cuBLAS extended to all tracks");
    println!("                except n=7000 (kept on the tensor-core padded-256 path).");
    println!("  n_queries=7000              -> track_t19 (FP16 padded, COMPUTE_32F_FAST_16F, TC)");
    println!("  n_queries=9000              -> track_t20 (FP16 unpadded, COMPUTE_32F, TILE=65536)");
    println!("  n_queries=11000/13000/15000 -> track_large (FP16 unpadded, COMPUTE_32F, TILE=32768)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_queries as usize {
        7000  => track_t19::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        9000  => track_t20::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        // The change: route n=11000/13000/15000 through the FP16 path instead of FP32 cuBLAS.
        // (Chain leader autovector_final3 sends these to track_shared which is plain FP32, TILE=4096.)
        _     => track_large::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}
