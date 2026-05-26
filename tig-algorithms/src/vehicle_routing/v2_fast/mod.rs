// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs` to identify algorithms
pub mod builder;
pub mod config;
pub mod evolution;
pub mod gene_pool;
pub mod instance;
pub mod operators;
pub mod route_eval;
pub mod runner;
pub mod solution;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() -> String {
    String::from("HGS-VRPTW with cache-optimized local search. Flat neighbor arrays, pre-allocated DP buffers, compact RouteEval.")
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    runner::Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))
}
