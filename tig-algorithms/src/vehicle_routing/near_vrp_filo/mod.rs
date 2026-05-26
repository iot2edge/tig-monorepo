mod builder;
mod config;
mod evolution;
mod gene_pool;
mod instance;
mod operators;
mod route_eval;
mod runner;
mod solution;

pub use runner::Solver;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn help() {
    println!("near_vrp_filo: HGS-VRPTW with FILO-style granular intra-route pruning.");
    println!("Forked from near_vrp_v1 (which forked fast_lane_v4). Adds a precomputed");
    println!("K-nearest neighbor table (Accorsi & Vigo 2021); intra-route relocate /");
    println!("swap / 2-opt / or-opt skip insertion positions where neither adjacent");
    println!("customer is a granular neighbor of the moving customer. Default level=4.");
}
