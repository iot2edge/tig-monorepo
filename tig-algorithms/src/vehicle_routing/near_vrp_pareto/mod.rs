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
    println!("near_vrp_v1: Deep HGS-VRPTW (default exploration_level=4).");
    println!("Forked from fast_lane_v4 with the deepest preset hardcoded as the");
    println!("default, so the on-chain runtime gets full-strength search even when");
    println!("no hyperparameters are passed.");
}
