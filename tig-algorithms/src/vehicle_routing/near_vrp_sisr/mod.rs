mod builder;
mod config;
mod evolution;
mod gene_pool;
mod instance;
mod operators;
mod route_eval;
mod runner;
mod sisr;
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
    println!("near_vrp_sisr: SISR (Slack Induction by String Removals) VRPTW solver.");
    println!("Replaces the HGS evolution body with a single-trajectory ruin-and-recreate");
    println!("loop (Christiaens & Vanden Berghe 2020) plus simulated-annealing acceptance.");
    println!("Reuses the LocalOps local search from fast_lane_v4 to polish each candidate.");
}
