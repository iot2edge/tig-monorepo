// Minimal test mod.rs — just save default solution to test linker
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
}

pub fn help() {
    println!("near_sat_cdcl minimal");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let _hp: Option<Hyperparameters> = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value(Value::Object(m.clone())).ok());
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution {
        variables: vec![false; nv],
    });
    Ok(())
}
