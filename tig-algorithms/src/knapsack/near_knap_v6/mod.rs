use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::knapsack::*;

fn objective(challenge: &Challenge, solution: &Solution) -> Option<u32> {
    challenge.evaluate_total_value(solution).ok()
}

fn push_if_valid(
    candidates: &mut Vec<Solution>,
    challenge: &Challenge,
    solution: Option<Solution>,
) {
    if let Some(sol) = solution {
        if objective(challenge, &sol).is_some() {
            candidates.push(sol);
        }
    }
}

fn run_save_style(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
    runner: fn(
        &Challenge,
        &dyn Fn(&Solution) -> Result<()>,
        &Option<Map<String, Value>>,
    ) -> Result<()>,
) -> Option<Solution> {
    let slot: RefCell<Option<Solution>> = RefCell::new(None);
    let capture = |solution: &Solution| -> Result<()> {
        *slot.borrow_mut() = Some(solution.clone());
        Ok(())
    };
    let _ = runner(challenge, &capture, hyperparameters);
    slot.into_inner()
}

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let mut candidates: Vec<Solution> = Vec::with_capacity(4);

        push_if_valid(
            &mut candidates,
            challenge,
            super::near_knap_v4::Solver::solve(challenge, None, hyperparameters)?,
        );
        push_if_valid(
            &mut candidates,
            challenge,
            super::near_knap_v5::Solver::solve(challenge, None, hyperparameters)?,
        );
        push_if_valid(
            &mut candidates,
            challenge,
            super::knap_apex::Solver::solve(challenge, None, hyperparameters)?,
        );
        push_if_valid(
            &mut candidates,
            challenge,
            run_save_style(
                challenge,
                hyperparameters,
                super::knap_fast::solve_challenge,
            ),
        );

        let best = candidates
            .into_iter()
            .max_by_key(|sol| objective(challenge, sol).unwrap_or(0));
        Ok(best)
    }
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!(
        "near_knap_v6: efficient portfolio over v4/v5/apex/fast with exact objective selection."
    );
}
