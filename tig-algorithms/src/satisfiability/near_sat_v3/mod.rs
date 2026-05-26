use anyhow::Result;
use serde_json::{Map, Number, Value};
use std::cell::RefCell;
use tig_challenges::satisfiability::*;

use super::{near_sat_v2, sat_vanguard};

fn solved(challenge: &Challenge, solution: &Solution) -> bool {
    solution.variables.len() == challenge.num_variables
        && challenge.clauses.iter().all(|clause| {
            clause.iter().any(|&literal| {
                let var_idx = literal.unsigned_abs() as usize - 1;
                let var_value = solution.variables[var_idx];
                (literal > 0 && var_value) || (literal < 0 && !var_value)
            })
        })
}

fn capped_vanguard_hp() -> Option<Map<String, Value>> {
    let mut hp = Map::new();
    let cap = Value::Number(Number::from_f64(100_000_000_000.0).unwrap());
    hp.insert("max_fuel_high".to_string(), cap.clone());
    hp.insert("max_fuel_low".to_string(), cap);
    Some(hp)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let best = RefCell::new(None::<Solution>);
    let capture = |solution: &Solution| -> Result<()> {
        if solved(challenge, solution) {
            *best.borrow_mut() = Some(solution.clone());
        } else if best.borrow().is_none() {
            *best.borrow_mut() = Some(solution.clone());
        }
        Ok(())
    };

    near_sat_v2::solve_challenge(challenge, &capture, hyperparameters)?;
    if let Some(solution) = best.borrow().as_ref() {
        if solved(challenge, solution) {
            let _ = save_solution(solution);
            return Ok(());
        }
    }

    let fallback = RefCell::new(best.borrow().clone());
    let capture_fallback = |solution: &Solution| -> Result<()> {
        if solved(challenge, solution) || fallback.borrow().is_none() {
            *fallback.borrow_mut() = Some(solution.clone());
        }
        Ok(())
    };

    let hp = capped_vanguard_hp();
    let _ = sat_vanguard::solve_challenge(challenge, &capture_fallback, &hp);

    if let Some(solution) = fallback.borrow().as_ref() {
        let _ = save_solution(solution);
    }
    Ok(())
}

pub fn help() {
    println!("near_sat_v3: near_sat_v2 first, capped sat_vanguard fallback portfolio.");
}
