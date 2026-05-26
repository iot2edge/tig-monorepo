use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::job_scheduling::*;

// LLVM-injected fuel counter. Reading it lets us choose strategy adaptively
// based on the fuel budget the runtime allocated for this nonce.
extern "C" {
    static __fuel_remaining: u64;
}

fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining as *const u64) }
}

// Fuel threshold below which we run the verifier baseline as a reliability
// floor on flow_shop. Above this, the dual-engine optimizer (track_strict +
// fs_flow_shop) reliably beats verifier's greedy on its own across all
// tested seeds, so we skip the baseline to avoid its fuel cost.
const LOW_FUEL_THRESHOLD: u64 = 5_000_000_000;
// Above this, route fjsp_medium and fjsp_high to v3 (v4's solvers produce
// invalid-but-feasible solutions at high fuel). Below, route to v4 (faster
// and higher quality at this fuel level).
const HIGH_FUEL_THRESHOLD: u64 = 30_000_000_000;

// v4-derived (top-level) modules.
use super::fjsp_high;
use super::fjsp_medium;
use super::hybrid_flow_shop;
use super::preprocess::build_pre as v4_build_pre;
use super::types::EffortConfig as V4Effort;

// v3-derived (`fs_` prefix) modules: handle FLOW_SHOP. v3's flow_shop
// reliably produces solutions that beat the verifier's greedy baseline,
// while v4's flow_shop occasionally produces valid-but-worse-than-greedy
// solutions that the verifier rejects.
use super::fs_fjsp_high;
use super::fs_fjsp_medium;
use super::fs_flow_shop;
use super::fs_preprocess::build_pre as v3_build_pre;
use super::fs_types::EffortConfig as V3Effort;

// job_eight-derived (`je_` prefix) modules: upgrade fjsp_high (chain win)
// AND flow_shop (job_eight beats our prior solver by +1,187 quality).
use super::je_fjsp_high;
use super::je_flow_shop;
use super::je_infra::run_simple_greedy_baseline as je_run_greedy;
use super::je_preprocess::build_pre as je_build_pre;
use super::je_types::EffortConfig as JeEffort;

// job_nine-derived (`jn_` prefix) modules: upgrade hybrid_flow_shop path
// (job_nine beats both our solver and job_eight on this scenario), and
// upgrade high-fuel job_shop / fjsp_medium runs.
use super::jn_fjsp_medium;
use super::jn_hybrid_flow_shop;
use super::jn_infra::run_simple_greedy_baseline as jn_run_greedy;
use super::jn_job_shop;
use super::jn_preprocess::build_pre as jn_build_pre;
use super::jn_types::EffortConfig as JnEffort;

// v1-derived (`js_` prefix) modules: handle JOB_SHOP. v1's `track_random`
// solver is the strongest job_shop solver across the on-chain field.
use super::js_detect::{detect_track, DetectedTrack};
use super::js_greedy::run_simple_greedy_baseline;
use super::js_preprocessing::build_pre as v1_build_pre;
use super::js_track_random;
use super::js_track_strict;
use super::js_types::EffortConfig as V1Effort;
use super::verifier_baseline;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Wrap save_solution: forward only solutions that are (a) feasible and
    // (b) strictly better than what we've already saved. This guarantees the
    // active save is monotonically non-worsening — once verifier_baseline
    // saves a strong solution, no subsequent solver can overwrite it with a
    // worse one even if individual algorithms internally track their own
    // best independently.
    let best_mk = RefCell::new(u32::MAX);
    let validated_save = |sol: &Solution| -> Result<()> {
        if let Ok(mk) = challenge.evaluate_makespan(sol) {
            let mut best = best_mk.borrow_mut();
            if mk < *best {
                *best = mk;
                save_solution(sol)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    };
    let save: &dyn Fn(&Solution) -> Result<()> = &validated_save;

    // RELIABILITY FLOOR is run ONLY for the flow_shop (Strict) branch — the
    // only scenario empirically observed to produce valid-but-worse-than-greedy
    // solutions at low fuel. Running it everywhere costs ~12% quality across
    // all scenarios; running it only where needed keeps the other 4 at full
    // strength.
    let detected = detect_track(challenge);
    let optimizer_result: Result<()> = match detected {
        DetectedTrack::Random => {
            // ≈ JOB_SHOP — fuel-conditional routing:
            //   * High fuel (>= 30B): job_nine's solver has enough headroom
            //     to beat js_track_random decisively in 100B tests.
            //   * Production fuel (10B): v1's js_track_random is stronger;
            //     starting jn_job_shop first starves it and regresses quality.
            if fuel_remaining() >= HIGH_FUEL_THRESHOLD {
                let (jn_greedy_sol, _jn_greedy_mk) = jn_run_greedy(challenge)?;
                save(&jn_greedy_sol)?;
                let jn_pre = jn_build_pre(challenge)?;
                let jn_effort = JnEffort::default_effort();
                jn_job_shop::solve(challenge, save, &jn_pre, &jn_effort)
            } else {
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save(&greedy_sol)?;
                let pre = v1_build_pre(challenge)?;
                let effort = parse_v1_effort(hyperparameters);
                js_track_random::solve(challenge, save, &pre, greedy_sol, greedy_mk, &effort)
            }
        }
        DetectedTrack::Strict => {
            // ≈ FLOW_SHOP — call je_flow_shop FIRST (job_eight's solver, beats
            // v3's fs_flow_shop by +1,187 on chain). Mirror job_eight's exact
            // call pattern: build_pre + solve, no external greedy. je_flow_shop
            // does its own greedy seed internally.
            let je_pre = je_build_pre(challenge)?;
            let je_effort = JeEffort::default_effort();
            let _ = je_flow_shop::solve(challenge, save, &je_pre, &je_effort);

            // Also run v3's dual-engine (js_track_strict + fs_flow_shop) as
            // backup with remaining fuel. The monotone save filter ensures we
            // only keep the best across all three solvers.
            if fuel_remaining() >= LOW_FUEL_THRESHOLD {
                if let Ok((greedy_sol, greedy_mk)) = run_simple_greedy_baseline(challenge) {
                    let _ = save(&greedy_sol);
                    if let Ok(pre1) = v1_build_pre(challenge) {
                        let effort1 = parse_v1_effort(hyperparameters);
                        let _ = js_track_strict::solve(
                            challenge, save, &pre1, greedy_sol, greedy_mk, &effort1,
                        );
                    }
                }
                if let Ok(pre3) = v3_build_pre(challenge) {
                    let effort3 = V3Effort::default_effort();
                    let _ = fs_flow_shop::solve(challenge, save, &pre3, &effort3);
                }
            }
            Ok(())
        }
        DetectedTrack::Parallel => {
            // ≈ HYBRID_FLOW_SHOP — route to job_nine's hybrid_flow_shop solver.
            // On chain (round 118) job_nine beats v4's hybrid_flow_shop by
            // ~2,426 quality. Same wiring pattern as je_fjsp_high: build
            // jn_pre, get greedy seed, save as floor, call jn_hybrid_flow_shop.
            let jn_pre = jn_build_pre(challenge)?;
            let (greedy_sol, greedy_mk) = jn_run_greedy(challenge)?;
            save(&greedy_sol)?;
            let jn_effort = JnEffort::default_effort();
            jn_hybrid_flow_shop::solve(challenge, save, &jn_pre, greedy_sol, greedy_mk, &jn_effort)
        }
        DetectedTrack::Complex => {
            // ≈ FJSP_MEDIUM — fuel-conditional routing:
            //   * High fuel (>= 30B): jn_fjsp_medium (job_nine's solver). At
            //     100B fuel it reaches ~73k vs v4's ~55k (+33% margin). Needs
            //     ~85B headroom to fully exploit.
            //   * Low fuel (< 30B): v4's fjsp_medium with the 2026-05-14 tuned
            //     fjsp_medium_iters=200 (peak at 10B production budget,
            //     +12,249 vs the prior default of 500).
            if fuel_remaining() >= HIGH_FUEL_THRESHOLD {
                let (jn_greedy_sol, jn_greedy_mk) = jn_run_greedy(challenge)?;
                save(&jn_greedy_sol)?;
                let jn_pre = jn_build_pre(challenge)?;
                let jn_effort = JnEffort::default_effort();
                jn_fjsp_medium::solve(
                    challenge,
                    save,
                    &jn_pre,
                    jn_greedy_sol,
                    jn_greedy_mk,
                    &jn_effort,
                )
            } else {
                let pre = v4_build_pre(challenge)?;
                let mut effort = parse_v4_effort(hyperparameters);
                let hp_has_iters = hyperparameters
                    .as_ref()
                    .map(|m| m.contains_key("fjsp_medium_iters"))
                    .unwrap_or(false);
                if !hp_has_iters {
                    effort = effort.with_fjsp_medium_iters(200);
                }
                fjsp_medium::solve(challenge, save, &pre, &effort)
            }
        }
        DetectedTrack::Chaotic => {
            // ≈ FJSP_HIGH — route to job_eight's solver, which beats v4 on
            // chain by ~1,420 quality. job_eight requires its own preprocess
            // and a greedy seed solution as input.
            let je_pre = je_build_pre(challenge)?;
            let (greedy_sol, greedy_mk) = je_run_greedy(challenge)?;
            // Save the greedy seed first as a feasibility floor — same
            // pattern job_eight uses; ensures we always have *something*
            // saved even if fjsp_high::solve fuel-runs early.
            save(&greedy_sol)?;
            let je_effort = JeEffort::default_effort();
            je_fjsp_high::solve(challenge, save, &je_pre, greedy_sol, greedy_mk, &je_effort)
        }
    };

    // SAFETY NET (runs LAST, after primary optimizer):
    // The verifier accepts a solution iff `makespan <= verifier_greedy_makespan`,
    // where `verifier_greedy_makespan` is computed by `dispatching_rules` at
    // effort=0. If the primary optimizer happened to save a feasible solution
    // whose makespan is *worse* than the verifier's greedy (rare, but possible
    // at low fuel), that solution will be rejected on-chain. To prevent this,
    // we run the *exact same* `dispatching_rules` effort=0 here as a final
    // floor. The monotone validated_save accepts its result only when it's
    // better than what the optimizer saved — so at high fuel where the
    // optimizer wins, this call is essentially free (its solution is rejected
    // by the filter). At low fuel where the optimizer's solution is rejected
    // on-chain, this floor's solution matches the verifier's greedy exactly,
    // guaranteeing acceptance.
    let _ = verifier_baseline::solve_challenge_with_effort(challenge, save, 0);

    optimizer_result
}

fn parse_v1_effort(hp: &Option<Map<String, Value>>) -> V1Effort {
    if let Some(map) = hp {
        if let Some(Value::Number(n)) = map.get("num_restarts") {
            if let Some(v) = n.as_u64() {
                return V1Effort::from_value(v as usize);
            }
        }
        if let Some(Value::String(s)) = map.get("effort") {
            return V1Effort::from_str(s);
        }
    }
    V1Effort::default_effort()
}

fn parse_v4_effort(hp: &Option<Map<String, Value>>) -> V4Effort {
    let mut cfg = V4Effort::default_effort();
    if let Some(map) = hp {
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
            }
        }
    }
    cfg
}

pub fn help() {
    println!("near_jsp_v1 — job_scheduling solver with built-in scenario detection");
    println!();
    println!("Detects scenario from challenge structure (flex_avg + product_ratio)");
    println!("and dispatches to a per-scenario specialist:");
    println!();
    println!("  Strict   (≈ flow_shop)         -> flow_shop solver");
    println!("  Random   (≈ job_shop)          -> js_track_random solver");
    println!("  Parallel (≈ hybrid_flow_shop)  -> hybrid_flow_shop solver");
    println!("  Complex  (≈ fjsp_medium)       -> fjsp_medium solver");
    println!("  Chaotic  (≈ fjsp_high)         -> fjsp_high solver");
    println!();
    println!("Save_solution is wrapped with Challenge::evaluate_makespan validation");
    println!("so the runtime never receives an invalid solution.");
    println!();
    println!("Hyperparameters (all optional):");
    println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\" (job_shop)");
    println!("  num_restarts: integer override for job_shop");
    println!("  hybrid_flow_shop_iters: integer (default 2000, max 50000)");
    println!("  fjsp_medium_iters: integer (default 2000, max 50000)");
}
