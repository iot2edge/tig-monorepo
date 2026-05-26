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

// Above this fuel level we take the high-fuel routes (jn_job_shop alone for
// Random; jn_fjsp_medium for Complex). The real chain budget (5T) is always
// above this; the low-fuel branches are retained only for local low-fuel tests.
const HIGH_FUEL_THRESHOLD: u64 = 30_000_000_000;

// v4-derived (top-level) modules. Only fjsp_medium is still wired (Complex
// low-fuel fallback); preprocess/types support it.
use super::fjsp_medium;
use super::preprocess::build_pre as v4_build_pre;
use super::types::EffortConfig as V4Effort;

// job_eight-derived (`je_` prefix) modules: fjsp_high (chain win).
use super::je_fjsp_high;
use super::je_infra::run_simple_greedy_baseline as je_run_greedy;
use super::je_preprocess::build_pre as je_build_pre;
use super::je_types::EffortConfig as JeEffort;

// job_nine-derived (`jn_` prefix) modules: upgrade hybrid_flow_shop path
// (job_nine beats both our solver and job_eight on this scenario), and
// upgrade high-fuel job_shop / fjsp_medium runs.
use super::jn_fjsp_medium;
use super::jn_flow_shop;
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
            // ≈ JOB_SHOP — dual-engine to handle seed-variant winners:
            //   On seeds {jsp_compare_2026_05_13, jsp_v4_robust_seed_2, jsp_fresh_c}
            //     js_track_random (v1) wins by +1,323 to +6,720.
            //   On seeds {jsp_fresh_a, jsp_fresh_b} jn_job_shop wins by 1,654 to 6,588.
            //   Run js_track_random FIRST with REDUCED num_restarts (250 vs default
            //   500, ~5B fuel), then jn_job_shop with leftover ~5B. Monotone save
            //   keeps the best of both per nonce.
            //   High fuel path keeps jn alone (jn dominates with >30B headroom).
            if fuel_remaining() >= HIGH_FUEL_THRESHOLD {
                let (jn_greedy_sol, _jn_greedy_mk) = jn_run_greedy(challenge)?;
                save(&jn_greedy_sol)?;
                let jn_pre = jn_build_pre(challenge)?;
                // BEAT JOB_NINE: job_nine runs jn_job_shop at iters=100000 and
                // stops early at 61,686 (192B, 3.8% of 5T). The solver is NOT
                // exhausted — given iters=300000 it converges (rc=0,
                // deterministic) to ~73,887 / 737B, i.e. +19.8% on nonce 0
                // (+3.8% on nonce 1, ~+10% avg). Bake 300000 as the default;
                // HP-overridable. Converges in ~35min/nonce.
                let mut jn_effort = JnEffort::default_effort();
                jn_effort.job_shop_iters = hyperparameters
                    .as_ref()
                    .and_then(|m| m.get("job_shop_iters"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(300_000);
                jn_job_shop::solve(challenge, save, &jn_pre, &jn_effort)
            } else {
                // First engine: js_track_random with capped restarts (frees fuel
                // for the second engine).
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save(&greedy_sol)?;
                let pre = v1_build_pre(challenge)?;
                let mut effort = parse_v1_effort(hyperparameters);
                // Default num_restarts is 500; reduce to free fuel for jn_job_shop.
                // Only override if HP didn't set explicit value.
                let hp_has_restarts = hyperparameters
                    .as_ref()
                    .map(|m| m.contains_key("num_restarts"))
                    .unwrap_or(false);
                if !hp_has_restarts {
                    effort.num_restarts = 250;
                }
                let _ = js_track_random::solve(challenge, save, &pre, greedy_sol, greedy_mk, &effort);

                // Second engine: jn_job_shop with leftover fuel (typically ~5B).
                if let Ok(jn_pre) = jn_build_pre(challenge) {
                    let jn_effort = JnEffort::default_effort();
                    let _ = jn_job_shop::solve(challenge, save, &jn_pre, &jn_effort);
                }
                Ok(())
            }
        }
        DetectedTrack::Strict => {
            // ≈ FLOW_SHOP — route to job_nine's flow_shop solver. job_nine runs
            // it at flow_shop_iters=1000 and converges in ~22s using only ~4.7B
            // of the 5T budget — leaving 99.9% of the fuel unused. jn_flow_shop's
            // solve() is effort-driven (flow_shop_iters scales strict_depth, the
            // Iterated-Greedy num_restarts, and per-restart iterations), so
            // cranking it lets us EXPLOIT the unused budget to BEAT job_nine on
            // flow_shop. flow_shop_iters is HP-tunable (default FS_ITERS) for
            // sweeping the quality/fuel curve. Default 5000 beats job_nine's
            // 1000-setting by +121 (36,310 vs 36,189, deterministic plateau)
            // for negligible extra fuel (4.79B vs 4.75B).
            const FS_ITERS: usize = 5000;
            let fs_iters = hyperparameters
                .as_ref()
                .and_then(|m| m.get("flow_shop_iters"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(FS_ITERS);
            let jn_effort = JnEffort {
                num_restarts: 500,
                job_shop_iters: 1000,
                flow_shop_iters: fs_iters,
                hybrid_flow_shop_iters: 1000,
                fjsp_medium_iters: 1000,
                fjsp_high_iters: 1000,
            };
            let (greedy_sol, greedy_mk) = jn_run_greedy(challenge)?;
            save(&greedy_sol)?;
            let jn_pre = jn_build_pre(challenge)?;
            jn_flow_shop::solve(challenge, save, &jn_pre, greedy_sol, greedy_mk, &jn_effort)
        }
        DetectedTrack::Parallel => {
            // ≈ HYBRID_FLOW_SHOP — route to job_nine's hybrid_flow_shop solver.
            // On chain (round 118) job_nine beats v4's hybrid_flow_shop by
            // ~2,426 quality. Same wiring pattern as je_fjsp_high: build
            // jn_pre, get greedy seed, save as floor, call jn_hybrid_flow_shop.
            let jn_pre = jn_build_pre(challenge)?;
            let (greedy_sol, greedy_mk) = jn_run_greedy(challenge)?;
            save(&greedy_sol)?;
            // job_nine converges at ~159B (3.2% of 5T) → big headroom.
            // hybrid_flow_shop_iters is HP-tunable to exploit it.
            let mut jn_effort = JnEffort::default_effort();
            if let Some(it) = hyperparameters
                .as_ref()
                .and_then(|m| m.get("hybrid_flow_shop_iters"))
                .and_then(|v| v.as_u64())
            {
                jn_effort.hybrid_flow_shop_iters = it as usize;
            }
            jn_hybrid_flow_shop::solve(challenge, save, &jn_pre, greedy_sol, greedy_mk, &jn_effort)
        }
        DetectedTrack::Complex => {
            // ≈ FJSP_MEDIUM — fuel-conditional routing:
            //   * High fuel (>= 30B): jn_fjsp_medium = job_nine's LEAN current
            //     fjsp_medium solver (ported faithfully). It terminates at
            //     ~267B fuel with quality ~42,895 — same throughput as the
            //     chain leader. (We previously carried a heavier 1,849-line
            //     fjsp_medium variant that ran ~3x the fuel for a noise-level
            //     +0.9%; capping it was unreliable — the iters knob is
            //     non-monotonic in fuel and a fuel-aware early-exit regressed
            //     quality to ~41,466 because the TLS fuel counter commits in
            //     chunks. The lean solver is faster AND higher quality, so it
            //     wins outright.) Mirrors job_nine's FjspMedium call exactly.
            //   * Low fuel (< 30B): v4's fjsp_medium with fjsp_medium_iters=200.
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
            // job_eight/job_nine BOTH ceiling at 57,705 on fjsp_high. NEGATIVE
            // RESULT (2026-05-23): tried beating it with jn_fjsp_heavy (the
            // heavy 1,849-line tabu/ILS/CP-kick FJSP search) given up to 1.5T
            // fuel + 20k iters — it ALSO converged to exactly 57,705 (~34min
            // wasted for zero gain). Three independent solvers hitting the same
            // 57,705 ⇒ this instance is at a genuine optimum/ceiling, not a
            // tuning gap. So fjsp_high stays je-only (fast, ~34B, 57,705).
            // fjsp_high_iters kept HP-tunable (harmless).
            let mut je_effort = JeEffort::default_effort();
            if let Some(it) = hyperparameters
                .as_ref()
                .and_then(|m| m.get("fjsp_high_iters"))
                .and_then(|v| v.as_u64())
            {
                je_effort.fjsp_high_iters = it as usize;
            }
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
