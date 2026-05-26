use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

// Runtime instrumentation globals are auto-injected by the build pipeline
// (LLVMFuelRTSig pass) when this .so is built. Defining them explicitly here
// causes "multiple definition" linker errors when other algos in the same
// crate (e.g. near_sat_cdcl) reference them. Use extern declarations instead.
extern "C" {
    static mut __fuel_remaining: u64;
    static mut __runtime_signature: u64;
    static mut __curr_memory_usage: u64;
    static mut __total_memory_usage: u64;
    static mut __max_memory_usage: u64;
    static mut __max_allowed_memory_usage: u64;
}

#[derive(Clone)]
struct Clause {
    lits: [i32; 3],
}

fn lit_satisfied(lit: i32, vars: &[bool]) -> bool {
    let v = lit.unsigned_abs() as usize - 1;
    (lit > 0 && vars[v]) || (lit < 0 && !vars[v])
}

fn eval_clause(cl: &Clause, vars: &[bool]) -> u8 {
    let mut c = 0;
    for &lit in &cl.lits {
        if lit_satisfied(lit, vars) {
            c += 1;
        }
    }
    c
}

fn rebuild_unsat(
    clauses: &[Clause],
    vars: &[bool],
    sat_count: &mut [u8],
    unsat: &mut Vec<usize>,
    in_unsat: &mut [bool],
    unsat_pos: &mut [usize],
) {
    unsat.clear();
    in_unsat.fill(false);
    unsat_pos.fill(usize::MAX);
    for (i, cl) in clauses.iter().enumerate() {
        let c = eval_clause(cl, vars);
        sat_count[i] = c;
        if c == 0 {
            unsat_pos[i] = unsat.len();
            unsat.push(i);
            in_unsat[i] = true;
        }
    }
}

fn add_unsat(ci: usize, unsat: &mut Vec<usize>, in_unsat: &mut [bool], unsat_pos: &mut [usize]) {
    if !in_unsat[ci] {
        in_unsat[ci] = true;
        unsat_pos[ci] = unsat.len();
        unsat.push(ci);
    }
}

fn remove_unsat(ci: usize, unsat: &mut Vec<usize>, in_unsat: &mut [bool], unsat_pos: &mut [usize]) {
    if in_unsat[ci] {
        in_unsat[ci] = false;
        let pos = unsat_pos[ci];
        unsat_pos[ci] = usize::MAX;
        if pos < unsat.len() {
            let moved = *unsat.last().unwrap();
            unsat.swap_remove(pos);
            if pos < unsat.len() {
                unsat_pos[moved] = pos;
            }
        }
    }
}

fn flip_var(
    var: usize,
    clauses: &[Clause],
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
    vars: &mut [bool],
    sat_count: &mut [u8],
    dirty: &mut Vec<usize>,
    unsat: &mut Vec<usize>,
    in_unsat: &mut [bool],
    unsat_pos: &mut [usize],
) {
    dirty.clear();
    dirty.extend_from_slice(&pos_occ[var]);
    dirty.extend_from_slice(&neg_occ[var]);
    dirty.sort_unstable();
    dirty.dedup();

    vars[var] = !vars[var];
    for &ci in dirty.iter() {
        sat_count[ci] = eval_clause(&clauses[ci], vars);
        if sat_count[ci] == 0 {
            add_unsat(ci, unsat, in_unsat, unsat_pos);
        } else {
            remove_unsat(ci, unsat, in_unsat, unsat_pos);
        }
    }
}

fn make_break(
    var: usize,
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
    vars: &[bool],
    sat_count: &[u8],
    clause_weights: Option<&[u16]>,
) -> (i32, i32) {
    let mut make = 0i32;
    let mut brk = 0i32;
    let make_occ = if vars[var] {
        &neg_occ[var]
    } else {
        &pos_occ[var]
    };
    let break_occ = if vars[var] {
        &pos_occ[var]
    } else {
        &neg_occ[var]
    };

    for &ci in make_occ {
        if sat_count[ci] == 0 {
            make += clause_weights.map_or(1, |w| w[ci] as i32);
        }
    }
    for &ci in break_occ {
        if sat_count[ci] == 1 {
            brk += clause_weights.map_or(1, |w| w[ci] as i32);
        }
    }
    (make, brk)
}

fn choose_var(
    cl: &Clause,
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
    vars: &[bool],
    sat_count: &[u8],
    clause_weights: Option<&[u16]>,
    rng: &mut SmallRng,
    noise_x1000: u32,
    mode: usize,
) -> usize {
    let mode_noise = match mode {
        2 => noise_x1000.saturating_mul(2).min(850),
        4 => noise_x1000 / 2,
        _ => noise_x1000,
    };
    if rng.gen_range(0..1000) < mode_noise {
        return cl.lits[rng.gen_range(0..3)].unsigned_abs() as usize - 1;
    }

    let mut vars3 = [0usize; 3];
    let mut mb = [(0i32, 0i32); 3];
    for (i, &lit) in cl.lits.iter().enumerate() {
        let v = lit.unsigned_abs() as usize - 1;
        vars3[i] = v;
        mb[i] = make_break(v, pos_occ, neg_occ, vars, sat_count, clause_weights);
    }

    if mode == 1 || mode == 5 {
        let mut best_i = 0usize;
        let mut best_s = i32::MIN;
        for i in 0..3 {
            let (make, brk) = mb[i];
            let zero = if brk == 0 { 60 } else { 0 };
            let score = if mode == 1 {
                make * 14 - brk * 18 + zero + rng.gen_range(0..7)
            } else {
                make * 18 - brk * 10 + zero / 2 + rng.gen_range(0..7)
            };
            if score > best_s {
                best_s = score;
                best_i = i;
            }
        }
        return vars3[best_i];
    }

    if mode == 3 {
        let mut best_i = 0usize;
        for i in 1..3 {
            let (make, brk) = mb[i];
            let (best_make, best_brk) = mb[best_i];
            if brk < best_brk || (brk == best_brk && make > best_make) {
                best_i = i;
            }
        }
        return vars3[best_i];
    }

    let mut weights = [0u32; 3];
    let mut total = 0u32;
    for i in 0..3 {
        let (make, brk) = mb[i];
        let m = (make + 1) as u32;
        let b = (brk + 1) as u32;
        let zero_break_bonus = if brk == 0 { 64 } else { 1 };
        let w = if mode == 4 {
            ((m * m * zero_break_bonus) / b).max(1)
        } else {
            ((m * m * m * zero_break_bonus) / (b * b)).max(1)
        };
        weights[i] = w;
        total += w;
    }

    let mut pick = rng.gen_range(0..total);
    for i in 0..3 {
        if pick < weights[i] {
            return vars3[i];
        }
        pick -= weights[i];
    }
    vars3[2]
}

fn init_assignment(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    restart: usize,
    rng: &mut SmallRng,
) -> Vec<bool> {
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let p = p_cnt[v];
        let n = n_cnt[v];
        vars[v] = match restart % 4 {
            0 => p >= n,
            1 => p > n + 1 || (p + n > 0 && rng.gen_bool(0.18)),
            2 => p >= n && !rng.gen_bool(0.08),
            _ => rng.gen_bool(0.5),
        };
    }
    vars
}

fn remember_elite(elite: &mut Vec<(usize, Vec<bool>)>, unsat_len: usize, vars: &[bool]) {
    if unsat_len > 10 {
        return;
    }
    if elite.iter().any(|(_, old)| old == vars) {
        return;
    }
    elite.push((unsat_len, vars.to_vec()));
    elite.sort_unstable_by_key(|x| x.0);
    if elite.len() > 18 {
        elite.pop();
    }
}

fn build_backbone(elite: &[(usize, Vec<bool>)], nv: usize) -> Option<Vec<i8>> {
    if elite.len() < 4 {
        return None;
    }
    let cutoff = (elite[0].0 + 2).min(6);
    let selected: Vec<&Vec<bool>> = elite
        .iter()
        .filter(|(u, _)| *u <= cutoff)
        .map(|(_, vars)| vars)
        .collect();
    if selected.len() < 4 {
        return None;
    }

    let mut backbone = vec![0i8; nv];
    let min_agree = (selected.len() * 7 + 7) / 8;
    for v in 0..nv {
        let trues = selected.iter().filter(|vars| vars[v]).count();
        if trues >= min_agree {
            backbone[v] = 1;
        } else if selected.len() - trues >= min_agree {
            backbone[v] = -1;
        }
    }
    Some(backbone)
}

fn backbone_assignment(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    backbone: &[i8],
    attempt: usize,
    rng: &mut SmallRng,
) -> Vec<bool> {
    let mut vars = init_assignment(nv, p_cnt, n_cnt, attempt, rng);
    let keep_x1000 = match attempt % 4 {
        0 => 960,
        1 => 910,
        2 => 850,
        _ => 760,
    };
    for v in 0..nv {
        if backbone[v] != 0 && rng.gen_range(0..1000) < keep_x1000 {
            vars[v] = backbone[v] > 0;
        }
    }
    vars
}

fn choose_backbone_var(
    cl: &Clause,
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
    vars: &[bool],
    sat_count: &[u8],
    backbone: &[i8],
    rng: &mut SmallRng,
    noise_x1000: u32,
) -> usize {
    let mut best_v = usize::MAX;
    let mut best_s = i32::MIN;
    for &lit in &cl.lits {
        let v = lit.unsigned_abs() as usize - 1;
        let (make, brk) = make_break(v, pos_occ, neg_occ, vars, sat_count, None);
        let penalty = if backbone[v] != 0 && (backbone[v] > 0) == vars[v] {
            9
        } else {
            0
        };
        let score = make * 16 - brk * 13 - penalty + rng.gen_range(0..5);
        if score > best_s {
            best_s = score;
            best_v = v;
        }
    }
    if best_v != usize::MAX && rng.gen_range(0..1000) >= noise_x1000 {
        best_v
    } else {
        cl.lits[rng.gen_range(0..3)].unsigned_abs() as usize - 1
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let clauses: Vec<Clause> = challenge
        .clauses
        .iter()
        .filter_map(|c| {
            if c.len() < 3 {
                None
            } else {
                Some(Clause {
                    lits: [c[0], c[1], c[2]],
                })
            }
        })
        .collect();
    let nc = clauses.len();

    let mut pos_occ = vec![Vec::<usize>::new(); nv];
    let mut neg_occ = vec![Vec::<usize>::new(); nv];
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    for (ci, cl) in clauses.iter().enumerate() {
        for &lit in &cl.lits {
            let v = lit.unsigned_abs() as usize - 1;
            if lit > 0 {
                pos_occ[v].push(ci);
                p_cnt[v] += 1;
            } else {
                neg_occ[v].push(ci);
                n_cnt[v] += 1;
            }
        }
    }

    let mut rng =
        SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let density = nc as f64 / nv.max(1) as f64;
    let initial_fuel = unsafe { __fuel_remaining };
    let fuel_mul = if initial_fuel >= 80_000_000_000 {
        3
    } else if initial_fuel >= 30_000_000_000 {
        2
    } else {
        1
    };
    let restarts = (if nv <= 100 {
        96
    } else if nv <= 500 {
        32
    } else {
        16
    }) * fuel_mul;
    let steps_per_restart = (if density < 4.0 {
        nv * 360
    } else if density < 4.35 {
        nv * 900
    } else {
        nv * 520
    }) * fuel_mul;
    let noise_x1000 = if density < 4.0 {
        220
    } else if density < 4.35 {
        100
    } else {
        260
    };

    let mut best_vars = vec![false; nv];
    let mut best_unsat = usize::MAX;
    let mut sat_count = vec![0u8; nc];
    let mut unsat = Vec::with_capacity(nc);
    let mut in_unsat = vec![false; nc];
    let mut unsat_pos = vec![usize::MAX; nc];
    let mut dirty = Vec::new();
    let mut elite = Vec::<(usize, Vec<bool>)>::new();

    for restart in 0..restarts {
        let mut vars = init_assignment(nv, &p_cnt, &n_cnt, restart, &mut rng);
        rebuild_unsat(
            &clauses,
            &vars,
            &mut sat_count,
            &mut unsat,
            &mut in_unsat,
            &mut unsat_pos,
        );

        let mut restart_best = unsat.len();
        let mut restart_best_vars = vars.clone();
        for _step in 0..steps_per_restart {
            if unsat.is_empty() {
                let _ = save_solution(&Solution { variables: vars });
                return Ok(());
            }
            if unsat.len() < best_unsat {
                best_unsat = unsat.len();
                best_vars.clone_from(&vars);
                remember_elite(&mut elite, unsat.len(), &vars);
            }
            if unsat.len() < restart_best {
                restart_best = unsat.len();
                restart_best_vars.clone_from(&vars);
            }

            let ci = unsat[rng.gen_range(0..unsat.len())];
            let cl = &clauses[ci];
            let mode = 0;
            let var = choose_var(
                cl,
                &pos_occ,
                &neg_occ,
                &vars,
                &sat_count,
                None,
                &mut rng,
                noise_x1000,
                mode,
            );
            flip_var(
                var,
                &clauses,
                &pos_occ,
                &neg_occ,
                &mut vars,
                &mut sat_count,
                &mut dirty,
                &mut unsat,
                &mut in_unsat,
                &mut unsat_pos,
            );
        }
        remember_elite(&mut elite, restart_best, &restart_best_vars);
    }

    if initial_fuel >= 80_000_000_000 {
        if let Some(backbone) = build_backbone(&elite, nv) {
            let attempts = if nv <= 100 {
                64
            } else if nv <= 500 {
                28
            } else {
                12
            };
            let repair_steps = if density < 4.35 { nv * 420 } else { nv * 260 };
            for attempt in 0..attempts {
                let mut vars =
                    backbone_assignment(nv, &p_cnt, &n_cnt, &backbone, attempt, &mut rng);
                rebuild_unsat(
                    &clauses,
                    &vars,
                    &mut sat_count,
                    &mut unsat,
                    &mut in_unsat,
                    &mut unsat_pos,
                );

                for _step in 0..repair_steps {
                    if unsat.is_empty() {
                        let _ = save_solution(&Solution { variables: vars });
                        return Ok(());
                    }
                    if unsat.len() < best_unsat {
                        best_unsat = unsat.len();
                        best_vars.clone_from(&vars);
                    }

                    let ci = unsat[rng.gen_range(0..unsat.len())];
                    let cl = &clauses[ci];
                    let var = choose_backbone_var(
                        cl,
                        &pos_occ,
                        &neg_occ,
                        &vars,
                        &sat_count,
                        &backbone,
                        &mut rng,
                        noise_x1000 + 60,
                    );
                    flip_var(
                        var,
                        &clauses,
                        &pos_occ,
                        &neg_occ,
                        &mut vars,
                        &mut sat_count,
                        &mut dirty,
                        &mut unsat,
                        &mut in_unsat,
                        &mut unsat_pos,
                    );
                }
            }
        }
    }

    let _ = save_solution(&Solution {
        variables: best_vars,
    });
    Ok(())
}

pub fn help() {
    println!("near_sat_v2: bounded focused WalkSAT/GSAT hybrid for random 3-SAT.");
}
