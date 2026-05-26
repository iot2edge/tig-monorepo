// STRUCTURAL INNOVATION: exact quadratic core optimization.
//
// The chain's QKP solvers (kqo_v5, v6, v7) all do "core-DP refinement" using a
// LINEAR 0-1 knapsack DP over a core of items. This treats core items as
// independent — picking item k contributes its current marginal contrib, but
// does NOT account for interactions BETWEEN core items.
//
// For this challenge (values[i] = 0, interaction_values[i][j] >= 0), almost all
// signal lives in the quadratic terms. So linear DP on the core systematically
// misses joint moves where two core items reinforce each other.
//
// This module replaces the linear DP with EXACT enumeration over 2^K subsets
// of a K-item core, computing the true quadratic objective for each candidate
// subset via Gray-code incremental updates.
//
// Complexity: O(K * 2^K). For K=22 → ~92M ops per call (~100ms). For K=24 →
// ~400M ops (~400ms). Practical cap is K=24; we adapt K based on n.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::knapsack::{Challenge, Solution};

/// Refine `solution` by exact quadratic enumeration over a K-item core selected
/// deterministically (low-contrib selected + high-contrib unselected).
/// Returns the improved solution (or the same one, unchanged) and its value.
pub fn refine(challenge: &Challenge, solution: &Solution, k_target: usize) -> (Solution, i64) {
    refine_with_core_strategy(challenge, solution, k_target, CoreStrategy::Boundary, 0)
}

#[derive(Copy, Clone)]
pub enum CoreStrategy {
    /// Lowest-contrib selected + highest-contrib unselected (default).
    Boundary,
    /// Items with highest INTERACTION STRENGTH within a candidate pool, then
    /// expanded by interaction-density to neighbors. Captures interactive cliques.
    InteractionDense,
    /// Random sample (anchored by seed).
    Random,
    /// Mix: half boundary, half random.
    BoundaryRandom,
}

fn refine_with_core_strategy(
    challenge: &Challenge,
    solution: &Solution,
    k_target: usize,
    strategy: CoreStrategy,
    seed: u64,
) -> (Solution, i64) {
    let n = challenge.num_items;
    let cap = challenge.max_weight;

    let mut in_sol = vec![false; n];
    for &i in &solution.items {
        in_sol[i] = true;
    }

    // Current value (full re-eval to avoid trusting upstream State).
    let mut cur_value: i64 = 0;
    let mut cur_weight: u32 = 0;
    let sel: Vec<usize> = (0..n).filter(|&i| in_sol[i]).collect();
    for &i in &sel {
        cur_value += challenge.values[i] as i64;
        cur_weight += challenge.weights[i];
        for &j in &sel {
            if j > i {
                cur_value += challenge.interaction_values[i][j] as i64;
            }
        }
    }

    // Per-item marginal contrib: what i contributes / would contribute right now.
    // For i selected: contrib[i] = v_i + sum_{j in sel, j != i} q_ij  (value lost if removed)
    // For i not selected: contrib[i] = v_i + sum_{j in sel} q_ij  (value gained if added)
    let mut contrib: Vec<i64> = vec![0; n];
    for i in 0..n {
        let mut c = challenge.values[i] as i64;
        for &j in &sel {
            if j != i {
                c += challenge.interaction_values[i][j] as i64;
            }
        }
        contrib[i] = c;
    }

    // Core selection: items most likely to flip.
    // - From SELECTED: lowest contrib (least committed) — these may want to leave
    // - From NOT-SELECTED: highest contrib (most tempting) — these may want to join
    let mut sel_scored: Vec<(usize, i64)> = Vec::new();
    let mut unsel_scored: Vec<(usize, i64)> = Vec::new();
    for i in 0..n {
        if challenge.weights[i] > cap {
            continue;
        }
        if in_sol[i] {
            sel_scored.push((i, contrib[i]));
        } else {
            unsel_scored.push((i, contrib[i]));
        }
    }
    sel_scored.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    unsel_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // Clamp k for safety. 2^26 = 67M would push fuel/time; cap at 24.
    let k_cap = 24usize.min(k_target);

    let mut core: Vec<usize> = Vec::with_capacity(k_cap);
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0x9E3779B97F4A7C15));

    match strategy {
        CoreStrategy::Boundary => {
            let half = k_cap / 2;
            let take_sel = sel_scored.len().min(half);
            let take_unsel = unsel_scored.len().min(k_cap - take_sel);
            for (idx, _) in sel_scored.iter().take(take_sel) {
                core.push(*idx);
            }
            for (idx, _) in unsel_scored.iter().take(take_unsel) {
                core.push(*idx);
            }
        }
        CoreStrategy::InteractionDense => {
            // Pool: top ~80 items by total interaction strength (sum of all q_ij).
            // Then greedily pick items maximally interactive with current core.
            let pool_size = (k_cap * 4).min(n);
            let mut interaction_sum: Vec<(usize, i64)> = (0..n)
                .filter(|&i| challenge.weights[i] <= cap)
                .map(|i| {
                    let s: i64 = challenge.interaction_values[i]
                        .iter()
                        .map(|&v| v as i64)
                        .sum();
                    (i, s)
                })
                .collect();
            interaction_sum.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            let pool: Vec<usize> = interaction_sum
                .iter()
                .take(pool_size)
                .map(|&(i, _)| i)
                .collect();

            // Seed: the highest-interaction item.
            if pool.is_empty() {
                return (solution.clone(), cur_value);
            }
            core.push(pool[0]);
            let mut in_core_set = vec![false; n];
            in_core_set[pool[0]] = true;

            // Iteratively add items maximally connected to current core.
            while core.len() < k_cap {
                let mut best_item: Option<usize> = None;
                let mut best_score: i64 = i64::MIN;
                for &cand in &pool {
                    if in_core_set[cand] {
                        continue;
                    }
                    let mut s: i64 = 0;
                    for &c in &core {
                        s += challenge.interaction_values[cand][c] as i64;
                    }
                    if s > best_score {
                        best_score = s;
                        best_item = Some(cand);
                    }
                }
                if let Some(it) = best_item {
                    core.push(it);
                    in_core_set[it] = true;
                } else {
                    break;
                }
            }
        }
        CoreStrategy::Random => {
            // Sample k_cap distinct items uniformly.
            let candidates: Vec<usize> = (0..n).filter(|&i| challenge.weights[i] <= cap).collect();
            if candidates.is_empty() {
                return (solution.clone(), cur_value);
            }
            let mut picked = vec![false; n];
            while core.len() < k_cap && core.len() < candidates.len() {
                let r = rng.gen_range(0..candidates.len());
                let it = candidates[r];
                if !picked[it] {
                    picked[it] = true;
                    core.push(it);
                }
            }
        }
        CoreStrategy::BoundaryRandom => {
            let half = k_cap / 3;
            let take_sel = sel_scored.len().min(half);
            let take_unsel = unsel_scored.len().min(half);
            let mut picked = vec![false; n];
            for (idx, _) in sel_scored.iter().take(take_sel) {
                if !picked[*idx] {
                    picked[*idx] = true;
                    core.push(*idx);
                }
            }
            for (idx, _) in unsel_scored.iter().take(take_unsel) {
                if !picked[*idx] {
                    picked[*idx] = true;
                    core.push(*idx);
                }
            }
            // Fill the rest with random items.
            let candidates: Vec<usize> = (0..n)
                .filter(|&i| challenge.weights[i] <= cap && !picked[i])
                .collect();
            while core.len() < k_cap && !candidates.is_empty() {
                let r = rng.gen_range(0..candidates.len());
                let it = candidates[r];
                if !picked[it] {
                    picked[it] = true;
                    core.push(it);
                }
            }
        }
    }
    if core.is_empty() {
        return (solution.clone(), cur_value);
    }
    let k = core.len();
    if k > 24 {
        return (solution.clone(), cur_value);
    }

    let in_core: Vec<bool> = {
        let mut v = vec![false; n];
        for &c in &core {
            v[c] = true;
        }
        v
    };

    // Fixed-out portion of current selection (selected, not in core).
    let fixed_sel: Vec<usize> = sel.iter().copied().filter(|&i| !in_core[i]).collect();
    let mut fixed_value: i64 = 0;
    let mut fixed_weight: u32 = 0;
    for &i in &fixed_sel {
        fixed_value += challenge.values[i] as i64;
        fixed_weight += challenge.weights[i];
        for &j in &fixed_sel {
            if j > i {
                fixed_value += challenge.interaction_values[i][j] as i64;
            }
        }
    }

    // For each core item: marginal contrib from fixed-out items + linear value.
    let mut f_contrib: Vec<i64> = vec![0; k];
    for (idx, &c) in core.iter().enumerate() {
        let mut s = challenge.values[c] as i64;
        for &f in &fixed_sel {
            s += challenge.interaction_values[c][f] as i64;
        }
        f_contrib[idx] = s;
    }

    // Pairwise interaction matrix among core items (flat row-major).
    let mut q_core: Vec<i64> = vec![0; k * k];
    for i in 0..k {
        for j in 0..k {
            q_core[i * k + j] = challenge.interaction_values[core[i]][core[j]] as i64;
        }
    }

    // Per-core-item weight (flat for cache).
    let w_core: Vec<u32> = core.iter().map(|&c| challenge.weights[c]).collect();

    // Gray-code enumeration. State:
    //   bits         : current core subset bitmask
    //   sub_weight   : sum of weights of core items in current subset
    //   sub_value    : fixed_value + sum_{i in subset} f_contrib[i] + sum_{i<j in subset} q_core[i,j]
    //   q_sum[i]     : sum_{j in current subset} q_core[i,j]  (used for incremental delta)
    let mut bits: u64 = 0;
    let mut sub_weight: u32 = fixed_weight;
    let mut sub_value: i64 = fixed_value;
    let mut q_sum: Vec<i64> = vec![0; k];

    let mut best_bits: u64 = 0;
    let mut best_value: i64 = if sub_weight <= cap {
        sub_value
    } else {
        i64::MIN
    };
    if sub_weight > cap {
        // Empty core means all fixed, sub_weight = fixed_weight ≤ cap always.
        // This branch is impossible by construction; keep for safety.
        return (solution.clone(), cur_value);
    }

    let n_subsets: u64 = 1u64 << k;
    // Standard binary reflected Gray code: at step g (1..n_subsets), flip bit
    // index = trailing_zeros(g).
    for g in 1u64..n_subsets {
        let flip = g.trailing_zeros() as usize;
        let mask = 1u64 << flip;
        let was_set = (bits & mask) != 0;

        // delta = f_contrib[flip] + q_sum[flip]
        // sign:  +delta when adding, -delta when removing
        let delta = f_contrib[flip] + q_sum[flip];
        if was_set {
            bits &= !mask;
            sub_weight -= w_core[flip];
            sub_value -= delta;
            // Decrement q_sum[j] for j != flip by q_core[j, flip]
            let row_base = flip * k;
            for j in 0..k {
                if j != flip {
                    q_sum[j] -= q_core[row_base + j];
                }
            }
        } else {
            bits |= mask;
            sub_weight += w_core[flip];
            sub_value += delta;
            // Increment q_sum[j] for j != flip by q_core[j, flip]
            let row_base = flip * k;
            for j in 0..k {
                if j != flip {
                    q_sum[j] += q_core[row_base + j];
                }
            }
        }

        if sub_weight <= cap && sub_value > best_value {
            best_value = sub_value;
            best_bits = bits;
        }
    }

    if best_value <= cur_value {
        return (solution.clone(), cur_value);
    }

    // Decode best_bits back into a full Solution.
    let mut out_items: Vec<usize> = fixed_sel.clone();
    for idx in 0..k {
        if (best_bits >> idx) & 1 == 1 {
            out_items.push(core[idx]);
        }
    }
    out_items.sort_unstable();
    let new_sol = Solution { items: out_items };
    (new_sol, best_value)
}

/// Run quadratic core refinement across a sweep of strategies and seeds. Each
/// pass starts from the CURRENT best (so improvements compound). Continues
/// until a full sweep yields no improvement OR max_passes is reached.
pub fn refine_iterative(
    challenge: &Challenge,
    solution: &Solution,
    k: usize,
    max_passes: usize,
) -> (Solution, i64) {
    let mut sol = solution.clone();
    let mut val: i64 = 0;
    for &i in &sol.items {
        val += challenge.values[i] as i64;
        for &j in &sol.items {
            if j > i {
                val += challenge.interaction_values[i][j] as i64;
            }
        }
    }

    // Sweep order: deterministic strategies first (Boundary, InteractionDense),
    // then randomized BoundaryRandom with varying seeds.
    let strategies: [(CoreStrategy, u64); 8] = [
        (CoreStrategy::Boundary, 0),
        (CoreStrategy::InteractionDense, 0),
        (CoreStrategy::BoundaryRandom, 17),
        (CoreStrategy::BoundaryRandom, 91),
        (CoreStrategy::Random, 137),
        (CoreStrategy::BoundaryRandom, 251),
        (CoreStrategy::Random, 389),
        (CoreStrategy::BoundaryRandom, 503),
    ];

    for _pass in 0..max_passes {
        let mut improved_this_pass = false;
        for &(strat, seed) in strategies.iter() {
            let (new_sol, new_val) = refine_with_core_strategy(challenge, &sol, k, strat, seed);
            if new_val > val {
                sol = new_sol;
                val = new_val;
                improved_this_pass = true;
            }
        }
        if !improved_this_pass {
            break;
        }
    }
    (sol, val)
}
