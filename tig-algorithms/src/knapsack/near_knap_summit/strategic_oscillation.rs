// STRUCTURAL INNOVATION 2: Strategic Oscillation with Tabu Memory.
//
// Reference: Glover & Laguna's strategic oscillation framework, adapted for
// QKP. Specifically based on the structural pattern in:
//   - Hifi, Wu (2014) — strategic oscillation for QKP variants
//   - Glover (1996) — tabu search with frequency memory
//
// Key structural difference from zenith's ILS + path-relink:
//   ILS:  perturb (small flip) → local search → accept-by-rule
//   SO:   ADD phase drifts INFEASIBLE (overload by alpha*W), then REMOVE
//         phase drifts back to feasibility along a DIFFERENT path.
//         Tabu memory prevents oscillating between same two states.
//
// This explores corners of the feasible region that ILS perturbations can't
// reach, because ILS perturbations stay close to current solution, while SO
// can wholesale REPLACE half the solution items in one cycle.
//
// For pure QKP (v_i = 0, q_ij >= 0), contrib[i] = sum_{j in current sel} q_ij.
// Marginal density score = contrib[i] / weight[i]. We use density-based
// add/remove with tabu controlling what's eligible.

use tig_challenges::knapsack::{Challenge, Solution};

struct OscState<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i64>,
    total_value: i64,
    total_weight: i64,
    tabu_until: Vec<u32>,
}

impl<'a> OscState<'a> {
    fn new(ch: &'a Challenge, init_sol: &Solution) -> Self {
        let n = ch.num_items;
        let mut s = OscState {
            ch,
            selected_bit: vec![false; n],
            contrib: vec![0; n],
            total_value: 0,
            total_weight: 0,
            tabu_until: vec![0; n],
        };
        // Seed contrib[i] = v_i (linear value, which is 0 for this challenge).
        for i in 0..n {
            s.contrib[i] = ch.values[i] as i64;
        }
        for &i in &init_sol.items {
            s.add_item(i);
        }
        s
    }

    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i];
        self.total_weight += self.ch.weights[i] as i64;
        let n = self.ch.num_items;
        for k in 0..n {
            self.contrib[k] += self.ch.interaction_values[i][k] as i64;
        }
        self.selected_bit[i] = true;
    }

    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j];
        self.total_weight -= self.ch.weights[j] as i64;
        let n = self.ch.num_items;
        for k in 0..n {
            self.contrib[k] -= self.ch.interaction_values[j][k] as i64;
        }
        self.selected_bit[j] = false;
    }

    fn to_solution(&self) -> Solution {
        let items: Vec<usize> = (0..self.ch.num_items)
            .filter(|&i| self.selected_bit[i])
            .collect();
        Solution { items }
    }
}

/// Run strategic oscillation starting from `init_sol`. Returns the best
/// FEASIBLE solution visited, with its total value.
pub fn oscillate(
    challenge: &Challenge,
    init_sol: &Solution,
    alpha_x100: u32, // alpha (overload fraction) × 100, e.g. 50 = 0.5
    tabu_tenure: u32,
    max_cycles: u32,
) -> (Solution, i64) {
    let n = challenge.num_items;
    let cap = challenge.max_weight as i64;
    let overload_cap = cap + (cap * alpha_x100 as i64) / 100;

    let mut state = OscState::new(challenge, init_sol);

    // Initial best = the input (which is feasible since came from solver).
    let mut best_sol = state.to_solution();
    let mut best_val = state.total_value;

    let mut iter_count: u32 = 0;
    for _cycle in 0..max_cycles {
        iter_count += 1;

        // ADD phase: drift to overload.
        // Each step: among NOT-selected and NOT-tabu items, pick the one with
        // highest contrib[i] / weight[i]. Tie-break by raw contrib[i].
        loop {
            if state.total_weight >= overload_cap {
                break;
            }
            let mut best_i: Option<usize> = None;
            let mut best_num: i64 = i64::MIN;
            let mut best_w: u32 = 1;
            for i in 0..n {
                if state.selected_bit[i] {
                    continue;
                }
                if state.tabu_until[i] > iter_count {
                    continue;
                }
                let w = challenge.weights[i];
                if w == 0 {
                    continue;
                }
                let c = state.contrib[i];
                // Compare c/w by cross-multiplication: c_a * w_b vs c_b * w_a.
                if best_i.is_none() {
                    best_i = Some(i);
                    best_num = c;
                    best_w = w;
                } else if c * (best_w as i64) > best_num * (w as i64) {
                    best_i = Some(i);
                    best_num = c;
                    best_w = w;
                }
            }
            match best_i {
                Some(i) => {
                    state.add_item(i);
                    state.tabu_until[i] = iter_count + tabu_tenure;
                }
                None => break,
            }
        }

        // REMOVE phase: drift back to feasibility.
        // Each step: among SELECTED and NOT-tabu items, pick the one with
        // lowest contrib[i] / weight[i] (least productive). If all selected
        // are tabu, ignore tabu (aspiration).
        loop {
            if state.total_weight <= cap {
                break;
            }
            let mut worst_i: Option<usize> = None;
            let mut worst_num: i64 = i64::MAX;
            let mut worst_w: u32 = 1;
            let mut worst_i_aspirate: Option<usize> = None;
            let mut worst_num_asp: i64 = i64::MAX;
            let mut worst_w_asp: u32 = 1;
            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }
                let w = challenge.weights[i];
                if w == 0 {
                    continue;
                }
                let c = state.contrib[i];
                let tabu = state.tabu_until[i] > iter_count;
                if !tabu {
                    if worst_i.is_none() {
                        worst_i = Some(i);
                        worst_num = c;
                        worst_w = w;
                    } else if c * (worst_w as i64) < worst_num * (w as i64) {
                        worst_i = Some(i);
                        worst_num = c;
                        worst_w = w;
                    }
                }
                if worst_i_aspirate.is_none() {
                    worst_i_aspirate = Some(i);
                    worst_num_asp = c;
                    worst_w_asp = w;
                } else if c * (worst_w_asp as i64) < worst_num_asp * (w as i64) {
                    worst_i_aspirate = Some(i);
                    worst_num_asp = c;
                    worst_w_asp = w;
                }
            }
            let to_remove = worst_i.or(worst_i_aspirate);
            match to_remove {
                Some(j) => {
                    state.remove_item(j);
                    state.tabu_until[j] = iter_count + tabu_tenure;
                }
                None => break,
            }
        }

        // Check feasibility + record best.
        if state.total_weight <= cap && state.total_value > best_val {
            best_val = state.total_value;
            best_sol = state.to_solution();
        }
    }

    (best_sol, best_val)
}

/// Multi-restart oscillation: try several (alpha, tabu_tenure) combos and
/// pick the best output.
pub fn oscillate_multi(
    challenge: &Challenge,
    init_sol: &Solution,
    init_val: i64,
    max_cycles_per_run: u32,
) -> (Solution, i64) {
    let n = challenge.num_items as u32;
    // Tenure scaled with problem size — more items → longer tabu.
    let tenure_short = (n / 20).max(8);
    let tenure_long = (n / 8).max(16);

    let configs: [(u32, u32); 6] = [
        (30, tenure_short), // 30% overload, short tabu — gentle exploration
        (50, tenure_short), // 50% overload, short tabu
        (50, tenure_long),  // 50% overload, long tabu — deeper memory
        (80, tenure_long),  // 80% overload, long tabu — wide oscillation
        (100, tenure_long), // 100% overload — almost full restart, with tabu structure
        (40, tenure_long),  // 40% overload, long tabu
    ];

    let mut best_sol = init_sol.clone();
    let mut best_val = init_val;
    for &(alpha, tenure) in configs.iter() {
        let (s, v) = oscillate(challenge, &best_sol, alpha, tenure, max_cycles_per_run);
        if v > best_val {
            best_val = v;
            best_sol = s;
        }
    }
    (best_sol, best_val)
}
