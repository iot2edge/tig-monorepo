// knap_fast — minimum-fuel QKP solver targeting TIG c003 production tracks.
//
// Design:
//  Phase 1 — multi-start construction + LS. 8 starts at n<=1500, 2 at n>1500:
//    0: live-contrib greedy.
//    1: tabu-style static T(i)/w sort-and-fill (matches tabu_search baseline construction).
//    2: pair-seed (best (i,j) by V_ij/(w_i+w_j)) + greedy.
//    3+: GRASP+RCL=6 (uniform pick from top-K).
//  Phase 2 — ILS at n<=1500: 6 rounds of kick + greedy refill + LS.
//  Phase 3 — SA at n<=1500: 6 bouts × 8000 iter pseudo-Metropolis 1-1.
// LS: best-improvement 1-1, add, 1-2, 2-1, 2-2 swaps until no improvement.
// Maintained sel/uns lists make LS scale O(|S|*|U|) instead of O(n^2).

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[inline]
fn rng_step(s: &mut u64) -> u32 {
    // Splitmix64
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z ^ (z >> 31)) & 0xFFFFFFFF) as u32
}

struct State {
    n: usize,
    cap: u32,
    selected: Vec<bool>,
    contrib: Vec<i64>,
    total_w: u32,
}

impl State {
    fn new_empty(ch: &Challenge) -> Self {
        let n = ch.num_items;
        Self {
            n,
            cap: ch.max_weight,
            selected: vec![false; n],
            contrib: vec![0i64; n],
            total_w: 0,
        }
    }

    fn reset(&mut self) {
        for s in self.selected.iter_mut() {
            *s = false;
        }
        for c in self.contrib.iter_mut() {
            *c = 0;
        }
        self.total_w = 0;
    }

    #[inline]
    fn slack(&self) -> u32 {
        self.cap.saturating_sub(self.total_w)
    }

    fn add_item(&mut self, ch: &Challenge, i: usize) {
        if self.selected[i] {
            return;
        }
        self.selected[i] = true;
        self.total_w += ch.weights[i];
        let row = &ch.interaction_values[i];
        for j in 0..self.n {
            self.contrib[j] += row[j] as i64;
        }
    }

    fn remove_item(&mut self, ch: &Challenge, i: usize) {
        if !self.selected[i] {
            return;
        }
        self.selected[i] = false;
        self.total_w -= ch.weights[i];
        let row = &ch.interaction_values[i];
        for j in 0..self.n {
            self.contrib[j] -= row[j] as i64;
        }
    }

    fn swap_in_place(&mut self, ch: &Challenge, r: usize, a: usize) {
        debug_assert!(self.selected[r] && !self.selected[a]);
        self.selected[r] = false;
        self.selected[a] = true;
        self.total_w = self.total_w - ch.weights[r] + ch.weights[a];
        let row_r = &ch.interaction_values[r];
        let row_a = &ch.interaction_values[a];
        for j in 0..self.n {
            self.contrib[j] += row_a[j] as i64 - row_r[j] as i64;
        }
    }

    fn total_value(&self, ch: &Challenge) -> i64 {
        // sum over pairs i<j in S of V_ij = (1/2) sum_{i in S} contrib[i]
        let mut v: i64 = 0;
        for i in 0..self.n {
            if self.selected[i] {
                v += self.contrib[i];
            }
        }
        v / 2
    }
}

fn total_interactions(ch: &Challenge) -> Vec<i64> {
    let n = ch.num_items;
    let mut t = vec![0i64; n];
    for i in 0..n {
        let mut s: i64 = 0;
        let row = &ch.interaction_values[i];
        for j in 0..n {
            s += row[j] as i64;
        }
        t[i] = s;
    }
    t
}

fn greedy_construct(state: &mut State, ch: &Challenge, total: &[i64], rng: &mut u64, noise: u32) {
    let n = ch.num_items;
    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }
        let any_sel = state.total_w > 0;
        let mut best_score: i64 = i64::MIN;
        let mut best_i: Option<usize> = None;
        for i in 0..n {
            if state.selected[i] {
                continue;
            }
            let w = ch.weights[i];
            if w == 0 || w > slack {
                continue;
            }
            let raw = if any_sel { state.contrib[i] } else { total[i] };
            if raw <= 0 {
                continue;
            }
            let mut score = (raw * 1000) / (w as i64);
            if noise != 0 {
                score += (rng_step(rng) & noise) as i64;
            }
            if score > best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => state.add_item(ch, i),
            None => break,
        }
    }
}

// Tabu-style construction: sort items once by T(i)/w(i), greedy fill in that order.
// Matches the tabu_search baseline starting solution.
fn static_total_construct(state: &mut State, ch: &Challenge, total: &[i64]) {
    let n = ch.num_items;
    let cap = state.cap;
    let mut order: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = ch.weights[i];
        if w == 0 || w > cap {
            continue;
        }
        // Even items with T(i)<=0 are kept — sort puts them last.
        let t = total[i];
        let s = (t * 1000) / (w as i64).max(1);
        order.push((s, i));
    }
    order.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for &(_, i) in &order {
        let w = ch.weights[i];
        if state.total_w + w <= cap {
            state.add_item(ch, i);
        }
    }
}

// GRASP-style construction with RCL: each pick chosen randomly from top-K candidates.
fn grasp_construct(
    state: &mut State,
    ch: &Challenge,
    total: &[i64],
    rng: &mut u64,
    rcl_size: usize,
) {
    let n = ch.num_items;
    let mut cands: Vec<(i64, usize)> = Vec::with_capacity(n);

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }
        let any_sel = state.total_w > 0;
        cands.clear();
        for i in 0..n {
            if state.selected[i] {
                continue;
            }
            let w = ch.weights[i];
            if w == 0 || w > slack {
                continue;
            }
            let raw = if any_sel { state.contrib[i] } else { total[i] };
            if raw <= 0 {
                continue;
            }
            let score = (raw * 1000) / (w as i64);
            cands.push((score, i));
        }
        if cands.is_empty() {
            break;
        }
        let take = cands.len().min(rcl_size);
        cands.select_nth_unstable_by(take - 1, |a, b| b.0.cmp(&a.0));
        // Pick uniformly from top-K (simple version of RCL).
        let pick = (rng_step(rng) as usize) % take;
        let chosen = cands[pick].1;
        state.add_item(ch, chosen);
    }
}

fn local_swap_1_1(state: &mut State, ch: &Challenge) -> bool {
    let n = ch.num_items;
    let mut any_improved = false;
    let mut sel_list: Vec<usize> = (0..n).filter(|&i| state.selected[i]).collect();
    let mut uns_list: Vec<usize> = (0..n).filter(|&i| !state.selected[i]).collect();
    loop {
        let mut best_delta: i64 = 0;
        let mut best_pair: Option<(usize, usize, usize, usize)> = None; // (r, a, r_idx, a_idx)
        for ri in 0..sel_list.len() {
            let r = sel_list[ri];
            let wr = ch.weights[r];
            let loss = state.contrib[r];
            let row_r = &ch.interaction_values[r];
            for ai in 0..uns_list.len() {
                let a = uns_list[ai];
                let wa = ch.weights[a];
                if wa == 0 {
                    continue;
                }
                let new_w = state.total_w as i64 - wr as i64 + wa as i64;
                if new_w > state.cap as i64 {
                    continue;
                }
                let gain_a = state.contrib[a] - row_r[a] as i64;
                let delta = gain_a - loss;
                if delta > best_delta {
                    best_delta = delta;
                    best_pair = Some((r, a, ri, ai));
                }
            }
        }
        match best_pair {
            Some((r, a, ri, ai)) => {
                state.swap_in_place(ch, r, a);
                sel_list[ri] = a;
                uns_list[ai] = r;
                any_improved = true;
            }
            None => break,
        }
    }
    any_improved
}

fn local_add(state: &mut State, ch: &Challenge) -> bool {
    let n = ch.num_items;
    let mut any_improved = false;
    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }
        let mut best_gain: i64 = 0;
        let mut best_i: Option<usize> = None;
        for i in 0..n {
            if state.selected[i] {
                continue;
            }
            let w = ch.weights[i];
            if w == 0 || w > slack {
                continue;
            }
            let g = state.contrib[i];
            if g > best_gain {
                best_gain = g;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                state.add_item(ch, i);
                any_improved = true;
            }
            None => break,
        }
    }
    any_improved
}

fn local_swap_1_2(state: &mut State, ch: &Challenge) -> bool {
    let n = ch.num_items;
    const K: usize = 24;
    let mut any_improved = false;
    let mut cands_buf: Vec<(i64, usize, u32, i64)> = Vec::with_capacity(n);

    'outer: loop {
        let sel_list: Vec<usize> = (0..n).filter(|&i| state.selected[i]).collect();
        let uns_list: Vec<usize> = (0..n).filter(|&i| !state.selected[i]).collect();

        let mut best_delta: i64 = 0;
        let mut best: Option<(usize, usize, usize)> = None;
        for &r in &sel_list {
            let wr = ch.weights[r];
            let loss = state.contrib[r];
            let new_cap = state.cap as i64 - state.total_w as i64 + wr as i64;
            if new_cap <= 0 {
                continue;
            }
            let row_r = &ch.interaction_values[r];

            cands_buf.clear();
            for &i in &uns_list {
                if i == r {
                    continue;
                }
                let wi = ch.weights[i];
                if wi == 0 || (wi as i64) > new_cap {
                    continue;
                }
                let net = state.contrib[i] - row_r[i] as i64;
                if net <= 0 {
                    continue;
                }
                cands_buf.push((net, i, wi, net * 1000 / (wi as i64)));
            }
            if cands_buf.len() < 2 {
                continue;
            }
            let take = cands_buf.len().min(K);
            cands_buf.select_nth_unstable_by(take - 1, |a, b| b.3.cmp(&a.3));
            let cands = &cands_buf[..take];

            for i in 0..cands.len() {
                let (na, a, wa, _) = cands[i];
                if (wa as i64) > new_cap {
                    continue;
                }
                let row_a = &ch.interaction_values[a];
                for j in (i + 1)..cands.len() {
                    let (nb, b, wb, _) = cands[j];
                    if (wa as i64) + (wb as i64) > new_cap {
                        continue;
                    }
                    let v_ab = row_a[b] as i64;
                    let gain = na + nb + v_ab;
                    let delta = gain - loss;
                    if delta > best_delta {
                        best_delta = delta;
                        best = Some((r, a, b));
                    }
                }
            }
        }
        match best {
            Some((r, a, b)) => {
                state.remove_item(ch, r);
                state.add_item(ch, a);
                state.add_item(ch, b);
                any_improved = true;
                continue 'outer;
            }
            None => break,
        }
    }
    any_improved
}

fn local_swap_2_1(state: &mut State, ch: &Challenge) -> bool {
    let n = ch.num_items;
    const KS: usize = 24;
    const KU: usize = 32;
    let mut any_improved = false;
    let mut sel_buf: Vec<(i64, usize, u32, i64)> = Vec::with_capacity(n);
    let mut uns_buf: Vec<(i64, usize, u32, i64)> = Vec::with_capacity(n);

    'outer: loop {
        sel_buf.clear();
        let mut uns_full: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            if state.selected[i] {
                let wi = ch.weights[i];
                if wi == 0 {
                    continue;
                }
                let pr = state.contrib[i] * 1000 / (wi as i64);
                sel_buf.push((state.contrib[i], i, wi, pr));
            } else {
                uns_full.push(i);
            }
        }
        if sel_buf.len() < 2 {
            return any_improved;
        }
        let sel_take = sel_buf.len().min(KS);
        sel_buf.select_nth_unstable_by(sel_take - 1, |a, b| a.3.cmp(&b.3));
        let sel: Vec<(i64, usize, u32, i64)> = sel_buf[..sel_take].to_vec();

        let mut best_delta: i64 = 0;
        let mut best: Option<(usize, usize, usize)> = None;

        for i in 0..sel.len() {
            let (lr1, r1, wr1, _) = sel[i];
            let row_r1 = &ch.interaction_values[r1];
            for j in (i + 1)..sel.len() {
                let (lr2, r2, wr2, _) = sel[j];
                let row_r2 = &ch.interaction_values[r2];
                let v_r1r2 = row_r1[r2] as i64;
                let total_loss = lr1 + lr2 - v_r1r2;
                let new_cap = state.cap as i64 - state.total_w as i64 + wr1 as i64 + wr2 as i64;
                if new_cap <= 0 {
                    continue;
                }
                uns_buf.clear();
                for &a in &uns_full {
                    if a == r1 || a == r2 {
                        continue;
                    }
                    let wa = ch.weights[a];
                    if wa == 0 || (wa as i64) > new_cap {
                        continue;
                    }
                    let gain_a = state.contrib[a] - row_r1[a] as i64 - row_r2[a] as i64;
                    if gain_a <= total_loss {
                        continue;
                    }
                    let pr = gain_a * 1000 / (wa as i64);
                    uns_buf.push((gain_a, a, wa, pr));
                }
                if uns_buf.is_empty() {
                    continue;
                }
                let uns_take = uns_buf.len().min(KU);
                uns_buf.select_nth_unstable_by(uns_take - 1, |a, b| b.3.cmp(&a.3));
                for &(gain_a, a, _wa, _) in &uns_buf[..uns_take] {
                    let delta = gain_a - total_loss;
                    if delta > best_delta {
                        best_delta = delta;
                        best = Some((r1, r2, a));
                    }
                }
            }
        }
        match best {
            Some((r1, r2, a)) => {
                state.remove_item(ch, r1);
                state.remove_item(ch, r2);
                state.add_item(ch, a);
                any_improved = true;
                continue 'outer;
            }
            None => break,
        }
    }
    any_improved
}

// 2-2 swap: remove (r1, r2), add (a, b). Bounded by top-K of each.
fn local_swap_2_2(state: &mut State, ch: &Challenge) -> bool {
    let n = ch.num_items;
    const KS: usize = 12;
    const KU: usize = 16;
    let mut any_improved = false;
    let mut sel_buf: Vec<(i64, usize, u32, i64)> = Vec::with_capacity(n);
    let mut uns_buf: Vec<(i64, usize, u32, i64)> = Vec::with_capacity(n);

    'outer: loop {
        sel_buf.clear();
        uns_buf.clear();
        for i in 0..n {
            let wi = ch.weights[i];
            if wi == 0 {
                continue;
            }
            if state.selected[i] {
                let pr = state.contrib[i] * 1000 / (wi as i64);
                sel_buf.push((state.contrib[i], i, wi, pr));
            } else if state.contrib[i] > 0 {
                let pr = state.contrib[i] * 1000 / (wi as i64);
                uns_buf.push((state.contrib[i], i, wi, pr));
            }
        }
        if sel_buf.len() < 2 {
            return any_improved;
        }
        let sel_take = sel_buf.len().min(KS);
        sel_buf.select_nth_unstable_by(sel_take - 1, |a, b| a.3.cmp(&b.3));
        let sel: Vec<(i64, usize, u32, i64)> = sel_buf[..sel_take].to_vec();
        if uns_buf.len() < 2 {
            return any_improved;
        }
        let uns_take = uns_buf.len().min(KU);
        uns_buf.select_nth_unstable_by(uns_take - 1, |a, b| b.3.cmp(&a.3));
        let uns: Vec<(i64, usize, u32, i64)> = uns_buf[..uns_take].to_vec();

        let mut best_delta: i64 = 0;
        let mut best: Option<(usize, usize, usize, usize)> = None;

        for i in 0..sel.len() {
            let (lr1, r1, wr1, _) = sel[i];
            let row_r1 = &ch.interaction_values[r1];
            for j in (i + 1)..sel.len() {
                let (lr2, r2, wr2, _) = sel[j];
                let row_r2 = &ch.interaction_values[r2];
                let v_r1r2 = row_r1[r2] as i64;
                let total_loss = lr1 + lr2 - v_r1r2;
                let new_cap = state.cap as i64 - state.total_w as i64 + wr1 as i64 + wr2 as i64;
                if new_cap <= 0 {
                    continue;
                }

                for p in 0..uns.len() {
                    let (_, a, wa, _) = uns[p];
                    if (wa as i64) > new_cap {
                        continue;
                    }
                    let row_a = &ch.interaction_values[a];
                    let gain_a = state.contrib[a] - row_r1[a] as i64 - row_r2[a] as i64;
                    for q in (p + 1)..uns.len() {
                        let (_, b, wb, _) = uns[q];
                        if (wa as i64) + (wb as i64) > new_cap {
                            continue;
                        }
                        let gain_b = state.contrib[b] - row_r1[b] as i64 - row_r2[b] as i64;
                        let v_ab = row_a[b] as i64;
                        // Total gain of adding {a,b} into S - {r1,r2}
                        let total_gain = gain_a + gain_b + v_ab;
                        let delta = total_gain - total_loss;
                        if delta > best_delta {
                            best_delta = delta;
                            best = Some((r1, r2, a, b));
                        }
                    }
                }
            }
        }

        match best {
            Some((r1, r2, a, b)) => {
                state.remove_item(ch, r1);
                state.remove_item(ch, r2);
                state.add_item(ch, a);
                state.add_item(ch, b);
                any_improved = true;
                continue 'outer;
            }
            None => break,
        }
    }
    any_improved
}

fn local_search(state: &mut State, ch: &Challenge) {
    loop {
        let mut imp = false;
        imp |= local_swap_1_1(state, ch);
        imp |= local_add(state, ch);
        imp |= local_swap_1_2(state, ch);
        imp |= local_swap_2_1(state, ch);
        imp |= local_swap_2_2(state, ch);
        if !imp {
            break;
        }
    }
}

// Pair seed: find best pair (i,j) by V_ij/(w_i+w_j) and seed the solution with both.
fn pair_seed(state: &mut State, ch: &Challenge) {
    let n = ch.num_items;
    let cap = state.cap;
    let mut best_score: i64 = 0;
    let mut best: Option<(usize, usize)> = None;
    for i in 0..n {
        let wi = ch.weights[i];
        if wi == 0 || wi > cap {
            continue;
        }
        let row_i = &ch.interaction_values[i];
        for j in (i + 1)..n {
            let wj = ch.weights[j];
            if wj == 0 || wi + wj > cap {
                continue;
            }
            let v = row_i[j] as i64;
            if v <= 0 {
                continue;
            }
            let s = (v * 1_000_000) / ((wi + wj) as i64);
            if s > best_score {
                best_score = s;
                best = Some((i, j));
            }
        }
    }
    if let Some((i, j)) = best {
        state.add_item(ch, i);
        state.add_item(ch, j);
    }
}

// Hub seed: force first pick to be top_hubs[k] where hubs sorted by T(i)/w(i).
fn hub_seed(state: &mut State, ch: &Challenge, total: &[i64], rank: usize) {
    let n = ch.num_items;
    let cap = state.cap;
    let mut hubs: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = ch.weights[i];
        if w == 0 || w > cap {
            continue;
        }
        if total[i] <= 0 {
            continue;
        }
        let s = (total[i] * 1000) / (w as i64);
        hubs.push((s, i));
    }
    if hubs.is_empty() {
        return;
    }
    hubs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let k = rank.min(hubs.len() - 1);
    state.add_item(ch, hubs[k].1);
}

// Restore state to a given selection (rebuild contrib, total_w from scratch).
fn restore_from(state: &mut State, ch: &Challenge, sel: &[bool]) {
    state.reset();
    for i in 0..state.n {
        if sel[i] {
            state.add_item(ch, i);
        }
    }
}

// Simulated annealing on 1-1 swaps. Tracks best; value updated incrementally.
// Maintains lists of selected/unselected for efficient random picks.
fn sa_anneal(state: &mut State, ch: &Challenge, rng: &mut u64, n_iter: usize) -> Vec<bool> {
    let n = ch.num_items;
    let mut cur_value = state.total_value(ch);
    let mut best_value = cur_value;
    let mut best_sel = state.selected.clone();

    if n_iter == 0 {
        return best_sel;
    }

    // Build selected / unselected indices
    let mut sel: Vec<usize> = (0..n).filter(|&i| state.selected[i]).collect();
    let mut uns: Vec<usize> = (0..n).filter(|&i| !state.selected[i]).collect();

    if sel.is_empty() || uns.is_empty() {
        return best_sel;
    }

    let t_start: i64 = 100;
    let t_end: i64 = 1;

    for it in 0..n_iter {
        let t = t_start - ((t_start - t_end) * it as i64) / (n_iter.max(1) as i64);

        let ri_idx = (rng_step(rng) as usize) % sel.len();
        let r = sel[ri_idx];
        let ai_idx = (rng_step(rng) as usize) % uns.len();
        let a = uns[ai_idx];
        let wr = ch.weights[r];
        let wa = ch.weights[a];
        if wa == 0 {
            continue;
        }
        let new_w = state.total_w as i64 - wr as i64 + wa as i64;
        if new_w < 0 || new_w > state.cap as i64 {
            continue;
        }
        let v_ar = ch.interaction_values[a][r] as i64;
        let delta = state.contrib[a] - v_ar - state.contrib[r];

        let accept = if delta >= 0 {
            true
        } else {
            let mag = -delta;
            let thresh = if t > 0 && t + mag > 0 {
                ((t * 100) / (t + mag)) as u32
            } else {
                0
            };
            (rng_step(rng) % 100) < thresh
        };

        if accept {
            state.swap_in_place(ch, r, a);
            cur_value += delta;
            // Update sel/uns lists
            sel[ri_idx] = a;
            uns[ai_idx] = r;
            if cur_value > best_value {
                best_value = cur_value;
                best_sel.copy_from_slice(&state.selected);
            }
        }
    }
    best_sel
}

// Kick: remove the k items with lowest contrib/w (worst contributors), force-add
// nothing — let greedy re-fill. This is a cheap diversification.
fn kick_worst(state: &mut State, ch: &Challenge, k: usize) {
    let mut worst: Vec<(i64, usize)> = Vec::with_capacity(state.n);
    for i in 0..state.n {
        if state.selected[i] {
            let w = ch.weights[i] as i64;
            let pr = if w > 0 {
                state.contrib[i] * 1000 / w
            } else {
                state.contrib[i]
            };
            worst.push((pr, i));
        }
    }
    if worst.is_empty() {
        return;
    }
    let take = worst.len().min(k);
    worst.select_nth_unstable_by(take.saturating_sub(1), |a, b| a.0.cmp(&b.0));
    let to_remove: Vec<usize> = worst[..take].iter().map(|x| x.1).collect();
    for r in to_remove {
        state.remove_item(ch, r);
    }
}

// Random-kick: remove k random selected items.
fn kick_random(state: &mut State, ch: &Challenge, k: usize, rng: &mut u64) {
    let n = state.n;
    let sel_indices: Vec<usize> = (0..n).filter(|&i| state.selected[i]).collect();
    if sel_indices.is_empty() {
        return;
    }
    let take = sel_indices.len().min(k);
    let mut chosen: Vec<usize> = Vec::with_capacity(take);
    let mut pool = sel_indices.clone();
    for _ in 0..take {
        if pool.is_empty() {
            break;
        }
        let idx = (rng_step(rng) as usize) % pool.len();
        chosen.push(pool.swap_remove(idx));
    }
    for r in chosen {
        state.remove_item(ch, r);
    }
}

fn solve(ch: &Challenge) -> Vec<bool> {
    let n = ch.num_items;
    let total = total_interactions(ch);

    let mut rng_seed: u64 = 0;
    for k in 0..8 {
        rng_seed |= (ch.seed[k] as u64) << (8 * k);
    }
    if rng_seed == 0 {
        rng_seed = 0xCAFEBABE_DEADBEEF;
    }

    let n_starts: usize = if n <= 1500 { 8 } else { 2 };
    let n_ils: usize = if n <= 1500 { 6 } else { 0 };
    let rcl_size: usize = 6;

    let mut st = State::new_empty(ch);
    let mut best_value: i64 = i64::MIN;
    let mut best_selected = vec![false; n];

    // Phase 1: diverse multi-start.
    // 0=greedy(live), 1=tabu-style static T(i)/w, 2=pair-seed, rest=GRASP+RCL.
    for start in 0..n_starts {
        st.reset();
        match start {
            0 => {
                greedy_construct(&mut st, ch, &total, &mut rng_seed, 0);
            }
            1 => {
                static_total_construct(&mut st, ch, &total);
            }
            2 => {
                pair_seed(&mut st, ch);
                greedy_construct(&mut st, ch, &total, &mut rng_seed, 0);
            }
            _ => {
                grasp_construct(&mut st, ch, &total, &mut rng_seed, rcl_size);
            }
        }
        local_search(&mut st, ch);

        let v = st.total_value(ch);
        if v > best_value {
            best_value = v;
            best_selected.copy_from_slice(&st.selected);
        }
    }

    // Phase 2: ILS — perturb best, re-optimize, accept if better.
    let n_selected_est = best_selected.iter().filter(|&&b| b).count();
    let kick_size = (n_selected_est / 12).max(2).min(20);

    for round in 0..n_ils {
        restore_from(&mut st, ch, &best_selected);
        if round % 2 == 0 {
            kick_worst(&mut st, ch, kick_size);
        } else {
            kick_random(&mut st, ch, kick_size, &mut rng_seed);
        }
        let noise = if round > 2 { 0xF } else { 0 };
        greedy_construct(&mut st, ch, &total, &mut rng_seed, noise);
        local_search(&mut st, ch);

        let v = st.total_value(ch);
        if v > best_value {
            best_value = v;
            best_selected.copy_from_slice(&st.selected);
        }
    }

    // Phase 3: Simulated annealing seeded from best, then LS to converge.
    // Skip SA at n>1500 to save fuel (LS alone clears thresholds at large n).
    let n_sa_bouts: usize = if n <= 1500 { 6 } else { 0 };
    let sa_iters_per_bout: usize = if n <= 1500 { 8000 } else { 0 };
    for _bout in 0..n_sa_bouts {
        restore_from(&mut st, ch, &best_selected);
        let sa_best = sa_anneal(&mut st, ch, &mut rng_seed, sa_iters_per_bout);
        restore_from(&mut st, ch, &sa_best);
        local_search(&mut st, ch);
        let v = st.total_value(ch);
        if v > best_value {
            best_value = v;
            best_selected.copy_from_slice(&st.selected);
        }
    }

    best_selected
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let selected = solve(challenge);
    let n = challenge.num_items;
    let items: Vec<usize> = (0..n).filter(|&i| selected[i]).collect();
    let _ = save_solution(&Solution { items });
    Ok(())
}

pub fn help() {
    println!("knap_fast: multi-start greedy + 1-1/1-2/2-1 swap + add. Minimum-fuel QKP.");
}
