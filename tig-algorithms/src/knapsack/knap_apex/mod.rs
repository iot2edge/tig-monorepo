// TIG's UI uses the pattern `tig_challenges::knapsack` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub window_k: Option<usize>,
    pub ils_rounds: Option<usize>,
    pub core_half_dp: Option<usize>,
}

#[derive(Clone, Copy)]
struct Rng {
    state: u64,
}
impl Rng {
    fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 {
            s = 1;
        }
        Self { state: s }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

// TopNeighbors: for each item i, the top-K items by V_ij value (descending).
// Built once per instance; lets sparse LS scan O(K) candidates per item instead of O(n).
struct TopNeighbors {
    k: usize,
    // neighbors[i*k .. (i+1)*k] = top-K item indices most-connected to i (V_ij descending).
    // Items with V_ij = 0 are filled with usize::MAX as a sentinel.
    neighbors: Vec<usize>,
}

impl TopNeighbors {
    fn new(ch: &Challenge, k: usize) -> Self {
        let n = ch.num_items;
        let mut neighbors = vec![usize::MAX; n * k];
        let mut buf: Vec<(i32, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            buf.clear();
            let row = &ch.interaction_values[i];
            for j in 0..n {
                if i != j && row[j] > 0 {
                    buf.push((row[j], j));
                }
            }
            // Partial sort descending by V_ij.
            let take = buf.len().min(k);
            if take > 0 {
                buf.select_nth_unstable_by(take.saturating_sub(1), |a, b| b.0.cmp(&a.0));
                for t in 0..take {
                    neighbors[i * k + t] = buf[t].1;
                }
            }
        }
        Self { k, neighbors }
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &[usize] {
        &self.neighbors[i * self.k..(i + 1) * self.k]
    }
}

struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n {
            contrib[i] = ch.values[i] as i32;
        }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
        }
    }

    #[inline(always)]
    fn slack(&self) -> u32 {
        self.ch.max_weight - self.total_weight
    }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for k in 0..n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_add(*row_ptr.add(k));
            }
        }
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for k in 0..n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_sub(*row_ptr.add(k));
            }
        }
        self.selected_bit[j] = false;
    }

    #[inline(always)]
    fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items)
            .filter(|&i| self.selected_bit[i])
            .collect()
    }

    fn clone_solution(&self) -> SolState {
        SolState {
            bits: self.selected_bit.clone(),
            contrib: self.contrib.clone(),
            value: self.total_value,
            weight: self.total_weight,
        }
    }

    fn restore_solution(&mut self, sol: &SolState) {
        self.selected_bit.clone_from(&sol.bits);
        self.contrib.clone_from(&sol.contrib);
        self.total_value = sol.value;
        self.total_weight = sol.weight;
    }
}

#[derive(Clone)]
struct SolState {
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
}

fn build_greedy_density(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    for i in 0..n {
        state.add_item(i);
    }
    while state.total_weight > cap {
        let mut worst = 0;
        let mut worst_s = i64::MAX;
        for i in 0..n {
            if state.selected_bit[i] {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w;
                if s < worst_s {
                    worst_s = s;
                    worst = i;
                }
            }
        }
        state.remove_item(worst);
    }
    for _ in 0..2 {
        let mut by_density: Vec<usize> = (0..n).collect();
        let contrib = &state.contrib;
        let weights = &state.ch.weights;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = weights[a] as i64;
            let wb = weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });
        let mut target = vec![false; n];
        let mut rem = cap;
        for &i in &by_density {
            if state.ch.weights[i] <= rem {
                target[i] = true;
                rem -= state.ch.weights[i];
            }
        }
        let mut to_rm = Vec::new();
        let mut to_add = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !target[i] {
                to_rm.push(i);
            }
            if !state.selected_bit[i] && target[i] {
                to_add.push(i);
            }
        }
        if to_rm.is_empty() && to_add.is_empty() {
            break;
        }
        for &r in &to_rm {
            state.remove_item(r);
        }
        for &a in &to_add {
            state.add_item(a);
        }
    }
}

fn build_greedy_value(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(state.ch.values[i]));
    for &i in &order {
        if state.total_weight + state.ch.weights[i] <= cap {
            state.add_item(i);
        }
    }
}

fn build_greedy_hub(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut hub_scores: Vec<(usize, i64)> = (0..n)
        .map(|i| {
            let s: i64 = state.ch.interaction_values[i]
                .iter()
                .map(|&v| v as i64)
                .sum();
            (i, s)
        })
        .collect();
    hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &hub_scores {
        if state.total_weight + state.ch.weights[i] <= cap {
            state.add_item(i);
        }
    }
}

fn build_greedy_synergy_weight(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut scores: Vec<(usize, i64)> = (0..n)
        .map(|i| {
            let avg_syn: i64 = if n > 1 {
                state.ch.interaction_values[i]
                    .iter()
                    .map(|&v| v as i64)
                    .sum::<i64>()
                    / (n as i64 - 1)
            } else {
                0
            };
            let w = (state.ch.weights[i] as i64).max(1);
            (i, (state.ch.values[i] as i64 + avg_syn) * 100 / w)
        })
        .collect();
    scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &scores {
        if state.total_weight + state.ch.weights[i] <= cap {
            state.add_item(i);
        }
    }
}

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }
        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = i64::MIN;
        let mut second_i: Option<usize> = None;
        let mut second_s: i64 = i64::MIN;
        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            if state.ch.weights[i] > slack {
                continue;
            }
            let c = state.contrib[i] as i64;
            if c <= 0 {
                continue;
            }
            let w = (state.ch.weights[i] as i64).max(1);
            let mut s = match mode {
                2 => c,
                3 => (c * 1000) / w + (state.ch.weights[i] as i64) * 3,
                _ => (c * 1000) / w,
            };
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }
            if s > best_s {
                second_s = best_s;
                second_i = best_i;
                best_s = s;
                best_i = Some(i);
            } else if s > second_s {
                second_s = s;
                second_i = Some(i);
            }
        }
        let pick = if mode >= 4 && second_i.is_some() {
            let m = if mode >= 5 { 1 } else { 3 };
            if (rng.next_u32() & m) == 0 {
                second_i
            } else {
                best_i
            }
        } else {
            best_i
        };
        if let Some(i) = pick {
            state.add_item(i);
        } else {
            break;
        }
    }
}

fn build_hub_pair_kth(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut pairs: Vec<(i32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if state.ch.weights[i] + state.ch.weights[j] <= cap {
                pairs.push((state.ch.interaction_values[i][j], i, j));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
    let mut used = Vec::new();
    let mut count = 0;
    for &(_, pi, pj) in &pairs {
        if used.contains(&pi) || used.contains(&pj) {
            continue;
        }
        if count == k {
            state.add_item(pi);
            state.add_item(pj);
            break;
        }
        used.push(pi);
        used.push(pj);
        count += 1;
    }
    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }
        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = 0;
        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            if state.ch.weights[i] > slack {
                continue;
            }
            let c = state.contrib[i] as i64;
            if c <= 0 {
                continue;
            }
            let w = (state.ch.weights[i] as i64).max(1);
            let s = (c * 1000) / w;
            if s > best_s {
                best_s = s;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            state.add_item(i);
        } else {
            break;
        }
    }
}

fn dp_refinement_hp(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let weights = &state.ch.weights;

    let mut by_density: Vec<usize> = (0..n).collect();
    by_density.sort_unstable_by(|&a, &b| {
        let na = contrib[a] as i64;
        let nb = contrib[b] as i64;
        let wa = weights[a] as i64;
        let wb = weights[b] as i64;
        (na * wb).cmp(&(nb * wa)).reverse()
    });
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    let mut rem = cap;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = weights[i];
        if w <= rem {
            rem -= w;
            idx_last_inserted = idx;
        } else if idx_first_rejected == n {
            idx_first_rejected = idx;
        }
    }

    let left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    let locked: Vec<usize> = by_density[..left].to_vec();
    let core: Vec<usize> = by_density[left..right].to_vec();

    let used_locked: u64 = locked.iter().map(|&i| weights[i] as u64).sum();
    let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
    let myk = core.len();
    if myk == 0 || rem_cap == 0 {
        return;
    }

    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for &it in &core {
        let wt = weights[it] as usize;
        total_core_weight += wt;
        if contrib[it] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap {
                all_pos_fit = false;
            }
        }
    }

    let target_sel = if all_pos_fit {
        let mut sel: Vec<usize> = locked.clone();
        for &it in &core {
            if contrib[it] > 0 {
                sel.push(it);
            }
        }
        sel.sort_unstable();
        sel
    } else {
        let myw = rem_cap.min(total_core_weight);
        let dp_size = myw + 1;
        let choose_size = myk * dp_size;
        if state.dp_cache.len() < dp_size {
            state.dp_cache.resize(dp_size, i64::MIN / 4);
        }
        if state.choose_cache.len() < choose_size {
            state.choose_cache.resize(choose_size, 0);
        }
        let init_val = i64::MIN / 4;
        for v in &mut state.dp_cache[..dp_size] {
            *v = init_val;
        }
        state.dp_cache[0] = 0;
        state.choose_cache[..choose_size].fill(0);

        let mut w_hi: usize = 0;
        for (t, &it) in core.iter().enumerate() {
            let wt = weights[it] as usize;
            if wt > myw {
                continue;
            }
            let val = contrib[it] as i64;
            let new_hi = (w_hi + wt).min(myw);
            for w in (wt..=new_hi).rev() {
                let cand = state.dp_cache[w - wt] + val;
                if cand > state.dp_cache[w] {
                    state.dp_cache[w] = cand;
                    state.choose_cache[t * dp_size + w] = 1;
                }
            }
            w_hi = new_hi;
        }

        let mut sel: Vec<usize> = locked.clone();
        let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = weights[it] as usize;
            if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                sel.push(it);
                w_star -= wt;
            }
        }
        sel.sort_unstable();
        sel
    };

    let mut to_rm = Vec::new();
    let mut to_add = Vec::new();
    let mut j = 0;
    let m = target_sel.len();
    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target {
            j += 1;
        }
        if state.selected_bit[i] && !in_target {
            to_rm.push(i);
        } else if in_target && !state.selected_bit[i] {
            to_add.push(i);
        }
    }
    for &r in &to_rm {
        state.remove_item(r);
    }
    for &a in &to_add {
        state.add_item(a);
    }
}

fn apply_best_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack == 0 {
        return false;
    }
    let n = state.ch.num_items;
    let mut best_i: Option<usize> = None;
    let mut best_d: i32 = 0;
    for i in 0..n {
        if state.selected_bit[i] {
            continue;
        }
        if state.ch.weights[i] > slack {
            continue;
        }
        let d = state.contrib[i];
        if d > best_d {
            best_d = d;
            best_i = Some(i);
        }
    }
    if let Some(i) = best_i {
        state.add_item(i);
        true
    } else {
        false
    }
}

// Build the frontier: union of TopNeighbors[s] for s in selected, restricted to unselected.
// Items in the frontier have high V_ij with at least one selected item, hence likely high contrib.
fn build_frontier(state: &State, selected: &[usize], tn: &TopNeighbors) -> Vec<usize> {
    let n = state.ch.num_items;
    let mut in_frontier = vec![false; n];
    let mut frontier: Vec<usize> = Vec::new();
    for &s in selected {
        for &j in tn.row(s) {
            if j == usize::MAX {
                break;
            }
            if state.selected_bit[j] || in_frontier[j] {
                continue;
            }
            in_frontier[j] = true;
            frontier.push(j);
        }
    }
    frontier
}

// Sparse 1-1 swap using TopNeighbors. For each selected r, candidate adds come from the
// frontier (union of TopNeighbors over selected items). This focuses search on items
// likely to integrate well with the rest of S.
fn apply_best_swap_1_1_tsn(state: &mut State, selected: &[usize], tn: &TopNeighbors) -> bool {
    let slack = state.slack();
    let frontier = build_frontier(state, selected, tn);
    if frontier.is_empty() {
        return false;
    }
    let mut best: Option<(usize, usize, i32)> = None;

    for &rm in selected {
        let w_rm = state.ch.weights[rm];
        let max_w = w_rm + slack;
        for &cand in &frontier {
            let wc = state.ch.weights[cand];
            if wc > max_w {
                continue;
            }
            let delta =
                state.contrib[cand] - state.contrib[rm] - state.ch.interaction_values[cand][rm];
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

// Sparse pair-add using TopNeighbors. Considers candidate pairs (a, b) where both are
// in the frontier (union of TopNeighbors of selected items).
fn apply_pair_add_tsn(state: &mut State, selected: &[usize], tn: &TopNeighbors) -> bool {
    let slack = state.slack();
    if slack < 2 {
        return false;
    }

    let frontier_full = build_frontier(state, selected, tn);
    let cands: Vec<usize> = frontier_full
        .into_iter()
        .filter(|&i| state.ch.weights[i] < slack)
        .collect();
    let m = cands.len();
    if m < 2 {
        return false;
    }

    let mut best_delta: i64 = 0;
    let mut best_pair: Option<(usize, usize)> = None;
    for ai in 0..m {
        let a = cands[ai];
        let wa = state.ch.weights[a];
        let ca = state.contrib[a] as i64;
        for bi in (ai + 1)..m {
            let b = cands[bi];
            if wa + state.ch.weights[b] > slack {
                continue;
            }
            let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
            if delta > best_delta {
                best_delta = delta;
                best_pair = Some((a, b));
            }
        }
    }
    if let Some((a, b)) = best_pair {
        state.add_item(a);
        state.add_item(b);
        true
    } else {
        false
    }
}

// TSN-based VND: 1-1 swap + pair-add + add, all sparse. Much faster per iteration than
// the full VND, allowing more ILS rounds in the same fuel.
fn local_search_vnd_tsn(state: &mut State, tn: &TopNeighbors) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..150 {
        if apply_best_add(state) {
            continue;
        }
        selected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] {
                selected_buf.push(i);
            }
        }
        if apply_best_swap_1_1_tsn(state, &selected_buf, tn) {
            continue;
        }
        if apply_pair_add_tsn(state, &selected_buf, tn) {
            continue;
        }
        break;
    }
}

// Lighter TSN VND: just 1-1 + add, fewer iterations. Used at very large n.
fn local_search_vnd_tsn_light(state: &mut State, tn: &TopNeighbors) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..80 {
        if apply_best_add(state) {
            continue;
        }
        selected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] {
                selected_buf.push(i);
            }
        }
        if apply_best_swap_1_1_tsn(state, &selected_buf, tn) {
            continue;
        }
        break;
    }
}

fn apply_best_swap_1_1(state: &mut State, selected: &[usize]) -> bool {
    let n = state.ch.num_items;
    let slack = state.slack();
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in selected {
        let w_rm = state.ch.weights[rm];
        let max_w = w_rm + slack;
        for cand in 0..n {
            if state.selected_bit[cand] {
                continue;
            }
            let wc = state.ch.weights[cand];
            if wc > max_w {
                continue;
            }
            let delta =
                state.contrib[cand] - state.contrib[rm] - state.ch.interaction_values[cand][rm];
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

fn apply_pair_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack < 2 {
        return false;
    }
    let n = state.ch.num_items;
    let unsel: Vec<usize> = (0..n)
        .filter(|&i| !state.selected_bit[i] && state.ch.weights[i] < slack)
        .collect();
    let m = unsel.len();
    if m < 2 {
        return false;
    }

    let mut best_delta: i64 = 0;
    let mut best_pair: Option<(usize, usize)> = None;
    for ai in 0..m {
        let a = unsel[ai];
        let wa = state.ch.weights[a];
        let ca = state.contrib[a] as i64;
        for bi in (ai + 1)..m {
            let b = unsel[bi];
            if wa + state.ch.weights[b] > slack {
                continue;
            }
            let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
            if delta > best_delta {
                best_delta = delta;
                best_pair = Some((a, b));
            }
        }
    }
    if let Some((a, b)) = best_pair {
        state.add_item(a);
        state.add_item(b);
        true
    } else {
        false
    }
}

fn apply_chain_move(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &rm in &sel {
        let w_rm = state.ch.weights[rm] as i64;
        let c_rm = state.contrib[rm] as i64;
        let budget = state.slack() as i64 + w_rm;

        for ui in 0..unsel.len() {
            let a1 = unsel[ui];
            let w_a1 = state.ch.weights[a1] as i64;
            if w_a1 >= budget {
                continue;
            }
            let c_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;

            for uj in (ui + 1)..unsel.len() {
                let a2 = unsel[uj];
                let w_a2 = state.ch.weights[a2] as i64;
                if w_a1 + w_a2 > budget {
                    continue;
                }

                let c_a2 = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                let syn = state.ch.interaction_values[a1][a2] as i64;
                let delta = c_a1 + c_a2 + syn - c_rm;

                if delta > best_delta {
                    let new_w = state.total_weight as i64 - w_rm + w_a1 + w_a2;
                    if new_w <= cap as i64 {
                        best_delta = delta;
                        best_move = Some((rm, a1, a2));
                    }
                }
            }
        }
    }

    if let Some((rm, a1, a2)) = best_move {
        state.remove_item(rm);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else {
        false
    }
}

fn apply_reverse_chain(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &add in &unsel {
        let w_add = state.ch.weights[add] as i64;
        let c_add = state.contrib[add] as i64;

        for si in 0..sel.len() {
            let r1 = sel[si];
            let w_r1 = state.ch.weights[r1] as i64;
            let c_r1 = state.contrib[r1] as i64;
            let c_add_r1 = state.ch.interaction_values[add][r1] as i64;

            for sj in (si + 1)..sel.len() {
                let r2 = sel[sj];
                let w_r2 = state.ch.weights[r2] as i64;
                let freed = w_r1 + w_r2;
                let new_w = state.total_weight as i64 - freed + w_add;
                if new_w > cap as i64 || new_w < 0 {
                    continue;
                }

                let c_r2 = state.contrib[r2] as i64;
                let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                let c_add_r2 = state.ch.interaction_values[add][r2] as i64;

                let lost = c_r1 + c_r2 - syn_r1_r2;
                let gained = c_add - c_add_r1 - c_add_r2;
                let delta = gained - lost;

                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some((r1, r2, add));
                }
            }
        }
    }

    if let Some((r1, r2, add)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(add);
        true
    } else {
        false
    }
}

// 1-3 swap: remove one selected r, add three unselected (a, b, c). Bounded:
//   - r picked from top-K worst-by-contrib selected
//   - (a, b, c) chosen from top-K best-by-contrib unselected with weight check
// Catches moves where multiple smaller items collectively replace one bulky/poor item.
fn apply_swap_1_3_bounded(state: &mut State, k: usize) -> bool {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut sel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    sel_ranked.sort_unstable_by_key(|&(_, c)| c);
    sel_ranked.truncate(k);

    let mut unsel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
    unsel_ranked.truncate(k);

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None;

    for &(rm, c_rm_i32) in &sel_ranked {
        let w_rm = state.ch.weights[rm] as i64;
        let c_rm = c_rm_i32 as i64;
        let budget = state.slack() as i64 + w_rm;
        if budget < 3 {
            continue;
        }

        for ai in 0..unsel_ranked.len() {
            let (a, _) = unsel_ranked[ai];
            let w_a = state.ch.weights[a] as i64;
            if w_a > budget {
                continue;
            }
            let c_a = state.contrib[a] as i64 - state.ch.interaction_values[a][rm] as i64;
            for bi in (ai + 1)..unsel_ranked.len() {
                let (b, _) = unsel_ranked[bi];
                let w_b = state.ch.weights[b] as i64;
                if w_a + w_b > budget {
                    continue;
                }
                let c_b = state.contrib[b] as i64 - state.ch.interaction_values[b][rm] as i64;
                let v_ab = state.ch.interaction_values[a][b] as i64;
                for ci in (bi + 1)..unsel_ranked.len() {
                    let (c, _) = unsel_ranked[ci];
                    let w_c = state.ch.weights[c] as i64;
                    if w_a + w_b + w_c > budget {
                        continue;
                    }
                    let c_c = state.contrib[c] as i64 - state.ch.interaction_values[c][rm] as i64;
                    let v_ac = state.ch.interaction_values[a][c] as i64;
                    let v_bc = state.ch.interaction_values[b][c] as i64;
                    let total_gain = c_a + c_b + c_c + v_ab + v_ac + v_bc;
                    let delta = total_gain - c_rm;
                    if delta > best_delta {
                        let new_w = state.total_weight as i64 - w_rm + w_a + w_b + w_c;
                        if new_w <= cap as i64 {
                            best_delta = delta;
                            best_move = Some((rm, a, b, c));
                        }
                    }
                }
            }
        }
    }
    if let Some((rm, a, b, c)) = best_move {
        state.remove_item(rm);
        state.add_item(a);
        state.add_item(b);
        state.add_item(c);
        true
    } else {
        false
    }
}

fn apply_swap_2_2_bounded(state: &mut State, k: usize) -> bool {
    let n = state.ch.num_items;
    let mut sel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    sel_ranked.sort_unstable_by_key(|&(_, c)| c);
    sel_ranked.truncate(k);

    let mut unsel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
    unsel_ranked.truncate(k);

    let cap = state.ch.max_weight;
    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None;

    for si in 0..sel_ranked.len() {
        let r1 = sel_ranked[si].0;
        let w_r1 = state.ch.weights[r1] as i64;
        let c_r1 = state.contrib[r1] as i64;
        for sj in (si + 1)..sel_ranked.len() {
            let r2 = sel_ranked[sj].0;
            let w_r2 = state.ch.weights[r2] as i64;
            let c_r2 = state.contrib[r2] as i64;
            let freed_weight = w_r1 + w_r2;
            let removed_syn = state.ch.interaction_values[r1][r2] as i64;
            let lost = c_r1 + c_r2 - removed_syn;
            let budget = state.slack() as i64 + freed_weight;

            for ui in 0..unsel_ranked.len() {
                let a1 = unsel_ranked[ui].0;
                let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 > budget {
                    continue;
                }
                let c_a1 = state.contrib[a1] as i64
                    - state.ch.interaction_values[a1][r1] as i64
                    - state.ch.interaction_values[a1][r2] as i64;
                for uj in (ui + 1)..unsel_ranked.len() {
                    let a2 = unsel_ranked[uj].0;
                    let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget {
                        continue;
                    }
                    let c_a2 = state.contrib[a2] as i64
                        - state.ch.interaction_values[a2][r1] as i64
                        - state.ch.interaction_values[a2][r2] as i64;
                    let added_syn = state.ch.interaction_values[a1][a2] as i64;
                    let delta = c_a1 + c_a2 + added_syn - lost;
                    if delta > best_delta {
                        let new_weight = state.total_weight as i64 - freed_weight + w_a1 + w_a2;
                        if new_weight <= cap as i64 {
                            best_delta = delta;
                            best_move = Some((r1, r2, a1, a2));
                        }
                    }
                }
            }
        }
    }
    if let Some((r1, r2, a1, a2)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else {
        false
    }
}

fn local_search_vnd_fast(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..80 {
        if apply_best_add(state) {
            continue;
        }
        selected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] {
                selected_buf.push(i);
            }
        }
        if apply_best_swap_1_1(state, &selected_buf) {
            continue;
        }
        break;
    }
}

fn local_search_vnd_medium(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..120 {
        if apply_best_add(state) {
            continue;
        }
        selected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] {
                selected_buf.push(i);
            }
        }
        if apply_best_swap_1_1(state, &selected_buf) {
            continue;
        }
        if apply_pair_add(state) {
            continue;
        }
        if apply_swap_2_2_bounded(state, k) {
            continue;
        }
        break;
    }
}

fn ils_vnd(state: &mut State, hp: &Hparams) {
    match hp.ils_vnd_level {
        0 => local_search_vnd_fast(state),
        1 => local_search_vnd_medium(state, hp.bounded_2_2_k),
        _ => local_search_vnd_heavy(state),
    }
}

fn local_search_vnd_heavy(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..300 {
        if apply_best_add(state) {
            continue;
        }
        selected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] {
                selected_buf.push(i);
            }
        }
        if apply_best_swap_1_1(state, &selected_buf) {
            continue;
        }
        if apply_pair_add(state) {
            continue;
        }
        if apply_swap_2_2_bounded(state, 25) {
            continue;
        }
        if apply_chain_move(state) {
            continue;
        }
        if apply_reverse_chain(state) {
            continue;
        }
        if apply_swap_1_3_bounded(state, 14) {
            continue;
        }
        break;
    }
}

fn simulated_annealing(state: &mut State, rng: &mut Rng, n_rounds: usize, n_iter: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    let mut sel: Vec<usize> = Vec::with_capacity(n);
    let mut unsel: Vec<usize> = Vec::with_capacity(n);
    let mut pos_in_sel = vec![0usize; n];
    let mut pos_in_unsel = vec![0usize; n];
    for i in 0..n {
        if state.selected_bit[i] {
            pos_in_sel[i] = sel.len();
            sel.push(i);
        } else {
            pos_in_unsel[i] = unsel.len();
            unsel.push(i);
        }
    }
    if sel.is_empty() || unsel.is_empty() {
        return;
    }

    let mut best_snap = state.clone_solution();

    let mut deltas: Vec<f64> = Vec::new();
    for _ in 0..100 {
        let rm = sel[rng.next_usize(sel.len())];
        let add = unsel[rng.next_usize(unsel.len())];
        let d = state.contrib[add] as f64
            - state.contrib[rm] as f64
            - state.ch.interaction_values[add][rm] as f64;
        if d < 0.0 {
            deltas.push(-d);
        }
    }
    if deltas.is_empty() {
        return;
    }
    deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p75 = deltas[deltas.len() * 3 / 4];
    let t0 = p75 / 0.693;
    if t0 < 1.0 {
        return;
    }

    let alpha = 0.95f64;
    let mut temp = t0;

    for _ in 0..n_rounds {
        for _ in 0..n_iter {
            if sel.is_empty() || unsel.is_empty() {
                continue;
            }

            let coin = rng.next_u32() % 10;
            if coin < 8 {
                let si = rng.next_usize(sel.len());
                let ui = rng.next_usize(unsel.len());
                let rm = sel[si];
                let add = unsel[ui];
                let w_new = state.total_weight - state.ch.weights[rm] + state.ch.weights[add];
                if w_new > cap {
                    continue;
                }
                let delta = state.contrib[add] as i64
                    - state.contrib[rm] as i64
                    - state.ch.interaction_values[add][rm] as i64;
                if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.replace_item(rm, add);
                    let last_sel = *sel.last().unwrap();
                    sel[si] = last_sel;
                    pos_in_sel[last_sel] = si;
                    sel.pop();
                    pos_in_sel[rm] = 0;

                    let last_unsel = *unsel.last().unwrap();
                    unsel[ui] = last_unsel;
                    pos_in_unsel[last_unsel] = ui;
                    unsel.pop();
                    pos_in_unsel[add] = 0;

                    pos_in_sel[add] = sel.len();
                    sel.push(add);
                    pos_in_unsel[rm] = unsel.len();
                    unsel.push(rm);
                }
            } else if coin == 8 {
                let slack = state.slack();
                if slack == 0 {
                    continue;
                }
                let ui = rng.next_usize(unsel.len());
                let add = unsel[ui];
                if state.ch.weights[add] > slack {
                    continue;
                }
                let delta = state.contrib[add] as i64;
                if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.add_item(add);
                    let last_unsel = *unsel.last().unwrap();
                    unsel[ui] = last_unsel;
                    pos_in_unsel[last_unsel] = ui;
                    unsel.pop();
                    pos_in_sel[add] = sel.len();
                    sel.push(add);
                }
            } else {
                let si = rng.next_usize(sel.len());
                let rm = sel[si];
                let delta = -(state.contrib[rm] as i64);
                if rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.remove_item(rm);
                    let last_sel = *sel.last().unwrap();
                    sel[si] = last_sel;
                    pos_in_sel[last_sel] = si;
                    sel.pop();
                    pos_in_unsel[rm] = unsel.len();
                    unsel.push(rm);
                }
            }

            if state.total_value > best_snap.value {
                best_snap = state.clone_solution();
            }
        }
        temp *= alpha;
    }

    if best_snap.value > state.total_value {
        state.restore_solution(&best_snap);
    }
}

fn crossover_frequency(population: &[SolState], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let pop_size = population.len();
    let mut freq = vec![0usize; n];
    for sol in population {
        for i in 0..n {
            if sol.bits[i] {
                freq[i] += 1;
            }
        }
    }
    let threshold = (pop_size * 3) / 4;
    let mut child_bits = vec![false; n];
    let mut child_weight: u32 = 0;
    let mut consensus: Vec<usize> = Vec::new();
    let mut exploratory: Vec<usize> = Vec::new();
    for i in 0..n {
        if freq[i] > threshold {
            consensus.push(i);
        } else if freq[i] > 0 {
            exploratory.push(i);
        }
    }
    for &i in &consensus {
        if child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
        }
    }
    for &i in &exploratory {
        if rng.next_u32() % 2 == 0 && child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
        }
    }
    child_bits
}

fn crossover_uniform(
    sol_a: &SolState,
    sol_b: &SolState,
    ch: &Challenge,
    rng: &mut Rng,
) -> Vec<bool> {
    let n = ch.num_items;
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    for i in 0..n {
        if sol_a.bits[i] && sol_b.bits[i] {
            if weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    for i in 0..n {
        if bits[i] {
            continue;
        }
        if sol_a.bits[i] || sol_b.bits[i] {
            if rng.next_u32() % 2 == 0 && weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    bits
}

// Path relinking: gradually transform `from` solution into `to` solution by greedy
// best-improvement swaps. Returns the best solution found along the path.
// At each step, picks the (rm, ad) pair from the symmetric difference that gives
// the best delta (subject to feasibility), then applies it. Also tries pure adds
// and pure removes when one side of the difference is exhausted.
fn path_relinking(from: &SolState, to: &SolState, ch: &Challenge) -> SolState {
    let n = ch.num_items;
    let cap = ch.max_weight;

    // Initialize working state from `from`
    let mut state = State::new_empty(ch);
    for i in 0..n {
        if from.bits[i] {
            state.add_item(i);
        }
    }

    // Symmetric difference
    let mut to_rm: Vec<usize> = (0..n).filter(|&i| from.bits[i] && !to.bits[i]).collect();
    let mut to_ad: Vec<usize> = (0..n).filter(|&i| !from.bits[i] && to.bits[i]).collect();

    let mut best = state.clone_solution();

    // Phase 1: balanced swaps along the symmetric diff
    while !to_rm.is_empty() && !to_ad.is_empty() {
        let mut best_delta = i64::MIN;
        let mut best_pair: Option<(usize, usize)> = None;
        for (ri, &rm) in to_rm.iter().enumerate() {
            let w_rm = ch.weights[rm] as i64;
            let c_rm = state.contrib[rm] as i64;
            for (ai, &ad) in to_ad.iter().enumerate() {
                let w_ad = ch.weights[ad] as i64;
                let new_w = state.total_weight as i64 - w_rm + w_ad;
                if new_w < 0 || new_w > cap as i64 {
                    continue;
                }
                let delta = state.contrib[ad] as i64 - ch.interaction_values[ad][rm] as i64 - c_rm;
                if delta > best_delta {
                    best_delta = delta;
                    best_pair = Some((ri, ai));
                }
            }
        }
        if let Some((ri, ai)) = best_pair {
            let rm = to_rm[ri];
            let ad = to_ad[ai];
            state.replace_item(rm, ad);
            to_rm.swap_remove(ri);
            to_ad.swap_remove(ai);
            if state.total_value > best.value {
                best = state.clone_solution();
            }
        } else {
            break;
        }
    }

    // Phase 2: pure adds (if to had extra items)
    while !to_ad.is_empty() {
        let mut best_delta: i64 = i64::MIN;
        let mut best_idx: Option<usize> = None;
        for (ai, &ad) in to_ad.iter().enumerate() {
            if state.total_weight + ch.weights[ad] > cap {
                continue;
            }
            let delta = state.contrib[ad] as i64;
            if delta > best_delta {
                best_delta = delta;
                best_idx = Some(ai);
            }
        }
        if let Some(ai) = best_idx {
            let ad = to_ad[ai];
            state.add_item(ad);
            to_ad.swap_remove(ai);
            if state.total_value > best.value {
                best = state.clone_solution();
            }
        } else {
            break;
        }
    }

    // Phase 3: pure removes (if from had extra items)
    while !to_rm.is_empty() {
        let mut best_delta: i64 = i64::MIN;
        let mut best_idx: Option<usize> = None;
        for (ri, &rm) in to_rm.iter().enumerate() {
            // Removing reduces value by contrib[rm] (which can be positive or negative)
            let delta = -(state.contrib[rm] as i64);
            if delta > best_delta {
                best_delta = delta;
                best_idx = Some(ri);
            }
        }
        if let Some(ri) = best_idx {
            let rm = to_rm[ri];
            state.remove_item(rm);
            to_rm.swap_remove(ri);
            if state.total_value > best.value {
                best = state.clone_solution();
            }
        } else {
            break;
        }
    }

    best
}

fn set_state_from_bits(state: &mut State, bits: &[bool]) {
    let n = state.ch.num_items;
    for i in (0..n).rev() {
        if state.selected_bit[i] {
            state.remove_item(i);
        }
    }
    for i in 0..n {
        if bits[i] {
            state.add_item(i);
        }
    }
}

fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let r = state.contrib[i] as f64 / (state.ch.weights[i] as f64).max(1.0);
        if state.selected_bit[i] {
            used_r.push((i, r));
        } else {
            unused_r.push((i, r));
        }
    }
    let ku = k.min(unused_r.len());
    if ku > 0 && ku < unused_r.len() {
        unused_r.select_nth_unstable_by(ku - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let ks = k.min(used_r.len());
    if ks > 0 && ks < used_r.len() {
        used_r.select_nth_unstable_by(ks - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    (
        unused_r[..ku].iter().map(|x| x.0).collect(),
        used_r[..ks].iter().map(|x| x.0).collect(),
    )
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
    for _ in 0..80 {
        let (mut best_unused, mut worst_used) = build_windows(state, window_k);
        let mut improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack {
                    continue;
                }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) {
                    ba = Some((c, d));
                }
            }
            if let Some((c, _)) = ba {
                state.add_item(c);
                improved = true;
                continue;
            }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w {
                        continue;
                    }
                    let d =
                        state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) {
                        bs = Some((c, rm, d));
                    }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                improved = true;
                continue;
            }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused
                .iter()
                .copied()
                .filter(|&i| state.ch.weights[i] < slack)
                .collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai + 1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack {
                            continue;
                        }
                        let d =
                            ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) {
                            bp = Some((a, b, d));
                        }
                    }
                }
                if let Some((a, b, _)) = bp {
                    state.add_item(a);
                    state.add_item(b);
                    improved = true;
                    continue;
                }
            }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let w_rm = state.ch.weights[rm] as i64;
                for &c in &best_unused {
                    let w_c = state.ch.weights[c] as i64;
                    if w_c >= w_rm {
                        continue;
                    }
                    let dw = (w_rm - w_c) as usize;
                    if dw == 0 || dw > 4 {
                        continue;
                    }
                    let d =
                        state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) {
                        bs = Some((c, rm, d));
                    }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                improved = true;
                continue;
            }
        }

        if state.slack() > 0 {
            let mut bs: Option<(usize, usize, f64)> = None;
            for &rm in &worst_used {
                let w_rm = state.ch.weights[rm] as i64;
                for &c in &best_unused {
                    let w_c = state.ch.weights[c] as i64;
                    if w_c <= w_rm {
                        continue;
                    }
                    let dw = w_c - w_rm;
                    if dw as usize > 4 || state.slack() < dw as u32 {
                        continue;
                    }
                    let d =
                        state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 {
                        let r = d as f64 / dw as f64;
                        if bs.map_or(true, |(_, _, br)| r > br) {
                            bs = Some((c, rm, r));
                        }
                    }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                improved = true;
                continue;
            }
        }

        if !improved {
            break;
        }
    }
}

// Deeper variant of windowed VND: larger window expansion + 2-2 swap inside window.
// Used as a heavier polish pass when we want quality over fuel.
fn local_search_vnd_windowed_deep(state: &mut State, window_k: usize) {
    let expanded_k = (window_k * 2).min(state.ch.num_items);
    for _ in 0..200 {
        let (mut best_unused, mut worst_used) = build_windows(state, expanded_k);
        let mut improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack {
                    continue;
                }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) {
                    ba = Some((c, d));
                }
            }
            if let Some((c, _)) = ba {
                state.add_item(c);
                improved = true;
                continue;
            }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w {
                        continue;
                    }
                    let d =
                        state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) {
                        bs = Some((c, rm, d));
                    }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                improved = true;
                continue;
            }
        }

        // 2-2 swap inside the window — bounded to top-K of each side to keep cost O(K^4) tractable.
        {
            const K22: usize = 12;
            let cap = state.ch.max_weight;
            // Build sorted top-K worst-by-contrib in worst_used (already small list)
            let ks = worst_used.len().min(K22);
            let ku = best_unused.len().min(K22);
            let mut best_delta: i64 = 0;
            let mut best_move: Option<(usize, usize, usize, usize)> = None;
            for si in 0..ks {
                let r1 = worst_used[si];
                let w_r1 = state.ch.weights[r1] as i64;
                let c_r1 = state.contrib[r1] as i64;
                for sj in (si + 1)..ks {
                    let r2 = worst_used[sj];
                    let w_r2 = state.ch.weights[r2] as i64;
                    let c_r2 = state.contrib[r2] as i64;
                    let removed_syn = state.ch.interaction_values[r1][r2] as i64;
                    let lost = c_r1 + c_r2 - removed_syn;
                    let budget = state.slack() as i64 + w_r1 + w_r2;
                    for ui in 0..ku {
                        let a1 = best_unused[ui];
                        let w_a1 = state.ch.weights[a1] as i64;
                        if w_a1 > budget {
                            continue;
                        }
                        let c_a1 = state.contrib[a1] as i64
                            - state.ch.interaction_values[a1][r1] as i64
                            - state.ch.interaction_values[a1][r2] as i64;
                        for uj in (ui + 1)..ku {
                            let a2 = best_unused[uj];
                            let w_a2 = state.ch.weights[a2] as i64;
                            if w_a1 + w_a2 > budget {
                                continue;
                            }
                            let c_a2 = state.contrib[a2] as i64
                                - state.ch.interaction_values[a2][r1] as i64
                                - state.ch.interaction_values[a2][r2] as i64;
                            let added_syn = state.ch.interaction_values[a1][a2] as i64;
                            let delta = c_a1 + c_a2 + added_syn - lost;
                            if delta > best_delta {
                                let new_w = state.total_weight as i64 - w_r1 - w_r2 + w_a1 + w_a2;
                                if new_w <= cap as i64 {
                                    best_delta = delta;
                                    best_move = Some((r1, r2, a1, a2));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((r1, r2, a1, a2)) = best_move {
                state.remove_item(r1);
                state.remove_item(r2);
                state.add_item(a1);
                state.add_item(a2);
                improved = true;
                continue;
            }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused
                .iter()
                .copied()
                .filter(|&i| state.ch.weights[i] < slack)
                .collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai + 1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack {
                            continue;
                        }
                        let d =
                            ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) {
                            bp = Some((a, b, d));
                        }
                    }
                }
                if let Some((a, b, _)) = bp {
                    state.add_item(a);
                    state.add_item(b);
                    improved = true;
                    continue;
                }
            }
        }

        if !improved {
            break;
        }
    }
}

// Cluster bomb perturbation: pick a random selected item as "epicenter", remove it
// plus the top-K selected items most synergistically connected to it. This breaks up
// correlated subsets (whole sub-cliques in S) instead of removing scattered low-density
// items. Very effective at escaping basins where all near-neighbors of a choice are
// also locally optimal.
fn cluster_bomb_perturb(state: &mut State, k: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    let selected: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    if selected.is_empty() {
        return;
    }
    let epicenter = selected[rng.next_usize(selected.len())];

    // Score remaining selected by interaction strength with epicenter (higher = more connected).
    let mut neighbors: Vec<(i32, usize)> = selected
        .iter()
        .filter(|&&i| i != epicenter)
        .map(|&i| (state.ch.interaction_values[epicenter][i], i))
        .collect();
    // Sort descending by interaction value; top-K are the cluster around the epicenter.
    neighbors.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let take = (k.saturating_sub(1)).min(neighbors.len());

    state.remove_item(epicenter);
    for &(_, i) in neighbors.iter().take(take) {
        state.remove_item(i);
    }
}

fn perturb_by_strategy(
    state: &mut State,
    strength: usize,
    stall_count: usize,
    strategy: usize,
    rng: &mut Rng,
    hp: &Hparams,
) {
    let selected = state.selected_items();
    if selected.is_empty() {
        return;
    }
    let mut removal_candidates: Vec<(usize, i64)>;

    match strategy {
        0 => {
            removal_candidates = selected
                .iter()
                .map(|&i| (i, state.contrib[i] as i64))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        }
        1 => {
            removal_candidates = selected
                .iter()
                .map(|&i| (i, -(state.ch.weights[i] as i64)))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        }
        2 => {
            removal_candidates = selected
                .iter()
                .map(|&i| {
                    let syn = state.contrib[i] as i64 - state.ch.values[i] as i64;
                    (i, syn)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        3 => {
            removal_candidates = selected
                .iter()
                .map(|&i| {
                    let w = (state.ch.weights[i] as i64).max(1);
                    (i, (state.contrib[i] as i64 * 1000) / w)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        4 => {
            removal_candidates = selected
                .iter()
                .map(|&i| {
                    let w = (state.ch.weights[i] as i64).max(1);
                    let density = (state.contrib[i] as i64 * 100) / w;
                    (i, state.ch.weights[i] as i64 - density)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        5 => {
            removal_candidates = selected
                .iter()
                .map(|&i| {
                    let w = (state.ch.weights[i] as i64).max(1);
                    (i, (state.contrib[i] as i64 * 10000) / (w * w))
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        6 => {
            removal_candidates = selected
                .iter()
                .map(|&i| (i, rng.next_u32() as i64))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        _ => {
            removal_candidates = selected
                .iter()
                .map(|&i| (i, -(state.contrib[i] as i64)))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }

    let base_remove = (selected.len() / hp.perturb_base_frac).max(2);
    let adaptive_mult = 1 + (stall_count / 2);
    let n_remove = (base_remove * adaptive_mult)
        .min(strength)
        .min(selected.len() * 2 / hp.perturb_max_frac);
    for j in 0..n_remove {
        if j < removal_candidates.len() {
            state.remove_item(removal_candidates[j].0);
        }
    }
}

fn greedy_reconstruct(state: &mut State, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

    match strategy % 4 {
        0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
        1 => candidates.sort_unstable_by(|&a, &b| {
            state.ch.weights[a]
                .cmp(&state.ch.weights[b])
                .then(state.contrib[b].cmp(&state.contrib[a]))
        }),
        2 => candidates.sort_unstable_by_key(|&i| {
            let syn: i64 = state.ch.interaction_values[i]
                .iter()
                .take(n.min(100))
                .map(|&v| v as i64)
                .sum();
            -(syn + state.contrib[i] as i64 / 10)
        }),
        _ => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            -((state.contrib[i] as i64 * 100) / w)
        }),
    }

    for &i in &candidates {
        if state.total_weight + state.ch.weights[i] <= cap {
            state.add_item(i);
        }
    }
}

struct Hparams {
    n_random_starts: usize,
    n_crossover_gen: usize,
    sa_rounds: usize,
    sa_iter: usize,
    n_sa_members: usize,
    ils_rounds: usize,
    ils_restart_interval: usize,
    perturb_base_frac: usize,
    perturb_max_frac: usize,
    ils_vnd_level: usize,
    bounded_2_2_k: usize,
    n_full_restarts: usize,
    use_hub_pair: bool,
    use_heavy_polish: bool,
    use_path_relinking: bool,
    use_tsn_in_ils: bool,
    final_sa_rounds: usize,
    final_sa_iter: usize,
    window_k: usize,
    core_half_dp: usize,
}

impl Hparams {
    // Defaults tuned for production fuel (500B-5T). Production winners use:
    //   - n_full_restarts: 3-12 (many short runs > one long run)
    //   - ils_rounds: 50-340 (much lower than v1 default 250)
    //   - window_k: 40-60 small-n, 52 large-n (much narrower than n)
    //   - core_half_dp: 10-32 (much smaller than v1 default 60)
    //   - n_crossover_gen: 15-26
    //   - sa_rounds: 40, sa_iter: 440, n_sa_members: 8
    //   - perturb_base_frac: 100, perturb_max_frac: 28 (very lax perturbation)
    //   - ils_restart_interval: 1 (restart every stalled round)
    fn for_size(n: usize, budget: u32) -> Self {
        if n <= 1200 {
            if budget <= 5 {
                Self {
                    n_random_starts: 4,
                    n_crossover_gen: 15,
                    sa_rounds: 40,
                    sa_iter: 440,
                    n_sa_members: 8,
                    ils_rounds: 50,
                    ils_restart_interval: 1,
                    perturb_base_frac: 100,
                    perturb_max_frac: 28,
                    ils_vnd_level: 0,
                    bounded_2_2_k: 14,
                    n_full_restarts: 12,
                    use_hub_pair: true,
                    use_heavy_polish: true,
                    use_path_relinking: false,
                    use_tsn_in_ils: false,
                    final_sa_rounds: 0,
                    final_sa_iter: 0,
                    window_k: 40,
                    core_half_dp: 10,
                }
            } else if budget <= 10 {
                Self {
                    n_random_starts: 4,
                    n_crossover_gen: 18,
                    sa_rounds: 40,
                    sa_iter: 440,
                    n_sa_members: 8,
                    ils_rounds: 50,
                    ils_restart_interval: 1,
                    perturb_base_frac: 100,
                    perturb_max_frac: 28,
                    ils_vnd_level: 0,
                    bounded_2_2_k: 14,
                    n_full_restarts: 10,
                    use_hub_pair: true,
                    use_heavy_polish: true,
                    use_path_relinking: false,
                    use_tsn_in_ils: false,
                    final_sa_rounds: 0,
                    final_sa_iter: 0,
                    window_k: 50,
                    core_half_dp: 14,
                }
            } else {
                // budget=25: fewer restarts, more ILS per restart
                Self {
                    n_random_starts: 4,
                    n_crossover_gen: 26,
                    sa_rounds: 24,
                    sa_iter: 440,
                    n_sa_members: 8,
                    ils_rounds: 50,
                    ils_restart_interval: 1,
                    perturb_base_frac: 100,
                    perturb_max_frac: 28,
                    ils_vnd_level: 0,
                    bounded_2_2_k: 0,
                    n_full_restarts: 12,
                    use_hub_pair: true,
                    use_heavy_polish: false,
                    use_path_relinking: false,
                    use_tsn_in_ils: false,
                    final_sa_rounds: 0,
                    final_sa_iter: 0,
                    window_k: 60,
                    core_half_dp: 14,
                }
            }
        } else {
            // Large n (e.g., 5000): production profile with moderate restarts + more ILS
            Self {
                n_random_starts: 3,
                n_crossover_gen: 8,
                sa_rounds: 0,
                sa_iter: 0,
                n_sa_members: 0,
                ils_rounds: 340,
                ils_restart_interval: 0,
                perturb_base_frac: 6,
                perturb_max_frac: 5,
                ils_vnd_level: 0,
                bounded_2_2_k: 0,
                n_full_restarts: 3,
                use_hub_pair: false,
                use_heavy_polish: false,
                use_path_relinking: false,
                use_tsn_in_ils: false,
                final_sa_rounds: 0,
                final_sa_iter: 0,
                window_k: 52,
                core_half_dp: 32,
            }
        }
    }

    fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
        let mut p = Self::for_size(n, budget);
        if let Some(m) = h {
            if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64()) {
                p.n_random_starts = v as usize;
            }
            if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64()) {
                p.n_crossover_gen = v as usize;
            }
            if let Some(v) = m.get("sa_rounds").and_then(|v| v.as_u64()) {
                p.sa_rounds = v as usize;
            }
            if let Some(v) = m.get("sa_iter").and_then(|v| v.as_u64()) {
                p.sa_iter = v as usize;
            }
            if let Some(v) = m.get("n_sa_members").and_then(|v| v.as_u64()) {
                p.n_sa_members = v as usize;
            }
            if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) {
                p.ils_rounds = v as usize;
            }
            if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) {
                p.ils_restart_interval = v as usize;
            }
            if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()) {
                p.perturb_base_frac = v as usize;
            }
            if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) {
                p.perturb_max_frac = v as usize;
            }
            if let Some(v) = m.get("ils_vnd_level").and_then(|v| v.as_u64()) {
                p.ils_vnd_level = v as usize;
            }
            if let Some(v) = m.get("bounded_2_2_k").and_then(|v| v.as_u64()) {
                p.bounded_2_2_k = v as usize;
            }
            if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) {
                p.n_full_restarts = v as usize;
            }
            if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) {
                p.window_k = v as usize;
            }
            if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) {
                p.core_half_dp = v as usize;
            }
        }
        p
    }
}

fn vnd_dispatch(state: &mut State, hp: &Hparams) {
    if hp.window_k < state.ch.num_items {
        local_search_vnd_windowed(state, hp.window_k);
    } else {
        ils_vnd(state, hp);
    }
}

fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..rng_offset * 100 {
        rng.next_u32();
    }
    let ch = hp.core_half_dp;

    // Build TopNeighbors once per instance — sparse iteration accelerates LS later.
    let tn_k = if n >= 4000 {
        80
    } else if n >= 2000 {
        96
    } else {
        128
    };
    let tn = TopNeighbors::new(challenge, tn_k);

    let mut population: Vec<SolState> = Vec::with_capacity(16);

    let n_greedy = if n <= 1200 { 4 } else { 3 };
    for variant in 0..n_greedy {
        let mut st = State::new_empty(challenge);
        match variant {
            0 => build_greedy_density(&mut st),
            1 => build_greedy_value(&mut st),
            2 => build_greedy_synergy_weight(&mut st),
            _ => build_greedy_hub(&mut st),
        }
        dp_refinement_hp(&mut st, ch);
        if hp.use_heavy_polish {
            local_search_vnd_heavy(&mut st);
        } else {
            vnd_dispatch(&mut st, hp);
        }
        population.push(st.clone_solution());
    }

    for mode in 4..(4 + hp.n_random_starts) {
        let mut st = State::new_empty(challenge);
        let m = if mode < 6 { mode } else { mode - 2 };
        construct_forward_incremental(&mut st, m, &mut rng);
        dp_refinement_hp(&mut st, ch);
        vnd_dispatch(&mut st, hp);
        population.push(st.clone_solution());
    }

    if hp.use_hub_pair {
        for k in 0..4 {
            let mut st = State::new_empty(challenge);
            build_hub_pair_kth(&mut st, k);
            dp_refinement_hp(&mut st, ch);
            vnd_dispatch(&mut st, hp);
            population.push(st.clone_solution());
        }
    }

    population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
    population.truncate(8);

    let mut state = State::new_empty(challenge);
    for gen in 0..hp.n_crossover_gen {
        let child_bits = crossover_frequency(&population, challenge, &mut rng);
        set_state_from_bits(&mut state, &child_bits);
        dp_refinement_hp(&mut state, ch);
        vnd_dispatch(&mut state, hp);
        population.push(state.clone_solution());

        if population.len() >= 2 {
            let a = gen % population.len().min(4);
            let b = (gen + 1) % population.len().min(4);
            if a != b {
                let child_bits =
                    crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                set_state_from_bits(&mut state, &child_bits);
                dp_refinement_hp(&mut state, ch);
                vnd_dispatch(&mut state, hp);
                population.push(state.clone_solution());
            }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
    }

    // Path relinking: combine top-2 solutions in both directions, polish each result.
    // Adds 2 new candidates to population.
    if hp.use_path_relinking && population.len() >= 2 {
        let n_pr_pairs: usize = if n <= 1200 { 3 } else { 1 };
        for pair_idx in 0..n_pr_pairs {
            let a_idx = pair_idx % population.len();
            let b_idx = (pair_idx + 1) % population.len();
            if a_idx == b_idx {
                continue;
            }
            let pr_result = path_relinking(&population[a_idx], &population[b_idx], challenge);
            state.restore_solution(&pr_result);
            vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());

            let pr_result_rev = path_relinking(&population[b_idx], &population[a_idx], challenge);
            state.restore_solution(&pr_result_rev);
            vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
    }

    if hp.sa_rounds > 0 {
        for pi in 0..hp.n_sa_members.min(population.len()) {
            state.restore_solution(&population[pi]);
            simulated_annealing(&mut state, &mut rng, hp.sa_rounds, hp.sa_iter);
            vnd_dispatch(&mut state, hp);
            let sol = state.clone_solution();
            if sol.value > population[pi].value {
                population.push(sol);
            }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
    }

    state.restore_solution(&population[0]);
    let mut best_val = state.total_value;
    let mut best_sel: Vec<usize> = state.selected_items();

    let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
    let zobrist_table: Vec<u64> = (0..n)
        .map(|i| {
            let mut h: u64 = 0x517CC1B727220A95;
            h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
            h
        })
        .collect();
    let compute_hash = |bits: &[bool]| -> u64 {
        let mut h: u64 = 0;
        for i in 0..n {
            if bits[i] {
                h ^= zobrist_table[i];
            }
        }
        h
    };
    tabu_hashes.push(compute_hash(&state.selected_bit));

    let mut stall_count = 0;
    for round in 0..hp.ils_rounds {
        let snap = state.clone_solution();

        dp_refinement_hp(&mut state, ch);
        vnd_dispatch(&mut state, hp);
        if hp.use_tsn_in_ils {
            local_search_vnd_tsn(&mut state, &tn);
        }

        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel = state.selected_items();
            stall_count = 0;
        }

        if state.total_value <= snap.value {
            state.restore_solution(&snap);
            stall_count += 1;

            if hp.ils_restart_interval > 0
                && stall_count > 0
                && stall_count % hp.ils_restart_interval == 0
            {
                let pi = (stall_count / hp.ils_restart_interval) % population.len();
                state.restore_solution(&population[pi]);
            }

            let strategy = round % 9;
            let strength = 5 + round / 4;
            // Strategy 8 = cluster bomb (NEW): targeted disruption of synergistic cluster
            if strategy == 8 {
                let bomb_size = (strength).max(4).min(state.selected_items().len() / 3 + 2);
                cluster_bomb_perturb(&mut state, bomb_size, &mut rng);
            } else {
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp);
            }
            greedy_reconstruct(&mut state, strategy);
            vnd_dispatch(&mut state, hp);
            if hp.use_tsn_in_ils {
                local_search_vnd_tsn(&mut state, &tn);
            }

            let h = compute_hash(&state.selected_bit);
            if tabu_hashes.contains(&h) {
                let extra_strength = 10 + round / 3;
                if round % 2 == 0 {
                    perturb_by_strategy(
                        &mut state,
                        extra_strength,
                        stall_count + 3,
                        6,
                        &mut rng,
                        hp,
                    );
                } else {
                    let bomb_size = extra_strength.max(6);
                    cluster_bomb_perturb(&mut state, bomb_size, &mut rng);
                }
                greedy_reconstruct(&mut state, 0);
                vnd_dispatch(&mut state, hp);
            }
            let h2 = compute_hash(&state.selected_bit);
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h2);
            } else {
                tabu_hashes[round % 128] = h2;
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                stall_count = 0;
            }
        } else {
            stall_count = 0;
            let h = compute_hash(&state.selected_bit);
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h);
            }
        }
    }

    let mut final_state = State::new_empty(challenge);
    for &i in &best_sel {
        final_state.add_item(i);
    }

    // Optional: Final SA bout to escape local opts the ILS phase may have settled into.
    if hp.final_sa_rounds > 0 {
        let v_before_sa = final_state.total_value;
        simulated_annealing(
            &mut final_state,
            &mut rng,
            hp.final_sa_rounds,
            hp.final_sa_iter,
        );
        if final_state.total_value > best_val {
            best_val = final_state.total_value;
            best_sel = final_state.selected_items();
        } else if final_state.total_value < v_before_sa {
            // SA may have left state worse — restore best
            final_state = State::new_empty(challenge);
            for &i in &best_sel {
                final_state.add_item(i);
            }
        }
    }

    if hp.use_heavy_polish {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_heavy(&mut final_state);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before {
                break;
            }
        }
        // Final deep windowed pass (the "deep" addition with 2-2 swap inside window).
        if final_state.ch.num_items <= 1500 {
            let v_before = final_state.total_value;
            local_search_vnd_windowed_deep(&mut final_state, hp.window_k.min(160));
            if final_state.total_value > best_val {
                best_val = final_state.total_value;
                best_sel = final_state.selected_items();
            } else if final_state.total_value < v_before {
                final_state = State::new_empty(challenge);
                for &i in &best_sel {
                    final_state.add_item(i);
                }
            }
        }
    } else {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before {
                break;
            }
        }
    }

    if final_state.total_value > best_val {
        Solution {
            items: final_state.selected_items(),
        }
    } else {
        Solution { items: best_sel }
    }
}

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let n = challenge.num_items;
        let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
        let budget_pct = if sum_w > 0 {
            ((challenge.max_weight as u64) * 100 / sum_w) as u32
        } else {
            10
        };
        let hp = Hparams::from_map(hyperparameters, n, budget_pct);
        let n_restarts = hp.n_full_restarts.max(1);
        let mut best_sol: Option<Solution> = None;
        let mut best_quality: i64 = i64::MIN;
        for restart in 0..n_restarts {
            let sol = run_one_instance(challenge, &hp, restart);
            let mut val: i64 = 0;
            for &i in &sol.items {
                val += challenge.values[i] as i64;
                for &j in &sol.items {
                    if j > i {
                        val += challenge.interaction_values[i][j] as i64;
                    }
                }
            }
            if val > best_quality {
                best_quality = val;
                best_sol = Some(sol);
            }
        }
        Ok(best_sol)
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
    println!("knap_apex: memetic+SA+ILS with path relinking and 1-3 swap.");
    println!("Built on the knap_quality_opt architecture; bigger DP, more SA, path relinking.");
}
