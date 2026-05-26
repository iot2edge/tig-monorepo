use super::builder::Builder;
use super::config::Config;
use super::gene_pool::{GenePool, Metric};
use super::instance::Instance;
use super::operators::LocalOps;
use super::route_eval::RouteEval;
use super::solution::Individual;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;
use tig_challenges::vehicle_routing::*;

// Pareto entry for split DP: (k=#routes, cost, pred_j, pred_idx_in_pareto[pred_j])
type ParetoEntry = (u32, i64, u32, u32);

pub struct Evolution<'a> {
    pub data: &'a Instance,
    pub params: Config,
    pub population: GenePool<'a>,
    split_dp: Vec<i64>,
    split_pred: Vec<usize>,
    // Pareto split state — reused across calls to avoid reallocation.
    pareto: Vec<Vec<ParetoEntry>>,
}

impl<'a> Evolution<'a> {
    pub fn new(data: &'a Instance, params: Config) -> Self {
        let population = GenePool::new(data);
        Self {
            data,
            params,
            population,
            split_dp: Vec::new(),
            split_pred: Vec::new(),
            pareto: Vec::new(),
        }
    }

    fn repair_and_maybe_add(&mut self, ls: &mut LocalOps, rng: &mut SmallRng) {
        let mut repaired_routes1: Vec<Vec<usize>> = Vec::new();
        ls.runls(&mut repaired_routes1, rng, &self.params, true, 100);
        let repaired1 = Individual::new_from_routes(self.data, &self.params, repaired_routes1);

        if repaired1.load_excess == 0 && repaired1.tw_violation == 0 {
            self.population.add(repaired1, &self.params);
        }
    }

    pub fn generate_initial_individual(
        &mut self,
        rng: &mut SmallRng,
        ls: &mut LocalOps,
        randomize: bool,
    ) {
        let mut routes: Vec<Vec<usize>> = Builder::build_routes(self.data, rng, randomize);
        ls.runls(&mut routes, rng, &self.params, false, 0);
        let ind = Individual::new_from_routes(self.data, &self.params, routes);
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;

        self.population.add(ind, &self.params);
        self.population
            .record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalOps) {
        let p1 = self.population.get_binary_tournament(rng);
        let mut p2 = self.population.get_binary_tournament(rng);
        while std::ptr::eq(p1, p2) {
            p2 = self.population.get_binary_tournament(rng);
        }
        let t2 = self.extract_giant_tour(&p2.routes);
        let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 };
        let target_routes =
            (p1.nb_routes + extra).clamp(self.data.lb_vehicles, self.data.nb_vehicles);

        let mut child_tour = self.crossover_rbx(p1, &t2, rng);
        self.mutate_tour(&mut child_tour, rng);

        let mut child_routes = self.split(&child_tour, target_routes);
        ls.runls(&mut child_routes, rng, &self.params, false, 0);
        let child = Individual::new_from_routes(self.data, &self.params, child_routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        self.population.add(child, &self.params);
        self.population
            .record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Option<(Vec<Vec<usize>>, i32)> {
        if let Some(save) = save_solution {
            let dummy_routes: Vec<Vec<usize>> =
                (1..self.data.nb_nodes).map(|i| vec![0, i, 0]).collect();
            let _ = save(&Solution {
                routes: dummy_routes,
            });
        }

        let mut ls = LocalOps::new(self.data, self.params);

        let diversity_boost = if self.data.nb_nodes < 1000 { 3 } else { 1 };
        for it in 0..(self.params.mu_start + diversity_boost) {
            self.generate_initial_individual(rng, &mut ls, it > 0);
        }

        let mut best_metric: Metric = self.population.best_metric();
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {
            self.generate_crossover_individual(rng, &mut ls);

            if it_total % self.params.nb_it_traces == 0 {
                self.population.print_trace(
                    it_total,
                    it_noimprov,
                    t0.elapsed().as_secs_f64(),
                    &self.params,
                );
            }

            let cur = self.population.best_metric();
            if cur.better_than(best_metric) {
                best_metric = cur;
                it_noimprov = 0;

                if let Some(best) = self.population.best_feasible() {
                    if let Some(save) = save_solution {
                        let _ = save(&Solution {
                            routes: best.routes,
                        });
                    }
                }
            } else {
                it_noimprov += 1;
            }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let mut best_routes = best.routes.clone();
            ls.runls(&mut best_routes, rng, &self.params, false, 0);
            let best_after = Individual::new_from_routes(self.data, &self.params, best_routes);
            let chosen = if best_after.tw_violation == 0
                && best_after.load_excess == 0
                && best_after.distance < best.distance
            {
                best_after
            } else {
                best
            };
            if let Some(save) = save_solution {
                let _ = save(&Solution {
                    routes: chosen.routes.clone(),
                });
            }
            Some((chosen.routes, chosen.cost as i32))
        } else {
            None
        }
    }

    fn mutate_tour(&mut self, tour: &mut Vec<usize>, rng: &mut SmallRng) {
        let _ = rng;

        let n = tour.len();
        if n < 3 {
            return;
        }

        let dist = |a: usize, b: usize| -> i64 {
            let (xa, ya) = self.data.node_positions[a];
            let (xb, yb) = self.data.node_positions[b];
            let dx = (xa as f64) - (xb as f64);
            let dy = (ya as f64) - (yb as f64);
            ((dx * dx + dy * dy).sqrt()) as i64
        };

        let mut best_i = 0usize;
        let mut best_remove_gain = i64::MIN;
        for i in 0..n {
            let a = if i == 0 { 0 } else { tour[i - 1] };
            let u = tour[i];
            let b = if i + 1 == n { 0 } else { tour[i + 1] };
            let gain = dist(a, u) + dist(u, b) - dist(a, b);
            if gain > best_remove_gain {
                best_remove_gain = gain;
                best_i = i;
            }
        }

        let i = best_i;
        let u = tour[i];
        let a = if i == 0 { 0 } else { tour[i - 1] };
        let b = if i + 1 == n { 0 } else { tour[i + 1] };

        let len = n - 1;
        let map_after = |k: usize| -> usize {
            if k < i {
                tour[k]
            } else {
                tour[k + 1]
            }
        };

        let mut best_ins = i;
        let mut best_delta = 0i64;

        for ins in 0..=len {
            if ins == i {
                continue;
            }
            let c = if ins == 0 { 0 } else { map_after(ins - 1) };
            let d = if ins == len { 0 } else { map_after(ins) };

            let delta =
                -(dist(a, u) + dist(u, b)) + dist(a, b) - dist(c, d) + dist(c, u) + dist(u, d);
            if delta < best_delta {
                best_delta = delta;
                best_ins = ins;
            }
        }

        if best_delta < 0 {
            let node = tour.remove(i);
            if best_ins <= tour.len() {
                tour.insert(best_ins, node);
            } else {
                tour.push(node);
            }
        }
    }

    pub fn split(&mut self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        // Pareto-dominance Bellman split (Vidal 2012): replaces the O(n^2 * k)
        // table dp[kk][j] with a per-prefix Pareto frontier of (k, cost) pairs.
        // Same minimum-cost partition; complexity drops to O(n^2 * |Pareto|)
        // where |Pareto| is typically small (just a handful of route-count
        // alternatives are non-dominated for any given prefix).
        let n = giant.len();
        if n == 0 {
            return Vec::new();
        }

        let k_cap = target_routes.max(1) as u32;
        let inf = i64::MAX / 4;

        // Reset pareto front buffers.
        if self.pareto.len() < n + 1 {
            self.pareto.resize(n + 1, Vec::new());
        }
        for p in self.pareto.iter_mut().take(n + 1) {
            p.clear();
        }
        // Base: zero routes covering empty prefix at cost 0.
        self.pareto[0].push((0u32, 0i64, 0u32, 0u32));

        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let depot = RouteEval::singleton(self.data, 0);

        for i in 0..n {
            if self.pareto[i].is_empty() {
                continue;
            }
            // Build the running route segment depot -> giant[i] -> ... -> giant[j-1]
            let mut acc = RouteEval::join2(
                self.data,
                &depot,
                &RouteEval::singleton(self.data, giant[i]),
            );
            for j in (i + 1)..=n {
                let route_cost = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                // Snapshot the Pareto front at i (avoid borrow conflict during
                // simultaneous read of pareto[i] and write of pareto[j]).
                let src_len = self.pareto[i].len();
                for src_idx in 0..src_len {
                    let (k_i, c_i, _, _) = self.pareto[i][src_idx];
                    let new_k = k_i + 1;
                    if new_k > k_cap {
                        continue;
                    }
                    let new_c = c_i.saturating_add(route_cost);
                    if new_c >= inf {
                        continue;
                    }
                    Self::pareto_insert(
                        &mut self.pareto[j],
                        (new_k, new_c, i as u32, src_idx as u32),
                    );
                }
                if acc.load > cap_limit {
                    break;
                }
                if j < n {
                    let next = RouteEval::singleton(self.data, giant[j]);
                    acc = RouteEval::join2(self.data, &acc, &next);
                }
            }
        }

        // Pareto front size diagnostics — disable after measurement.
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static SUM_FRONT: AtomicU64 = AtomicU64::new(0);
            static MAX_FRONT: AtomicU64 = AtomicU64::new(0);
            static N_J: AtomicU64 = AtomicU64::new(0);
            static N_CALLS: AtomicU64 = AtomicU64::new(0);
            let mut local_sum = 0u64;
            let mut local_max = 0u64;
            for p in self.pareto.iter().take(n + 1) {
                local_sum += p.len() as u64;
                if (p.len() as u64) > local_max {
                    local_max = p.len() as u64;
                }
            }
            SUM_FRONT.fetch_add(local_sum, Ordering::Relaxed);
            let prev_max = MAX_FRONT.load(Ordering::Relaxed);
            if local_max > prev_max {
                MAX_FRONT.store(local_max, Ordering::Relaxed);
            }
            N_J.fetch_add((n + 1) as u64, Ordering::Relaxed);
            let calls = N_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls % 500 == 0 {
                let sum = SUM_FRONT.load(Ordering::Relaxed);
                let total_j = N_J.load(Ordering::Relaxed);
                let max = MAX_FRONT.load(Ordering::Relaxed);
                eprintln!(
                    "[par] split call {}  avg_front={:.2}  max_front={}  k_cap={}  this_call_max={}",
                    calls,
                    sum as f64 / total_j as f64,
                    max,
                    k_cap,
                    local_max,
                );
            }
        }

        // Pick best (k <= target, min cost) at j = n.
        let best = self.pareto[n]
            .iter()
            .filter(|e| e.0 <= k_cap)
            .min_by_key(|e| e.1)
            .copied();

        let (best_k, _best_cost, _, _) = match best {
            Some(e) => e,
            None => {
                // No valid partition — fall back to one-customer-per-route.
                let mut routes: Vec<Vec<usize>> = Vec::with_capacity(n);
                for &id in giant {
                    routes.push(vec![0, id, 0]);
                }
                return routes;
            }
        };

        // Backtrack via (pred_j, pred_idx).
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(best_k as usize);
        let mut cur_j: u32 = n as u32;
        // Locate the chosen entry index in pareto[n].
        let mut cur_idx: u32 = self.pareto[n]
            .iter()
            .position(|&(k, c, pj, pi)| (k, c, pj, pi) == best.unwrap())
            .unwrap() as u32;
        for _ in 0..best_k {
            let entry = self.pareto[cur_j as usize][cur_idx as usize];
            let (_k, _c, pred_j, pred_idx) = entry;
            let i = pred_j as usize;
            let j = cur_j as usize;
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            for p in i..j {
                r.push(giant[p]);
            }
            r.push(0);
            routes.push(r);
            cur_j = pred_j;
            cur_idx = pred_idx;
        }
        routes.reverse();
        routes
    }

    /// Insert into a Pareto front sorted by `k` ascending. By Pareto invariant
    /// `cost` is strictly decreasing in `k` (lower-k entries have higher cost),
    /// so this does an inline domination check + prune.
    #[inline]
    fn pareto_insert(front: &mut Vec<ParetoEntry>, e: ParetoEntry) {
        let (k_new, c_new, _, _) = e;
        // If any existing point dominates `e`, drop it.
        for &(k, c, _, _) in front.iter() {
            if k <= k_new && c <= c_new && (k < k_new || c < c_new) {
                return;
            }
        }
        // Drop existing points dominated by `e`.
        front.retain(|&(k, c, _, _)| !(k_new <= k && c_new <= c && (k_new < k || c_new < c)));
        front.push(e);
    }

    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let (x0, y0) = (
            self.data.node_positions[0].0 as f64,
            self.data.node_positions[0].1 as f64,
        );
        let mut route_angles: Vec<(f64, usize)> = Vec::with_capacity(routes.len());

        for (r_idx, r) in routes.iter().enumerate() {
            if r.len() <= 2 {
                continue;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                cnt += 1;
            }
            let bx = sum_x / (cnt as f64);
            let by = sum_y / (cnt as f64);
            let angle = (by - y0).atan2(bx - x0);
            route_angles.push((angle, r_idx));
        }

        route_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tour = Vec::with_capacity(self.data.nb_nodes - 1);
        for &(_, r_idx) in &route_angles {
            let r = &routes[r_idx];
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                if id != 0 {
                    tour.push(id);
                }
            }
        }
        tour
    }

    fn crossover_rbx(&self, p1: &Individual, t2: &Vec<usize>, rng: &mut SmallRng) -> Vec<usize> {
        let n = self.data.nb_nodes - 1;
        if n == 0 {
            return Vec::new();
        }

        let mut cand: Vec<usize> = Vec::new();
        for (idx, r) in p1.routes.iter().enumerate() {
            if r.len() > 2 {
                cand.push(idx);
            }
        }
        if cand.is_empty() {
            return t2.clone();
        }

        cand.shuffle(rng);
        let keep = rng.gen_range(1..=cand.len().min(3));
        cand.truncate(keep);

        let mut used = vec![false; self.data.nb_nodes];
        let mut child: Vec<usize> = Vec::with_capacity(n);

        for &ri in &cand {
            let r = &p1.routes[ri];
            for &id in r.iter().skip(1).take(r.len() - 2) {
                if !used[id] {
                    used[id] = true;
                    child.push(id);
                }
            }
        }

        for &id in t2 {
            if !used[id] {
                used[id] = true;
                child.push(id);
            }
        }

        if child.len() < n {
            for id in 1..self.data.nb_nodes {
                if !used[id] {
                    used[id] = true;
                    child.push(id);
                    if child.len() == n {
                        break;
                    }
                }
            }
        } else if child.len() > n {
            child.truncate(n);
        }
        child
    }
}
