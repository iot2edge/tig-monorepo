// SISR (Slack Induction by String Removals) ruin-and-recreate operators.
// Christiaens & Vanden Berghe (2020), "Slack Induction by String Removals
// for Vehicle Routing Problems."

use super::instance::Instance;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;

pub struct SisrOps<'a> {
    data: &'a Instance,
    nearest: Vec<Vec<usize>>,
}

impl<'a> SisrOps<'a> {
    pub fn new(data: &'a Instance) -> Self {
        let n = data.nb_nodes;
        let mut nearest: Vec<Vec<usize>> = vec![Vec::new(); n];
        let cap = n.saturating_sub(2).min(60);
        for i in 1..n {
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(n - 2);
            for j in 1..n {
                if j == i {
                    continue;
                }
                prox.push((data.dm(i, j), j));
            }
            prox.sort_by_key(|&(d, _)| d);
            nearest[i] = prox.into_iter().take(cap).map(|(_, j)| j).collect();
        }
        Self { data, nearest }
    }

    pub fn ruin(
        &self,
        routes: &mut Vec<Vec<usize>>,
        c_bar: f64,
        l_max: usize,
        alpha_split: f64,
        rng: &mut SmallRng,
    ) -> Vec<usize> {
        let n = self.data.nb_nodes;
        if n < 3 {
            return Vec::new();
        }

        // Build customer -> route_idx lookup.
        let mut cust_route: Vec<i32> = vec![-1; n];
        for (r_idx, route) in routes.iter().enumerate() {
            for &c in &route[1..route.len() - 1] {
                cust_route[c] = r_idx as i32;
            }
        }

        // Average route length (number of customers, excluding depot endpoints).
        let nb_customers: usize = routes.iter().map(|r| r.len().saturating_sub(2)).sum();
        let nb_active_routes = routes.iter().filter(|r| r.len() > 2).count().max(1);
        let avg_route_len = (nb_customers as f64 / nb_active_routes as f64).max(1.0);

        // L_s_max = min(l_max, avg_route_len), then k_s_max via SISR formula.
        let ls_max = (l_max as f64).min(avg_route_len).max(1.0);
        let ks_max = (4.0 * c_bar / (1.0 + ls_max) - 1.0).max(1.0);
        let ks_target: usize = rng.gen_range(1..=(ks_max.floor() as usize + 1).max(1));

        // Pick a seed customer (uniform over all customers).
        let seed: usize = rng.gen_range(1..n);

        // Walk through nearest-to-seed list; ruin a string in each *new* route we hit.
        let mut removed: Vec<usize> = Vec::new();
        let mut route_touched: Vec<bool> = vec![false; routes.len()];
        let mut neighbor_iter = std::iter::once(seed).chain(self.nearest[seed].iter().copied());

        let mut strings_done = 0usize;
        for c_neigh in neighbor_iter.by_ref() {
            if strings_done >= ks_target {
                break;
            }
            let r_idx = cust_route[c_neigh];
            if r_idx < 0 {
                continue;
            }
            let r_idx = r_idx as usize;
            if route_touched[r_idx] {
                continue;
            }
            let route_len_inner = routes[r_idx].len().saturating_sub(2);
            if route_len_inner == 0 {
                continue;
            }
            let l_cap = (ls_max as usize).min(route_len_inner).max(1);
            let l_actual: usize = rng.gen_range(1..=l_cap);

            // Locate c_neigh's position in this route.
            let pos_seed = routes[r_idx]
                .iter()
                .position(|&c| c == c_neigh)
                .unwrap_or(1);

            // Choose a contiguous string of length l_actual containing pos_seed.
            // The string spans inner indices [start_inner .. start_inner + l_actual)
            // where inner indices are 1..=route_len_inner.
            let earliest_start = (pos_seed.saturating_sub(l_actual - 1)).max(1);
            let latest_start = pos_seed.min(route_len_inner + 1 - l_actual);
            let start = if earliest_start >= latest_start {
                earliest_start
            } else {
                rng.gen_range(earliest_start..=latest_start)
            };

            // Decide simple removal vs split removal.
            // Split removal keeps a "preserved" middle of m customers in place
            // and removes the rest of an extended length-(l+m) window.
            let do_split =
                rng.gen_bool(alpha_split) && l_actual >= 2 && route_len_inner >= l_actual + 1;
            if do_split {
                let m_max = (route_len_inner - l_actual).min(l_actual);
                let m: usize = if m_max >= 1 {
                    rng.gen_range(1..=m_max)
                } else {
                    1
                };
                // Window is [start .. start + l_actual + m), pick which sub-segment
                // of length m to keep.
                let window_end = (start + l_actual + m).min(route_len_inner + 1);
                let window_len = window_end - start;
                let preserve_offset_max = window_len.saturating_sub(m);
                if preserve_offset_max < 2 {
                    // No room for a true split — fall back to plain removal.
                    let take_end = (start + l_actual).min(route_len_inner + 1);
                    for _ in start..take_end {
                        let removed_id = routes[r_idx].remove(start);
                        removed.push(removed_id);
                    }
                } else {
                    let preserve_offset = rng.gen_range(1..preserve_offset_max);
                    let preserve_start = start + preserve_offset;
                    let preserve_end = preserve_start + m;
                    // Remove right segment first so left indices stay valid.
                    for _ in preserve_end..window_end {
                        let removed_id = routes[r_idx].remove(preserve_end);
                        removed.push(removed_id);
                    }
                    for _ in start..preserve_start {
                        let removed_id = routes[r_idx].remove(start);
                        removed.push(removed_id);
                    }
                }
            } else {
                let take_end = (start + l_actual).min(route_len_inner + 1);
                for _ in start..take_end {
                    let removed_id = routes[r_idx].remove(start);
                    removed.push(removed_id);
                }
            }

            route_touched[r_idx] = true;
            strings_done += 1;
        }

        removed
    }

    /// Greedy "blink" cheapest insertion: for each removed customer (in random
    /// order), find the cheapest TW+capacity-feasible insertion across all routes
    /// and the option of opening a new route. Falls back to opening a new route
    /// if no feasible spot exists; if fleet is full, force-inserts with violation
    /// (the LS phase will repair under penalty).
    pub fn recreate_greedy(
        &self,
        routes: &mut Vec<Vec<usize>>,
        mut removed: Vec<usize>,
        rng: &mut SmallRng,
        blink_rate: f64,
    ) {
        // Sort by demand desc, but with a small randomized tie-break.
        // SISR paper uses random / demand-ordered / far-from-depot orderings;
        // demand-desc is robust.
        removed.sort_by(|&a, &b| self.data.demands[b].cmp(&self.data.demands[a]));
        // Light shuffle of equal-demand groups.
        for i in 1..removed.len() {
            if self.data.demands[removed[i]] == self.data.demands[removed[i - 1]]
                && rng.gen_bool(0.5)
            {
                removed.swap(i, i - 1);
            }
        }

        for c in removed {
            let mut best: Option<(usize /*r*/, usize /*pos*/, i64 /*delta*/)> = None;
            for r_idx in 0..routes.len() {
                if let Some((pos, delta)) = self.cheapest_insertion(&routes[r_idx], c) {
                    match best {
                        None => {
                            best = Some((r_idx, pos, delta));
                        }
                        Some((_, _, d)) => {
                            if delta < d && !rng.gen_bool(blink_rate) {
                                best = Some((r_idx, pos, delta));
                            }
                        }
                    }
                }
            }

            match best {
                Some((r_idx, pos, _)) => {
                    routes[r_idx].insert(pos, c);
                }
                None => {
                    if routes.len() < self.data.nb_vehicles {
                        routes.push(vec![0, c, 0]);
                    } else {
                        // Force-insert at the position with smallest distance penalty,
                        // ignoring TW feasibility. The LS will repair under penalty.
                        let (r_idx, pos) = self.cheapest_force(&routes, c);
                        routes[r_idx].insert(pos, c);
                    }
                }
            }
        }

        // Drop empty routes.
        routes.retain(|r| r.len() > 2);
    }

    /// Cheapest TW+capacity-feasible insertion into a single route.
    /// Returns (pos, delta_distance) or None if no feasible insertion exists.
    fn cheapest_insertion(&self, route: &[usize], c: usize) -> Option<(usize, i64)> {
        let data = self.data;
        let demand: i32 = route[1..route.len() - 1]
            .iter()
            .map(|&v| data.demands[v])
            .sum();
        if demand + data.demands[c] > data.max_capacity {
            return None;
        }

        let n_pos = route.len();
        let mut best: Option<(usize, i64)> = None;

        // Forward arrival times for each position in current route.
        let mut arrival = vec![0i32; n_pos];
        let mut depart = vec![0i32; n_pos];
        // Position 0 is depot — start at 0.
        depart[0] = 0;
        for p in 1..n_pos {
            let from = route[p - 1];
            let to = route[p];
            let arr = depart[p - 1] + data.dm(from, to);
            arrival[p] = arr.max(data.start_tw[to]);
            depart[p] = arrival[p] + data.service_times[to];
            if arrival[p] > data.end_tw[to] {
                return None; // Current route is already infeasible — reject.
            }
        }

        for ins in 1..n_pos {
            let prev = route[ins - 1];
            let next = route[ins];
            let arr_c = (depart[ins - 1] + data.dm(prev, c)).max(data.start_tw[c]);
            if arr_c > data.end_tw[c] {
                continue;
            }
            let depart_c = arr_c + data.service_times[c];
            let arr_next = (depart_c + data.dm(c, next)).max(data.start_tw[next]);
            if arr_next > data.end_tw[next] {
                continue;
            }

            // Walk forward from `next` to verify feasibility through end.
            let mut feasible = true;
            let mut prev_node = next;
            let mut t_now = arr_next + data.service_times[next];
            let mut q = ins + 1;
            while q < n_pos {
                let to = route[q];
                let arr = (t_now + data.dm(prev_node, to)).max(data.start_tw[to]);
                if arr > data.end_tw[to] {
                    feasible = false;
                    break;
                }
                t_now = arr + data.service_times[to];
                prev_node = to;
                q += 1;
            }
            if !feasible {
                continue;
            }

            let delta = (data.dm(prev, c) + data.dm(c, next) - data.dm(prev, next)) as i64;
            let pick = match best {
                None => true,
                Some((_, d)) => delta < d,
            };
            if pick {
                best = Some((ins, delta));
            }
        }

        best
    }

    /// Force-insert at the cheapest position regardless of TW feasibility.
    /// Used when fleet is exhausted and no clean spot exists.
    fn cheapest_force(&self, routes: &[Vec<usize>], c: usize) -> (usize, usize) {
        let data = self.data;
        let mut best: Option<(usize, usize, i64)> = None;
        for (r_idx, route) in routes.iter().enumerate() {
            for ins in 1..route.len() {
                let prev = route[ins - 1];
                let next = route[ins];
                let delta = (data.dm(prev, c) + data.dm(c, next) - data.dm(prev, next)) as i64;
                let pick = match best {
                    None => true,
                    Some((_, _, d)) => delta < d,
                };
                if pick {
                    best = Some((r_idx, ins, delta));
                }
            }
        }
        best.map(|(r, p, _)| (r, p)).unwrap_or((0, 1))
    }
}
