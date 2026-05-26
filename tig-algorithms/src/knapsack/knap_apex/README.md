# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_apex
* **Copyright:** 2026 Tyler
* **Identity of Submitter:** Tyler
* **Identity of Creator of Algorithmic Method:** Tyler
* **Unique Algorithm Identifier (UAI):** null

## Description

`knap_apex` is a memetic QKP solver built on the public `knap_quality_opt` architecture (multi-start construction → DP refinement → variable-neighborhood descent → frequency/uniform crossover → simulated annealing → iterated local search → final heavy polish). The submitted version contributes:

### Algorithmic additions over knap_quality_opt v1

1. **Cluster-bomb perturbation** (new ILS strategy 8). When ILS stalls, picks a random selected item as "epicenter" and removes it plus the K most-synergistic selected items by V-value. Breaks up correlated sub-cliques in S, complementing the existing low-density / low-value perturbations. (This name was extracted from `knap_quality_opt_v6`'s binary symbol table — the function exists in v6 but its source is not public, so this is a reimplementation.)

2. **Deep windowed VND** (`local_search_vnd_windowed_deep`). Variant of windowed VND that uses a 2× expanded candidate window and adds a bounded 2-2 swap (top-12 worst-used vs top-12 best-unused) inside the window. Used as a final polish stage after heavy polish converges.

3. **TopNeighbors sparse data structure** with `apply_best_swap_1_1_tsn` and `apply_pair_add_tsn`. Per-item top-K-by-V_ij list lets LS scan O(K) candidates per move instead of O(n). At n=5000 with K=80 this is ~60× per-move-iteration speedup. Currently kept as supplemental polish (off by default — `use_tsn_in_ils=false`); the sparse structure itself is built once per instance and used for both `local_search_vnd_tsn` and `apply_pair_add_tsn`.

4. **1-3 swap operator** (`apply_swap_1_3_bounded`). Replaces one selected item with three unselected items, bounded by top-K of each side. Catches moves where multiple smaller items collectively replace one bulky/poor item — a move that 1-1, 1-2, 2-2 cannot find.

5. **Path relinking** primitive (`path_relinking`). Greedy best-pair walk along the symmetric difference between two population members, with end-state polishing. Currently disabled by hyperparameter (didn't lift quality vs the cost in benchmarks); kept in source for future tuning.

6. **Final-stage SA pass** (gated by `final_sa_rounds`). A fresh simulated annealing bout from the best-known solution after ILS converges, followed by another heavy-polish + DP-refinement loop. Adds a second escape opportunity beyond the population-level SA executed earlier in the pipeline.

### Production-tuned hyperparameter defaults

Defaults are calibrated against actual on-chain production benchmark hyperparameters (extracted from the TIG mainnet API `/get-benchmarks` for c003 in recent rounds). Production winners on c003 consistently use:

* `n_full_restarts: 12` for n=1000 tracks (3 for n=5000) — many short runs > one long run
* `ils_rounds: 50` for n=1000 (340 for n=5000) — much lower than v1 default 250
* `window_k: 40-60` for n=1000, `52` for n=5000 — much narrower than n
* `core_half_dp: 10-32` — much smaller DP window than v1 default 60
* `n_crossover_gen: 15-26`, `sa_rounds: 40, sa_iter: 440, n_sa_members: 8`
* `perturb_base_frac: 100, perturb_max_frac: 28, ils_restart_interval: 1`

`knap_apex` uses these as defaults so it lands close to the production frontier without HP override; players can still pass overrides via the standard `from_map` interface.

## References and Acknowledgments

### 1. Academic Papers
- Pisinger, D., "The quadratic knapsack problem — a survey," Discrete Applied Mathematics, 2007
- Lourenco, H.R., Martin, O.C., Stutzle, T., "Iterated Local Search," Handbook of Metaheuristics, 2003
- Glover, F., "A template for scatter search and path relinking," Lecture Notes in Computer Science, 1998
- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P., "Optimization by simulated annealing," Science, 1983
- Mladenovic, N., Hansen, P., "Variable neighborhood search," Computers & Operations Research, 1997

### 2. Code References
- knap_quality_opt (TIG, by NVX) — base architecture for multi-start / SA / ILS / crossover / DP refinement.
  https://github.com/tig-foundation/tig-monorepo/tree/main/tig-algorithms/src/knapsack/knap_quality_opt
- knap_quality_opt_v6 (TIG, on-chain a126) — binary symbol table inspected to identify additional move primitives (`cluster_bomb_perturb`, `local_search_vnd_windowed_deep`, `local_search_vnd_tsn`, `path_relink`, `TopNeighbors`); reimplemented from naming + general knapsack literature.
- near_knap_v4 / near_knap_v5 (TIG, by Tyler) — earlier ILS+VND iterations on the same problem.

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
