# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** near_knap_v5
* **Copyright:** 2026 Tyler
* **Identity of Submitter:** Tyler
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

This algorithm is a derivative of `near_knap_v4` (TIG knapsack), substantially
extending it with population-based search and post-search refinement.

### 1. Code References
- `near_knap_v4` (TIG) – https://github.com/tig-foundation/tig-monorepo
- `knap_quality_opt` (TIG) – https://github.com/tig-foundation/tig-monorepo
  (reference for the bounded 2-2 swap operator and the frequency-based and
   uniform crossover operators)


## Additional Notes

`near_knap_v5` keeps the multi-start ILS architecture of `near_knap_v4` and
adds the following changes:

1. **Sparse-neighbor CSR layout.** The per-item neighbor list (used in
   `add_item` / `remove_item` and every swap loop) is stored as a flat
   `(pairs, offsets)` CSR structure instead of `Vec<Vec<(u16, i16)>>`. This
   removes one pointer indirection per neighbor read with no semantic change.

2. **i64 scoring in `apply_best_swap_neigh_any`.** The score expression in the
   neighbor-swap operator is converted from i128 to i64 (magnitudes always fit:
   `delta * 1_000_000` ≤ 2^52 for realistic values).

3. **N-conditional knob bumps.** For n ≥ 2000, an extra initial start is added
   and DP refinement passes are doubled (capped at n < 3500). For n < 2000,
   `n_perturbation_rounds` is increased by 15 to use available fuel headroom.
   The post-ILS round cap at n ≥ 2000 is raised from 16 to 20.

4. **Bounded 2-2 swap operator** added to the VND chain (after
   `apply_best_swap_neigh_any`). Picks the k weakest selected items and the k
   strongest unselected items, then exhaustively tests all (r1, r2) → (a1, a2)
   four-tuples. The operator is gated to even-numbered VND iterations to bound
   its cost.

5. **Population-based multi-start with crossover (n ≥ 2000).** The initial
   multi-start phase keeps a population of up to 6 polished solutions, then
   runs 2-4 crossover generations (more for smaller n). Each generation
   produces:
   - A frequency-crossover child: items appearing in > 75% of the population
     are inherited as "consensus"; items appearing in any member are
     randomly inherited as "exploratory" filler.
   - A uniform-crossover child between the top two population members:
     intersection items first, then a random sample from the symmetric
     difference.
   Each child is polished (DP + micro-QKP + VND) and added to the population;
   the population is re-sorted and truncated after each generation. The top
   two members become the intensification and diversification states for the
   ILS phase.

6. **Original hybrid step (n < 2000).** At small n the population/crossover
   phase is bypassed; the original intersection-fill + union-prune hybrid step
   from `near_knap_v4` runs unchanged.

7. **Post-ILS basin-relinking polish.** After the main ILS loop converges, the
   algorithm runs additional polish passes seeded from:
   - `best_sel`
   - the intensification basin (`state_int`)
   - the diversification basin (`state_div`)
   - the intersection of `state_int` and `state_div`
   - the (capacity-pruned) union of `state_int` and `state_div`

   The polish loop is iterated up to 3 passes with early-break on no-improve.
   Each polish runs DP refinement, micro-QKP, 1-2 exchange, and the windowed
   VND. Adoption is monotone — the polished result is kept only if it strictly
   improves on the running best.

8. **Stochastic kick-and-polish.** Following the relinking polishes, the
   algorithm performs a sequence of "kick" passes: random K items are removed
   from `best_sel` (for an n-conditional list of K values), and the remainder
   is re-polished via the same helper. At n < 4000 the kick array is iterated
   for two passes (different RNG state). At n ≥ 4000 only one pass runs to
   stay within the per-nonce fuel budget.

9. **Contrib-based kicks.** A final round of kicks where the K items removed
   are the K items with the lowest current `contrib` value (rather than
   random). This biases the kick toward removing the weakest contributors,
   freeing capacity for stronger replacements.

10. **Fuel-aware checkpointing.** The algorithm reads `__fuel_remaining` (a
    runtime-exported global maintained by the LLVM fuel-tracking pass) and
    skips post-ILS polish, kick, and contrib-kick passes when remaining fuel
    drops below an n-dependent safety threshold. In addition, every update to
    the running best solution invokes the `save_solution` callback, so even
    if the LLVM-injected fuel exhaustion terminates the algorithm mid-search,
    the latest improvement is already on disk. This guarantees a valid
    solution is produced for every nonce, including very large `n_items`
    where fuel may be tight.


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
