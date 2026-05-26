# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** near_vrp_v1
* **Copyright:** 2026 Tyler
* **Identity of Submitter:** Tyler
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

This algorithm reuses the source code of a prior TIG vehicle_routing
submission, `fast_lane_v4`. All algorithmic content (the HGS-VRPTW
hybrid genetic search, the Pareto-dominance Split DP, the ALNS
perturbation, and the local-search neighborhoods) originates with the
author of `fast_lane_v4`. This submission introduces a one-line change
to the default configuration.

### Code references
- `fast_lane_v4` (TIG, c002_a103) — https://github.com/tig-foundation/tig-monorepo
  Source of every file in `near_vrp_v1/`. Reused verbatim except for one
  line in `config.rs::Config::defaults` (see "Additional Notes" below).


## Additional Notes

`near_vrp_v1` exists to address one specific issue with the on-chain
behavior of the existing `fast_lane_v*` family.

The configuration in `config.rs` defines five `exploration_level` presets
(0 through 4):
- Level 0: single local-search pass, `max_it_total = 0` — no genetic iterations.
- Level 4: deep HGS, `max_it_total = 5,000`, `mu = 10`, `lambda = 10`.

`fast_lane_v4` defines `Config::defaults` to return preset 0. Because the
on-chain runtime never passes hyperparameters, every fast_lane variant
on-chain runs at level 0 — its weakest mode — and never exercises the
deep HGS that the algorithm was designed for. Worse, level 0 sometimes
saves an "initialization fallback" solution that exceeds the fleet-size
constraint, which is why `fast_lane_v4` and `fast_lane_v5` ship with a
100% invalid rate at the on-chain track sizes (n ∈ {600, 700, 800, 900,
1000}) despite being valid algorithms in principle.

The single change in `near_vrp_v1` is in `config.rs`:

```rust
pub fn defaults(nb_nodes: usize) -> Self {
    Self::preset(4, nb_nodes)   // was: Self::preset(0, nb_nodes)
}
```

With this change, the algorithm runs at full strength regardless of
whether hyperparameters are passed. User-supplied hyperparameters
(including `exploration_level`) still override the default through
`Config::initialize`, so all of the original behavior remains accessible
when explicitly requested.

### Hyperparameters (all optional, inherited from fast_lane_v4)

- `exploration_level`: `0` | `1` | `2` | `3` | `4` — controls preset depth.
- Any field of `Config` (e.g. `max_it_total`, `mu`, `granularity`) can be
  individually overridden after the preset is selected.


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
