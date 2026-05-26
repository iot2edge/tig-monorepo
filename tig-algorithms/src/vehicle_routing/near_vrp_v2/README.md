# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** near_vrp_v1
* **Copyright:** 2026
* **Identity of Submitter:** Tyler
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional information

near_vrp_v1 is a forked variant of fast_lane_v4 that defaults to the deepest exploration preset (level 4) instead of the fastest (level 0).

The on-chain runtime passes no hyperparameters, so the existing fast_lane_v* family runs in single-LS-pass mode and fails to fully exploit the per-nonce fuel budget (5 trillion). near_vrp_v1 hardcodes the deep HGS preset (5,000 iterations, mu=10, lambda=10) at the source level, so it always runs at full strength regardless of how the runtime invokes it.

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
