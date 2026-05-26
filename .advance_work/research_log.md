# Advance research log — QKP novelty hunt

Started: 2026-04-30. Submitter: Tyler. Target challenge: c003 knapsack.

## Context

Pursuing an Advance submission to TIG for the Quadratic Knapsack Problem.
Previous code submission `near_knap_v5` is a strong CPU implementation (+7-11%
quality over `near_knap_v4`, the upstream). The Advance route requires a
*genuinely novel* algorithmic METHOD, not just a stronger heuristic.

## TIG-specific QKP structure (the only thing we have leverage on)

The instance generation in [tig-challenges/src/knapsack/mod.rs](../tig-challenges/src/knapsack/mod.rs)
produces V_ij with very specific structure that generic QKP solvers ignore:

- **V_ij = floor(Jaccard(P_i, P_j) × 1000)** where P_i is the project set assigned to participant i
- V_ij ∈ [0, 1000]; symmetric; non-negative
- d_ij = 1 - V_ij/1000 is a **metric** (Jaccard distance satisfies triangle inequality on sets — Levandowsky & Winter 1971)
- The participants are sampled in clusters from a lognormal-cardinality subset distribution, so V tends to have **community structure** + **low effective rank**
- Item weights uniform in [1, 10]; capacity = budget% × sum_weights
- v_i = 0 (no linear term — pure quadratic)

Implications:
- The pure-quadratic + uniform-weight + metric-V combination is a structurally
  narrower problem than "general QKP". Approximation theorems and algorithms
  that exploit any of (metric, low-rank, no linear term, bounded weights) are
  potentially harder-to-prior-art when claimed for this problem class.

## Investigation candidates (ranked by novelty potential)

### A. Jaccard / metric structure exploitation [INVESTIGATING]
- Hypothesis: V being a Jaccard similarity matrix gives properties (metric d, low rank,
  bounded eigenvalues) that generic QKP solvers don't use.
- Status: hostile prior-art agent dispatched 2026-04-30. Awaiting results.

### B. Specific budget-allocation rule with approximation guarantee [DEFERRED]
- Pursue only if (A) doesn't pan out.
- Path: prove a competitive ratio for some specific community-budget split rule.

### C. Specific stitcher beating coordinate-descent [DEFERRED]
- Pursue only if (A) doesn't pan out.
- Empirical work; needs benchmark suite.

### D. Pivot to a different challenge [BACKUP]
- vehicle_routing, hypergraph, energy_arbitrage may have more open territory.

## Killed angles (don't revisit)

| Angle | Killed by |
|---|---|
| Community-detection decomposition (Leiden/Louvain on V_ij) | DeCODe (Allman/Tang/Daoutidis 2018-19) — generic optimization decomposition method, packaged & commercialized; D-Wave qbsolv — production decompose-solve-stitch on QUBO, which subsumes QKP |
| GPU-port of QKP (parallel multi-start, GA, etc.) | Patvardhan 2016 (quantum-inspired EA for QKP); SUKP CUDA paper 2024; multilevel UBQP literature |
| QUBO reformulation + Ising solver | Roch et al. 2023, 2025; Tahara et al. 2025 (Extended Ising Machine on QKP) |
| Tabu search variants | Yang/Wang/Chu 2013; Lai 2019 (extensive prior art) |
| Memetic / path-relinking | Chen & Hao 2016; Zhou & Hao 2022; Chen/Hao/Glover 2016 |
| Constructive RL for QKP | Pu et al. 2025; Wang et al. 2022 |
| Breakpoints / parametric heuristic | Hochbaum et al. 2025 |
| Propagation-based DP heuristic | Fomeni 2024 |

## Standard benchmarks for any future claim

- Billionnet-Soutif (2004): n ∈ {100, 200, 300}
- Pisinger generators (2007): n up to ~6000
- Caprara-Pisinger-Toth (1999): dense, n up to 400
- [phil85/benchmark-instances-for-qkp](https://github.com/phil85/benchmark-instances-for-qkp) — consolidates 7 collections

## Hard truths

- **The "X applied to QKP" pattern keeps failing prior art.** Generic methods
  for QUBO (DeCODe, qbsolv, multilevel) subsume most ports. Inventiveness
  requires a TIGHT tie to a specific problem property.
- **Inventive distance** is the criterion: how much non-obvious work is
  required to get from prior art to your specific method. "Apply Louvain
  to QKP graph" = small distance = obvious. "Exploit specific spectral
  property of Jaccard matrices to derive bounded-error rounding scheme" =
  larger distance = potentially inventive.
- **Real estimate of effort to a defensible Advance**: 4-8 weeks of focused
  work with no guarantee of success, even on a narrow angle. The fee is
  refundable, but the time isn't.

## Next decision point

After the Jaccard prior-art agent reports:
- If it finds NO killing prior art on Jaccard-structured QKP → pursue
  mathematical analysis of low-rank / metric V exploitation
- If it finds prior art → consider direction (D) or accept that v5 ships alone

---

## Update 2026-04-30: Jaccard angle — prior art OK, math partially broken

### Prior art status
- **No direct prior art** on QKP + Jaccard exploitation. Closest hits don't
  invalidate:
  - Pferschy-Schauer 2013 (QKP on bounded-treewidth/planar): exploits
    sparsity/topology, not Jaccard structure. Doesn't apply (Jaccard V is dense
    in eigenvalue terms).
  - Tang et al. 2023 (SDP relaxation + Burer-Monteiro): low-rank, but assumes
    sparsity. Different regime.
  - Bouchard, Aregui & Denoeux 2013: proved full Jaccard matrix is PSD —
    we want to use this as mathematical leverage.
  - Gower 1971: 1-Jaccard is Euclidean-embeddable metric — also leverage.

### Empirical spectrum check (`.advance_work/spectrum/`)

Generated 27 TIG QKP instances (n ∈ {500, 1000, 2000} × budget ∈ {10, 50, 90} ×
3 nonces each). Computed eigendecomposition of V.

**Critical finding: TIG's V is NOT PSD.** Roughly 60-70% of eigenvalues are
NEGATIVE on every instance. Top |eigenvalues| are positive but a substantial
negative spectrum exists.

**Why**: TIG sets `interaction_values[i][i] = 0` (zero diagonal). The
Bouchard/Gower PSD theorems are for the FULL Jaccard matrix where diagonal
J(P_i, P_i) = 1. Zeroing the diagonal converts PSD → indefinite. So the
direct Cholesky/Gram embedding into Euclidean space DOES NOT work.

**What IS still useful**:
- **Effective rank is genuinely low**: ~30% of n captures 90% of energy,
  ~40% captures 95%. Low-rank approximation is justified empirically.
- **Stable rank** (||V||_F² / ||V||_2²) is 12-40 across instances, much less
  than n.
- The 1-Jaccard distance still satisfies triangle inequality (set property,
  unrelated to diagonal). Metric arguments still apply.

### Implications for the Advance angle

**Originally hoped-for**:  V = ΦΦ^T → x^T V x = ‖Σ x_i φ_i‖² → geometric
norm-maximization. **Killed by indefinite V.**

**Still viable narrower angles**:

a. **PSD augmentation + Lagrangian recovery**: V' = V + 1000·I is PSD (full
   Jaccard with diag=1000). Cholesky V' = ΦΦ^T gives geometric reformulation
   for the modified objective x^T V' x = x^T V x + 1000·|S|. The +1000|S|
   term means we'd be solving "QKP + cardinality bonus", which biases toward
   bigger teams. Since team size ≈ capacity/5.5 (roughly fixed by capacity),
   this might be approximately equivalent to QKP up to a constant. Needs
   careful analysis: is the duality gap small? Can we recover QKP-optimal from
   augmented-optimal?

b. **Eigendecomposition-based move evaluation**: V = Σᵢ λᵢ vᵢvᵢ^T (always
   exists for symmetric V). Maintain spectral state s_i = vᵢ · x. Move
   evaluation in O(k) instead of O(neighbor_count). For dense V this is a win;
   for our sparse V it's neutral-to-loss because neighbor lists are tiny.

c. **Top-k positive eigenspace pruning**: ignore the negative-eigenvalue
   subspace (which is "anti-correlation" structure). Use top-k positive
   eigenspace as a "feature space" for warm-starting and move ordering.
   Bounded approximation if negative eigenvalues are small relative to positive.

d. **Rank-r relaxation as warm-start**: solve the rank-r approximation
   exactly (LP-roundable continuous problem in low dim), use as warm-start
   for full local search. The rank-r choice tied to Jaccard's natural effective
   rank is the novel ingredient.

### Honest status

The Jaccard angle has SOME real substance (low-rank, metric, no direct prior
art) but turning it into a *proven* novel method is a multi-week math + code
research project. The PSD shortcut died, so the cleanest geometric
reformulation needs a workaround (option a above). The rank-r warm-start
(option d) is the most pragmatic candidate but its "novelty" depends on
whether the specific Jaccard-tied rank choice is non-obvious to a person
skilled in art.

**Realistic time estimate to a defensible Advance from here**: 4-8 weeks of
sustained work covering:
1. Math: prove approximation bound for chosen narrowing (1-2 weeks)
2. Implementation: integrate into CPU QKP solver (2-3 weeks)
3. Benchmarks: vs near_knap_v5, knap_quality_opt, on Billionnet/Pisinger
   instances (1 week)
4. Evidence document writing (1 week)

This is real research. Single chat sessions can move it forward incrementally
but cannot complete it.
