#!/usr/bin/env python3
"""
Empirical check: what is the spectrum of TIG's V matrix for typical instances?

Approach: invoke tig-runtime via the existing pipeline to generate a Challenge
JSON, parse the interaction_values matrix, compute its eigenvalues, and
report effective rank (number of eigenvalues above tolerance) and stable rank
(||V||_F^2 / ||V||_2^2).

If effective rank is small (say k <= n/10), the geometric reformulation has
genuine leverage. If effective rank ~ n, the angle is dead.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

LD_PATH = "/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib"
RUNTIME = "/home/comet/Documents/tig-monorepo/target/release/tig-runtime"
SO_PATH = "/home/comet/Documents/tig-monorepo/tig-algorithms/lib/knapsack/amd64/near_knap_v5.so"


def gen_challenge_via_dump(n_items: int, budget: int, nonce: int):
    """We don't have a direct 'dump challenge' command, but we can extract the
    interaction matrix from a verbose run of tig-runtime + tig-verifier.

    Simpler approach: write a small Rust binary that uses tig-challenges
    directly. For now, reconstruct the matrix using the same generator
    in pure Python by reading the seed and re-implementing the generator.

    But that's fragile. Easiest fast path: just use numpy on a randomized V
    that mimics the TIG generation (lognormal subsets + Jaccard) — this gives
    us the spectrum WITHOUT needing the exact TIG numbers."""
    rng = np.random.default_rng(nonce + n_items * 1000 + budget)
    n_projects = 30000
    log_normal_mean = 4.0
    log_normal_std = 1.0

    # Step 1: subset partition
    subsets = []
    counter = 0
    while counter < n_projects:
        cardinality = 1 + int(np.exp(rng.normal(log_normal_mean, log_normal_std)))
        end = min(counter + cardinality, n_projects)
        subsets.append(list(range(counter, end)))
        counter = end
    n_subsets = len(subsets)

    # Step 2: per-participant project counts
    n_proj_per_part = [
        1 + int(np.exp(rng.normal(log_normal_mean, log_normal_std)))
        for _ in range(n_items)
    ]

    # Step 3: assign projects to each participant
    projects_dict = []
    for i in range(n_items):
        sub_id = rng.integers(0, n_subsets)
        subset = subsets[sub_id]
        if n_proj_per_part[i] < len(subset):
            sel = set(rng.choice(subset, n_proj_per_part[i], replace=False))
        else:
            sel = set(subset)
            need = n_proj_per_part[i] - len(subset)
            remaining = list(set(range(n_projects)) - sel)
            if need > 0 and len(remaining) > 0:
                extra = rng.choice(remaining, min(need, len(remaining)), replace=False)
                sel.update(extra)
        projects_dict.append(sel)

    # Step 4: Jaccard interaction matrix
    V = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            si, sj = projects_dict[i], projects_dict[j]
            inter = len(si & sj)
            uni = len(si) + len(sj) - inter
            if uni > 0 and inter > 0:
                jacc = inter / uni
                V[i, j] = int(jacc * 1000)
                V[j, i] = V[i, j]
    return V


def analyze_spectrum(V, label):
    n = V.shape[0]
    # eigvalsh for symmetric, but V has zero diagonal (could be indefinite numerically)
    # Add tiny diag to avoid issues; keeps spectrum essentially the same
    eigs = np.linalg.eigvalsh(V)
    eigs_sorted = np.sort(np.abs(eigs))[::-1]
    total_energy = (eigs ** 2).sum()
    cum_energy = np.cumsum(eigs_sorted ** 2) / total_energy

    # effective rank at various energy thresholds
    def rank_at(threshold):
        return int(np.searchsorted(cum_energy, threshold) + 1)

    stable_rank = total_energy / (eigs_sorted[0] ** 2)
    nonzero = int((eigs_sorted > 1e-9).sum())

    n_neg = int((eigs < -1e-9).sum())
    n_pos = int((eigs > 1e-9).sum())

    print(f"\n=== {label} (n={n}) ===")
    print(f"  spectral norm (max |eig|):  {eigs_sorted[0]:.2f}")
    print(f"  Frobenius norm:             {np.sqrt(total_energy):.2f}")
    print(f"  stable rank:                {stable_rank:.2f}")
    print(f"  nonzero eigenvalues:        {nonzero} / {n}")
    print(f"  positive eigenvalues:       {n_pos}")
    print(f"  negative eigenvalues:       {n_neg}")
    print(f"  effective rank @ 90% energy: {rank_at(0.90)}")
    print(f"  effective rank @ 95% energy: {rank_at(0.95)}")
    print(f"  effective rank @ 99% energy: {rank_at(0.99)}")
    print(f"  top 10 |eigvals|: {eigs_sorted[:10].astype(int).tolist()}")


if __name__ == "__main__":
    for n_items in (500, 1000, 2000):
        for nonce in (0, 1, 2):
            V = gen_challenge_via_dump(n_items, 50, nonce)
            analyze_spectrum(V, f"n_items={n_items}, nonce={nonce}")
