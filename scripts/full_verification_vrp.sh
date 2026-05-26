#!/usr/bin/env bash
# Verify near_vrp_v1 reliability and quality across:
#  - All on-chain track sizes (n=600/700/800/900/1000)
#  - Default + alt seed
#  - Production fuel (5T)
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=vehicle_routing
REPO=/home/comet/Documents/tig-monorepo
ALGO=near_vrp_v1

run() {
    local label="$1"; local n="$2"; local seed="$3"; local fuel="$4"; local nonces="$5"
    out=$(python3 "$REPO/scripts/test_algorithm" "$ALGO" "n_nodes=$n" null \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --seed "$seed" --nonces "$nonces" --workers 4 --fuel "$fuel" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    fin=$(echo "$last" | grep -oE '#finished: [0-9]+' | grep -oE '[0-9]+')
    printf "%-45s q=%-8s finished=%-3s invalid=%s\n" "$label" "${q:-ERR}" "${fin:-?}" "${inv:-?}"
}

echo "=== Test A: default seed (rand_hash), 5T fuel, all 5 sizes ==="
for n in 600 700 800 900 1000; do
    run "n=$n / rand_hash / 5T / 10n" "$n" rand_hash 5000000000000 10
done
echo
echo "=== Test B: alt seed, 5T fuel, all 5 sizes ==="
for n in 600 700 800 900 1000; do
    run "n=$n / alt_seed_xyz / 5T / 10n" "$n" alt_seed_xyz 5000000000000 10
done
