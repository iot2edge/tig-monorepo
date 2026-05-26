#!/usr/bin/env bash
# Verify both issues are fixed:
#  - Issue 1: invalid rate on flow_shop at random seeds (target: 0/50)
#  - Issue 2: behavior under low fuel (target: no catastrophic invalid)
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo
ALGO=near_jsp_v1

run() {
    local label="$1"; local sc="$2"; local seed="$3"; local fuel="$4"; local nonces="$5"
    out=$(python3 "$REPO/scripts/test_algorithm" "$ALGO" "n=50,s=$sc" null \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --seed "$seed" --nonces "$nonces" --workers 12 --fuel "$fuel" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    fin=$(echo "$last" | grep -oE '#finished: [0-9]+' | grep -oE '[0-9]+')
    printf "%-50s q=%-7s finished=%-3s invalid=%s\n" "$label" "${q:-ERR}" "${fin:-?}" "${inv:-?}"
}

echo "=== Test A: default seed (rand_hash), 10B fuel, all 5 scenarios ==="
for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
    run "$sc / rand_hash / 10B / 50n" "$sc" rand_hash 10000000000 50
done
echo
echo "=== Test B: alt seed, 10B fuel, all 5 scenarios ==="
for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
    run "$sc / alt_seed_xyz / 10B / 50n" "$sc" alt_seed_xyz 10000000000 50
done
echo
echo "=== Test C: low fuel (3B), default seed, all 5 scenarios ==="
for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
    run "$sc / rand_hash / 3B / 20n" "$sc" rand_hash 3000000000 20
done
