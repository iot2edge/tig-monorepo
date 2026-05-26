#!/usr/bin/env bash
# Hyperparameter sweep across (algo, scenario, hp) combinations.
# Output: each line "algo|scenario|hp|quality|invalid"
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo
NONCES="${NONCES:-20}"
WORKERS="${WORKERS:-12}"
FUEL="${FUEL:-10000000000}"

run_test() {
    local algo="$1"; local sc="$2"; local hp="$3"
    out=$(python3 "$REPO/scripts/test_algorithm" "$algo" "n=50,s=$sc" "$hp" \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    echo "$algo|$sc|$hp|${q:-ERR}|${inv:-?}"
}

# Combinations: (algo, scenario, hp)
CASES=(
    # v2 with effort=extreme on every scenario (auto-detect)
    'adaptive_js_v2|flow_shop|{"effort":"extreme"}'
    'adaptive_js_v2|hybrid_flow_shop|{"effort":"extreme"}'
    'adaptive_js_v2|job_shop|{"effort":"extreme"}'
    'adaptive_js_v2|fjsp_medium|{"effort":"extreme"}'
    'adaptive_js_v2|fjsp_high|{"effort":"extreme"}'
    # v3 with correct track + bumped iters
    'adaptive_js_v3|flow_shop|{"track":"flow_shop"}'
    'adaptive_js_v3|hybrid_flow_shop|{"track":"hybrid_flow_shop","hybrid_flow_shop_iters":10000}'
    'adaptive_js_v3|hybrid_flow_shop|{"track":"hybrid_flow_shop","hybrid_flow_shop_iters":20000}'
    'adaptive_js_v3|job_shop|{"track":"job_shop","job_shop_iters":50000}'
    'adaptive_js_v3|job_shop|{"track":"job_shop","job_shop_iters":100000}'
    'adaptive_js_v3|fjsp_medium|{"track":"fjsp_medium","fjsp_medium_iters":10000}'
    'adaptive_js_v3|fjsp_medium|{"track":"fjsp_medium","fjsp_medium_iters":20000}'
    'adaptive_js_v3|fjsp_high|{"track":"fjsp_high"}'
    # v4 with bumped iters
    'adaptive_js_v4|hybrid_flow_shop|{"track":"hybrid_flow_shop","hybrid_flow_shop_iters":10000}'
    'adaptive_js_v4|hybrid_flow_shop|{"track":"hybrid_flow_shop","hybrid_flow_shop_iters":20000}'
    'adaptive_js_v4|job_shop|{"track":"job_shop","job_shop_iters":50000}'
    'adaptive_js_v4|job_shop|{"track":"job_shop","job_shop_iters":100000}'
    'adaptive_js_v4|fjsp_medium|{"track":"fjsp_medium","fjsp_medium_iters":10000}'
    'adaptive_js_v4|fjsp_medium|{"track":"fjsp_medium","fjsp_medium_iters":20000}'
    # v1 with effort=extreme
    'adaptive_js|flow_shop|{"effort":"extreme"}'
    'adaptive_js|job_shop|{"effort":"extreme"}'
    # v3/v4 flow_shop fallback test
    'adaptive_js_v3|flow_shop|{"track":"flow_shop"}'
)

OUT="${OUT_FILE:-/tmp/param_sweep_results.txt}"
: > "$OUT"
echo "Running ${#CASES[@]} tests..."
for case in "${CASES[@]}"; do
    IFS='|' read -r algo sc hp <<< "$case"
    echo "[$(date +%H:%M:%S)] $algo / $sc / $hp" >&2
    result=$(run_test "$algo" "$sc" "$hp")
    echo "$result" | tee -a "$OUT"
done

echo
echo "=== RESULTS ==="
column -ts'|' "$OUT"
