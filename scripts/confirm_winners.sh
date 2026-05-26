#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo
NONCES=50
WORKERS=12
FUEL=10000000000

run() {
    local label="$1"; local algo="$2"; local sc="$3"; local hp="$4"
    out=$(python3 "$REPO/scripts/test_algorithm" "$algo" "n=50,s=$sc" "$hp" \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    printf "%-40s q=%-8s invalid=%s\n" "$label" "${q:-ERR}" "${inv:-?}"
}

# Confirm winners
run "v3 flow_shop track=flow_shop"        adaptive_js_v3   flow_shop    '{"track":"flow_shop"}'
run "v1 flow_shop effort=extreme"         adaptive_js      flow_shop    '{"effort":"extreme"}'
run "v4 fjsp_medium iters=5000"           adaptive_js_v4   fjsp_medium  '{"track":"fjsp_medium","fjsp_medium_iters":5000}'
run "v4 fjsp_medium iters=10000"          adaptive_js_v4   fjsp_medium  '{"track":"fjsp_medium","fjsp_medium_iters":10000}'
run "v4 fjsp_medium iters=2000 (default)" adaptive_js_v4   fjsp_medium  '{"track":"fjsp_medium"}'
# Also: try v3 flow_shop with effort
run "v1 flow_shop default (baseline)"     adaptive_js      flow_shop    'null'
