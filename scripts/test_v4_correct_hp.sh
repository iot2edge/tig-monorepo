#!/usr/bin/env bash
# Test v4 with the correct `track` hyperparameter for each scenario.
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo

for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
  hp="{\"track\":\"$sc\"}"
  echo ">>> v4  /  $sc  (hp=$hp)"
  out=$(python3 "$REPO/scripts/test_algorithm" adaptive_js_v4 "n=50,s=$sc" "$hp" \
    --tig-runtime-path "$REPO/target/release/tig-runtime" \
    --tig-verifier-path "$REPO/target/release/tig-verifier" \
    --lib-dir "$REPO/tig-algorithms/lib" \
    --nonces 50 --workers 12 --fuel 10000000000 2>&1 | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
  q=$(echo "$out" | grep -oE 'avg_quality: [-0-9,]+' | tail -1 | sed 's/avg_quality: //;s/,//g')
  echo "    -> q=$q"
done
