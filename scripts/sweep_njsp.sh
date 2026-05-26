#!/usr/bin/env bash
# Sweep one algorithm across all 5 job_scheduling scenarios.
# Usage: ALGO=near_jsp_v1 NONCES=50 WORKERS=12 ./sweep_njsp.sh
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo
ALGO="${ALGO:-near_jsp_v1}"
NONCES="${NONCES:-50}"
WORKERS="${WORKERS:-12}"
FUEL="${FUEL:-10000000000}"

TOTAL=0
for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
  echo ">>> $ALGO  /  $sc"
  out=$(python3 "$REPO/scripts/test_algorithm" "$ALGO" "n=50,s=$sc" null \
    --tig-runtime-path "$REPO/target/release/tig-runtime" \
    --tig-verifier-path "$REPO/target/release/tig-verifier" \
    --lib-dir "$REPO/tig-algorithms/lib" \
    --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" --ignore-invalid 2>&1)
  q=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1 | grep -oE 'avg_quality: [-0-9,]+' | tail -1 | sed 's/avg_quality: //;s/,//g')
  inv=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1 | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
  printf "    -> q=%s (invalid=%s)\n" "$q" "$inv"
  TOTAL=$((TOTAL + ${q:-0}))
done
echo
echo "TOTAL across 5 scenarios: $TOTAL"
