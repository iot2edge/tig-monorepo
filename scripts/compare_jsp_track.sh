#!/usr/bin/env bash
# Compare JSP algos but pass `track` HP per scenario for solvers that need it.
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling
REPO=/home/comet/Documents/tig-monorepo

ALGOS="${ALGOS:-near_jsp_v4 job_eight job_nine}"
NONCES="${NONCES:-10}"
WORKERS="${WORKERS:-12}"
FUEL="${FUEL:-10000000000}"
SEED="${SEED:-rand_hash}"
RT="$REPO/target/release/tig-runtime-all"
VF="$REPO/target/release/tig-verifier-all"

echo "JSP track-aware comparator: nonces=$NONCES fuel=$FUEL seed=$SEED"
echo "| algo | flow_shop | hybrid_flow_shop | job_shop | fjsp_medium | fjsp_high | TOTAL |"
echo "|---|---|---|---|---|---|---|"

for algo in $ALGOS; do
  row="| $algo "
  total=0
  for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
    # Pass track HP for chain algos that need it; near_jsp_v4 auto-detects so leave null
    if [[ "$algo" == near_jsp_* ]]; then
      hp='null'
    else
      hp='{"track":"'"$sc"'"}'
    fi
    out=$(python3 "$REPO/scripts/test_algorithm" "$algo" "n=50,s=$sc" "$hp" \
      --tig-runtime-path "$RT" --tig-verifier-path "$VF" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --seed "$SEED" --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    if [ "${inv:-0}" != "0" ]; then
      row+="| ${q:-0} (${inv}inv) "
    else
      row+="| ${q:-0} "
    fi
    total=$((total + ${q:-0}))
  done
  row+="| $total |"
  echo "$row"
done
