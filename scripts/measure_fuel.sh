#!/usr/bin/env bash
# Measure fuel use of an algorithm across all 5 job_scheduling scenarios.
# Direct calls to tig-runtime so we can read fuel_consumed from the OutputData JSON.
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
REPO=/home/comet/Documents/tig-monorepo
ALGO="${ALGO:-near_jsp_v1}"
SO="$REPO/tig-algorithms/lib/job_scheduling/amd64/$ALGO.so"
NONCES="${NONCES:-10}"
FUEL="${FUEL:-100000000000}"   # 100B cap (very high so we see true usage)

OUT=$(mktemp -d)
trap "rm -rf $OUT" EXIT

for sc in flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high; do
  fuels=()
  for n in $(seq 0 $((NONCES - 1))); do
    "$REPO/target/release/tig-runtime" \
      "{\"algorithm_id\":\"\",\"challenge_id\":\"c007\",\"track_id\":\"n=50,s=$sc\",\"block_id\":\"\",\"player_id\":\"\"}" \
      rand_hash "$n" "$SO" --fuel "$FUEL" --output "$OUT" >/dev/null 2>&1 || true
    f=$(python3 -c "import json,sys; d=json.load(open('$OUT/$n.json')); print(d.get('fuel_consumed', 'NA'))" 2>/dev/null || echo NA)
    fuels+=("$f")
  done
  python3 -c "
import sys
vals = [int(x) for x in '${fuels[@]}'.split() if x.isdigit()]
if not vals:
    print(f'$sc: no data')
else:
    avg = sum(vals)/len(vals)
    mn = min(vals); mx = max(vals)
    print(f'$sc: n={len(vals)}/$NONCES  avg={avg/1e9:.2f}B  min={mn/1e9:.2f}B  max={mx/1e9:.2f}B')
"
done
