#!/usr/bin/env bash
# Sweep job_scheduling algorithms × scenarios. Prints a results matrix.
set -u

export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=job_scheduling

REPO=/home/comet/Documents/tig-monorepo
ALGOS=(adaptive_js adaptive_js_v2 adaptive_js_v3 adaptive_js_v4 job_two)
SCENARIOS=(flow_shop hybrid_flow_shop job_shop fjsp_medium fjsp_high)
NONCES="${NONCES:-8}"
WORKERS="${WORKERS:-8}"
FUEL="${FUEL:-10000000000}"

OUT=/tmp/job_sched_sweep.tsv
: > "$OUT"
echo -e "algo\tscenario\tavg_quality\twall_seconds" >> "$OUT"

for algo in "${ALGOS[@]}"; do
  for sc in "${SCENARIOS[@]}"; do
    echo ">>> $algo  /  $sc" >&2
    t0=$(date +%s)
    out=$(python3 "$REPO/scripts/test_algorithm" "$algo" "n=50,s=$sc" null \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" 2>&1 | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    t1=$(date +%s)
    q=$(echo "$out" | grep -oE 'avg_quality: [-0-9,]+' | tail -1 | sed 's/avg_quality: //;s/,//g')
    [[ -z "$q" ]] && q="ERR"
    echo -e "${algo}\t${sc}\t${q}\t$((t1-t0))" >> "$OUT"
    echo "    -> q=$q (wall=$((t1-t0))s)" >&2
  done
done

echo
echo "=== Results matrix (avg_quality) ==="
python3 - <<'PY'
import csv
from collections import defaultdict
rows = list(csv.DictReader(open("/tmp/job_sched_sweep.tsv"), delimiter="\t"))
algos = []
for r in rows:
    if r["algo"] not in algos: algos.append(r["algo"])
scenarios = []
for r in rows:
    if r["scenario"] not in scenarios: scenarios.append(r["scenario"])
m = defaultdict(dict)
for r in rows:
    m[r["algo"]][r["scenario"]] = r["avg_quality"]
hdr = ["algo"] + scenarios
print("\t".join(hdr))
for a in algos:
    print("\t".join([a] + [m[a].get(s, "?") for s in scenarios]))
PY
