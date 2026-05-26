#!/usr/bin/env bash
# Apples-to-apples comparator for vehicle_routing algorithms.
# Runs each algo at each on-chain track size with the same fuel and seed,
# prints a clean Markdown table of avg_quality and #invalid.
set -u
export LD_LIBRARY_PATH=/home/comet/.rustup/toolchains/nightly-2025-02-10-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH:-}
export CHALLENGE=vehicle_routing
REPO=/home/comet/Documents/tig-monorepo

ALGOS=("${ALGOS:-fast_lane_v6 v2_fast new_adaptive near_vrp_v1}")
SIZES=("${SIZES:-600 700 800 900 1000}")
NONCES="${NONCES:-5}"
WORKERS="${WORKERS:-4}"
FUEL="${FUEL:-5000000000000}"
SEED="${SEED:-rand_hash}"

# Sanity: tig-runtime must be c002. Rebuild if not.
if ! strings "$REPO/target/release/tig-runtime" | grep -q "as c002::Track"; then
    echo "tig-runtime not c002 — rebuilding..."
    (cd "$REPO" && cargo clean -p tig-runtime -p tig-verifier > /dev/null 2>&1)
    (cd "$REPO" && cargo build -r -p tig-runtime --features vehicle_routing -p tig-verifier > /dev/null 2>&1) || { echo "rebuild failed"; exit 1; }
fi

run_one() {
    local algo="$1"; local n="$2"
    out=$(python3 "$REPO/scripts/test_algorithm" "$algo" "n_nodes=$n" null \
      --tig-runtime-path "$REPO/target/release/tig-runtime" \
      --tig-verifier-path "$REPO/target/release/tig-verifier" \
      --lib-dir "$REPO/tig-algorithms/lib" \
      --seed "$SEED" --nonces "$NONCES" --workers "$WORKERS" --fuel "$FUEL" --ignore-invalid 2>&1)
    last=$(echo "$out" | tr '\r' '\n' | grep -E 'avg_quality:' | tail -1)
    q=$(echo "$last" | grep -oE 'avg_quality: [-0-9,]+' | sed 's/avg_quality: //;s/,//g')
    inv=$(echo "$last" | grep -oE '#invalid: [0-9]+' | grep -oE '[0-9]+')
    fin=$(echo "$last" | grep -oE '#finished: [0-9]+' | grep -oE '[0-9]+')
    elapsed=$(echo "$last" | grep -oE 'elapsed: [0-9.]+s' | grep -oE '[0-9.]+')
    echo "${q:-0}|${fin:-0}|${inv:-0}|${elapsed:-0}"
}

echo "Comparator: nonces=$NONCES workers=$WORKERS fuel=$FUEL seed=$SEED"
echo "Algos: ${ALGOS[*]}"
echo "Sizes: ${SIZES[*]}"
echo

# Build header
header="| Algo "
for n in $SIZES; do header+="| n=$n "; done
header+="|"
echo "$header"

sep="|---"
for n in $SIZES; do sep+="|---"; done
sep+="|"
echo "$sep"

# Run grid
for algo in $ALGOS; do
    row="| $algo "
    for n in $SIZES; do
        result=$(run_one "$algo" "$n")
        q=$(echo "$result" | cut -d'|' -f1)
        inv=$(echo "$result" | cut -d'|' -f3)
        fin=$(echo "$result" | cut -d'|' -f2)
        if [ "$inv" != "0" ]; then
            row+="| ${q} (${inv} inv) "
        else
            row+="| ${q} "
        fi
    done
    row+="|"
    echo "$row"
done
