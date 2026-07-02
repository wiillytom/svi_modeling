#!/bin/bash
# Launch one or more live gatherers + the Streamlit chain in a single terminal.
# Ctrl+C stops every spawned process cleanly.
#
# Usage:
#   ./run_live.sh                 # default: gather BOTH eth and btc, 1 s polling
#   ./run_live.sh both 1          # explicit both
#   ./run_live.sh eth 1           # just eth
#   ./run_live.sh btc 1           # just btc

set -euo pipefail

WHICH="${1:-both}"
POLL="${2:-1}"

# Use whatever python/streamlit are on PATH — activate the intended conda env
# BEFORE running this script:
#     conda activate natixis_internship
# Override by exporting ENV_PY / ENV_ST if you need explicit paths.
ENV_PY="${ENV_PY:-python}"
ENV_ST="${ENV_ST:-streamlit}"

cd "$(dirname "$0")"
mkdir -p "2 - Data/live"

# Resolve the list of currencies we'll gather
case "$WHICH" in
    both) CCYS=(eth btc) ;;
    eth)  CCYS=(eth)     ;;
    btc)  CCYS=(btc)     ;;
    *)    echo "first arg must be one of: both, eth, btc" >&2; exit 1 ;;
esac

# Start one gatherer per requested currency
GATHERER_PIDS=()
for ccy in "${CCYS[@]}"; do
    log="2 - Data/live/live_gather_${ccy}.log"
    echo "▶  starting gatherer  ($ccy, every ${POLL}s)  log=$log"
    "$ENV_PY" volatility_surface/live_gather.py --currency "$ccy" --poll "$POLL" \
        > "$log" 2>&1 &
    GATHERER_PIDS+=($!)
done

cleanup() {
    echo
    for pid in "${GATHERER_PIDS[@]}"; do
        echo "■  stopping gatherer pid=$pid"
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${GATHERER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "▶  spawned gatherer pids: ${GATHERER_PIDS[*]}"
echo "▶  starting Streamlit app — open the printed URL in your browser"
echo

"$ENV_ST" run volatility_surface/streamlit_chain.py
