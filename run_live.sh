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

# Guard against launching with the wrong conda env's Python/Streamlit — this
# has silently happened before (base env shadows the pinned one on PATH) and
# surfaces as a confusing TypeError deep inside the Streamlit UI instead of
# a clear message here.
REQUIRED_ST=$(grep -i '^streamlit==' requirements.txt | cut -d= -f3)
FOUND_ST=$("$ENV_PY" -c "import streamlit; print(streamlit.__version__)" 2>/dev/null || echo "MISSING")
if [ "$FOUND_ST" != "$REQUIRED_ST" ]; then
    echo "✖  Wrong environment: '$ENV_PY' resolves to streamlit $FOUND_ST, expected $REQUIRED_ST." >&2
    echo "   Run 'conda activate natixis_internship' before ./run_live.sh, or set ENV_PY/ENV_ST explicitly." >&2
    exit 1
fi
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
    "$ENV_PY" volatility_surface/utils/live_gather.py --currency "$ccy" --poll "$POLL" \
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

"$ENV_ST" run volatility_surface/utils/streamlit_chain.py
