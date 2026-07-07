# Volatility Surface Modelling — Crypto Options

Calibrates implied-volatility surfaces (SVI, SSVI,
eSSVI, SABR) on **Deribit ETH / BTC options**, evaluates competing fitting
objectives across snapshots, and powers a live trader-style option chain
backed by a per-second data pipeline and a fast-path eSSVI recalibration.

---

## Quick start — live chain

One command launches the gatherer + the Streamlit app and shuts both down on
`Ctrl+C`:

```bash
./run_live.sh                 # defaults: ETH, 1 s polling
./run_live.sh btc 1           # BTC instead
```

What happens:

1. `live_gather.py` (background) pulls Deribit every second over REST,
   subscribes to its websocket for top-of-book sizes, cleans, and appends to
   a rolling parquet at `2 - Data/live/live_eth.parquet`.
2. `streamlit_chain.py` (foreground) reads the latest snapshot from that
   parquet, runs a **30 ms theta-only eSSVI update**, and repaints the chain
   table every second.
3. The first page load triggers a one-time **full eSSVI fit** (~50 s) which is
   cached. After that, only the fast path runs.

Both processes run from the `natixis_internship` conda env and communicate
only via the parquet file (atomic rename, safe to read concurrently).

---

## File map

### `volatility_surface/core/` — calibration & pricing

| file | role |
|---|---|
| `calibration/calibrator.py` | All `calibrate_*` functions (per-slice SVI, global eSSVI/SSVI, SABR, theta-only update) |
| `calibration/objectives.py` | Objective function registry (`iv_wmse`, `vega_wmse`, `iv_zweighted`, `band`, …) |
| `pricing/pricing_models.py` | Black-Scholes / Garman-Kohlhagen pricing |
| `pricing/volatility_dataframe.py` | Row-wise IV inversion via `vollib` |

### `volatility_surface/models/` — volatility models

| file | role |
|---|---|
| `vol_models.py` | Model classes: `RawSVI`, `SSVI`, `eSSVI`, `SABR` |
| `rw_parabolic.py` | Reiswich-Wystup simplified parabolic (delta-space) |

### `volatility_surface/utils/` — data, live pipeline & apps

| file | role |
|---|---|
| `data_handling.py` | Offline cleaning: CSV → cleaned parquet |
| `option_data_gathering.py` | Deribit REST helpers (`get_book_summary`, `get_index_price`) |
| `arbitrage_checker.py` | Static-arbitrage detection and heatmap plotting |
| `live_gather.py` | Live Deribit poller. REST for book summary + WS for sizes + rolling parquet writer |
| `streamlit_chain.py` | Live option-chain trading screen, eSSVI fast-path refits, maturity tabs |
| `streamlit_svi.py` | Interactive SVI / SVI-JW explorer with risk-neutral density |
| `benchmark_objectives.py` | Sweep of 10 fitting objectives on global eSSVI across snapshots → plot |
| `benchmark_rw_parabolic.py` | RW parabolic vs {SVI, SSVI, eSSVI, SABR} benchmark |

### `volatility_surface/plots/`

| file | role |
|---|---|
| `vol_plots.py` | Calibration plots: smile / surface / metrics |

### `volatility_surface/notebooks/`

| file | role |
|---|---|
| `project_outline.ipynb` | Start-to-finish walkthrough (run this first) |
| `notebook.ipynb` | Main research notebook |
| `results.ipynb` | Result inspection / plots |

### Top-level

| file | role |
|---|---|
| `run_live.sh` | One-command launcher for the live pipeline |
| `2 - Data/parquets/` | Cleaned archive parquets — input to the benchmark and offline notebooks |
| `2 - Data/live/` | Rolling live parquet (60 min window) + gatherer log |
| `2 - Data/btc/`, `eth/`, `market_making/` | Raw per-snapshot CSVs |
| `notebook.ipynb` | Top-level scratch notebook |

---

## Data flow

```
       Deribit                          Deribit
   REST book_summary               WS  ticker.{ins}.100ms
        │ 1 Hz                           │ continuous
        ▼                                ▼
   ┌────────────────────────────────────────────┐
   │ live_gather.py                              │
   │   process_data() → clean_live()             │
   │   merge top-of-book sizes from WS state     │
   │   atomic append → 60-min rolling parquet    │
   └─────────────────────┬──────────────────────┘
                         │
                         ▼
              2 - Data/live/live_eth.parquet
                         │
                         ▼
   ┌────────────────────────────────────────────┐
   │ streamlit_chain.py                          │
   │   reads latest snapshot                     │
   │   first tick:  calibrate_global_essvi   ~50 s
   │   each tick:   ...essvi_update(theta_only)  │
   │   OTM-only filter at calibration time       │
   │   renders chain table per maturity tab      │
   └────────────────────────────────────────────┘
```

---

## Other scripts

### Objective benchmark

```bash
python volatility_surface/utils/benchmark_objectives.py            # all 15 snapshots
python volatility_surface/utils/benchmark_objectives.py --quick    # first 4 expiries each
python volatility_surface/utils/benchmark_objectives.py --random 10 --seed 42
```

Outputs `benchmark_objectives_essvi.pkl` (raw per-slice metrics) and
`benchmark_objectives_essvi.png` (boxplots + per-snapshot lines + Pareto
scatter).

### Static SVI explorer

```bash
streamlit run volatility_surface/utils/streamlit_svi.py
```

Interactive sliders for Raw SVI (`a, b, ρ, m, σ`) and SVI Jump-Wings
(`v_t, ψ_t, p_t, c_t, ṽ_t`); plots the smile with the analytic risk-neutral
density overlay and flags butterfly arbitrage in red.

### Offline cleaning

```bash
python volatility_surface/utils/data_handling.py            # rebuilds the eth/btc archive parquets
```

---

## Environment

Everything runs from a single conda env: **`natixis_internship`**
(Python 3.12). `run_live.sh` hard-codes the env's Python and Streamlit
binaries.

Set it up from a fresh checkout:

```bash
conda create -n natixis_internship python=3.12 -y
conda activate natixis_internship
pip install -r requirements.txt
```

`requirements.txt` pins everything the code touches: the scientific stack
(`numpy / pandas / scipy / matplotlib / pyarrow`), `vollib` for IV inversion,
`streamlit` for the UI, and `requests + websocket-client` for the Deribit
REST + WS feeds.

---

## Models — quick reference

| model | params per slice | global params | notes |
|---|---|---|---|
| **RawSVI** | `a, b, ρ, m, σ` | – | most flexible, no calendar coupling |
| **SSVI** | `θ` (ATM total variance) | `η, γ, ρ` | provably arb-free under GJ 2014 conditions |
| **eSSVI** | `θ, η, γ` | `ρ_0, ρ_∞, λ` | `ρ(θ) = ρ_∞ + (ρ_0 − ρ_∞)·exp(−λθ)` |
| **SABR** | `α, ρ, ν` (β fixed) | – | classical Hagan; optional Obłój correction for β<1 |

Calibration data convention: bid/ask columns are **`bid_iv`** and **`ask_iv`**
(not `bid`/`ask`). Greeks (vega, delta, gamma, theta) are pulled directly
from the Deribit API.

---

## Calibration design choices

- **OTM-only at calibration time.** ITM rows are kept in the parquet so the
  trader screen can show real quotes on both sides of every strike, but the
  calibration data is filtered to `(call & k≥0) ∨ (put & k≤0)` to avoid
  parity-driven double-weighting.
- **Default objective is `iv_zweighted`** (Gatheral vega-weighting with a
  configurable `β`, optional `1/spread²` term). Empirically robust on ETH
  across snapshots.
- **Theta-only fast path** in `calibrate_global_essvi_update` keeps every
  shape parameter (ρ_0, ρ_∞, λ, η_i, γ_i) frozen and refits only the n ATM
  variance levels via 1-D Brent searches. ~30 ms for ~10 expiries.
