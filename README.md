# Volatility Surface Modelling — Crypto Options

Calibrates implied-volatility surfaces (SVI, SSVI, eSSVI, Global eSSVI/Mingone,
SABR) on **Deribit ETH / BTC options**, evaluates competing fitting objectives
across snapshots, powers a live trader-style option chain, and backtests option
strategies built on top of the surface.

**Start here:** [OVERVIEW.md](OVERVIEW.md) — the handover document (architecture,
CLI reference, results, next steps). [HANDOVER.md](HANDOVER.md) is the long
version; [passation.md](passation.md) / [passation_synthese.md](passation_synthese.md)
are the French equivalents.

---

## Quick start — live chain

Activate the env **first** (the script uses whatever `python` / `streamlit` are on
PATH and refuses to start on the wrong one):

```bash
conda activate natixis_internship
./run_live.sh                 # defaults: BOTH currencies, 1 s polling
./run_live.sh eth 1           # ETH only
./run_live.sh gif eth 7 30 1  # record the chain table to a GIF (needs a gatherer running)
```

What happens:

1. `live_gather.py` (one background process per currency) pulls Deribit every
   second over REST, subscribes to its websocket for top-of-book sizes, and
   appends the **unfiltered** chain to a rolling parquet at
   `2 - Data/live/live_<ccy>.parquet`.
2. `streamlit_chain.py` (foreground) reads the latest snapshot, runs a **~30 ms
   theta-only eSSVI update**, and repaints the chain table every second.
3. The first page load triggers a one-time **full fit** (~50 s) which is cached.
   After that, only the fast path runs.

Both processes communicate only via the parquet file (atomic rename, safe to
read concurrently).

> ⚠️ Behind the Natixis firewall the Deribit API is unreachable. The screen then
> only runs against archived data under `2 - Data/`.

---

## File map

### `volatility_surface/models/` — volatility models

| file | role |
|---|---|
| `vol_models.py` | `VolModel` ABC + `RawSVI`, `SSVI`, `eSSVI`, `GlobalESSVI` (Mingone), `SABR`; registry `get_model()` |
| `rw_parabolic.py` | Reiswich-Wystup simplified parabolic (delta-space) |

### `volatility_surface/core/` — calibration & pricing

| file | role |
|---|---|
| `calibration/calibrator.py` | All `calibrate_*` functions (per-slice SVI/SABR, global eSSVI/SSVI/SABR, Mingone box, warm-start updates), `crossedness`, `compare` |
| `calibration/objectives.py` | Objective registry (`iv_wmse`, `vega_wmse`, `iv_zweighted`, `band`, …) + evaluation metrics |
| `pricing/pricing_models.py` | Black-76 / Black-Scholes / Garman-Kohlhagen, greeks, `inverse_delta`, vectorised `gk_implied_vol` |
| `pricing/volatility_dataframe.py` | Bid/ask IV inversion (`add_bid_ask_iv`, polars-native) |

### `volatility_surface/utils/` — data, live pipeline & apps

| file | role |
|---|---|
| `data_handling.py` | Offline cleaning: `clean_df` (pandas, one snapshot), `clean_df_pl` / `clean_bulk_parquet_chunked` (polars, bulk), `load_uncleaned` for EDA |
| `bulk_concat.py` | ~170k raw CSVs → one cleaned parquet, batched + parallel + streaming |
| `validate_dataset.py` | Sanity-check a cleaned parquet before trusting a backtest on it |
| `option_data_gathering.py` | Deribit REST helpers (`get_book_summary`, `get_index_price`) |
| `deribit_perp.py` / `deribit_funding.py` | Perp OHLCV history / realised funding-rate history |
| `arbitrage_checker.py` | Static-arbitrage detection and heatmap plotting |
| `live_gather.py` | Live Deribit poller. REST book summary + WS sizes + rolling parquet writer |
| `streamlit_chain.py` | Live option-chain trading screen, fast-path refits, maturity tabs |
| `streamlit_svi.py` | Interactive SVI / SVI-JW explorer with risk-neutral density |
| `chain_gif.py` | Headless chain-table → GIF recorder |
| `benchmark_objectives.py` | Sweep of 10 fitting objectives on global eSSVI across snapshots |
| `benchmark_penalty_cal.py` | `penalty_cal` sweep for per-slice SVI / SABR |
| `benchmark_rw_parabolic.py` | RW parabolic vs {SVI, SSVI, eSSVI, SABR} |
| `make_calibration_gifs.py` | Nelder-Mead vs Differential-Evolution convergence animations |

### `volatility_surface/backtest/` — strategies

| file | role |
|---|---|
| `roll_engine.py` | Systematic roll backtester (Lucic & Sepp 2024 §5.2-5.3): 13 structures × Long/Short, hourly delta-hedge with Net Delta, coin/USD accounting |
| `roll_results.py` | Paper-style Table 1 (Total / P.a. / Vol / Sharpe / MaxDD / Skew / α / β / R²), attribution, regime splits |
| `run_roll.py` | **CLI** for the above — the main entry point |
| `results_io.py` | Load results from disk instead of recomputing (~1 h per full pass) |
| `engine.py` | Relative-value engine: trade "red cells" (theo outside the quoted book) |
| `signal.py` | `bs_pricing`, `detect_red_cells` — shared by the live screen, the GIF recorder and the backtest |
| `results.py` / `live_plot.py` | Event-log aggregation / live P&L chart in a notebook |
| `leverage.py` | Leverage-effect study: LHAR, block bootstrap, Heston Monte Carlo, causal regimes |
| `vrp_signal.py` | Volatility risk premium and its predictability |
| `spot_vol_corr.py` | Implied vs realised spot-vol correlation; discretisation-loss view |
| `regime_hmm.py` | Gaussian HMM with a strict filtered (tradeable) / smoothed (descriptive) split |

### `volatility_surface/plots/`

| file | role |
|---|---|
| `vol_plots.py` | Calibration plots: smile / surface / total variance / vega / metrics / term structure |

### `volatility_surface/notebooks/`

| file | role |
|---|---|
| `project_outline.ipynb` | The surface library, start to finish (**run this first**) |
| `options_strategies_summary.ipynb` | The strategies: construction, backtester, results, diagnosis |
| `notebook.ipynb`, `results.ipynb` | Earlier research / result-inspection notebooks |

> ⚠️ Do **not** edit an `.ipynb` on disk while it is open in Jupyter — autosave
> silently overwrites external changes.

### Top-level

| path | role |
|---|---|
| `run_live.sh` | One-command launcher for the live pipeline (+ GIF mode) |
| `2 - Data/parquets/` | Cleaned archive parquets + perp history — input to the benchmarks and backtests |
| `2 - Data/live/` | Rolling live parquet (60 min window) + gatherer logs |
| `2 - Data/funding/` | Realised Deribit funding history |
| `2 - Data/btc/`, `eth/`, `market_making/` | Raw per-snapshot CSVs |
| `results/` | Leverage-effect study output: `summary.md`, `montecarlo.md`, `tables/`, `figures/` |
| `calibration_results/` | Pickled calibration runs |
| `Publications Natixis/` | The source papers |

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
   │   first tick:  full fit             ~50 s   │
   │   each tick:   ..._update(theta_only) ~30 ms│
   │   OTM-only filter at calibration time       │
   │   renders chain table per maturity tab      │
   └────────────────────────────────────────────┘
```

---

## Other scripts

### Strategy backtest

```bash
python volatility_surface/backtest/run_roll.py --options <chunks-dir-or-parquet> --perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" --all --frequency weekly --coin ETH --csv results_weekly.csv
```

`--all` runs the whole 26-strategy catalogue in one pass. Add `--max-snaps 5000`
for a smoke run first, `--execution both` to diff realistic spread-crossing
against the paper's flat-50bp convention. Full flag reference in
[OVERVIEW.md §4](OVERVIEW.md).

### Objective benchmark

```bash
python volatility_surface/utils/benchmark_objectives.py            # all snapshots
python volatility_surface/utils/benchmark_objectives.py --quick    # first 4 expiries each
python volatility_surface/utils/benchmark_objectives.py --random 10 --seed 42
```

Outputs `benchmark_objectives_essvi.pkl` (raw per-slice metrics) and
`benchmark_objectives_essvi.png` (boxplots + per-snapshot lines + Pareto scatter).

### Static SVI explorer

```bash
streamlit run volatility_surface/utils/streamlit_svi.py
```

Sliders for Raw SVI (`a, b, ρ, m, σ`) and SVI Jump-Wings; plots the smile with the
analytic risk-neutral density and flags butterfly arbitrage in red.

### Offline cleaning

```bash
python volatility_surface/utils/data_handling.py     # rebuilds the eth/btc archive parquets
```

---

## Environment

Everything runs from a single conda env: **`natixis_internship`** (Python 3.12).

```bash
conda create -n natixis_internship python=3.12 -y
conda activate natixis_internship
pip install -r requirements.txt
```

`requirements.txt` pins the scientific stack (`numpy / pandas / polars / scipy /
matplotlib / pyarrow`), `vollib` for IV inversion, `streamlit` for the UI, and
`requests + websocket-client` for the Deribit feeds.

Two pins that matter: `polars[rtcompat]` (the plain wheel crashes on this Mac's
CPU) and `streamlit==1.58.0` (`run_live.sh` checks it and refuses to start on a
mismatch — the base env shadowing the pinned one on PATH has bitten before).

---

## Models — quick reference

| model | params per slice | global params | notes |
|---|---|---|---|
| **RawSVI** | `a, b, ρ, m, σ` | – | most flexible, no calendar coupling |
| **SSVI** | `θ` | `η, γ, ρ` | arb-free under GJ 2014: `η(1+|ρ|) ≤ 2`, `γ ∈ (0, 0.5]` |
| **eSSVI** | `θ` | `η, γ, ρ_0, ρ_∞, λ` | `ρ(θ) = ρ_∞ + (ρ_0 − ρ_∞)·exp(−λθ)`. **5 + n params** |
| **Global eSSVI** | `θ, ρ, ψ` | – (box) | Mingone 2022: every point in the reparametrised box is arbitrage-free by construction |
| **SABR** | `α, ρ, ν` (β fixed) | – | Hagan; the Obłój correction is a **no-op at β = 1** |

Calibration data convention: bid/ask columns are **`bid_iv`** and **`ask_iv`**
(not `bid`/`ask`) — the wrong name silently returns `NaN` for `spread_hit_rate`.
Greeks are pulled directly from the Deribit API.

---

## Calibration design choices

- **OTM-only at calibration time.** ITM rows are kept in the parquet so the
  trader screen can show real quotes on both sides of every strike, but the
  calibration data is filtered to `(call & k≥0) ∨ (put & k≤0)` to avoid
  parity-driven double-weighting.
- **The live screen uses `vega_wmse`.** `iv_zweighted(β=0.4)` won on ~10 ETH
  dates but the edge vanished across the full snapshot set, so plain
  vega-weighting is the robust choice. (`calibrate_*` still defaults to
  `iv_wmse` when no objective is passed.) The `1/spread²` term was removed from
  `vega_wmse` — it degraded `vwrmse`/`iv_rrmse` on 24/24 benchmarked snapshots.
- **Theta-only fast path** in `calibrate_global_essvi_update` freezes every shape
  parameter and refits only the n ATM variance levels via 1-D Brent searches.
  ~30 ms for ~10 expiries — this is what makes the live screen possible.
- **Two ways to enforce no-arbitrage:** soft penalties (`calibrate_global_essvi`)
  or Mingone's structural box (`calibrate_global_essvi_mingone`). The box is
  measurably free — same fit quality as an unconstrained per-slice fit.
