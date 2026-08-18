# Handover — Overview
Deribit ETH/BTC options: volatility surface calibration, a live pricing screen, and two backtesters.

---

## 1. The project in five lines

 **(a)** A surface library: clean Deribit quotes → fit a surface (SVI / SSVI / eSSVI) → check for static arbitrage → serve it in a per-second trading screen with a 30 ms warm-start recalibration. 
 **(b)** A replication of Lucic & Sepp's systematic roll backtester (26 structures, weekly/monthly rolls, hourly delta-hedged) and a relative-value engine trading "red cells". 
**Result:** no variant produced a positive Sharpe ratio.  No vol premium for Lucic-Sepp strategies on 2024-2026 data, and too much transaction costs for the red-cell arbitrage.


---

## 2. Getting started

```bash
./run_live.sh                              # ETH+BTC gatherers + Streamlit screen
```

This creates the trading screen, but because of Natixis's firewall, we can't pull data from the Deribit API. Therefore, we only use it with archived data pulled either from Simon's computer (inside of 2 - Data) or data that we gather from our own PC.

**The two notebooks are the docs**:

| notebook | covers |
|---|---|
| [project_outline.ipynb](volatility_surface/notebooks/project_outline.ipynb) | the surface library, end to end |
| [options_strategies_summary.ipynb](volatility_surface/notebooks/options_strategies_summary.ipynb) | the strategies, the results, the diagnosis |

> ⚠️ `volatility_surface/notebooks/clean_chunks_full/` and `2 - Data/parquets/eth_hourly_2024.parquet` are on the desk machine only. Every `[archive]` cell, every `run_roll.py` call and every premium figure depends on them. Retrieve or rebuild them ([bulk_concat.py](volatility_surface/utils/bulk_concat.py)) before anything else.


---

## 3. Architecture

| layer | files | role |
|---|---|---|
| **Data** | [data_handling.py](volatility_surface/utils/data_handling.py), [bulk_concat.py](volatility_surface/utils/bulk_concat.py), [validate_dataset.py](volatility_surface/utils/validate_dataset.py) | Deribit CSV/parquet → calibration-ready frame (`k`, `t`, `w`, `mark_iv`, `bid_iv`/`ask_iv`, `vega`). OTM filter; ~31% of quotes survive |
| **Models** | [vol_models.py](volatility_surface/models/vol_models.py)| `VolModel` → `w(k, params)`. Registry `get_model(...)` |
| **Calibration** | [calibrator.py](volatility_surface/core/calibration/calibrator.py), [objectives.py](volatility_surface/core/calibration/objectives.py) | every `calibrate_*`, the weighted objectives and the metrics |
| **Pricing** | [pricing_models.py](volatility_surface/core/pricing/pricing_models.py) | Black-76 / BS / Garman-Kohlhagen, greeks, `inverse_delta` |
| **Arbitrage** | [arbitrage_checker.py](volatility_surface/utils/arbitrage_checker.py) | Gatheral–Jacquier conditions: calendar `∂_t w ≥ 0`, butterfly `g(k) ≥ 0` |
| **Plots** | [vol_plots.py](volatility_surface/plots/vol_plots.py) | every figure |
| **Live** | [live_gather.py](volatility_surface/utils/live_gather.py), [streamlit_chain.py](volatility_surface/utils/streamlit_chain.py), [run_live.sh](run_live.sh) | rolling parquet → screen, 30 ms refit per tick |
| **Backtest** | `backtest/` — [roll_engine.py](volatility_surface/backtest/roll_engine.py) + [run_roll.py](volatility_surface/backtest/run_roll.py) + [roll_results.py](volatility_surface/backtest/roll_results.py), [engine.py](volatility_surface/backtest/engine.py) + [signal.py](volatility_surface/backtest/signal.py) | systematic (Lucic & Sepp) and relative value |
| **Diagnostics** | `backtest/` — [leverage.py](volatility_surface/backtest/leverage.py), [vrp_signal.py](volatility_surface/backtest/vrp_signal.py), [spot_vol_corr.py](volatility_surface/backtest/spot_vol_corr.py), [regime_hmm.py](volatility_surface/backtest/regime_hmm.py) | why the strategies returned nothing |
| **Outputs** | [results/summary.md](results/summary.md), [results/montecarlo.md](results/montecarlo.md), [results/tables/](results/tables/) | the leverage-effect study, written up |


**Models available:**

| model | parameters | note |
|---|---|---|
| Raw SVI | 5/slice | most flexible, no calendar coupling |
| SSVI | 3 global + 1/slice | arb-free if η(1+\|ρ\|) ≤ 2, γ ∈ (0, 0.5] |
| eSSVI | 5 global + n | ρ(θ) = ρ_∞ + (ρ_0−ρ_∞)e^(−λθ) |
| Global eSSVI (Mingone) | 3n via a box | every point in the box is arbitrage-free — no penalties |




---

## 4. Running things from the CLI

To run python scripts from my natixis computer, I had to to use 
```bash
C:\Users\%USERNAME\AppData\Local\miniforge3\python.exe
```
### The roll backtester — [run_roll.py](volatility_surface/backtest/run_roll.py)

The main entry point. It wraps [roll_engine.py](volatility_surface/backtest/roll_engine.py) (the simulation) and [roll_results.py](volatility_surface/backtest/roll_results.py) (the paper-style table); neither of those is run directly.

```bash
python volatility_surface/backtest/run_roll.py --options volatility_surface/notebooks/clean_chunks_full 
--perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" --all --frequency weekly --coin ETH 
--funding "2 - Data/funding/eth_perp_funding_2024_2026.parquet" --csv results_weekly.csv
```

That is the reference run: whole catalogue (26 strategies), weekly rolls, realised funding, table written to CSV. ~1 h on the full archive; all 26 cost roughly the wall-clock of one, so never loop over `--structure`.

| flag | what it does |
|---|---|
| `--options` / `--perp` | **required.** Cleaned option parquet *or* a directory of `chunk_*.parquet`; perp OHLCV parquet |
| `--all` | whole catalogue + paper-style table. Without it, `--structure` runs one |
| `--structure "Short Straddle"` | one structure. 13 bases × Long/Short: ATM/25D/10D Call & Put, Straddle, 25D/10D Strangle, Call Spread, Put Spread, 25D RR, 25D ButterFly |
| `--frequency` | `daily\|weekly\|monthly\|quarterly`, or comma-separated with `--all` |
| `--execution` | `spread` (default, crosses the real book) · `paper` (fill-at-mid + 50 bp, Assumption 5.1) · **`both`** → runs each and diffs them |
| `--accounting` | `coin` (default) or `usd` |
| `--funding` / `--funding-annual-rate` | realised Deribit parquet (Eq 29), or a constant fallback; 0 = omitted |
| `--regime` | `bear1\|bear2\|bear3\|bull1\|bull2` — ⚠️ hand-drawn from the finished chart, descriptive only |
| `--start` / `--end` | `YYYY-MM-DD` (UTC, end exclusive) |
| `--hedge-band` | no-trade band, fraction of NAV (default 0.05) |
| `--max-snaps 5000` | **smoke run — use this first** to check the wiring |
| `--csv` / `--dump-run NAME` | summary table / full result dicts (NAV paths, event log) |

**Start with a smoke run.** It exercises the whole path in minutes:

```bash
python volatility_surface/backtest/run_roll.py --options volatility_surface/notebooks/clean_chunks_full 
--perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" --all --max-snaps 5000
```

**Is the premium eaten by the spread, or absent?** `--execution both` answers it in one pass:

```bash
python volatility_surface/backtest/run_roll.py --options volatility_surface/notebooks/clean_chunks_full 
--perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" --all --frequency weekly --execution both
```

**Reading results back.** `--dump-run` then [results_io.py](volatility_surface/backtest/results_io.py) so notebooks read artefacts instead of recomputing an hour of backtest:

```python
from volatility_surface.backtest import results_io as RIO
RIO.available()                  # what's on disk
RIO.load_table("results_weekly") # the summary table
RIO.load_run("weekly_band10")    # full dicts: NAV, events, params
```


### Diagnostics

```bash
# volatility risk premium + predictive test (--compare-estimators, --skew for the extra tables)
python volatility_surface/backtest/vrp_signal.py --options "2 - Data/parquets/eth_hourly_2024.parquet" 
--perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet" --frequency weekly --rv-frequency 1h

# spot-vol correlation; --levels switches to the vol-level / discretisation-loss view
python volatility_surface/backtest/spot_vol_corr.py --options "2 - Data/parquets/eth_hourly_2024.parquet" 
--perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet" --bar-freq 1h 
--estimators close,parkinson,garman_klass,rogers_satchell --out spot_vol_corr.png

# HMM regimes — always pass --walk-forward, the default is smoothed and look-ahead contaminated
python volatility_surface/backtest/regime_hmm.py --options "2 - Data/parquets/eth_hourly_2024.parquet" 
--perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet" --states 2 --walk-forward
```

`--bar-freq` on `spot_vol_corr.py` changes the values significantly depending on the frequency

> ⚠️ **[leverage.py](volatility_surface/backtest/leverage.py) has no CLI**, despite the usage line in its own docstring. It imports `argparse` but has no `__main__` block — running it exits 0 and produces nothing. Import it instead (`LEV.daily_measures`, `LEV.lhar`, `LEV.e1_correlations`, `LEV.mc_recovery`), as the strategies notebook does. Adding the missing `main()` is a 20-minute fix worth doing.

### Data pipeline

```bash
# raw CSV tree -> one cleaned parquet (hourly grid; keeps the FIRST file of each hour, verified)
python volatility_surface/utils/bulk_concat.py --root "2 - Data/eth" --out "2 - Data/parquets/eth_hourly.parquet" --interval-minutes 60 
--workers 8
python volatility_surface/utils/bulk_concat.py --root "2 - Data/eth" --dry-run          # count files first

# sanity-check before trusting any backtest built on it — [WARN] look, [FAIL] stop
python volatility_surface/utils/validate_dataset.py --path "2 - Data/parquets/eth_hourly.parquet"

# pull fresh perp OHLCV / rebuild the archive parquets
python volatility_surface/utils/deribit_perp.py --instrument ETH-PERPETUAL --start 2024-06-01 --resolution 1 
--out "2 - Data/parquets/eth_perp_1min.parquet"
python volatility_surface/utils/data_handling.py
```

`deribit_funding.py` is import-only — call `fetch_funding(...)` and write the parquet yourself.

### Live system and benchmarks

```bash
./run_live.sh                 # both gatherers + Streamlit screen
./run_live.sh eth 1           # ETH only, 1 s polling
./run_live.sh gif eth 7 30 1  # record the chain table to a GIF (needs a gatherer running)

streamlit run volatility_surface/utils/streamlit_svi.py     # static SVI / SVI-JW explorer

python volatility_surface/utils/benchmark_objectives.py --quick        # 10 objectives on global eSSVI
python volatility_surface/utils/benchmark_penalty_cal.py --n-snapshots 1   # penalty_cal sweep (~1 h at defaults)
python volatility_surface/utils/make_calibration_gifs.py --currency eth    # NM vs DE slides
```

**Library-only, no CLI:** `roll_results.py`, `results_io.py`, `engine.py` (the red-cell backtester — call `E.run_backtest(...)`), `leverage.py`, `deribit_funding.py`, and everything under `core/`, `models/`, `plots/`.

---

## 5. What the project measured

**Surfaces.**  eSSVI ≈ 0.0103 VWMSE. Mingone's no-arbitrage box costs essentially nothing (31.2% vs 31.4% red cells on ETH). **But eSSVI has a structural red-cell floor of ~18% (ETH) / ~28% (BTC)** — 3 parameters per slice cannot bend to crypto wings; RawSVI reaches ~2–5% but is not necessarily rbitrage-free. The tradeoff between the two is the no-arb cost essentially

**Strategies: no positive Sharpe**, across 26 structures × ETH/BTC × weekly/monthly × coin/USD accounting. Three-part diagnosis:

| finding | figure |
|---|---|
| **No premium to harvest** (ATM IV 62.9% vs RV 63.7%) | **−0.82%**, t = −0.44 |
| **Discretisation loss** of an hourly hedge | **3–6 vol points**, state-dependent |
| **Residual directional beta** via vanna (long/short call) | **±0.13**|
| **ETH spot–vol correlation**  | **ρ ≈ −0.21**, CI [−0.37, −0.04] |
| Option vs perp execution cost | **~11% round trip** against 5 bp |

In plain terms: **the strategies paid a large, certain cost to express a small, uncertain edge in the most expensive available instrument.**

**Two methodological results worth more than the backtest:**
- A Heston Monte Carlo shows the correlation estimator is **attenuated 2.5×** (divide by 0.40) — the flat plot was the correct answer to a question asked with too little precision.
- Signed variation (RS⁺−RS⁻)/RV is **structurally blind** to a diffusive leverage effect: it measures jump asymmetry, not the correlation.

---

## 6. The traps that cost the most

1. **Hedging an inverse option with the plain Black delta.** You need the Net Delta `Δ̃ = Δ − V/S` (Lucic & Sepp, Cor. 1). Without it, a systematic drift that reads as alpha one way and as a cost the other. **The numeraire is part of the instrument.**
2. **Mistaking a fitting floor for opportunity.** eSSVI's 18–28% red cells come from functional form, not the market.
3. **A flat 50 bp on mid** (the paper's Assumption 5.1) understates crypto execution by an order of magnitude. We cross the real spread.
4. **Look-ahead through the inputs.** `roll_engine.REGIMES` was drawn on the *finished* chart; `hmmlearn.predict()` is Viterbi, hence smoothed. Causal replacements: `leverage.causal_regimes`, `regime_hmm.walk_forward_states`.
5. **Interpreting a correlation before calibrating the estimator.** The simulation should have come first.
6. **One snapshot is an anecdote, not a result.** `iv_zweighted(β=0.4)` won on 10 ETH dates and lost on the full set → the live system uses `vega_wmse`.
7. **Columns are `bid_iv`/`ask_iv`**, not `bid`/`ask` — the wrong name returns `NaN` silently.

Full list (24 traps): [HANDOVER.md §11](HANDOVER.md).

---

## 7. Next steps, ranked

1. **Build the constant-maturity, constant-moneyness IV surface and regress ΔIV on Δlog F**, then compare the implied ρ to the −0.21 measured from returns. If implied is materially more negative, the **skew premium** is the tradeable object — not the level. The highest-value remaining measurement.
2. **Re-run the roll strategies with vanna hedged explicitly**, now that the correlation driving it is quantified.
3. **Repeat the leverage study on BTC** — sign agreement across two assets is worth more than another ETH robustness check.
4. Library side: `speed_mode="gradient"` (arbitrage-free eSSVI by structural reparameterisation rather than by penalty).

**Documented dead ends — do not retry:** hinge/band and soft-count losses for red cells; tightening Mingone's butterfly bound; `py_vollib_vectorized`; `%matplotlib widget`; eSSVI's second power-law family.

---

> Every wrong answer in this project came from a component that could not see what it was being asked about — a hedge ratio missing a numeraire term, a cost model an order of magnitude too small, a fitting floor mistaken for opportunity, a label that encoded the future. **None of them announced itself as an error. They all produced plausible numbers.**

*Full detail: [HANDOVER.md](HANDOVER.md) — 15 sections.*
