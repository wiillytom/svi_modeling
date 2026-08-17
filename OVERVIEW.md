# Handover — Overview

**Natixis CIB — Quant Analyst Internship.** Deribit ETH/BTC options: volatility surface calibration, a live pricing screen, and two backtesters.
Full version: [HANDOVER.md](HANDOVER.md).

---

## 1. The project in five lines

Two halves. **(a)** A surface library: clean Deribit quotes → fit a surface (SVI / SSVI / eSSVI / Mingone / SABR) → check for static arbitrage → serve it in a per-second trading screen with a 30 ms warm-start recalibration. That is working infrastructure. **(b)** The strategies built on top: a replication of Lucic & Sepp's systematic roll backtester (26 structures, weekly/monthly rolls, hourly delta-hedged) and a relative-value engine trading "red cells".

**Result: no variant produced a positive Sharpe ratio.** That is not a bug — it is the finding, and the diagnosis is where the value is (§4).

---

## 2. Getting started

```bash
conda activate natixis_internship          # underscore, not hyphen
./run_live.sh                              # ETH+BTC gatherers + Streamlit screen
```

Single env `natixis_internship` (Python 3.12). The base env has neither `vollib`, `polars`, nor the pinned `streamlit`.

**The two notebooks *are* the documentation** — run them before reading any code:

| notebook | covers |
|---|---|
| [project_outline.ipynb](volatility_surface/notebooks/project_outline.ipynb) | the surface library, end to end |
| [options_strategies_summary.ipynb](volatility_surface/notebooks/options_strategies_summary.ipynb) | the strategies, the results, the diagnosis |

> ⚠️ **Blocker #1:** `volatility_surface/notebooks/clean_chunks_full/` and `2 - Data/parquets/eth_hourly_2024.parquet` are **on the desk machine only**. Every `[archive]` cell, every `run_roll.py` call and every premium figure depends on them. Retrieve or rebuild them ([bulk_concat.py](volatility_surface/utils/bulk_concat.py)) before anything else.

> ⚠️ Never edit a `.ipynb` on disk while it is open in Jupyter — autosave silently overwrites.

---

## 3. Architecture

| layer | files | role |
|---|---|---|
| **Data** | [data_handling.py](volatility_surface/utils/data_handling.py), [bulk_concat.py](volatility_surface/utils/bulk_concat.py), [validate_dataset.py](volatility_surface/utils/validate_dataset.py) | Deribit CSV/parquet → calibration-ready frame (`k`, `t`, `w`, `mark_iv`, `bid_iv`/`ask_iv`, `vega`). OTM filter; ~31% of quotes survive |
| **Models** | [vol_models.py](volatility_surface/models/vol_models.py), [rw_parabolic.py](volatility_surface/models/rw_parabolic.py) | `VolModel` → `w(k, params)`. Registry `get_model(...)` |
| **Calibration** | [calibrator.py](volatility_surface/core/calibration/calibrator.py), [objectives.py](volatility_surface/core/calibration/objectives.py) | every `calibrate_*`, the weighted objectives and the metrics |
| **Pricing** | [pricing_models.py](volatility_surface/core/pricing/pricing_models.py) | Black-76 / BS / Garman-Kohlhagen, greeks, `inverse_delta` |
| **Arbitrage** | [arbitrage_checker.py](volatility_surface/utils/arbitrage_checker.py) | Gatheral–Jacquier conditions: calendar `∂_t w ≥ 0`, butterfly `g(k) ≥ 0` |
| **Plots** | [vol_plots.py](volatility_surface/plots/vol_plots.py) | every figure |
| **Live** | [live_gather.py](volatility_surface/utils/live_gather.py), [streamlit_chain.py](volatility_surface/utils/streamlit_chain.py), [run_live.sh](run_live.sh) | 1 Hz REST + WS → rolling parquet → screen, 30 ms refit per tick |
| **Backtest** | [roll_engine.py](volatility_surface/backtest/roll_engine.py), [engine.py](volatility_surface/backtest/engine.py) | systematic (Lucic & Sepp) and relative value |
| **Diagnostics** | [leverage.py](volatility_surface/backtest/leverage.py), [vrp_signal.py](volatility_surface/backtest/vrp_signal.py), [spot_vol_corr.py](volatility_surface/backtest/spot_vol_corr.py), [regime_hmm.py](volatility_surface/backtest/regime_hmm.py) | why the strategies returned nothing |
| **Outputs** | [results/summary.md](results/summary.md), [results/montecarlo.md](results/montecarlo.md), [results/tables/](results/tables/) | the leverage-effect study, written up |

**Models available:**

| model | parameters | note |
|---|---|---|
| Raw SVI | 5/slice | most flexible, no calendar coupling |
| SSVI | 3 global + 1/slice | arb-free if η(1+\|ρ\|) ≤ 2, γ ∈ (0, 0.5] |
| eSSVI | **5 global + n** | ρ(θ) = ρ_∞ + (ρ_0−ρ_∞)e^(−λθ) |
| Mingone | 3n via a box | **every point in the box is arbitrage-free** — no penalties |
| SABR | 3/slice, β fixed | Hagan; Obłój correction is a no-op at β = 1 |

> ⚠️ The notebooks and README still say "3 global + 3/slice" for eSSVI — **stale** since the paper-fidelity fix (2026-07-20).

---

## 4. What the project measured

**Surfaces.** SABR `vwrmse` ≈ 0.0065 vs eSSVI ≈ 0.0103. Mingone's no-arbitrage box costs essentially nothing (31.2% vs 31.4% red cells on ETH). **But eSSVI has a structural red-cell floor of ~18% (ETH) / ~28% (BTC)** — 3 parameters per slice cannot bend to crypto wings; RawSVI reaches ~2–5% but is not arbitrage-free. **That gap is the price of the guarantee.**

**Strategies: no positive Sharpe**, across 26 structures × ETH/BTC × weekly/monthly × coin/USD accounting. Three-part diagnosis:

| finding | figure |
|---|---|
| **No premium to harvest** (ATM IV 62.9% vs RV 63.7%) | **−0.82%**, t = −0.44 |
| **Discretisation loss** of an hourly hedge | **3–6 vol points**, state-dependent |
| **Residual directional beta** via vanna (long/short call) | **±0.13**, spot correlation ±0.80 |
| **ETH spot–vol correlation** (LHAR γ_d = −8.75, t = −5.76) | **ρ ≈ −0.21**, CI [−0.37, −0.04] |
| Option vs perp execution cost | **~11% round trip** against 5 bp |

In plain terms: **the strategies paid a large, certain cost to express a small, uncertain edge in the most expensive available instrument.**

**Two methodological results worth more than the backtest:**
- A Heston Monte Carlo shows the correlation estimator is **attenuated 2.5×** (divide by 0.40) — the flat plot was the correct answer to a question asked with too little precision.
- Signed variation (RS⁺−RS⁻)/RV is **structurally blind** to a diffusive leverage effect: it measures jump asymmetry, not the correlation.

---

## 5. The traps that cost the most

1. **Hedging an inverse option with the plain Black delta.** You need the Net Delta `Δ̃ = Δ − V/S` (Lucic & Sepp, Cor. 1). Without it, a systematic drift that reads as alpha one way and as a cost the other. **The numeraire is part of the instrument.**
2. **Mistaking a fitting floor for opportunity.** eSSVI's 18–28% red cells come from functional form, not the market.
3. **A flat 50 bp on mid** (the paper's Assumption 5.1) understates crypto execution by an order of magnitude. We cross the real spread.
4. **Look-ahead through the inputs.** `roll_engine.REGIMES` was drawn on the *finished* chart; `hmmlearn.predict()` is Viterbi, hence smoothed. Causal replacements: `leverage.causal_regimes`, `regime_hmm.walk_forward_states`.
5. **Interpreting a correlation before calibrating the estimator.** The simulation should have come first.
6. **One snapshot is an anecdote, not a result.** `iv_zweighted(β=0.4)` won on 10 ETH dates and lost on the full set → the live system uses `vega_wmse`.
7. **Columns are `bid_iv`/`ask_iv`**, not `bid`/`ask` — the wrong name returns `NaN` silently.

Full list (24 traps): [HANDOVER.md §11](HANDOVER.md).

---

## 6. Next steps, ranked

1. **Build the constant-maturity, constant-moneyness IV surface and regress ΔIV on Δlog F**, then compare the implied ρ to the −0.21 measured from returns. If implied is materially more negative, the **skew premium** is the tradeable object — not the level. The highest-value remaining measurement.
2. **Re-run the roll strategies with vanna hedged explicitly**, now that the correlation driving it is quantified.
3. **Repeat the leverage study on BTC** — sign agreement across two assets is worth more than another ETH robustness check.
4. Library side: `speed_mode="gradient"` (arbitrage-free eSSVI by structural reparameterisation rather than by penalty).

**Documented dead ends — do not retry:** hinge/band and soft-count losses for red cells; tightening Mingone's butterfly bound; `py_vollib_vectorized`; `%matplotlib widget`; eSSVI's second power-law family.

---

> Every wrong answer in this project came from a component that could not see what it was being asked about — a hedge ratio missing a numeraire term, a cost model an order of magnitude too small, a fitting floor mistaken for opportunity, a label that encoded the future. **None of them announced itself as an error. They all produced plausible numbers.**

*Full detail: [HANDOVER.md](HANDOVER.md) — 15 sections.*
