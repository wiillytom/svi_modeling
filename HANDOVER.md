# Crypto Volatility Surfaces & Option Strategies — Handover

**Natixis CIB — Quant Analyst Internship**
Deribit ETH/BTC options: surface calibration, live pricing screen, and two systematic backtesters.

---

## How to read this

Two audiences, one document.

- **Next intern:** read §0 → §1 → §2, run the two notebooks, then use §3–§10 as reference when you touch a module. §11 (traps) will save you the most time. §13 tells you what to work on.
- **Interview prep:** §1 is the elevator version. §4 and §5 carry the formulas. §10 is every number the project measured, with its confidence interval. §11 is the "tell me about a time something went wrong" material — the good answers are all there.

Everything below links to real files. Paths are relative to the project root.

---

## 0. TL;DR — the whole project in one page

Two halves, built in order.

**Half 1 — the surface library** (`volatility_surface/core`, `models`, `plots`, `utils`).
Clean Deribit quotes → fit an implied-vol surface with one of five parameterisations → check it for static arbitrage → serve it in a live per-second trading screen with a 30 ms warm-start recalibration. That is a working piece of infrastructure.

**Half 2 — the strategies built on top** (`volatility_surface/backtest`).
Two engines: a faithful replication of Lucic & Sepp (2024)'s systematic roll backtester (26 structures, weekly/monthly rolls, hourly delta-hedged, coin or USD accounting), and an opportunistic relative-value engine trading "red cells" (model vs book disagreements) from the live screen.

**Headline result: no variant produced a positive Sharpe ratio.** That is not a failure of the code — it is the finding, and the diagnosis is the valuable part:

1. There was **no volatility risk premium** to harvest in ETH over the sample (IV − RV = −0.82%, t = −0.44).
2. Hourly delta-hedging **structurally gives up 3–6 vol points** of realised variation (discretisation loss).
3. What P&L remained was **residual directional beta through vanna** (β = ±0.13, spot correlation ±0.80), not volatility.
4. Execution costs **~11% round-trip** on a crypto option spread against 5 bp on the perp.

Along the way the project produced a defensible spot–vol correlation for ETH (**ρ ≈ −0.21, CI [−0.37, −0.04]**), which nothing in the desk's prior work had measured.

```
   Deribit REST/WS ──► data_handling / live_gather ──► calibration-ready frame
                                                            │
                        ┌───────────────────────────────────┼──────────────────┐
                        ▼                                   ▼                  ▼
                   vol_models          ◄── calibrator ──►  objectives    arbitrage_checker
                  (SVI/SSVI/eSSVI/           (fit)        (weights +          (GJ checks)
                   Mingone/SABR/RW)                        metrics)
                        │                                   │
                        ├──────────► streamlit_chain  (live screen, 30 ms refit)
                        │
                        └──────────► backtest/ ──► roll_engine  (Lucic & Sepp systematic)
                                                └► engine       (red-cell relative value)
                                                └► leverage / vrp_signal / spot_vol_corr / regime_hmm
                                                     (the diagnostics that explained the results)
```

---

## 1. First 30 minutes

### Environment

Everything runs from one conda env, **`natixis_internship`** (Python 3.12). The base env does **not** have `vollib`, `polars`, or the pinned `streamlit` — running from it produces confusing failures deep inside the UI rather than an import error.

```bash
conda create -n natixis_internship python=3.12 -y && conda activate natixis_internship && pip install -r requirements.txt
```

Explicit interpreter when you need it: `/opt/anaconda3/envs/natixis_internship/bin/python3`.

Notable pins in [requirements.txt](requirements.txt): `streamlit==1.58.0` (the launcher hard-checks this), `numpy 2.3.5`, `vollib 1.0.11`, `polars[rtcompat]==1.43.0` — the plain `polars` wheel crashes on this Mac's CPU.

### Run the two notebooks

They *are* the documentation. Everything else in this file is a reference index into them.

| notebook | covers | runtime |
|---|---|---|
| [project_outline.ipynb](volatility_surface/notebooks/project_outline.ipynb) | the surface library, start to finish — cleaning, models, calibration, objectives, arbitrage, pricing, fast path, live system | fast, except 3 global calibrations at ~1–2 min each |
| [options_strategies_summary.ipynb](volatility_surface/notebooks/options_strategies_summary.ipynb) | the strategies — construction, backtester internals, results, and the three-part diagnosis | ~70 s (archive-dependent cells degrade gracefully) |

> ⚠️ **Notebook editing trap.** Do not edit `.ipynb` files on disk while they are open in a Jupyter tab. Jupyter holds its own in-memory copy and its autosave silently overwrites external changes. This cost two debugging sessions. Paste code into cells instead, or confirm the notebook is closed first.

### Launch the live screen

```bash
conda activate natixis_internship && ./run_live.sh
```

Starts both gatherers (ETH + BTC) and the Streamlit chain, shuts everything down on `Ctrl+C`. See [run_live.sh](run_live.sh) for the `eth`/`btc`/`gif` variants.

### Where the data is

| path | what | on this laptop? |
|---|---|---|
| [`2 - Data/parquets/{eth,btc}_options_data_cleaned.parquet`](2%20-%20Data/parquets/) | calibration-ready archive snapshots — the input to every offline notebook and benchmark | ✅ |
| `2 - Data/parquets/eth_perp_1min_2024_2026.parquet` | ETH perp 1-min OHLCV, 1.14 M bars, zero gaps — the spine of every diagnostic | ✅ (40 MB) |
| `2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet` | shorter perp window used by the roll backtests | ✅ |
| [`2 - Data/funding/`](2%20-%20Data/funding/) | realised Deribit hourly funding (Eq 29 input) | ✅ |
| [`2 - Data/live/`](2%20-%20Data/live/) | rolling 60-min live parquets + gatherer logs | ✅ (522 MB) |
| [`2 - Data/{eth,btc,market_making}/`](2%20-%20Data/) | raw per-snapshot CSVs | ✅ |
| `volatility_surface/notebooks/clean_chunks_full/` | **the full 2024–2026 cleaned option archive** | ❌ **desk machine only** |
| `2 - Data/parquets/eth_hourly_2024.parquet` | hourly option panel for the IV/premium studies | ❌ **desk machine only** |

**The two missing files are the single biggest onboarding blocker.** Every cell marked `[archive]` in the strategies notebook, every `run_roll.py` invocation, and every implied-vol premium number depends on them. They can be rebuilt from raw CSVs with [bulk_concat.py](volatility_surface/utils/bulk_concat.py) — budget serious wall-clock — or copied from the desk machine. Do that first.

---

## 2. Repository map

```
Internship Natixis/
├── volatility_surface/
│   ├── models/          vol_models.py, rw_parabolic.py         — the parameterisations
│   ├── core/
│   │   ├── calibration/ calibrator.py, objectives.py           — the fitting engine
│   │   └── pricing/     pricing_models.py, volatility_dataframe.py
│   ├── plots/           vol_plots.py                           — every figure
│   ├── utils/           data pipeline, live system, benchmarks
│   ├── backtest/        two engines + four diagnostic modules
│   └── notebooks/       the two walkthroughs
├── 2 - Data/            raw CSVs, cleaned parquets, live feeds, funding
├── results/             leverage-effect study output (summary.md, tables/, figures/)
├── calibration_results/ pickled calibration runs
├── presentation_assets/ figures & GIFs for slides
├── Publications Natixis/ the four source papers
├── run_live.sh          one-command live launcher
└── README.md, CLAUDE.md
```

~15,800 lines of Python across 35 modules. Line counts are a rough guide to where the complexity sits: `calibrator.py` 2459, `vol_plots.py` 999, `objectives.py` 955, `streamlit_chain.py` 870, `roll_engine.py` 843, `vol_models.py` 778.

---

## 3. Layer 1 — Data pipeline

**Files:** [data_handling.py](volatility_surface/utils/data_handling.py) · [option_data_gathering.py](volatility_surface/utils/option_data_gathering.py) · [bulk_concat.py](volatility_surface/utils/bulk_concat.py) · [validate_dataset.py](volatility_surface/utils/validate_dataset.py) · [volatility_dataframe.py](volatility_surface/core/pricing/volatility_dataframe.py) · [deribit_perp.py](volatility_surface/utils/deribit_perp.py) · [deribit_funding.py](volatility_surface/utils/deribit_funding.py)

Raw Deribit book summary (one row per listed instrument) → calibration-ready frame:

| column | meaning |
|---|---|
| `k` | log-moneyness `ln(K/F)` |
| `t` | time to expiry, years |
| `w` | total variance `σ_mark² · t` |
| `mark_iv` | Deribit mark IV |
| `bid_iv` / `ask_iv` | IVs inverted from bid/ask **prices** |
| `vega` | analytic Black vega — the default fitting weight |
| `underlying_price` | per-expiry **forward** F |
| `estimated_delivery_price` | **spot** index S |

> ⚠️ The bid/ask IV columns are **`bid_iv`/`ask_iv`**, not `bid`/`ask`. Passing the wrong name silently returns `NaN` for `spread_hit_rate` rather than raising. Always pass `bid_col='bid_iv', ask_col='ask_iv'` explicitly.

**Filters applied by `clean_df(filter_rows=True)`:** `volume > 0`, no-arbitrage boundary (put: `F·bid < K`; call: `bid < 1`), OTM-only (`call & k≥0` ∨ `put & k≤0`), and dropping rows where bid/ask IV inversion fails.

The OTM filter exists so put–call parity duplicates don't double-weight the same strike. Only **~31% of listed ETH quotes survive** the full pipeline — which is why `load_uncleaned()` exists: run EDA on the unfiltered universe, otherwise you hide exactly the composition (illiquidity, ITM spread width) that motivated the filters.

**Three variants, know which you want:**

| function | input | engine | use |
|---|---|---|---|
| `clean_df` | one CSV = one snapshot | pandas | the original archive |
| `clean_df_pl` | bulk multi-snapshot parquet | polars | the 2024–2026 backtest data |
| `clean_bulk_parquet_chunked` | bulk → directory of `chunk_*.parquet` | polars | what the backtesters iterate |
| `live_gather.clean_live` | live REST response | pandas | the live feed (**no** row filtering — see §7) |

`clean_df_pl` was verified byte-for-byte (~1e-14) against the pandas baseline. It detects snapshot boundaries itself by timestamp gap (`snapshot_gap_ms=1000`; real snapshots are ~1 min apart, intra-snapshot poll jitter ~10–20 ms).

**Downsampling matters more than it looks.** In [bulk_concat.py](volatility_surface/utils/bulk_concat.py), `--interval-minutes` thins by choosing which *files* to open — the IV inversion inside `clean_df` is essentially all the cost, so an hourly build over a minute dump is ~60× less work, not just less storage. Verified: keeping the **first** file of each hour reproduces a minute-data backtest to 0.00e+00; keeping the **last** shifts every hedge and roll by 59 minutes and moves returns by up to 1.7 percentage points.

Run [validate_dataset.py](volatility_surface/utils/validate_dataset.py) before trusting any backtest on a newly built parquet. The 2024–2026 build is stitched from at least three Deribit export vintages, and a units or coverage change would corrupt results silently rather than crash.

---

## 4. Layer 2 — The models

**File:** [vol_models.py](volatility_surface/models/vol_models.py) (+ [rw_parabolic.py](volatility_surface/models/rw_parabolic.py))

Every model subclasses `VolModel` and implements `w(k, params)` — total implied variance. IV is `√(w/t)`. Registry: `get_model("svi" | "ssvi" | "essvi" | "global_essvi" | "sabr", **kwargs)`.

### The five parameterisations

**Raw SVI** (Gatheral) — 5 params per slice, no calendar coupling:

```
w(k) = a + b·[ρ(k − m) + √((k − m)² + σ²)]
```

**SSVI** (Gatheral–Jacquier 2012/2014) — 3 global + 1 per slice:

```
w(k) = (θ/2)·[1 + ρφk + √((φk + ρ)² + 1 − ρ²)],   φ(θ) = η / θ^γ
```

Free of static arbitrage when **η(1 + |ρ|) ≤ 2** and **γ ∈ (0, 0.5]**. Three `phi_type` options are implemented: `"power"` (default, above), `"heston"` (single λ, the other original GJ form), `"blend"` (η / [θ^γ(1+θ)^(1−γ)]). The blend's arbitrage condition is *not* the power-law's — re-deriving Theorem 4.2(iii) at γ=½ gives η²(1+|ρ|) ≤ 4 — so `validate()` falls back to the exact numerical `min_g` check for it.

**eSSVI** (Hendriks & Martini 2019) — the only change vs SSVI is promoting ρ to a function of θ:

```
ρ(θ) = ρ_∞ + (ρ_0 − ρ_∞)·e^(−λθ)
```

ρ is more negative at short maturities and flattens out; constant-ρ SSVI cannot capture that. **Parameter count: 5 global (η, γ, ρ_0, ρ_∞, λ) + n per-slice (θ_i) = 5 + n.**

> ⚠️ **The notebooks and README still say "3 global + 3 per slice" — that is stale.** The code fitted η/γ per-slice until 2026-07-20, which gave eSSVI an unfair capacity advantage and was not the paper's model. It now fits a single global power law. Fix this text when you touch it; if asked in an interview, this is a good example of "the implementation had drifted from the paper and the benchmark was measuring the wrong thing."

**GlobalESSVI / Mingone (2022)** — eSSVI re-written in (θ, ρ, ψ) with ψ := θφ, so no functional form for φ is assumed at all:

```
w(k) = ½·[θ + ρψk + √((ψk + θρ)² + θ²(1 − ρ²))]
```

**SABR** (Hagan et al. 2002) — 3 params per slice (α, ρ, ν), β fixed at 1.0 by default. The **Obłój (2008)** correction to `z(K,F)` is implemented but is a **no-op at β = 1** (both reduce to (ν/α)·log(F/K)) — it only bites for β < 1. The docstring spells this out; worth knowing so you don't chase a difference that cannot exist.

**Reiswich–Wystup parabolic** — the FX approach, a parabola in delta space anchored on 3 quotes:

```
σ(Δ) = σ_ATM − 2σ_RR(Δ − 0.5) + 16σ_S(Δ − 0.5)²
```

### Arbitrage machinery on every model

Butterfly is checked via the Gatheral **g(k)** function:

```
g(k) = (1 − k·w'/(2w))² − (w'²/4)·(1/w + ¼) + w''/2
```

with the Breeden–Litzenberger density `p(k) = g(k)/√(2πw) · exp(−d_−²/2)`, `d_− = −k/√w − √w/2`. Negative density ⇔ butterfly arbitrage.

`SSVI`, `eSSVI`, `GlobalESSVI` and `RawSVI` each **override `min_g` with closed-form derivatives** on a 50-point grid instead of the base class's 600-point finite-difference scan — ~15× faster, and it checks the *actual single-slice* condition rather than the stricter surface-level GJ bound. That distinction is load-bearing at short maturities, where a slice can be butterfly-free even when η(1+|ρ|) > 2.

Calendar is `crossedness(model, p1, p2) = max(0, max_k [w(k,t₁) − w(k,t₂)])`.

---

## 5. Layer 3 — Calibration

**Files:** [calibrator.py](volatility_surface/core/calibration/calibrator.py) · [objectives.py](volatility_surface/core/calibration/objectives.py)

### Entry points

| function | model | params | typical cost |
|---|---|---|---|
| `calibrate_snapshot` | any per-slice (RawSVI, SABR) | 5n / 3n | seconds |
| `calibrate_global_ssvi` | SSVI | 3 + n | ~1–2 min |
| `calibrate_global_essvi` | eSSVI | 5 + n | ~1–2 min cold, faster in `speed_mode="fast"` |
| `calibrate_global_essvi_mingone` | GlobalESSVI | 3n via the no-arb box | **0.1–0.4 s** (ETH/BTC median) |
| `calibrate_global_sabr` | SABR + calendar coupling | 3n | minutes |
| `calibrate_global_essvi_update` | eSSVI warm start | — | **~30–200 ms** (`theta_only`) / ~1–3 s (`full_polish`) |
| `calibrate_snapshot_sabr_update` | SABR **or RawSVI** warm start | — | ~5 s |

All return the same result dict, which every plot and check consumes:

```python
{"model_name": str, "objective": str, "expiries": [float],
 "params": [dict], "metrics": {name: [float per slice]},
 "n_points": [int], "_model": VolModel}
```

Aggregate with `np.nanmean(result["metrics"]["spread_hit_rate"])` — per-slice lists can contain `NaN` from skipped slices, so plain `sum()/len()` is wrong.

### Two ways to enforce no-arbitrage — and this is the interesting design tension

**Soft penalties** (`calibrate_global_essvi`): calendar crossedness and butterfly violations enter the objective as weighted penalty terms (`penalty_cal=500`, `penalty_but=200`). Flexible, but the optimiser gets little gradient signal and the problem can degenerate.

**Structural** (`calibrate_global_essvi_mingone`): reparameterise N eSSVI slices as `ρ_1..ρ_N ∈ (−1,1)ᴺ`, `θ_1 > 0`, `a_2..a_N > 0`, `c_1..c_N ∈ (0,1)`, such that **every point in that box is automatically free of both calendar and butterfly arbitrage** (Mingone Prop 3.1). No penalties, no arbitrage check during optimisation at all. Solved with `scipy.optimize.least_squares` at the paper's own `max_nfev=1000, ftol=1e-8`.

The box was audited on 2026-07-17: 20k random draws from the eq.(4) domain, **zero violations** of eq.(3), zero butterfly, zero calendar crossings, including the subtle backward look-ahead recursion for `C_ψ₁`. The implementation is faithful.

**And the box costs essentially nothing.** Fitting each slice with a totally free (θ, ρ, ψ) and no cross-slice constraint gives the *same* red-cell rate as the constrained box (ETH 31.2% vs 31.4%, BTC 36.1% vs 36.1%). So upgrading the GJ butterfly bound to Mingone's tighter necessary-and-sufficient bound (paper §2.2.2) would buy ~nothing. Don't spend time on it.

### Objectives

Registry `OBJECTIVES` in [objectives.py](volatility_surface/core/calibration/objectives.py); all share the signature `f(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None, vegas=None, bid=None, ask=None)`.

The one worth understanding is **`iv_zweighted`**:

```
weight_j = vega_j · exp(β·z_j²) / spread_j²,      z_j = k_j / (σ_j √t)
```

Because vega ∝ exp(−z²/2), this is equivalently `weight ≈ vega^(1−2β)` — a smooth power family with one monotone, interpretable knob on wing attention:

- β = 0 → pure vega (ATM-dominated)
- β = 0.5 → uniform per unit of standardised moneyness
- β > 0.5 → active wing emphasis, use with care

Available: `w_mse`, `iv_mse`, `iv_wmse`, `price_mse`, `iv_rmse`, `vega_wmse`, `vega_wmse_linear`, `band`, `vega_wmse_band`, `iv_zweighted`, `iv_uniform_z`, `iv_convex_blend`, `iv_replication`.

### Metrics

```python
DEFAULT_METRICS = ["iv_rrmse", "spread_hit_rate", "price_vwrrmse", "w_rmse",
                   "vwrmse", "vwrrmse", "iv_rmse_atm", "iv_rmse_otm"]
```

- `vwrmse` — vega-weighted absolute RMSE on IV (matches the FactSet formula)
- `spread_hit_rate` — fraction of fitted IVs landing inside `[bid_iv, ask_iv]`

### The warm-start fast path

`calibrate_global_essvi_update(mode="theta_only")` freezes every shape parameter (ρ_0, ρ_∞, λ, η, γ) and refits only the n ATM variance levels θ_i via independent 1-D Brent searches. **That is what makes the live screen possible** — 30 ms against ~50 s for a cold fit. `mode="full_polish"` runs a short Nelder–Mead from the previous solution when the shape may have moved.

`calibrate_snapshot_sabr_update` was generalised to accept RawSVI as well as SABR — the body was already model-agnostic, only the `isinstance` check was too strict.

---

## 6. Layer 4 — Arbitrage checking & plots

**Files:** [arbitrage_checker.py](volatility_surface/utils/arbitrage_checker.py) · [vol_plots.py](volatility_surface/plots/vol_plots.py)

`check_arbitrage(result)` evaluates the fitted surface on a dense (k, t) grid and tests the two Gatheral–Jacquier conditions — **calendar** `∂_t w ≥ 0` and **butterfly** `g(k,t) ≥ 0` — returning violation statistics. `plot_arbitrage_map` renders a 2×2 diagnostic with violations overlaid in red.

Plotting API (all take a result dict, all return a `plt.Figure`): `plot_surface`, `plot_slices`, `plot_total_variance`, `plot_vega`, `plot_price_vs_bidask`, `plot_fitted_term_structure`, `plot_rho_comparison`, `plot_metrics`, `plot_compare`, `plot_metric_by_date`, `plot_all`.

Conventions worth preserving: truncated RdPu palette (`CMAP`/`PALETTE`); matplotlib stays **inline/static** (`%matplotlib widget` was tested and reverted as too laggy); `plot_total_variance` was tried in rotating 3D and reverted to 2D because 2D is easier to copy into slides; `shuffle_colors=True` alternates strictly dark/light rather than running a gradient, so the tail of the sequence keeps contrast.

> ⚠️ `ax.suptitle()` does not exist — it's a `Figure` method. Use `ax.text(0.5, 1.02, …, transform=ax.transAxes)` with an increased `set_title(pad=26)`.

---

## 7. Layer 5 — Pricing & inverse options

**File:** [pricing_models.py](volatility_surface/core/pricing/pricing_models.py)

Three engines share one Black skeleton, differing only in the carry term: **Black-76** (forward measure, no rates), **Black-Scholes** (drift r), **Garman-Kohlhagen** (drift r − q, where q absorbs crypto funding/staking/basis). GK is the workhorse; BS is the q = 0 case.

Deribit publishes spot S (`estimated_delivery_price`) and the per-expiry forward F (`underlying_price`), so the implied rate is **r = ln(F/S)/t**.

### The one detail that most changes results

Deribit options are **inverse** — quoted and settled in the coin, not USD:

```
payoff_coin = (S_T − K)⁺ / S_T   (call),    (K − S_T)⁺ / S_T   (put)
```

That `1/S_T` makes the coin payoff non-linear in spot even with expiry fixed, and asymmetric between calls and puts. Because the numeraire is the coin rather than cash, the correct hedge ratio is **not** the Black delta — it is the premium-adjusted **Net Delta** (Lucic & Sepp 2024, Corollary 1):

```
Δ̃(t,S) = Δ(t,S) − V(t,S)/S
```

implemented as `inverse_delta` and wired through **every** engine. Hedging with the plain Black delta leaves a systematic residual: puts are where it bites, since the Net delta is unbounded below and a put that goes deep ITM mid-week is under-hedged by up to 60% of notional.

Also available: `gk_vega`, `gk_vanna`, `gk_volga`, `gk_implied_vol`, `vv_price` (vanna-volga on 3 anchors), `compare_pricing_models`.

**`gk_implied_vol` was rewritten for ~45× speed** on large frames by shrinking the active row set each Newton iteration. Two subtle bugs surfaced and are worth remembering if you ever touch it: (1) gate the *step itself* on `not converged`, not just the drop-from-active-set — near-zero vega with a tiny diff can still compute a huge step and clip a good row to the bound; (2) write `sigma[idx]` **after** stepping, not before, or a non-convergent row is one iteration stale (a real period-2 oscillating row was found landing on the wrong phase). Residual known limitation, shared with the old solver: deep-OTM/short-T rows where the model price underflows, so `|diff| < tol` is satisfiable without σ meaning anything. Jaeckel's "Let's Be Rational" handles it; plain Newton structurally cannot.

---

## 8. Layer 6 — The live system

**Files:** [live_gather.py](volatility_surface/utils/live_gather.py) · [streamlit_chain.py](volatility_surface/utils/streamlit_chain.py) · [streamlit_svi.py](volatility_surface/utils/streamlit_svi.py) · [chain_gif.py](volatility_surface/utils/chain_gif.py) · [signal.py](volatility_surface/backtest/signal.py) · [run_live.sh](run_live.sh)

```
   Deribit REST (book summary, 1 Hz) ──┐   Deribit WS (ticker.*, top-of-book sizes)
                                       ▼            │
                             live_gather.py ◄───────┘
                               clean_live() → derive t, k, w, vega, bid_iv/ask_iv
                               NO row filtering — the parquet is the raw chain
                               atomic rename → rolling 60-min parquet
                                       │
                                       ▼
                             streamlit_chain.py
                               read latest snapshot
                               first tick : full eSSVI / SABR / SVI  (~50 s, cached)
                               each tick  : theta_only update        (~30 ms)
                               OTM filter applied only to the calibration frame
                               render chain table + maturity tabs + arb banner
```

The two processes communicate **only through the parquet file**, with an atomic `os.replace`, so concurrent reads are safe.

Design point worth stating in an interview: the gatherer deliberately writes the **unfiltered** chain. The trading screen must show real quotes on both sides of every strike; the OTM/no-arb filters are applied downstream, only to the frame handed to the calibrator (`streamlit_chain._clean_for_calibration`). Filtering at write time would have made the screen wrong to save nothing.

The screen mirrors a Deribit chain: invariant greeks left, smile vol, call/put markets either side of a black STRIKE column (ATM highlighted), WS bid/ask sizes, and theo prices that turn **red** when they fall outside the quoted spread. A banner flags calendar-spread arbitrage across fitted expiries.

`detect_red_cells` in [signal.py](volatility_surface/backtest/signal.py) is the **single** implementation of that red-cell test, shared by the screen, the GIF recorder and the backtest engine — it exists because the same logic had already been duplicated once, and a third copy was where to stop.

Side apps: [streamlit_svi.py](volatility_surface/utils/streamlit_svi.py) is an interactive Raw SVI / SVI-JW explorer with the analytic risk-neutral density overlaid and butterfly arbitrage flagged in red. [chain_gif.py](volatility_surface/utils/chain_gif.py) records the chain table to a GIF headlessly (needs a gatherer running).

---

## 9. Layer 7 — The backtesters

Two genuinely different engines that share pricing and nothing else.

| | [roll_engine.py](volatility_surface/backtest/roll_engine.py) | [engine.py](volatility_surface/backtest/engine.py) |
|---|---|---|
| **idea** | systematic — fixed calendar, fixed structure | opportunistic relative value |
| **source** | Lucic & Sepp (2024) §5.2–5.3 | ours, from the live screen's red cells |
| **entry** | every roll date, unconditionally | when the calibrated surface disagrees with the book |
| **selection** | ATM / 25Δ / 10Δ by rule | whichever cells are flagged |
| **exit** | held to maturity | signal nulls, better cell appears, or expiry |
| **sizing** | contracts ∝ Coin NAV (auto-compounding) | vega-targeted ($1,000 vega per position) |
| **hedge** | hourly, perp, Net Delta | same |
| **surface** | not needed — trades listed strikes | full eSSVI every 15 min, warm-started between |

### 9a. The systematic engine

**Files:** [roll_engine.py](volatility_surface/backtest/roll_engine.py) · [roll_results.py](volatility_surface/backtest/roll_results.py) · [run_roll.py](volatility_surface/backtest/run_roll.py) · [results_io.py](volatility_surface/backtest/results_io.py)

13 base structures × Long/Short = **26 strategies**: ATM/25Δ/10Δ calls and puts, straddle, 25Δ/10Δ strangle, call spread, put spread, 25Δ risk reversal, 25Δ butterfly. `run_all_strategies` runs the **whole catalog in a single pass** over the data — the expensive part (reading and iterating ~165 daily chunks) is paid once and each strategy carries its own light state, so 26 strategies cost roughly the wall-clock of one.

**Per-snapshot state machine:**

```
┌── roll instant? ──────────────────────────────────────────┐
│   _accrue    book hedge P&L + funding up to this instant   │
│   _settle    realise the expiring structure at intrinsic   │
│   _open      select legs, size to NAV, pay entry costs     │
│   _rehedge   force=True — the book changed wholesale       │
└────────────────────────────────────────────────────────────┘
┌── hedge hour? ────────────────────────────────────────────┐
│   _accrue        perp P&L (Eq 26) + funding (Eq 29)        │
│   _compute_book  target delta + mid mark  (PURE)           │
│   _rehedge       only if |drift| > band × NAV; charge 5 bp │
└────────────────────────────────────────────────────────────┘
```

Three details decide whether the NAV path means anything:

1. **`_accrue` runs *before* the book changes at a roll**, otherwise the last holding period's hedge P&L is silently dropped. Funding accrues pro-rata on elapsed time, so it is correct whether called on the hourly grid or off-grid at a roll.
2. **`_compute_book` is pure** — the same function serves both hedging and marking, so the two cannot disagree.
3. **The no-trade band is absolute** (coin units, scaled by NAV), not relative. Delta-neutral structures sit at target ≈ 0, so a relative band triggers on every tick.

**Leg selection** (`_select_leg`): ATM = the two strikes nearest the forward, then whichever has **maximum open interest**; (delta, d) = the two strikes bracketing |Black delta| = d, then max OI again. The delta comes from **each strike's own mid IV**, not a fitted surface — deliberately, so selection is independent of whether the calibration was good that day. Tie-breaking on OI matters more than it looks: the nearest strike is often barely quoted, and picking it produces fills that never existed.

**Execution — where we deviate from the paper, on purpose:**

| | paper (Assumption 5.1) | this implementation |
|---|---|---|
| option entry | 50 bp on mid | **cross the real spread** (long buys the ask, short sells the bid) |
| option marking | — | **mid, every hedge hour** |
| perp | 5 bp of traded notional | same |
| settlement | — | intrinsic coin payoff, no spread |
| explicit fee | — | `option_fee_bps`, default 0, **on top** |

`execution="paper"` reproduces the paper's convention for comparison; `execution="spread"` is the default. Two consequences: the half-spread surfaces automatically the instant a position opens, and hourly mid-marking makes the NAV **continuous**, so Vol / Sharpe / MaxDD are meaningful rather than step-functions between rolls.

**Hedge P&L (Eq 26):** `(F_h − F_{h−1})/F_h × Δ^coin`. The `F_t^{Tk}/F_t` perp-vs-dated scaling of Eq 25 is deliberately omitted to stay identical to `engine.py` (a ~1e-3 funding-basis term for 7-day options). **Funding (Eq 29)** uses the realised hourly Deribit rate with the correct sign — a long perp *pays* when the rate is positive.

**Accounting:** `"coin"` (default, §5.1.2 Eqs 39–44, Deribit-native) or `"usd"` (Eqs 46–52). Under USD, option P&L is `S_t·V(t) − S_{t₀}·V(t₀)` and sizing is `N = Π^USD/S_t`.

**Reporting** ([roll_results.py](volatility_surface/backtest/roll_results.py)) reproduces the paper's Table 1 columns — Total, P.a./CAGR, annualised Vol, Sharpe, MaxDD, Skew, α_AN, β, R² — with daily log-returns of Coin NAV annualised by √365, zero risk-free rate, and α/β/R² from OLS of **weekly** strategy returns on weekly coin returns (α × 52, R² adjusted). The coin buy-and-hold is the first row by construction (β = 1, α = 0, R² = 1).

```bash
PYTHONPATH="/Users/macbookair/Internship Natixis" /opt/anaconda3/envs/natixis_internship/bin/python3 volatility_surface/backtest/run_roll.py --options volatility_surface/notebooks/clean_chunks_full --perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" --all --frequency weekly --coin ETH --funding "2 - Data/funding/eth_perp_funding_2024_2026.parquet" --csv results_weekly.csv
```

A full-catalog pass over the 2024–2026 archive is roughly an hour. [results_io.py](volatility_surface/backtest/results_io.py) exists so notebooks **read** those artefacts rather than recompute them: `load_table` for the summary CSV, `dump_run`/`load_run` for the full result dicts (NAV paths, event log, roll log) that a summary CSV cannot serve.

### 9b. The relative-value engine

**File:** [engine.py](volatility_surface/backtest/engine.py) · [results.py](volatility_surface/backtest/results.py) · [live_plot.py](volatility_surface/backtest/live_plot.py)

Every snapshot: calibrate a global eSSVI surface, re-price every listed option, flag a **red cell** where the model's Black-76 coin price falls outside that row's `[bid_price, ask_price]`. A **price-space** test, not IV-space — deliberately, so it matches exactly what a trader watching the live screen sees.

```
theo > ask  → 'buy'   (model says the market is too cheap)
theo < bid  → 'sell'  (model says the market is too rich)
```

Enter vega-sized ($1,000 vega) on each *new* red cell, delta-hedge the whole book with the perp via Net Delta, exit when the signal nulls, when a better opportunity appears at the same strike+expiry, or at expiry into intrinsic. Full eSSVI recalibration every 15 minutes.

Two controls are built in, and they are the reason the result is trustworthy:

- **`invert_signal=True`** reruns the identical selection with buy↔sell swapped, *without* flipping costs (spread-crossing is paid either way). If the inverted book doesn't lose roughly what the normal one makes, the P&L is transaction costs, not signal.
- **`model_name="svi"`** swaps eSSVI (3 params/slice) for RawSVI (5 params/slice). The drop in red-cell count is the part of the "opportunity" that was functional form rather than mispricing.

`_make_calibrator` is what makes that swap a one-argument change.

### 9c. The diagnostic modules

These were written *after* the backtests returned nothing, to find out why. They are arguably the most reusable output of the project.

| module | question | headline |
|---|---|---|
| [leverage.py](volatility_surface/backtest/leverage.py) | Is there a leverage effect in ETH, from the perp alone? | **ρ ≈ −0.21**, CI [−0.37, −0.04] |
| [vrp_signal.py](volatility_surface/backtest/vrp_signal.py) | Does (RV_past − IV) predict (RV_future − IV)? | No: corr +0.127, t = 1.25, R² 1.6% |
| [spot_vol_corr.py](volatility_surface/backtest/spot_vol_corr.py) | Implied vs realised spot–vol correlation; which vol does a hedger earn? | discretisation loss **3–6 vol points** |
| [regime_hmm.py](volatility_surface/backtest/regime_hmm.py) | Can a *causal* market-state signal replace the hand-drawn regimes? | Yes as a classifier; nothing to switch between |

**`leverage.py` — the four estimands, and why the distinction decides the answer:**

| estimand | verdict |
|---|---|
| **contemporaneous** corr(r_t, Δlog RV_t) | **mechanically contaminated** — r_t is inside RV_t, so the covariance is carried by E[r³]. This is a realised-*skewness* estimator wearing a correlation's clothes. Never lead with it. |
| **predictive** corr(r_t, Δlog RV_{t+1}) | **clean** — the honest correlation-based proxy |
| **signed variation** (RS⁺−RS⁻)/RV | thought to be best; **the Monte Carlo refuted it** (see §10) |
| **LHAR γ⁻** (Corsi–Reno) | **the primary test** — controls for vol persistence and separates the two return signs |

**`regime_hmm.py` — the filtered/smoothed distinction, enforced by naming:**

- **FILTERED** P(s_t | x_1..x_t) — uses only the past. Tradeable. → `walk_forward_states`, which also refits parameters on an expanding window so not even the fitted means and covariances contain the future.
- **SMOOTHED** P(s_t | x_1..x_T) — uses the whole sample. **Descriptive only.** → `smoothed_states`, deliberately named so it cannot be used by accident.

`hmmlearn.predict()` (Viterbi) is smoothed. A backtest built on it looks excellent and means nothing.

---

## 10. Everything the project measured

### Surface fitting

| finding | numbers |
|---|---|
| SABR vs eSSVI, absolute fit (ETH, ~10 dates) | SABR `vwrmse` ≈ **0.0065**, eSSVI ≈ **0.0103** |
| eSSVI `spread_hit_rate` | 0.608 with `iv_wmse`; **0.724** with `iv_zweighted(β=0.4)` |
| β grid (spread_hit_rate) | 0.0→0.684, 0.1→0.712, 0.2→0.708, 0.3→0.718, **0.4→0.725**, 0.5→0.713 |
| exact `iv_uniform_z` | 0.668 — **worse** than pure vega; the Gaussian approximation acts as regularisation |
| **but** β = 0.4 did not survive | the edge vanished on the full snapshot set → the live system uses plain `vega_wmse` |
| RW parabolic vs SABR (same param count) | **3–10× worse** on Deribit — right tool for a 3-quote FX market, wrong for a 30-strike crypto chain |
| γ on crypto | pegs at the GJ bound of 0.5 (the eSSVI paper found 0.42 on DJX) — expected given fatter crypto wings, not a bug |
| `speed_mode` fast vs thorough | equivalent in ~96% of cases; one real failure found (2026-06-12 12:45, ETH **and** BTC simultaneously) → thorough stays the default for n ≤ 25 |
| DE phases removed | benchmarked across every ETH/BTC snapshot: neither the global DE phase nor DE warm-start ever beat the NM warm-start, at ~10× the cost. Saves ~40 s per cold fit |

### Red cells (Mingone objective study)

Rate at which theo falls outside the quoted IV book, live ETH/BTC, 10 snapshots each, fitting the bid/ask mid:

| objective | ETH | BTC | note |
|---|---|---|---|
| IV space, vega² weight | **30.4%** | **35.7%** | current default (`vega2_mid`) |
| price space, ω = 1 | 31.0% | 34.1% | the paper's own §4.2 choice |
| price space, ω = 1/vega² | 46.3% | 53.2% | floated in §4.1 |
| IV space, unweighted | 49.3% | 54.6% | |
| IV space, vega¹ weight | 39.1% | 47.0% | the pre-2026-07 default |

Rows 1–2 agree to ~1 point, rows 3–4 to ~1–3 points — that is the first-order identity `C_mkt − C_model ≈ vega·(iv_obs − iv_fit)` confirmed numerically. **A constant-weight price objective IS a vega²-weighted IV objective.** `vega2_mid` is the default purely on cost (price-space pricing through the normal CDF on every LM Jacobian evaluation is 3–20× slower); use `price_flat` offline, where on BTC it is genuinely ~2 points better.

**The structural floor:** differential evolution minimising the red *count* directly, per-slice and unconstrained, cannot beat **~18% (ETH) / ~28% (BTC)** for eSSVI. RawSVI reaches ~2–5% / ~11–17% but is not arbitrage-free. **That gap is the price of the guarantee** — it is not recoverable by better calibration. eSSVI has 3 params/slice; crypto smiles need RawSVI's extra `m`/`a` freedom.

### Strategy results

**No variant produced a positive Sharpe ratio** — across 26 structures × ETH/BTC × weekly/monthly × coin/USD accounting, with realistic spread-crossing execution and realised funding. Several had positive *total returns*; so did holding the coin. The point of a delta-hedged vol strategy is that it shouldn't need the coin to go up, and none cleared that bar.

### The diagnosis, in three parts

**I — the strategies traded direction, not volatility.**

| structure | β vs coin | corr with spot |
|---|---|---|
| long call | **+0.13** | +0.81 |
| short call | **−0.13** | −0.80 |
| call-neutral (straddle, strangle, butterfly) | **−0.03** | — |

An hourly delta hedge removes delta. **It does not remove vanna.** Between hedges a delta-hedged book picks up roughly `vanna × Δσ × ΔS`, and Δσ is driven by ΔS through the spot–vol correlation.

**II — there was no premium to harvest.**

| quantity | value |
|---|---|
| mean ATM IV (30d constant maturity) | 62.9% |
| mean subsequent realised vol | 63.7% |
| **premium (IV − RV)** | **−0.82%** |
| t-statistic | **−0.44**, CI [−4.5%, +2.8%] vol pts |

Options were on average very slightly *cheap*. The timing rule fails too: (RV_past − IV) predicts (RV_future − IV) at corr +0.127, t = 1.25, R² 1.6%. ETH realised vol *is* persistent (weekly autocorrelation +0.27) — implied vol already knows.

**And the hedger cannot even earn the theoretical premium.** A delta-hedged position earns realised variance *at its hedging frequency*. Only close-to-close is monetisable; Parkinson / Garman-Klass / Rogers-Satchell read the bar's high and low, i.e. variation *inside* the hedging interval:

| estimator | vs close, ETH 2024–2026, 30d window, 1h bars |
|---|---|
| rv_close | 65.5% |
| parkinson | 68.8% (**+3.3**) |
| garman_klass | 70.0% (**+4.6**) |
| rogers_satchell | 71.9% (**+6.5**) |

State-dependent: near zero at 50% vol, ~15 points at the peaks. Against a theoretical premium of −0.82%, there was never a margin.

**III — the spot–vol correlation is real but weak.**

Sample: ETH perp, 1-min bars, 2024-06-01 → 2026-08-03, 1,142,674 bars, 794 daily observations, realised measures at 5-minute sampling (288 obs/day, 8.3% relative standard error). Seed 20260810. See [results/summary.md](results/summary.md).

```
LHAR (Corsi–Reno):
log RV_{t+1} = c + HAR(log RV) + γ_d·r⁻_t + γ_w·r⁻_{t-4:t} + γ_m·r⁻_{t-21:t} + γ_p·r⁺_t

γ_d = −8.75   HAC se 1.52   t = −5.76   p < 1e-8
γ_w = −8.40   HAC se 3.87   t = −2.17   p = 0.030
γ_p = +3.09   HAC se 1.17   t = +2.64   p = 0.008
R² = 0.376,  n = 772
```

A **−1% day raises next-day realised variance by 8.8%** (vol ~4.3%); a **+1% day raises it 3.1%**. Both raise vol — that is the magnitude effect — but **down moves raise it 2.8× more**, and that ratio is the leverage effect. News impact curve confirms the shape: mean +0.123 across down-return bins vs −0.110 across up bins.

**ρ_ETH ≈ −0.21, 95% CI [−0.37, −0.04]** — roughly **one third the strength of an equity index** (−0.60 to −0.80).

**Frequency matters more than the estimator.** Full-sample ETH 2024–2026:

| estimator | 1D bars | 1h bars |
|---|---|---|
| close-to-close | −0.09 | **−0.2191** |
| parkinson | −0.26 | −0.256 |
| garman_klass | −0.2952 | −0.2610 |
| rogers_satchell | −0.3149 | −0.265 |

All 1h values have t < −5. At daily bars close-to-close averages the effect away; at hourly it recovers most of it. **Never quote a spot–vol correlation without naming `bar_freq`.** The practical implication: `roll_engine`'s hourly hedge *does* capture the leverage effect — the mechanics were never the blocker.

**Honest reporting of what is *not* significant:** the E5 signed variation (−0.0025, p = 0.72), realised skewness (−0.0198, p = 0.68 — full-sample daily skew is in fact **+0.15**), jump-robust versions (BV −0.062, CI includes zero; MedRV −0.045, p = 0.13 — part of the effect lives in the jump component), volatility feedback at every horizon 1–10 days, and all regime differences. Across ~25 tests, the LHAR survives Holm correction comfortably; the headline correlation (p = 0.0059) **does not survive on its own**.

**Horizon term structure** — the effect peaks at the **daily** horizon and disappears beyond it. **That is the opposite of equities**, where it strengthens with horizon. Worth flagging as a genuine difference rather than smoothing over.

### The Monte Carlo that reframed everything

[results/montecarlo.md](results/montecarlo.md). Heston at 5-min steps for 794 days — matching the real sample exactly — 12 paths per ρ, run through the **same estimator functions** used on real data.

| true ρ | estimated corr(r_t, Δlog RV_{t+1}) | sd | E5 (RS⁺−RS⁻)/RV |
|---|---|---|---|
| −0.8 | −0.307 | 0.039 | **+0.0055** |
| −0.5 | −0.201 | 0.033 | +0.0016 |
| −0.2 | −0.073 | 0.034 | −0.0013 |
| 0.0 | +0.015 | 0.034 | −0.0015 |
| +0.3 | +0.128 | 0.029 | −0.0046 |

**Two conclusions, one of which contradicted the brief:**

1. **The correlation estimator passes, but is attenuated ~2.5×.** Monotone, tight (sd ≈ 0.03), correctly signed in 100% of paths at every negative ρ. Mean attenuation **0.399** — divide any observed value by ≈0.40 to read it as ρ. Intrinsic errors-in-variables, not a bug: daily RV is a noisy estimate of integrated variance and that noise is independent of the return.
2. **E5 signed variation is structurally blind to a diffusive leverage effect.** At true ρ = −0.8 it reads **+0.0055 — the wrong sign** — and varies by only 0.01 across the whole range with no monotone ordering. In a diffusion, returns are conditionally symmetric given the current variance level no matter how the variance process correlates with price; the leverage correlation lives in the *joint dynamics across time*, not the within-day sign split. E5 detects **asymmetric jumps**, a different mechanism. So the empirical E5 null is **not** evidence against a leverage effect. Relabel it as signed jump asymmetry.

**And the flat rolling-correlation plot was the correct answer.** At true ρ = −0.70, a 90-observation rolling estimate sits at −0.280 and **crosses zero 0 times across ~8,400 window-observations**. A genuinely strong leverage effect would never have produced the plot in question. At the true ρ ≈ −0.21 the expected rolling estimate is ≈ −0.085 with a per-window standard error of ≈ 0.107 — individual windows are uninformative and ±0.4 excursions are ordinary. **The rolling view is the wrong display for an effect this size.**

### Methodology worth reusing

Newey–West HAC standard errors (Bartlett, bandwidth `floor(4(n/100)^(2/9))`; 10 lags for the asymmetric regression, 22 for LHAR). Stationary block bootstrap (Politis–Romano, geometric blocks, 2000 replicates) for correlation CIs. Placebo null by shuffling returns while preserving both marginals. Estimators unit-tested against closed forms: RV/BV/MedRV recover σ² on GBM to within 0.8%; BV rises 46% against RV's 1016% on an injected 10% jump; signed variation is exactly 0.0000 on symmetric input.

---

## 11. Traps — read this before you debug anything

Consolidated from both notebooks' "Trials & errors" sections and the project memory. Every one of these produced *plausible numbers*, not an error.

### Modelling & calibration

1. **eSSVI fitting η/γ per-slice.** Not the paper's model — closer to per-slice RawSVI in disguise, and it gave eSSVI an unfair capacity advantage in every benchmark. Fixed 2026-07-20 to a single global power law. *The implementation had quietly drifted from the paper it claimed to implement.*
2. **Calendar penalty on θ-monotonicity alone.** Once ρ varies with θ, θ₂ ≥ θ₁ is **necessary but not sufficient** (Hendriks & Martini Prop 3.5) — the wing slopes θφ(1 ± ρ) must also be ordered. Replaced with real crossedness on a dense k-grid.
3. **A `1/spread²` term in `obj_iv_vega_wmse`.** Benchmarked on 24 real snapshots: degraded `vwrmse`/`iv_rrmse` on **24/24** with no reliable gain on `spread_hit_rate`. Removed.
4. **A `exp(k)=K` factor in `_bs_vega`.** Inconsistent with `pricing_models.gk_vega`; moved the vega peak to k* = 1.5σ²T instead of the standard 0.5σ²T. Only affected calibrations where no `vega_col=` was supplied.
5. **β = 0.4 over-fitted on ~10 ETH dates.** The edge vanished on the full snapshot set. **One snapshot is an anecdote, not a result** — this became the project's default working method.
6. **Hinge/band loss on [bid, ask] to minimise red cells is a trap.** Squared excess prefers many small violations over few large ones, so it optimises the wrong thing (31.8% → **54.2%**, even warm-started from a good fit). A saturating arctan/Cauchy soft-count doesn't work either — the Jacobian is zero wherever the residual is inside the band, so `least_squares` terminates at x₀ without moving. **Don't retry these.**
7. **Mistaking a fitting floor for opportunity.** eSSVI's 3 params/slice produce an 18–28% red-cell rate on ETH chains from functional form alone. Most flagged "mispricings" were the model, not the market.
8. **SVI-JW degeneracy.** At ρ ≈ 0 and ψ ≈ 0 the JW→raw map is singular; a tiny skew change produced a huge σ and a flat smile. Fixed with an explicit degeneracy guard + UI warning.

### Pricing & backtesting

9. **Hedging an inverse option with the plain Black delta.** Missing the −V/S adjustment gave every structure a slow systematic drift that read as alpha one way and cost the other. **The numeraire is part of the instrument.**
10. **Inverse-option units.** Black-76 returns USD; Deribit quotes in the coin. Every theo had to be divided by F — otherwise the whole chain flagged false arbitrage.
11. **Charging a flat 50 bp on mid.** The paper's Assumption 5.1 understates crypto option execution by an order of magnitude. Crossing the real spread moved several strategies from "marginal" to "clearly negative" — and hourly mid-marking made the NAV continuous enough for Sharpe and MaxDD to mean anything.
12. **A relative no-trade band.** Delta-neutral structures sit at target ≈ 0, so a relative band triggers on every tick. It must be absolute, in coin units, scaled by NAV.
13. **Accruing after the book changed.** Calling `_accrue` after a roll silently drops the last holding period's hedge P&L. **Ordering is load-bearing in a state machine.**

### Statistics & look-ahead

14. **Smoothed HMM states.** `hmmlearn.predict()` is Viterbi — it uses the whole sample. A backtest on it looks excellent and means nothing.
15. **Hand-drawn regimes.** `roll_engine.REGIMES` was labelled peak-to-trough from the *finished* price chart. "This strategy works in bear markets" built on that is circular. Causal replacements: `leverage.causal_regimes` (trailing 90-day drawdown) and `regime_hmm.walk_forward_states`. **Look-ahead arrives through the inputs you didn't think of as estimates.**
16. **Interpreting a correlation before calibrating the estimator.** Weeks spent reading a rolling spot–vol correlation that Monte Carlo later showed was attenuated 2.5× and, at n = 90, unable to distinguish −0.2 from zero. **The simulation should have come first.**
17. **The contemporaneous specification.** corr(r_t, ΔRV_t) puts r_t inside RV_t. Reads −0.093 vs −0.085 for the clean t+1 version — similar here, but they are different estimands and only the second is the leverage effect.
18. **`rv_window` semantics changed** in `spot_vol_corr.build` from a **count of bars** to **calendar days**. That change alone moved a measured correlation from −0.02 to −0.18 with nothing else altered, and cost a debugging session.

### Environment & tooling

19. **`matplotlib.use("Agg")` inside a plotting helper** switched the whole kernel headless, silently swallowing every figure.
20. **Wrong conda env.** The base env shadows the pinned one on PATH; symptom is a confusing `TypeError` deep inside the Streamlit UI. `run_live.sh` now hard-checks the Streamlit version and refuses to start.
21. **Streamlit `@st.cache_resource` evicts on file save**, forcing a 50 s recompute on every model switch. Fixed by keeping the warm-start fit in `st.session_state` keyed by `(currency, model)`.
22. **`implied_volatility_dataframe` returned a scalar `None`** on invalid quotes, collapsing the `df[["bid_iv","ask_iv"]]` assignment with *"Columns must be same length as key"* and freezing the gatherer. Fixed by always returning a `(nan, nan)` pair.
23. **`py_vollib_vectorized` is a dead end — don't retry.** Requires an ancient `numba`; any version new enough to have a Python 3.12 wheel fails to type-check the package's jitted internals. Worse, `vollib` and `py-lets-be-rational` are *different PyPI distributions that both ship a module named `lets_be_rational`* — installing or uninstalling one corrupts the other via on-disk file collision. Never have both installed.
24. **Editing an open `.ipynb` on disk.** Jupyter's autosave silently overwrites external changes. Cost two sessions.

---

## 12. Papers → code

| paper | file | what it gave us |
|---|---|---|
| Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces* — [`SVI Gatheral Jacquier.pdf`](Publications%20Natixis/) | `SSVI` in [vol_models.py](volatility_surface/models/vol_models.py), `calibrate_global_ssvi` | the SSVI surface, the η(1+\|ρ\|) ≤ 2 / γ ∈ (0, 0.5] conditions, the g(k) butterfly test |
| Hendriks & Martini, *eSSVI* (SSRN 2971502) — [`ssrn-2971502.pdf`](Publications%20Natixis/) | `eSSVI`, `calibrate_global_essvi` | ρ(θ) as an exponential family (§5.3.1); Prop 3.5 / Thm 4.1 calendar conditions |
| Mingone (2022), *No-arbitrage global parametrization for the eSSVI surface*, arXiv:2204.00312 | `GlobalESSVI`, `calibrate_global_essvi_mingone` | the (ρ, θ₁, a, c) box in which **every** point is arbitrage-free (Prop 3.1); the ρ ∈ ]−0.95, 0.95[ numerical bound (§4.4); the least_squares routine and its defaults (§4.1) |
| Hagan et al. (2002) + Obłój (2008) — [`Arbitrage Free SABR.pdf`](Publications%20Natixis/) | `SABR` | the lognormal expansion; the corrected z(K,F), a no-op at β = 1 |
| **Lucic & Sepp (2024), *Valuation and hedging of cryptocurrency inverse options*** (SSRN 4606748) — [`BTC_inverse_options.pdf`](Publications%20Natixis/) | `inverse_delta`, all of [roll_engine.py](volatility_surface/backtest/roll_engine.py), [roll_results.py](volatility_surface/backtest/roll_results.py) | **Corollary 1** (Net Delta Δ̃ = Δ − V/S); §5.2–5.3 structure catalog; Eq 26 hedge P&L; Eq 29 funding; Eqs 39–44 coin / 46–52 USD accounting; Assumption 5.1 costs; Table 1 metric definitions |
| Reiswich & Wystup (2010), CPQF WP 20 | [rw_parabolic.py](volatility_surface/models/rw_parabolic.py) | the simplified parabolic smile, Theorem 1 eqs (34)–(35) |
| Corsi & Renò (LHAR) | `leverage.lhar` | the leverage-HAR specification that became the primary test |
| Barndorff-Nielsen, Kinnebrock & Shephard | `leverage.e5_signed_variation` | signed variation — measured, and shown to answer a different question |

---

## 13. Open threads, ranked

**From the strategies work (highest expected value first):**

1. **Build the constant-maturity, constant-moneyness IV surface and regress ΔIV on Δlog F.** Compare the *implied* ρ to the −0.21 measured from returns. If implied is materially more negative, the **skew risk premium** is the tradeable object — not the level. §13 of the strategies notebook gives the baseline. This is the highest-value remaining measurement, and the one thing the data can currently support.
2. **Re-run the roll strategies with vanna hedged explicitly**, now that the correlation driving it is quantified. The one modification with a measured rationale behind it.
3. **Repeat the leverage study on BTC.** Sign agreement across two assets is worth more than another ETH robustness check.

**From the surface library:**

4. **`speed_mode="gradient"` — structural, not penalty-based, arb-free eSSVI.** The idea: build θ_i via cumulative softplus (monotone by construction), bound η via the analytic §5.4 butterfly bound instead of numerical `min_g`, and enforce the §5.3.1 ρ-family admissibility inequality directly. The objective then reduces to pure vega²-weighted fit error — smooth (C^∞ where w > 0) — so L-BFGS-B or SLSQP becomes principled, replacing the NM+penalty `"thorough"` path. Still non-convex (needs a warm start); the butterfly bound's `min(·,·)` gives SLSQP a kink, but that is a valid constraint form. Benchmark against the current path on fit quality / measured arb-freedom / wall-clock.
5. **Parallelise the SABR warm-start** (per-slice, embarrassingly parallel) to reach 1 Hz on the live screen.
6. **Re-invert bid/ask IVs under each pricing engine** for a fully consistent model bake-off.
7. **Extend the RW parabola** (cubic/quartic in delta) to test whether the crypto wing miss is one free parameter away from being fixed.

**Documentation debt:**

8. The "3 global + 3 per slice" eSSVI description in [project_outline.ipynb](volatility_surface/notebooks/project_outline.ipynb) §5 and [README.md](README.md) is stale — it is 5 global + n per-slice.
9. `results_io` expects a `Backtest results/` directory at the project root (overridable with `$ROLL_RESULTS_DIR`) that does not exist on this laptop.

**Dead ends — documented so nobody re-runs them:**

- eSSVI's second power-law family from Hendriks & Martini: mathematically equivalent to the first when η/γ are fitted per slice. No gain. Not implemented.
- Tightening the Mingone butterfly bound from GJ to Martini–Mingone: the box already costs ~nothing.
- Hinge/band and soft-count losses for red cells (see trap #6).
- `py_vollib_vectorized` (see trap #23).
- `%matplotlib widget` — too laggy, already reverted.

---

## 14. Complete file index

### Models & calibration

| file | role |
|---|---|
| [models/vol_models.py](volatility_surface/models/vol_models.py) | `VolModel` ABC, `RawSVI`, `SSVI`, `eSSVI`, `GlobalESSVI`, `SABR`, `MODELS`/`get_model` |
| [models/rw_parabolic.py](volatility_surface/models/rw_parabolic.py) | Reiswich–Wystup parabolic; `fit_paper_style`, `fit_least_squares`, `w_of_k` |
| [core/calibration/calibrator.py](volatility_surface/core/calibration/calibrator.py) | every `calibrate_*`, `fit_single_slice`, `load_snapshot`, `crossedness`, `enforce_calendar_arbfree`, `summary`, `compare` |
| [core/calibration/objectives.py](volatility_surface/core/calibration/objectives.py) | `OBJECTIVES`, `METRICS`, `DEFAULT_METRICS`, `evaluate_all`, `optimize_beta` |

### Pricing

| file | role |
|---|---|
| [core/pricing/pricing_models.py](volatility_surface/core/pricing/pricing_models.py) | BS / GK / Black-76, greeks, `inverse_delta`, vectorised `gk_implied_vol`, vanna-volga |
| [core/pricing/volatility_dataframe.py](volatility_surface/core/pricing/volatility_dataframe.py) | `add_bid_ask_iv` (polars-native), `implied_volatility_dataframe` |

### Data pipeline

| file | role |
|---|---|
| [utils/data_handling.py](volatility_surface/utils/data_handling.py) | `clean_df`, `clean_df_pl`, `clean_bulk_parquet_chunked`, `parquet_creation`, `load_uncleaned` |
| [utils/option_data_gathering.py](volatility_surface/utils/option_data_gathering.py) | Deribit REST helpers |
| [utils/bulk_concat.py](volatility_surface/utils/bulk_concat.py) | ~170k CSV → one parquet, batched + parallel + streaming |
| [utils/validate_dataset.py](volatility_surface/utils/validate_dataset.py) | pre-backtest sanity check ([WARN]/[FAIL]) |
| [utils/deribit_perp.py](volatility_surface/utils/deribit_perp.py) | perp OHLCV history (the Eq 26 price path) |
| [utils/deribit_funding.py](volatility_surface/utils/deribit_funding.py) | realised hourly funding (the Eq 29 carry) |

### Live system

| file | role |
|---|---|
| [utils/live_gather.py](volatility_surface/utils/live_gather.py) | 1 Hz REST + WS poller → rolling parquet (atomic rename) |
| [utils/streamlit_chain.py](volatility_surface/utils/streamlit_chain.py) | the trading screen (eSSVI / SABR / SVI selectable) |
| [utils/streamlit_svi.py](volatility_surface/utils/streamlit_svi.py) | interactive SVI / SVI-JW explorer with RND overlay |
| [utils/chain_gif.py](volatility_surface/utils/chain_gif.py) | headless chain-table → GIF recorder |
| [run_live.sh](run_live.sh) | one-command launcher (+ env guard) |

### Analysis & benchmarks

| file | role |
|---|---|
| [utils/arbitrage_checker.py](volatility_surface/utils/arbitrage_checker.py) | `check_arbitrage`, `plot_arbitrage_map` |
| [plots/vol_plots.py](volatility_surface/plots/vol_plots.py) | every calibration figure |
| [utils/benchmark_objectives.py](volatility_surface/utils/benchmark_objectives.py) | 10-objective sweep on global eSSVI → `.pkl` + `.png` |
| [utils/benchmark_penalty_cal.py](volatility_surface/utils/benchmark_penalty_cal.py) | `penalty_cal` sweep across four decades for SVI/SABR |
| [utils/benchmark_rw_parabolic.py](volatility_surface/utils/benchmark_rw_parabolic.py) | RW vs {SVI, SSVI, eSSVI, SABR} |
| [utils/make_calibration_gifs.py](volatility_surface/utils/make_calibration_gifs.py) | NM vs DE convergence animations for slides |

### Backtesting

| file | role |
|---|---|
| [backtest/roll_engine.py](volatility_surface/backtest/roll_engine.py) | systematic engine, structure catalog, roll grids, state machine |
| [backtest/roll_results.py](volatility_surface/backtest/roll_results.py) | paper-style Table 1, attribution, regime splits, `execution_table` |
| [backtest/run_roll.py](volatility_surface/backtest/run_roll.py) | CLI for the above |
| [backtest/results_io.py](volatility_surface/backtest/results_io.py) | load results from disk instead of recomputing |
| [backtest/engine.py](volatility_surface/backtest/engine.py) | red-cell relative-value engine |
| [backtest/signal.py](volatility_surface/backtest/signal.py) | `bs_pricing`, `detect_red_cells`, `implied_carry_rate`, `detect_calendar_arb` |
| [backtest/results.py](volatility_surface/backtest/results.py) | event-log aggregation for `engine.py` |
| [backtest/live_plot.py](volatility_surface/backtest/live_plot.py) | incremental live P&L chart in a Jupyter cell |
| [backtest/leverage.py](volatility_surface/backtest/leverage.py) | the full leverage-effect study (LHAR, bootstrap, Heston MC, causal regimes) |
| [backtest/vrp_signal.py](volatility_surface/backtest/vrp_signal.py) | volatility risk premium and its predictability |
| [backtest/spot_vol_corr.py](volatility_surface/backtest/spot_vol_corr.py) | implied vs realised spot–vol correlation; the level/discretisation view |
| [backtest/regime_hmm.py](volatility_surface/backtest/regime_hmm.py) | hand-rolled Gaussian HMM with filtered/smoothed separation enforced |

### Outputs

| path | contents |
|---|---|
| [results/summary.md](results/summary.md) | the leverage-effect study, written up |
| [results/montecarlo.md](results/montecarlo.md) | the estimator calibration — **read before any empirical number** |
| [results/tables/](results/tables/) | 13 CSVs: signature plot, E1/E2/E3/E5, asymmetric regression, LHAR, horizon term structure, influence, spike-decay, causal regimes, MC recovery, placebo null |
| [results/figures/](results/figures/) | `leverage_shape.png` (news impact curve), `horizon_term_structure.png` |
| [calibration_results/](calibration_results/) | pickled calibration runs by model/date/objective |
| [presentation_assets/](presentation_assets/) | slide figures and GIFs (surface comparisons, DE-vs-NM convergence, live chain recording, Pareto, architecture) |
| `benchmark_*.pkl` / `.png` (root) | benchmark outputs |

---

## 15. The one-line version

> Every wrong answer in this project came from a component that could not see what it was being asked about — a hedge ratio missing a numeraire term, a cost model an order of magnitude too small, a band defined against the wrong quantity, a fitting floor mistaken for opportunity, a label that encoded the future, an estimator attenuated 2.5× that nobody had calibrated. **None of them announced itself as an error. They all produced plausible numbers.**

---

*Draft — last updated 2026-08-17. Update §13 as threads close.*
