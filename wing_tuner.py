"""
wing_tuner.py
=============
Grid-search over the ATM weight multiplier in make_bid_ask_loss to find the
value that:

  1. ATM constraint  : fitted IV stays within bid-ask for all options with
                       |k| <= atm_threshold  (hard requirement)
  2. Wing objective  : minimise mean violation outside bid-ask for options
                       with |k| > atm_threshold  (soft objective)

The search is two-stage:
  - Coarse pass over a wide grid (fast, finds the ballpark)
  - Fine pass over a narrow interval around the coarse winner

Speed strategy
--------------
Only the first evaluation runs a full cold-start calibration.
Every subsequent evaluation warm-starts from the previous result via
calibrate_global_essvi_update(mode="full_polish"), which takes ~1–3 s
instead of ~40–50 s and produces fits of identical quality.
Set warm_start=False to force independent cold starts (slower but fully
decoupled — useful for debugging or benchmarking).

ATM model floor
---------------
If no weight satisfies the ATM constraint, the report will show the
ATM floor: the minimum violation the model can achieve given its structural
constraints (global rho, arbitrage-free butterfly).  For eSSVI this is
typically 1–5 vol points when bid-ask spreads are very tight.  Increasing
atm_weight beyond the point where the curve flattens will not improve ATM
fit further — the residual is irreducible by any objective weighting.

Usage
-----
    from wing_tuner import tune_wing_weight

    best_ww, best_result, summary = tune_wing_weight(
        snapshot,
        bid_col      = "bid_iv",
        ask_col      = "ask_iv",
        vega_col     = "vega",
        atm_threshold = 0.2,
    )
    print(f"Optimal wing_weight: {best_ww:.3f}")

The returned best_result is a standard calibration result dict that can be
passed directly to any plotting or comparison function.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Callable

from objectives  import make_zone_wmse, make_bid_ask_loss
from calibrator  import calibrate_global_essvi, calibrate_global_essvi_update
from vol_models  import eSSVI


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_violations(
    df:          pd.DataFrame,
    result:      dict,
    bid_col:     str,
    ask_col:     str,
    atm_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each option in the snapshot compute how far the fitted IV lies outside
    the bid-ask spread:
        violation_i = max(0, iv_fit - iv_ask) + max(0, iv_bid - iv_fit)
    Zero means the fit is inside the spread.

    Returns two arrays: (atm_violations, wing_violations).
    """
    model = result["_model"]
    atm_v, wing_v = [], []

    for t_exp, params in zip(result["expiries"], result["params"]):
        sub    = df[df["t"] == t_exp]
        k      = sub["k"].values
        iv_fit = model.iv(k, params, t_exp)

        bid = sub[bid_col].values.copy()
        ask = sub[ask_col].values.copy()
        # Normalise if stored as percentage (>5 is a safe sentinel for raw %)
        if bid.max() > 5:
            bid /= 100.0
        if ask.max() > 5:
            ask /= 100.0

        viol     = np.maximum(0.0, iv_fit - ask) + np.maximum(0.0, bid - iv_fit)
        atm_mask = np.abs(k) <= atm_threshold

        atm_v.extend(viol[atm_mask].tolist())
        wing_v.extend(viol[~atm_mask].tolist())

    return np.array(atm_v), np.array(wing_v)


def _score(atm_arr: np.ndarray, wing_arr: np.ndarray) -> dict:
    """Summarise violation arrays into scalar metrics."""
    return {
        "atm_max_viol":      float(np.max(atm_arr))          if len(atm_arr)  > 0 else 0.0,
        "atm_mean_viol":     float(np.mean(atm_arr))          if len(atm_arr)  > 0 else 0.0,
        "atm_frac_outside":  float(np.mean(atm_arr  > 1e-4))  if len(atm_arr)  > 0 else 0.0,
        "wing_max_viol":     float(np.max(wing_arr))           if len(wing_arr) > 0 else 0.0,
        "wing_mean_viol":    float(np.mean(wing_arr))          if len(wing_arr) > 0 else 0.0,
        "wing_frac_outside": float(np.mean(wing_arr > 1e-4))   if len(wing_arr) > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_weight(
    wing_weight:   float,
    df:            pd.DataFrame,
    model,
    calibrate_fn:  Callable,
    bid_col:       str,
    ask_col:       str,
    vega_col:      str,
    atm_threshold: float,
    prev_result:   dict = None,
) -> Tuple[dict, dict, float]:
    """
    Calibrate with one atm_weight value using make_bid_ask_loss and return
    (scores, result, elapsed).

    The `wing_weight` parameter controls the ATM zone multiplier
    (atm_weight in make_bid_ask_loss): higher values force the optimizer to
    prioritise staying within bid-ask near ATM over the wings.

    If `prev_result` is provided the calibration warm-starts from those
    parameters using calibrate_global_essvi_update(mode="full_polish"),
    cutting run time from ~45 s to ~1–3 s per evaluation while matching
    cold-start fit quality.
    """
    obj = make_bid_ask_loss(atm_threshold=atm_threshold, atm_weight=wing_weight)
    t0  = time.time()

    if prev_result is not None:
        try:
            r = calibrate_global_essvi_update(
                df, prev_result,
                objective=obj,
                mode="full_polish",
                vega_col=vega_col,
                bid_col=bid_col,
                ask_col=ask_col,
                verbose=False,
            )
        except Exception:
            # Graceful fallback to cold start if update path fails
            r = calibrate_fn(df, model=model, objective=obj,
                             vega_col=vega_col, bid_col=bid_col, ask_col=ask_col,
                             verbose=False)
    else:
        r = calibrate_fn(df, model=model, objective=obj,
                         vega_col=vega_col, bid_col=bid_col, ask_col=ask_col,
                         verbose=False)

    elapsed = time.time() - t0

    atm_arr, wing_arr = _compute_violations(df, r, bid_col, ask_col, atm_threshold)
    scores = _score(atm_arr, wing_arr)
    return scores, r, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TUNER
# ─────────────────────────────────────────────────────────────────────────────

def tune_wing_weight(
    df:             pd.DataFrame,
    bid_col:        str   = "bid_iv",
    ask_col:        str   = "ask_iv",
    vega_col:       str   = "vega",
    atm_threshold:  float = 0.2,
    coarse_grid:    Optional[List[float]] = None,
    n_fine:         int   = 5,
    atm_tol:        float = 1e-4,
    model                 = None,
    calibrate_fn:   Optional[Callable] = None,
    warm_start:     bool  = True,
    verbose:        bool  = True,
) -> Tuple[float, dict, pd.DataFrame]:
    """
    Two-stage grid search for the optimal ATM weight multiplier in
    make_bid_ask_loss.

    Stage 1 — Coarse pass
        Evaluates a wide grid.  Default: [1, 2, 3, 5, 8, 12, 20].
        Values > 1 give ATM options proportionally more weight than wings.
        Values < 1 give wings more weight — rarely useful for the ATM constraint.

    Stage 2 — Fine pass
        Narrows to n_fine points around the coarse winner.

    Speed
        The first evaluation does a full cold-start calibration.
        All subsequent evaluations warm-start from the previous result via
        calibrate_global_essvi_update(mode="full_polish"), reducing each
        step from ~40-50 s to ~1-3 s.  Set warm_start=False to force
        independent cold starts (useful for benchmarking).

    ATM model floor
        If no weight satisfies the ATM constraint, the report shows the
        minimum achievable violation alongside the bid-ask spread widths.
        A flat ATM curve (no improvement past a certain weight) indicates
        the model has reached its structural floor — the residual is
        irreducible by any objective weighting.

    Selection rule
        Among weights where atm_max_viol <= atm_tol, pick the one with the
        smallest wing_mean_viol.
        If none satisfy the ATM constraint, return the weight with the
        smallest atm_max_viol.

    Parameters
    ----------
    df             : snapshot DataFrame (output of load_snapshot)
    bid_col        : column name for bid implied vol (same units as mark_iv)
    ask_col        : column name for ask implied vol
    vega_col       : column name for precomputed BS vega
    atm_threshold  : log-strike boundary between ATM and wing zones
    coarse_grid    : explicit list of atm_weight values for the coarse pass
                     (default: [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    n_fine         : extra points in the fine pass (default 5)
    atm_tol        : max acceptable ATM violation in implied vol units
    model          : VolModel instance  (default: eSSVI())
    calibrate_fn   : calibration function  (default: calibrate_global_essvi)
    warm_start     : warm-start each evaluation from the previous result
    verbose        : print progress table

    Returns
    -------
    best_weight    : float — optimal wing_weight
    best_result    : dict  — calibration result dict for the optimal weight
    summary        : pd.DataFrame — all evaluated (wing_weight, metrics) rows,
                     sorted by wing_weight, with a 'selected' boolean column
    """
    # ── Defaults ──────────────────────────────────────────────────────────────
    if coarse_grid is None:
        # ATM weight multiplier: >1 means ATM options get proportionally more
        # weight than wings.  Start at 1 (equal) and go up to 20.
        coarse_grid = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    if model is None:
        model = eSSVI()
    if calibrate_fn is None:
        calibrate_fn = calibrate_global_essvi

    # ── Validate columns ──────────────────────────────────────────────────────
    for col in [bid_col, ask_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    # ── Bid-ask spread statistics (computed once for context) ──────────────────
    bid_vals = df[bid_col].values.copy().astype(float)
    ask_vals = df[ask_col].values.copy().astype(float)
    if bid_vals.max() > 5:
        bid_vals /= 100.0
    if ask_vals.max() > 5:
        ask_vals /= 100.0
    spread_vals  = ask_vals - bid_vals
    atm_mask_df  = np.abs(df["k"].values) <= atm_threshold
    sp_atm       = spread_vals[atm_mask_df]
    sp_wing      = spread_vals[~atm_mask_df]

    # ── Printing helpers ──────────────────────────────────────────────────────
    HDR = (f"{'ww':>6}  {'ATM_max':>9}  {'ATM_%out':>8}  "
           f"{'Wing_mean':>9}  {'Wing_%out':>9}  {'time':>6}")

    def _print_row(ww, sc, elapsed, tag=""):
        print(f"  {ww:>6.3f}  {sc['atm_max_viol']:>9.4f}  "
              f"{sc['atm_frac_outside']*100:>7.1f}%  "
              f"{sc['wing_mean_viol']:>9.4f}  "
              f"{sc['wing_frac_outside']*100:>8.1f}%  "
              f"{elapsed:>5.1f}s  {tag}")

    if verbose:
        print(f"\n{'═'*72}")
        print(f"  Bid-ask hinge loss tuner  |  atm_threshold={atm_threshold}")
        print(f"  ATM zone : |k| ≤ {atm_threshold}   Wing zone : |k| > {atm_threshold}")
        print(f"  Objective : make_bid_ask_loss  (zero inside spread, hinge² outside)")
        print(f"  ATM tolerance : {atm_tol}  ({atm_tol*100:.2f} vol points)")
        print(f"  Warm-start    : {'on — full_polish after first cold start' if warm_start else 'off — independent cold starts'}")
        print(f"{'─'*72}")
        print(f"  Bid-ask spread — ATM  (|k|≤{atm_threshold}):  "
              f"mean={sp_atm.mean():.4f}  median={np.median(sp_atm):.4f}  "
              f"min={sp_atm.min():.4f}  ({len(sp_atm)} pts)")
        print(f"  Bid-ask spread — Wings (|k|>{atm_threshold}):  "
              f"mean={sp_wing.mean():.4f}  median={np.median(sp_wing):.4f}  "
              f"min={sp_wing.min():.4f}  ({len(sp_wing)} pts)")
        print(f"  ┄  ATM violation floor ≈ spread half-width = {sp_atm.mean()/2:.4f}")
        print(f"  ┄  If ATM_max stays above {sp_atm.min():.4f} across all weights,")
        print(f"  ┄  the model has reached its structural floor — not a tuning issue.")
        print(f"{'═'*72}")

    # ── Stage 1: coarse pass ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  Stage 1 — Coarse pass ({len(coarse_grid)} evaluations)")
        print(f"  {HDR}")
        print(f"  {'─'*len(HDR)}")

    all_rows, all_results = [], {}
    last_result = None   # warm-start chain

    for ww in coarse_grid:
        sc, r, elapsed = _evaluate_weight(
            ww, df, model, calibrate_fn, bid_col, ask_col, vega_col, atm_threshold,
            prev_result=last_result if warm_start else None,
        )
        if warm_start:
            last_result = r
        all_results[ww] = r
        row = {"wing_weight": ww, **sc, "elapsed_s": elapsed}
        all_rows.append(row)
        if verbose:
            tag = "✓ ATM ok" if sc["atm_max_viol"] <= atm_tol else ""
            _print_row(ww, sc, elapsed, tag)

    # ── Stage 2: fine pass around the coarse winner ───────────────────────────
    coarse_df   = pd.DataFrame(all_rows)
    clean_c     = coarse_df[coarse_df["atm_max_viol"] <= atm_tol]
    coarse_best = (clean_c["wing_mean_viol"].idxmin()
                   if len(clean_c) > 0
                   else coarse_df["atm_max_viol"].idxmin())
    best_ww_c   = float(coarse_df.loc[coarse_best, "wing_weight"])

    # Build fine grid: n_fine points between the neighbours of the coarse winner
    idx_c     = coarse_grid.index(best_ww_c)
    lo        = coarse_grid[max(0, idx_c - 1)]
    hi        = coarse_grid[min(len(coarse_grid) - 1, idx_c + 1)]
    fine_grid = [w for w in np.linspace(lo, hi, n_fine + 2)[1:-1]
                 if not any(abs(w - existing) < 1e-6
                            for existing in coarse_grid)]

    if verbose and fine_grid:
        print(f"\n  Stage 2 — Fine pass ({len(fine_grid)} evaluations, "
              f"ww ∈ [{lo:.3f}, {hi:.3f}])")
        print(f"  {HDR}")
        print(f"  {'─'*len(HDR)}")

    # Fine pass warm-start chain begins from the coarse winner result
    last_result = all_results[best_ww_c]

    for ww in fine_grid:
        sc, r, elapsed = _evaluate_weight(
            ww, df, model, calibrate_fn, bid_col, ask_col, vega_col, atm_threshold,
            prev_result=last_result if warm_start else None,
        )
        if warm_start:
            last_result = r
        all_results[ww] = r
        row = {"wing_weight": ww, **sc, "elapsed_s": elapsed}
        all_rows.append(row)
        if verbose:
            tag = "✓ ATM ok" if sc["atm_max_viol"] <= atm_tol else ""
            _print_row(ww, sc, elapsed, tag)

    # ── Select best ───────────────────────────────────────────────────────────
    summary = pd.DataFrame(all_rows).sort_values("wing_weight").reset_index(drop=True)

    clean = summary[summary["atm_max_viol"] <= atm_tol]

    if len(clean) == 0:
        best_idx = int(summary["atm_max_viol"].idxmin())
        atm_ok   = False
    else:
        best_idx = int(clean["wing_mean_viol"].idxmin())
        atm_ok   = True

    best_ww     = float(summary.loc[best_idx, "wing_weight"])
    best_result = all_results[best_ww]
    summary["selected"] = summary["wing_weight"] == best_ww

    # ── Report ────────────────────────────────────────────────────────────────
    if verbose:
        atm_floor = summary["atm_max_viol"].min()
        print(f"\n{'═'*72}")
        if not atm_ok:
            print(f"  ⚠  No weight satisfies ATM constraint (atm_tol={atm_tol}).")
            print(f"     ATM violation floor = {atm_floor:.4f}  "
                  f"(min bid-ask half-width = {sp_atm.min()/2:.4f})")
            if atm_floor > sp_atm.min() / 2:
                print(f"     The floor exceeds the tightest spread — this is a model")
                print(f"     limitation (global rho + arbitrage constraints).  Consider:")
                print(f"     • calibrate_snapshot() for per-slice fits (more flexible)")
                print(f"     • widening atm_tol to {atm_floor*1.05:.4f}")
            print(f"     Returning weight with minimum ATM violation.")
        print(f"  Optimal wing_weight  = {best_ww:.4f}")
        print(f"  ATM  max violation   = {summary.loc[best_idx,'atm_max_viol']:.4f}  "
              f"({summary.loc[best_idx,'atm_frac_outside']*100:.1f}% of ATM pts outside)")
        print(f"  Wing mean violation  = {summary.loc[best_idx,'wing_mean_viol']:.4f}  "
              f"({summary.loc[best_idx,'wing_frac_outside']*100:.1f}% of wing pts outside)")
        print(f"{'═'*72}\n")

    return best_ww, best_result, summary


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PLOT  (optional helper)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tuning_summary(summary: pd.DataFrame, atm_tol: float = 1e-4):
    """
    Plot ATM and wing violations as a function of wing_weight.
    Shades the feasible region where ATM constraint is satisfied.
    Marks the selected weight with a vertical line.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Wing-weight tuning: bid-ask violations vs wing_weight", fontsize=13)

    ww    = summary["wing_weight"].values
    sel   = summary["selected"].values
    clean = summary["atm_max_viol"].values <= atm_tol

    for ax, y_col, ylabel, title in [
        (axes[0], "atm_max_viol",    "Max violation (IV units)", "ATM zone  |k| ≤ threshold"),
        (axes[1], "wing_mean_viol",  "Mean violation (IV units)", "Wing zone  |k| > threshold"),
    ]:
        y = summary[y_col].values

        # Feasible region shading
        if clean.any():
            ax.axvspan(ww[clean].min(), ww[clean].max(),
                       color="green", alpha=0.08, label="ATM constraint satisfied")

        ax.plot(ww, y, "o-", color="#4c9be8", lw=2, ms=6, label=y_col)

        # ATM tolerance line on ATM plot
        if y_col == "atm_max_viol":
            ax.axhline(atm_tol, color="red", ls="--", lw=1, label=f"tolerance={atm_tol}")

        # Selected weight
        best_ww = float(summary.loc[sel, "wing_weight"].values[0])
        ax.axvline(best_ww, color="orange", ls="--", lw=1.5, label=f"best ww={best_ww:.3f}")
        ax.scatter([best_ww], [summary.loc[sel, y_col].values[0]],
                   color="orange", s=100, zorder=5)

        ax.set_xlabel("wing_weight")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
