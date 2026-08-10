"""
Spot-vol correlation on the option history — the quantity behind the skew, and
behind the residual beta that turned out to drive the roll-strategy results.

Two correlations, deliberately kept separate because they answer different
questions:

    IMPLIED   corr(spot return, change in ATM implied vol)
              How the market REPRICES vol when spot moves. This is what the
              skew embeds, and what makes a call-exposed structure carry vanna
              (delta-hedging removes delta, not the change in vega with spot).

    REALISED  corr(spot return, change in realised vol)
              The actual leverage effect in the returns themselves.

A gap between the two is informative: implied more negative than realised means
the market pays more for downside vol protection than the underlying's own
behaviour warrants.

`--levels` switches from the correlation view to the LEVEL view: every estimator's
annualised vol through time, spot on the twin axis. Different question, same panel:

    iv - rv_close             the premium an hourly hedger can actually earn.
    rv_range - rv_close       the DISCRETISATION LOSS — variation inside the
                              hedging interval, which close-to-close cannot see
                              and a hedger rebalancing at `bar_freq` never
                              monetises. On ETH 2024-2026 this runs +3.3 vol
                              points (Parkinson) to +6.5 (Rogers-Satchell), and
                              it is state-dependent: near zero at 50% vol,
                              ~15 points at the peaks.

The implied series is built at CONSTANT MATURITY. Taking "the front expiry's ATM
vol" instead produces a sawtooth as maturity decays and then jumps at each roll,
and the correlation would then partly measure that artefact rather than the
market. Interpolation is done in total variance w = sigma^2 * T, linear in T,
which is the standard no-arbitrage-friendly choice.

    python volatility_surface/backtest/spot_vol_corr.py \
        --options "2 - Data/parquets/eth_hourly_2024.parquet" \
        --perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet" \
        --out spot_vol_corr.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl


# --------------------------------------------------------------------------- #
# Constant-maturity ATM implied vol
# --------------------------------------------------------------------------- #
def atm_iv_constant_maturity(option_path: str, target_days: float = 30.0,
                             resample: str = "1D") -> pd.Series:
    """ATM implied vol at a fixed `target_days` maturity, one point per
    `resample` bucket.

    ATM is taken as the strike nearest the forward, averaging the call and put
    quotes (equal by parity, and averaging halves the quote noise). Expiries
    bracketing the target are interpolated in total variance; if the target is
    outside the listed range the nearest expiry is used flat, which is honest
    for the very front of the curve where nothing shorter exists.
    """
    cols = ["creation_timestamp_x", "expiration_timestamp_ms", "strike",
            "option_type", "underlying_price", "bid_iv", "ask_iv", "t"]
    df = pl.read_parquet(option_path, columns=cols).to_pandas()
    df["iv"] = df[["bid_iv", "ask_iv"]].mean(axis=1)
    df = df[np.isfinite(df["iv"]) & (df["iv"] > 0)]
    df["dt"] = pd.to_datetime(df["creation_timestamp_x"], unit="ms", utc=True)
    df["bucket"] = df["dt"].dt.floor(resample)
    # one snapshot per bucket — the first, so the series is on a clean grid
    keep = df.groupby("bucket")["creation_timestamp_x"].transform("min")
    df = df[df["creation_timestamp_x"] == keep]

    out = {}
    stamps = {}
    tgt = target_days / 365.0
    for bucket, snap in df.groupby("bucket"):
        stamps[bucket] = int(snap["creation_timestamp_x"].iloc[0])
        # ATM vol per expiry
        per_exp = []
        for _, sl in snap.groupby("expiration_timestamp_ms"):
            F = float(sl["underlying_price"].iloc[0])
            T = float(sl["t"].iloc[0])
            if T <= 0:
                continue
            near = sl.loc[(sl["strike"] - F).abs().nsmallest(2).index]
            if near.empty:
                continue
            per_exp.append((T, float(near["iv"].mean())))
        if not per_exp:
            continue
        per_exp.sort()
        Ts = np.array([p[0] for p in per_exp])
        ivs = np.array([p[1] for p in per_exp])
        if len(Ts) == 1 or tgt <= Ts[0] or tgt >= Ts[-1]:
            out[bucket] = float(ivs[np.argmin(np.abs(Ts - tgt))])
            continue
        j = int(np.searchsorted(Ts, tgt))
        T0, T1, v0, v1 = Ts[j - 1], Ts[j], ivs[j - 1], ivs[j]
        w0, w1 = v0 ** 2 * T0, v1 ** 2 * T1          # interpolate TOTAL VARIANCE
        w = w0 + (w1 - w0) * (tgt - T0) / (T1 - T0)
        out[bucket] = float(np.sqrt(max(w, 0.0) / tgt))
    ser = pd.Series(out, name=f"iv_{int(target_days)}d").sort_index()
    # Carry the ACTUAL snapshot instant, not just the bucket it fell in: the spot
    # has to be sampled at exactly these times or the return and the IV change
    # cover different intervals. Resampling the perp independently to the same
    # calendar put the two ~24h apart and collapsed a strong injected leverage
    # effect (-1.2 coefficient) to a measured correlation of -0.04.
    ser.attrs["stamps"] = pd.Series(stamps).sort_index().reindex(ser.index)
    return ser


def perp_at_times(perp_path: str, stamps_ms: pd.Series) -> pd.Series:
    """Perp close AT OR BEFORE each timestamp — same never-look-ahead rule as
    `roll_engine._perp_at`, vectorised."""
    d = pd.read_parquet(perp_path, columns=["timestamp_ms", "close"]).sort_values("timestamp_ms")
    ts, px = d["timestamp_ms"].to_numpy(), d["close"].to_numpy()
    idx = np.searchsorted(ts, stamps_ms.to_numpy(), side="right") - 1
    return pd.Series(np.where(idx >= 0, px[np.clip(idx, 0, None)], np.nan),
                     index=stamps_ms.index, name="spot")





# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
def realised_vol_by_estimator(perp_path: str, index: pd.DatetimeIndex,
                              estimators: tuple[str, ...], window_days: int = 30,
                              bar_freq: str = "1h") -> pd.DataFrame:
    """Trailing realised vol at each point of `index`, one column per estimator.

    Built from `vrp_signal.perp_variance` so there is a single implementation of
    each estimator in the project. The rolling window is TIME-based, not a count
    of bars, so a gap in the perp shortens the sample rather than silently
    reaching further back than intended.

    Estimators needing OHLC are skipped (with a note) when the perp file has only
    closes, instead of aborting the whole build.
    """
    from volatility_surface.backtest import vrp_signal as V

    out = {}
    for est in estimators:
        try:
            v = V.perp_variance(perp_path, freq=bar_freq, estimator=est)
        except KeyError as e:
            print(f"  [skip] {est}: {e}")
            continue
        per = v.attrs["periods"]
        rv = np.sqrt(v.rolling(f"{window_days}D").mean().clip(lower=0) * per)
        # asof onto the option-snapshot grid: value AT OR BEFORE each stamp, so
        # the vol never contains a bar the snapshot could not have seen
        out[f"rv_{est}"] = rv.reindex(rv.index.union(index)).ffill().reindex(index)
    return pd.DataFrame(out, index=index)


def build(option_path: str, perp_path: str, target_days: float = 30.0,
          resample: str = "1D", rv_window: int = 30,
          estimators: tuple[str, ...] = ("close",), bar_freq: str = "1h") -> pd.DataFrame:
    """Panel of spot, implied vol and realised vol (one column per estimator),
    all sampled at the option-snapshot instants.

    `rv_window` is in DAYS (it used to be a count of observations; with several
    bar frequencies in play a count is ambiguous).
    """
    iv = atm_iv_constant_maturity(option_path, target_days, resample)
    if iv.empty:
        return pd.DataFrame()
    spot = perp_at_times(perp_path, iv.attrs["stamps"])
    df = pd.DataFrame({"spot": spot, "iv": iv})
    df["ret"] = np.log(df["spot"] / df["spot"].shift(1))
    df["d_iv"] = df["iv"].diff()

    rv = realised_vol_by_estimator(perp_path, df.index, estimators, rv_window, bar_freq)
    for c in rv.columns:
        df[c] = rv[c]
        df["d_" + c] = rv[c].diff()
    return df.dropna(subset=["ret"])


def rv_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("rv_")]


def _corr_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 5:
        return np.nan, np.nan, n
    c = float(np.corrcoef(x[m], y[m])[0, 1])
    t = c * np.sqrt((n - 2) / max(1 - c ** 2, 1e-12))
    return c, t, n


def full_window(df: pd.DataFrame) -> pd.DataFrame:
    """One row for the implied correlation, one per realised-vol estimator.

    The estimators disagree on the LEVEL of vol (range estimators run ~10% low
    at 1-min-into-hourly resolution, see vrp_signal.ESTIMATORS), but the
    correlation is scale-free — so a spread across estimators here is about the
    SHAPE of the vol response to spot, not about that level bias."""
    rows = [{"measure": "implied  corr(ret, dIV)", **dict(zip(
        ("corr", "t_stat", "n"), _corr_t(df["ret"], df["d_iv"])))}]
    for c in rv_columns(df):
        est = c[3:]
        rows.append({"measure": f"realised corr(ret, dRV) [{est}]",
                     **dict(zip(("corr", "t_stat", "n"), _corr_t(df["ret"], df["d_" + c])))})
    out = pd.DataFrame(rows).set_index("measure")
    return out.assign(corr=out["corr"].round(4), t_stat=out["t_stat"].round(2))


def rolling(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    cols = {"implied": df["ret"].rolling(window_days).corr(df["d_iv"])}
    for c in rv_columns(df):
        cols[c[3:]] = df["ret"].rolling(window_days).corr(df["d_" + c])
    return pd.DataFrame(cols)


def _backdrop(ax, regimes: dict | None, spot: pd.Series | None,
              log_spot: bool = False) -> list:
    """Regime bands and the spot on a twin right axis — the context both the
    correlation and the level view share. Returns the legend handles.

    The spot is drawn in grey and behind the data on purpose: it is there to read
    the curves against, not as another signal competing for attention. Regime
    bands get explicit legend entries — without them the two shades are
    indistinguishable at low alpha and the reader cannot tell a bull band from a
    bear band, or either from an unclassified gap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from volatility_surface.plots.vol_plots import PALETTE

    BULL, BEAR = PALETTE[1], PALETTE[6]
    handles = []
    if regimes:
        for name, (a, b) in regimes.items():
            is_bull = name.startswith("bull")
            ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"),
                       color=BULL if is_bull else BEAR, alpha=0.22 if is_bull else 0.13, lw=0)
        handles += [Patch(facecolor=BULL, alpha=0.22, label="bull regime"),
                    Patch(facecolor=BEAR, alpha=0.13, label="bear regime"),
                    Patch(facecolor="white", edgecolor="lightgrey", label="unclassified")]

    if spot is not None:
        ax2 = ax.twinx()
        ax2.plot(spot.index, spot.values, color="0.45", lw=1.0, alpha=0.75, zorder=1)
        if log_spot:
            # ETH spans 1500-4800 over 2024-2026; on a linear axis the early
            # years compress into a flat line
            ax2.set_yscale("log")
        ax2.set_ylabel("spot (USD)", color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
        ax2.set_zorder(1)
        ax.set_zorder(2)
        ax.patch.set_visible(False)   # else the left axes' background hides the spot
        handles.append(plt.Line2D([], [], color="0.45", lw=1.0, label="spot (right axis)"))
    return handles


def plot(roll: pd.DataFrame, full: pd.DataFrame, out_path: str,
         window_days: int = 90, regimes: dict | None = None,
         spot: pd.Series | None = None, implied: bool = True,
         log_spot: bool = False) -> None:
    """Rolling correlations on the left axis; optionally the spot on a twin
    right axis for context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from volatility_surface.plots.vol_plots import PALETTE

    fig, ax = plt.subplots(figsize=(13, 5.5))
    handles = _backdrop(ax, regimes, spot, log_spot)

    lines = []
    fi = full["corr"].to_dict()
    if implied:
        l, = ax.plot(roll.index, roll["implied"], color=PALETTE[6], lw=1.9, zorder=4,
                     label=f"implied  ΔIV   full {fi.get('implied  corr(ret, dIV)', float('nan')):+.2f}")
        lines.append(l)
    styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    shades = [PALETTE[3], PALETTE[2], PALETTE[4], PALETTE[1]]
    for i, est in enumerate([c for c in roll.columns if c != "implied"]):
        l, = ax.plot(roll.index, roll[est], color=shades[i % len(shades)], lw=1.2,
                     ls=styles[i % len(styles)], zorder=3,
                     label=f"realised ΔRV [{est}]  full "
                           f"{fi.get(f'realised corr(ret, dRV) [{est}]', float('nan')):+.2f}")
        lines.append(l)
    ax.axhline(0, color="grey", lw=0.8, zorder=2)
    if implied:
        ax.set_ylim(-1, 1)
    else:
        # the implied correlation sits near -1 and squashes everything else onto
        # a sliver of the axis; without it, scale to what is actually plotted
        vals = roll[[c for c in roll.columns if c != "implied"]].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = max(0.05, 0.15 * (hi - lo))
            ax.set_ylim(max(-1.0, lo - pad), min(1.0, hi + pad))
    ax.set_ylabel("correlation")
    ax.set_title(f"Spot–vol correlation, {window_days}-day rolling window")
    ax.legend(handles=lines + handles, loc="lower left", fontsize=7.5,
              framealpha=0.92, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------- #
# Levels — same panel, the other question
# --------------------------------------------------------------------------- #
def levels(df: pd.DataFrame) -> pd.DataFrame:
    """Level statistics per series, IN VOL POINTS, plus the two gaps worth naming.

    `vs_close` is the discretisation loss: how much more vol the estimator sees
    than a hedger rebalancing at `bar_freq` can capture. `premium` is implied
    minus that estimator's realised — what selling vol would have earned if the
    strategy tracked THAT quantity. Only the `rv_close` row is harvestable at the
    hedging frequency; the others price vol that is never monetised.

    `n` is deliberately left unscaled — multiplying the whole frame by 100 to get
    vol points turns the observation count into nonsense.
    """
    rows = []
    for c in rv_columns(df) + (["iv"] if "iv" in df else []):
        s = df[c].dropna()
        rows.append({"series": c, "n": len(s), "mean": s.mean(),
                     "median": s.median(), "min": s.min(), "max": s.max(),
                     "std": s.std()})
    out = pd.DataFrame(rows).set_index("series")
    if "rv_close" in out.index:
        out["vs_close"] = out["mean"] - out.loc["rv_close", "mean"]
    if "iv" in out.index:
        out["premium"] = out.loc["iv", "mean"] - out["mean"]
        out.loc["iv", "premium"] = np.nan
    scale = [c for c in out.columns if c != "n"]
    out[scale] = out[scale] * 100
    return out.round(2)


def plot_levels(df: pd.DataFrame, out_path: str, regimes: dict | None = None,
                spot: bool = True, log_spot: bool = False, implied: bool = True,
                window_days: int | None = None, bar_freq: str | None = None,
                target_days: float = 30.0) -> None:
    """Annualised vol per estimator on the left axis, spot on the right.

    Implied is drawn heaviest and in black because the quantity of interest is
    its distance from `rv_close` — that gap is the harvestable premium and should
    be the first thing the eye lands on. `rv_close` comes first among the
    realised lines and stays solid: it is the reference the others are read
    against, not one dashed line among four.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from volatility_surface.plots.vol_plots import PALETTE

    fig, ax = plt.subplots(figsize=(13, 5.5))
    handles = _backdrop(ax, regimes, df["spot"] if spot and "spot" in df else None,
                        log_spot)

    # dark end of the ramp only: the regime bands are drawn in the light end, and
    # a light line over a light band is unreadable. The linestyle carries the
    # distinction between estimators, the colour only has to stay visible.
    cols = sorted(rv_columns(df), key=lambda c: c != "rv_close")
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    shades = [PALETTE[7], PALETTE[5], PALETTE[4], PALETTE[3], PALETTE[6]]
    lines = []
    for i, c in enumerate(cols):
        l, = ax.plot(df.index, df[c] * 100, color=shades[i % len(shades)],
                     ls=styles[i % len(styles)], lw=1.4, zorder=3,
                     label=f"realised [{c[3:]}]   mean {df[c].mean():.1%}")
        lines.append(l)

    if implied and "iv" in df:
        l, = ax.plot(df.index, df["iv"] * 100, color="black", lw=2.0, zorder=5,
                     label=f"implied ATM {target_days:.0f}d   mean {df['iv'].mean():.1%}")
        lines.insert(0, l)

    ax.set_ylabel("annualised volatility (%)")
    ax.set_title("Volatility levels"
                 + (f" — {window_days}d trailing window" if window_days else "")
                 + (f", {bar_freq} bars" if bar_freq else ""))
    ax.legend(handles=lines + handles, loc="upper right", fontsize=7.5,
              framealpha=0.92, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved -> {out_path}")


def main() -> None:
    from volatility_surface.backtest import roll_engine as R

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", required=True)
    ap.add_argument("--perp", required=True)
    ap.add_argument("--target-days", type=float, default=30.0,
                    help="constant maturity for the implied vol series (default 30)")
    ap.add_argument("--resample", default="1D", help="1D or 1h")
    ap.add_argument("--rv-window", type=int, default=30,
                    help="trailing window IN DAYS for realised vol")
    ap.add_argument("--estimators", default="close",
                    help="comma-separated realised-vol estimators, or 'all' "
                         "(close,parkinson,garman_klass,rogers_satchell)")
    ap.add_argument("--bar-freq", default="1h", help="bar frequency the estimators run on")
    ap.add_argument("--roll-window", type=int, default=90, help="rolling correlation window")
    ap.add_argument("--out", default="spot_vol_corr.png")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-regimes", action="store_true")
    ap.add_argument("--no-spot", action="store_true",
                    help="omit the spot on the right axis")
    ap.add_argument("--no-implied", action="store_true",
                    help="omit the implied (ΔIV) line; the y-axis then scales to "
                         "the realised correlations instead of staying on [-1, 1]")
    ap.add_argument("--levels", action="store_true",
                    help="plot vol LEVELS per estimator instead of correlations")
    ap.add_argument("--log-spot", action="store_true",
                    help="log scale on the spot axis")
    a = ap.parse_args()

    from volatility_surface.backtest.vrp_signal import ESTIMATORS
    ests = ESTIMATORS if a.estimators == "all" else tuple(
        e.strip() for e in a.estimators.split(","))
    df = build(a.options, a.perp, a.target_days, a.resample, a.rv_window,
               estimators=ests, bar_freq=a.bar_freq)
    print(f"\n{len(df)} observations, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  "
          f"[{a.rv_window}d trailing window on {a.bar_freq} bars]")
    print(f"ATM IV at constant {a.target_days:.0f}d: mean {df['iv'].mean():.1%}")
    for c in rv_columns(df):
        print(f"  {c:26s} mean {df[c].mean():.1%}")

    if a.levels:
        print(f"\n{'=' * 78}\nVOLATILITY LEVELS (vol points)\n{'=' * 78}")
        print(levels(df).to_string())
        print(f"\n  vs_close = discretisation loss: variation inside the hedging")
        print(f"             interval, invisible to a hedger rebalancing every {a.bar_freq}.")
        print("  premium  = implied minus that estimator's realised. Only the rv_close")
        print("             row is harvestable by a hedger at this frequency.")
        plot_levels(df, a.out, None if a.no_regimes else R.REGIMES,
                    spot=not a.no_spot, log_spot=a.log_spot,
                    implied=not a.no_implied, window_days=a.rv_window,
                    bar_freq=a.bar_freq, target_days=a.target_days)
        if a.csv:
            df.to_csv(a.csv)
            print(f"saved -> {a.csv}")
        return

    full = full_window(df)
    print(f"\n{'=' * 62}\nFULL-WINDOW SPOT-VOL CORRELATION\n{'=' * 62}")
    print(full.to_string())
    print("\n  Negative = vol rises when spot falls (the leverage effect).")
    print("  This is what a negative risk reversal prices, and what gives a")
    print("  call-exposed delta-hedged structure its residual directional beta.")

    roll = rolling(df, a.roll_window)
    print(f"\n{'=' * 62}\nROLLING {a.roll_window}-DAY\n{'=' * 62}")
    print(roll.describe().round(3).to_string())

    plot(roll, full, a.out, a.roll_window,
         None if a.no_regimes else R.REGIMES,
         spot=None if a.no_spot else df["spot"], implied=not a.no_implied,
         log_spot=a.log_spot)
    if a.csv:
        df.join(roll.add_prefix("roll_")).to_csv(a.csv)
        print(f"saved -> {a.csv}")


if __name__ == "__main__":
    main()
