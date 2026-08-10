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
def build(option_path: str, perp_path: str, target_days: float = 30.0,
          resample: str = "1D", rv_window: int = 30) -> pd.DataFrame:
    iv = atm_iv_constant_maturity(option_path, target_days, resample)
    if iv.empty:
        return pd.DataFrame()
    spot = perp_at_times(perp_path, iv.attrs["stamps"])
    per = {"1D": 365, "1h": 8760}.get(resample, 365)
    df = pd.DataFrame({"spot": spot, "iv": iv})
    df["ret"] = np.log(df["spot"] / df["spot"].shift(1))
    df["rv"] = df["ret"].rolling(rv_window).std() * np.sqrt(per)
    df["d_iv"] = df["iv"].diff()
    df["d_rv"] = df["rv"].diff()
    return df.dropna(subset=["ret"])


def _corr_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 5:
        return np.nan, np.nan, n
    c = float(np.corrcoef(x[m], y[m])[0, 1])
    t = c * np.sqrt((n - 2) / max(1 - c ** 2, 1e-12))
    return c, t, n


def full_window(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lbl, col in [("implied  corr(ret, d_IV)", "d_iv"),
                     ("realised corr(ret, d_RV)", "d_rv")]:
        c, t, n = _corr_t(df["ret"], df[col])
        rows.append({"measure": lbl, "corr": round(c, 4), "t_stat": round(t, 2), "n": n})
    return pd.DataFrame(rows).set_index("measure")


def rolling(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    return pd.DataFrame({
        "implied": df["ret"].rolling(window_days).corr(df["d_iv"]),
        "realised": df["ret"].rolling(window_days).corr(df["d_rv"]),
    })


def plot(roll: pd.DataFrame, full: pd.DataFrame, out_path: str,
         window_days: int = 90, regimes: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from volatility_surface.plots.vol_plots import PALETTE

    fig, ax = plt.subplots(figsize=(12, 5))
    if regimes:
        for name, (a, b) in regimes.items():
            ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"),
                       color=PALETTE[1] if name.startswith("bull") else PALETTE[5],
                       alpha=0.12, lw=0)
            ax.text(pd.Timestamp(a, tz="UTC"), 0.97, name, fontsize=7,
                    color="grey", va="top", ha="left", transform=ax.get_xaxis_transform())

    ax.plot(roll.index, roll["implied"], color=PALETTE[6], lw=1.6,
            label=f"implied — corr(spot ret, ΔIV)   full: {full.loc['implied  corr(ret, d_IV)', 'corr']:+.2f}")
    ax.plot(roll.index, roll["realised"], color=PALETTE[3], lw=1.3, ls="--",
            label=f"realised — corr(spot ret, ΔRV)  full: {full.loc['realised corr(ret, d_RV)', 'corr']:+.2f}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("correlation")
    ax.set_title(f"Spot–vol correlation, {window_days}-day rolling window")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
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
    ap.add_argument("--rv-window", type=int, default=30, help="periods for realised vol")
    ap.add_argument("--roll-window", type=int, default=90, help="rolling correlation window")
    ap.add_argument("--out", default="spot_vol_corr.png")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-regimes", action="store_true")
    a = ap.parse_args()

    df = build(a.options, a.perp, a.target_days, a.resample, a.rv_window)
    print(f"\n{len(df)} observations, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    print(f"ATM IV at constant {a.target_days:.0f}d: mean {df['iv'].mean():.1%}, "
          f"realised vol ({a.rv_window}p): mean {df['rv'].mean():.1%}")

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
         None if a.no_regimes else R.REGIMES)
    if a.csv:
        df.join(roll.add_prefix("roll_")).to_csv(a.csv)
        print(f"saved -> {a.csv}")


if __name__ == "__main__":
    main()
