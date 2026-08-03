"""
Does (past realised vol - implied vol) predict (future realised vol - implied
vol)? — the question behind any "enter when RV > IV by X" rule.

Why this test and not a backtest: a delta-hedged strategy's P&L is a very noisy
estimator of the volatility risk premium (it adds discrete-hedging error, gamma
slippage and transaction costs on top of the quantity of interest), and at ~110
weekly observations the Sharpe standard error is ~0.68 — nothing below |1.34| is
distinguishable from zero. Measuring the premium and its predictability directly
uses the same data far more efficiently, and answers whether there is anything
to harvest BEFORE any execution assumption is made.

The trap the test exists to catch: ETH realised vol IS persistent (weekly
autocorrelation +0.27, R^2 7.5% on 2024-2026), so "RV has been high" does
predict "RV will be high". But implied vol already knows that — market makers
see the same history. The signal only has value if RV_past - IV predicts
RV_future - IV, i.e. if the market UNDER-reacts to recent realised vol. That is
what `predictive_test` measures.

Sign convention, so the trade direction is never ambiguous:
    premium := IV - RV_future
      > 0  options were expensive relative to what happened -> SELLING vol won
      < 0  options were cheap                              -> BUYING vol won
    signal  := RV_past - IV
      > 0  recent realised vol above what is priced        -> the "buy vol" case

    python volatility_surface/backtest/vrp_signal.py \
        --options "2 - Data/parquets/eth_hourly_2024.parquet" \
        --perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet"
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from volatility_surface.backtest import roll_engine as R

MS_D = 86_400_000


# --------------------------------------------------------------------------- #
# Realised volatility
# --------------------------------------------------------------------------- #
def hourly_log_returns(perp_path: str) -> pd.Series:
    """Hourly log returns of the perp, indexed by UTC timestamp."""
    d = pd.read_parquet(perp_path)
    d["dt"] = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    h = d.set_index("dt")["close"].resample("1h").last().dropna()
    return np.log(h / h.shift(1)).dropna()


def realised_vol(rets: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Annualised realised vol over [t0, t1) from hourly returns."""
    seg = rets[(rets.index >= t0) & (rets.index < t1)]
    if len(seg) < 24:  # under a day of data: too few points to annualise from
        return np.nan
    return float(seg.std(ddof=1) * np.sqrt(24 * 365))


# --------------------------------------------------------------------------- #
# Implied volatility at each roll
# --------------------------------------------------------------------------- #
def atm_iv_series(option_path: str, grid: list[int]) -> pd.DataFrame:
    """ATM mid implied vol at each roll date, for the option expiring at the
    NEXT roll date — exactly the contract a weekly roll strategy would buy or
    sell, so the premium measured here is the one those strategies trade."""
    rows = []
    for i, roll_ts in enumerate(grid[:-1]):
        target_exp = grid[i + 1]
        got = None
        # take the first snapshot at or after the roll instant
        for ts, snap in R._iter_snapshots(option_path, max_snaps=1,
                                          start_ms=roll_ts, end_ms=roll_ts + 6 * 3_600_000):
            got = (ts, snap)
            break
        if got is None:
            continue
        ts, snap = got
        legs = R._select_structure(snap, target_exp, [("ATM", "C", 1.0), ("ATM", "P", 1.0)],
                                   expiry_tol_ms=min(MS_D, (target_exp - roll_ts) // 2))
        if legs is None:
            continue
        rows.append({
            "roll_ts": roll_ts,
            "date": pd.Timestamp(roll_ts, unit="ms", tz="UTC"),
            "expiry": pd.Timestamp(target_exp, unit="ms", tz="UTC"),
            # average the call and put ATM IVs — they should agree by parity, and
            # averaging halves the quote noise
            "iv": float(np.mean([l["entry_iv"] for l in legs])),
            "strike": legs[0]["strike"], "forward": legs[0]["F_expiry"],
        })
    return pd.DataFrame(rows)


def build_panel(option_path: str, perp_path: str, frequency: str = "weekly") -> pd.DataFrame:
    """One row per roll: IV, past RV, forward RV, premium and signal."""
    rets = hourly_log_returns(perp_path)
    lo = int(rets.index[0].timestamp() * 1000)
    hi = int(rets.index[-1].timestamp() * 1000)
    grid = R.roll_grid(lo, hi, frequency)
    df = atm_iv_series(option_path, grid)
    if df.empty:
        return df

    horizon = pd.Timedelta(milliseconds=int(np.median(np.diff(grid))))
    df["rv_past"] = [realised_vol(rets, d - horizon, d) for d in df["date"]]
    df["rv_fut"] = [realised_vol(rets, d, e) for d, e in zip(df["date"], df["expiry"])]
    df["premium"] = df["iv"] - df["rv_fut"]     # >0: selling vol won
    df["signal"] = df["rv_past"] - df["iv"]     # >0: the "buy vol" case
    return df.dropna(subset=["iv", "rv_past", "rv_fut"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _tstat(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else np.nan


def unconditional_premium(df: pd.DataFrame) -> None:
    """Is there a premium AT ALL, before any timing rule?"""
    p = df["premium"].to_numpy()
    t = _tstat(p)
    print(f"  observations            : {len(p)}")
    print(f"  mean IV                 : {df['iv'].mean():.1%}")
    print(f"  mean realised (forward) : {df['rv_fut'].mean():.1%}")
    print(f"  mean premium (IV - RV)  : {p.mean():+.2%}   t = {t:+.2f}")
    if np.isfinite(t) and abs(t) < 2:
        print("  -> not distinguishable from zero: on this sample there is no "
              "unconditional premium to harvest, in EITHER direction")
    elif t > 0:
        print("  -> options were systematically EXPENSIVE: short vol had an edge before costs")
    else:
        print("  -> options were systematically CHEAP: long vol had an edge before costs")


def predictive_test(df: pd.DataFrame) -> None:
    """Does the signal predict the premium? This is the decisive test — a
    signal that merely tracks vol persistence adds nothing, because implied vol
    already prices that."""
    x = df["signal"].to_numpy()
    y = -df["premium"].to_numpy()   # RV_fut - IV: positive = buying vol paid off
    n = len(x)
    c = float(np.corrcoef(x, y)[0, 1])
    b, a = np.polyfit(x, y, 1)
    t = c * np.sqrt((n - 2) / (1 - c ** 2)) if abs(c) < 1 else np.nan
    print(f"  corr(RV_past - IV, RV_fut - IV) : {c:+.3f}   t = {t:+.2f}   n = {n}")
    print(f"  slope                           : {b:+.3f}   R^2 = {c ** 2:.1%}")
    if not np.isfinite(t) or abs(t) < 2:
        print("  -> NO predictive power. An 'enter when RV > IV by X' rule has nothing")
        print("     to exploit here; any threshold that looks good is fitted noise.")
    else:
        print("  -> the signal carries information; a threshold rule is worth building")


def threshold_table(df: pd.DataFrame, thresholds=(-0.10, -0.05, 0.0, 0.05, 0.10)) -> None:
    """Mean outcome conditional on the signal clearing a threshold.

    Read this as a diagnostic, NOT as a parameter search. With ~110 observations
    the standard error on each bucket mean is large, and picking the best cell
    is how backtests get overfitted — the whole table is one sample."""
    print(f"  {'signal >':>10} {'n':>5} {'mean RV_fut - IV':>18} {'t':>7} {'hit rate':>9}")
    y = -df["premium"]
    for th in thresholds:
        m = df["signal"] > th
        if m.sum() < 5:
            continue
        seg = y[m].to_numpy()
        print(f"  {th:>+10.0%} {int(m.sum()):>5} {seg.mean():>17.2%} "
              f"{_tstat(seg):>7.2f} {float((seg > 0).mean()):>8.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", required=True)
    ap.add_argument("--perp", required=True)
    ap.add_argument("--frequency", default="weekly")
    ap.add_argument("--csv", default=None, help="write the per-roll panel here")
    a = ap.parse_args()

    df = build_panel(a.options, a.perp, a.frequency)
    if df.empty:
        print("no usable rolls — check that the option file covers the perp's range")
        return

    print(f"\n{'=' * 66}\n1. IS THERE A PREMIUM AT ALL? (unconditional, before costs)\n{'=' * 66}")
    unconditional_premium(df)
    print(f"\n{'=' * 66}\n2. DOES 'RV > IV' PREDICT ANYTHING? (the decisive test)\n{'=' * 66}")
    predictive_test(df)
    print(f"\n{'=' * 66}\n3. OUTCOME BY SIGNAL THRESHOLD (diagnostic, not a parameter search)\n{'=' * 66}")
    threshold_table(df)

    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\nsaved panel -> {a.csv}")


if __name__ == "__main__":
    main()
