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
#: Annualisation factor per sampling frequency.
_PERIODS = {"1min": 525_600, "5min": 105_120, "15min": 35_040,
            "1h": 8_760, "4h": 2_190, "1D": 365}


def perp_log_returns(perp_path: str, freq: str = "1h") -> pd.Series:
    """Log returns of the perp resampled to `freq`, indexed by UTC timestamp.

    The sampling frequency is the one real choice in this measurement. Hourly is
    the default because `roll_engine` delta-hedges hourly, and a delta-hedged
    option's P&L tracks the realised vol AT ITS HEDGING FREQUENCY — so hourly RV
    is the quantity those strategies actually trade against. Measured on ETH
    2024-2026 the choice barely matters (62.5% daily to 67.0% at 1-minute, the
    latter inflated by bid-ask bounce), i.e. less than the premium's own standard
    error, so the headline result is not an artefact of it.
    """
    if freq not in _PERIODS:
        raise ValueError(f"freq must be one of {list(_PERIODS)}")
    d = pd.read_parquet(perp_path)
    d["dt"] = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    h = d.set_index("dt")["close"].resample(freq).last().dropna()
    r = np.log(h / h.shift(1)).dropna()
    r.attrs["periods"] = _PERIODS[freq]
    r.attrs["freq"] = freq
    return r


#: kept so older callers/notebooks keep working
hourly_log_returns = perp_log_returns


def realised_vol(rets: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Annualised realised vol over [t0, t1) at the series' own frequency."""
    seg = rets[(rets.index >= t0) & (rets.index < t1)]
    periods = rets.attrs.get("periods", 8_760)
    # need enough points to estimate a std at all — scale the floor with the
    # sampling frequency rather than hard-coding an hourly assumption
    if len(seg) < max(10, periods // 365):
        return np.nan
    return float(seg.std(ddof=1) * np.sqrt(periods))


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


# --------------------------------------------------------------------------- #
# Skew
# --------------------------------------------------------------------------- #
#: Deltas quoted per side. Resolved leg by leg (not via `_select_structure`) so a
#: thin chain missing the 10D wing costs that one column, not the whole row.
_SKEW_TARGETS = [("atm", "ATM", "C"), ("atm_p", "ATM", "P"),
                 ("c25", ("delta", 0.25), "C"), ("p25", ("delta", 0.25), "P"),
                 ("c10", ("delta", 0.10), "C"), ("p10", ("delta", 0.10), "P")]


def iv_by_delta_series(option_path: str, grid: list[int]) -> pd.DataFrame:
    """Implied vol at each quoted delta, per roll date, for the option expiring
    at the next roll — the same contracts the roll strategies trade."""
    rows = []
    for i, roll_ts in enumerate(grid[:-1]):
        target_exp = grid[i + 1]
        got = None
        for ts, snap in R._iter_snapshots(option_path, max_snaps=1,
                                          start_ms=roll_ts, end_ms=roll_ts + 6 * 3_600_000):
            got = (ts, snap)
            break
        if got is None:
            continue
        _, snap = got
        uniq = np.unique(snap["expiration_timestamp_ms"].to_numpy())
        j = int(np.argmin(np.abs(uniq - target_exp)))
        if abs(uniq[j] - target_exp) > min(MS_D, (target_exp - roll_ts) // 2):
            continue
        sl = snap[snap["expiration_timestamp_ms"] == uniq[j]]
        if sl.empty:
            continue
        F = float(sl["underlying_price"].iloc[0])
        row = {"date": pd.Timestamp(roll_ts, unit="ms", tz="UTC"),
               "expiry": pd.Timestamp(target_exp, unit="ms", tz="UTC"), "forward": F}
        for key, moneyness, otype in _SKEW_TARGETS:
            leg = R._select_leg(sl, F, moneyness, otype)
            iv = R._mid_iv(leg) if leg is not None else np.nan
            row[f"iv_{key}"] = float(iv) if np.isfinite(iv) and iv > 0 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ATM: average the call and put quotes (equal by parity; averaging halves the
    # quote noise), then the standard FX/crypto skew decomposition.
    df["iv_atm"] = df[["iv_atm", "iv_atm_p"]].mean(axis=1)
    df["rr25"] = df["iv_c25"] - df["iv_p25"]                       # <0: puts richer
    df["rr10"] = df["iv_c10"] - df["iv_p10"]
    df["bf25"] = (df["iv_c25"] + df["iv_p25"]) / 2 - df["iv_atm"]  # smile convexity
    df["call_skew"] = df["iv_c25"] - df["iv_atm"]                  # the leg that drove
    df["put_skew"] = df["iv_p25"] - df["iv_atm"]                   # the regime split
    return df


def regime_skew_table(option_path: str, perp_path: str,
                      regimes: dict[str, tuple[str, str]],
                      frequency: str = "weekly", rv_freq: str = "1h") -> pd.DataFrame:
    """Implied skew vs realised asymmetry, per regime.

    The point is the comparison, not either column alone. Implied skew being
    negative is normal — crypto returns are left-skewed, so puts SHOULD carry a
    higher vol. What matters is whether the priced asymmetry matched the realised
    one: a regime where the market priced a big negative risk reversal while
    returns realised positively skewed is a regime where calls were too cheap,
    and long-call structures win — which is the mechanism behind the bull1/bull2
    reversal seen in the regime results.
    """
    rets = perp_log_returns(perp_path, freq=rv_freq)
    lo = int(rets.index[0].timestamp() * 1000)
    hi = int(rets.index[-1].timestamp() * 1000)
    df = iv_by_delta_series(option_path, R.roll_grid(lo, hi, frequency))
    if df.empty:
        return df

    out = []
    for name, (a, b) in regimes.items():
        m = (df["date"] >= pd.Timestamp(a, tz="UTC")) & (df["date"] < pd.Timestamp(b, tz="UTC"))
        seg = df[m]
        if seg.empty:
            continue
        r = rets[(rets.index >= pd.Timestamp(a, tz="UTC")) & (rets.index < pd.Timestamp(b, tz="UTC"))]
        out.append({
            "regime": name, "n_rolls": len(seg),
            "iv_atm": seg["iv_atm"].mean(),
            "rr25": seg["rr25"].mean(),
            "rr10": seg["rr10"].mean(),
            "bf25": seg["bf25"].mean(),
            "call_skew": seg["call_skew"].mean(),
            "put_skew": seg["put_skew"].mean(),
            "realised_vol": float(r.std(ddof=1) * np.sqrt(rets.attrs["periods"])) if len(r) > 2 else np.nan,
            "realised_skew": float(pd.Series(r).skew()) if len(r) > 2 else np.nan,
        })
    return pd.DataFrame(out).set_index("regime").round(4)


def build_panel(option_path: str, perp_path: str, frequency: str = "weekly",
                rv_freq: str = "1h") -> pd.DataFrame:
    """One row per roll: IV, past RV, forward RV, premium and signal."""
    rets = perp_log_returns(perp_path, freq=rv_freq)
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
    ap.add_argument("--frequency", default="weekly", help="roll calendar: daily|weekly|monthly|quarterly")
    ap.add_argument("--rv-frequency", default="1h", choices=list(_PERIODS),
                    help="sampling frequency for realised vol (default 1h, matching "
                         "roll_engine's hourly delta hedge)")
    ap.add_argument("--csv", default=None, help="write the per-roll panel here")
    ap.add_argument("--skew", action="store_true",
                    help="report implied skew vs realised asymmetry per regime instead "
                         "of the premium/predictability tests")
    a = ap.parse_args()

    if a.skew:
        tbl = regime_skew_table(a.options, a.perp, R.REGIMES,
                                frequency=a.frequency, rv_freq=a.rv_frequency)
        if tbl.empty:
            print("no usable rolls")
            return
        print(f"\n{'=' * 78}\nIMPLIED SKEW vs REALISED ASYMMETRY, BY REGIME"
              f"\n  rr25 = IV(25d call) - IV(25d put)   <0 means puts richer (the normal state)"
              f"\n  bf25 = smile convexity | call_skew / put_skew are vs ATM"
              f"\n  realised_skew = third moment of {a.rv_frequency} returns over the regime"
              f"\n{'=' * 78}")
        print(tbl.to_string())
        print("\n  Read the LAST TWO columns against rr25: a very negative rr25 (calls cheap"
              "\n  relative to puts) alongside a positive realised_skew is a regime where the"
              "\n  market under-priced upside — long-call structures win there.")
        if a.csv:
            tbl.to_csv(a.csv)
            print(f"\nsaved -> {a.csv}")
        return

    df = build_panel(a.options, a.perp, a.frequency, rv_freq=a.rv_frequency)
    if df.empty:
        print("no usable rolls — check that the option file covers the perp's range")
        return

    print(f"\n{'=' * 66}\n1. IS THERE A PREMIUM AT ALL? (unconditional, before costs)"
          f"\n   [rolls: {a.frequency} | realised vol sampled at {a.rv_frequency}]\n{'=' * 66}")
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
