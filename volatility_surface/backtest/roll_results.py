"""
Reporting for `roll_engine.run_roll_backtest` — the headline metrics of
Lucic & Sepp (2024) Table 1 (Total, CAGR/p.a., annualised vol, Sharpe, MaxDD,
skew) plus a Coin P&L attribution across the paper's components (option premium
vs. delta-hedge vs. funding vs. transaction costs).

Returns/vol/Sharpe are computed on the Coin NAV path (`result["nav"]`) resampled
to daily, with zero risk-free rate (the paper's convention, §5.1). Sharpe uses
daily log-returns annualised by sqrt(365) — crypto trades every day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_nav(nav: pd.DataFrame) -> pd.Series:
    """Last NAV of each UTC day, indexed by date."""
    if nav.empty:
        return pd.Series(dtype=float)
    s = nav.copy()
    s["date"] = pd.to_datetime(s["timestamp"], unit="ms", utc=True).dt.floor("D")
    return s.groupby("date")["coin_nav"].last()


def summarize_rolls(result: dict) -> dict:
    """Headline performance + P&L attribution for one roll backtest."""
    nav = result["nav"]
    events = pd.DataFrame(result["events"])
    daily = _daily_nav(nav)

    metrics: dict = {"structure": result["params"]["structure"],
                     "frequency": result["params"]["frequency"]}

    if len(daily) >= 2 and (daily > 0).all():
        logret = np.log(daily / daily.shift(1)).dropna()
        n_days = (daily.index[-1] - daily.index[0]).days or 1
        years = n_days / 365.0
        total = daily.iloc[-1] / daily.iloc[0] - 1.0
        cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        vol = logret.std(ddof=1) * np.sqrt(365.0)
        sharpe = (logret.mean() * 365.0) / vol if vol > 0 else np.nan
        running_max = daily.cummax()
        maxdd = float((daily / running_max - 1.0).min())
        skew = float(logret.skew())
        metrics.update({"total_return": float(total), "cagr": float(cagr),
                        "ann_vol": float(vol), "sharpe": float(sharpe),
                        "max_dd": maxdd, "skew": skew,
                        "final_nav_coin": float(daily.iloc[-1])})
    else:
        metrics.update({"total_return": np.nan, "cagr": np.nan, "ann_vol": np.nan,
                        "sharpe": np.nan, "max_dd": np.nan, "skew": np.nan,
                        "final_nav_coin": float(daily.iloc[-1]) if len(daily) else np.nan})

    if not events.empty:
        metrics["attribution"] = {
            "option_pnl": float(events["option_pnl"].sum()),
            "option_cost": float(events["option_cost"].sum()),
            "hedge_pnl": float(events["hedge_pnl"].sum()),
            "funding": float(events["funding"].sum()),
            "rebalance_cost": float(events["rebalance_cost"].sum()),
        }
        a = metrics["attribution"]
        metrics["attribution"]["net_pnl"] = (a["option_pnl"] - a["option_cost"]
                                             + a["hedge_pnl"] + a["funding"]
                                             - a["rebalance_cost"])
    metrics["n_rolls"] = sum(1 for r in result["rolls"] if r["action"] == "open")
    return metrics


def print_summary(metrics: dict) -> None:
    print(f"{metrics['structure']} / {metrics['frequency']} rolls "
          f"— {metrics.get('n_rolls', 0)} rolls")
    print(f"  Total return : {metrics['total_return']:+.1%}")
    print(f"  CAGR (p.a.)  : {metrics['cagr']:+.1%}")
    print(f"  Ann. vol     : {metrics['ann_vol']:.1%}")
    print(f"  Sharpe       : {metrics['sharpe']:.2f}")
    print(f"  Max drawdown : {metrics['max_dd']:.1%}")
    print(f"  Skew (daily) : {metrics['skew']:+.2f}")
    print(f"  Final NAV    : {metrics['final_nav_coin']:.4f} coin")
    a = metrics.get("attribution")
    if a:
        print("  P&L attribution (coin):")
        print(f"    option premium : {a['option_pnl']:+.4f}")
        print(f"    option cost    : -{a['option_cost']:.4f}")
        print(f"    delta-hedge    : {a['hedge_pnl']:+.4f}")
        print(f"    funding        : {a['funding']:+.4f}")
        print(f"    rebalance cost : -{a['rebalance_cost']:.4f}")
        print(f"    NET            : {a['net_pnl']:+.4f}")
