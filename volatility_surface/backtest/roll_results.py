"""
Reporting for `roll_engine` — the headline metrics of Lucic & Sepp (2024)
Table 1 (Total, P.a./CAGR, annualised Vol, Sharpe, MaxDD, Skew, alpha_AN, beta,
R^2) plus a Coin P&L attribution, and `results_table` to assemble a whole
catalog into the paper's table layout.

Conventions (paper §5.1, Table 1 caption):
  - Vol / Sharpe / Skew: daily log-returns of the Coin NAV, annualised by
    sqrt(365); zero risk-free rate.
  - alpha_AN / beta / R^2: OLS of the strategy's WEEKLY log-returns on the coin's
    weekly log-returns; alpha annualised by x52, R^2 adjusted.
  - The benchmark row ("ETH"/"BTC") is the coin buy-and-hold (from the perp/spot
    path recorded in each nav frame's `coin_px`), with beta=1, alpha=0, R^2=1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_last(nav: pd.DataFrame, col: str) -> pd.Series:
    s = nav.copy()
    s["date"] = pd.to_datetime(s["timestamp"], unit="ms", utc=True).dt.floor("D")
    return s.groupby("date")[col].last()


def _perf_stats(daily: pd.Series) -> dict:
    """Total / CAGR / ann.vol / Sharpe / MaxDD / skew from a daily level series."""
    out = {"total_return": np.nan, "cagr": np.nan, "ann_vol": np.nan,
           "sharpe": np.nan, "max_dd": np.nan, "skew": np.nan,
           "final": float(daily.iloc[-1]) if len(daily) else np.nan}
    if len(daily) < 2 or not (daily > 0).all():
        return out
    logret = np.log(daily / daily.shift(1)).dropna()
    years = ((daily.index[-1] - daily.index[0]).days or 1) / 365.0
    out["total_return"] = float(daily.iloc[-1] / daily.iloc[0] - 1.0)
    out["cagr"] = float((daily.iloc[-1] / daily.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = logret.std(ddof=1) * np.sqrt(365.0)
    out["ann_vol"] = float(vol)
    out["sharpe"] = float((logret.mean() * 365.0) / vol) if vol > 0 else np.nan
    out["max_dd"] = float((daily / daily.cummax() - 1.0).min())
    out["skew"] = float(logret.skew())
    return out


def _weekly_logret(nav: pd.DataFrame, col: str) -> pd.Series:
    s = nav.copy()
    s.index = pd.to_datetime(s["timestamp"], unit="ms", utc=True)
    wk = s[col].resample("W-FRI").last().dropna()
    return np.log(wk / wk.shift(1)).dropna()


def _regress(nav: pd.DataFrame) -> tuple[float, float, float]:
    """(alpha_AN, beta, R^2_adj) of weekly strategy vs weekly coin returns."""
    if nav.empty or "coin_px" not in nav.columns:
        return np.nan, np.nan, np.nan
    rs = _weekly_logret(nav, "coin_nav")
    rb = _weekly_logret(nav, "coin_px")
    j = rs.index.intersection(rb.index)
    if len(j) < 3:
        return np.nan, np.nan, np.nan
    x, y = rb.loc[j].to_numpy(), rs.loc[j].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    yhat = alpha + beta * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(x)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if (n > 2 and np.isfinite(r2)) else r2
    return float(alpha * 52.0), float(beta), float(r2_adj)


def summarize_rolls(result: dict) -> dict:
    """Headline performance + regression + P&L attribution for one backtest."""
    nav = result["nav"]
    daily = _daily_last(nav, "coin_nav") if not nav.empty else pd.Series(dtype=float)
    m = {"structure": result["params"]["structure"], "frequency": result["params"]["frequency"],
         "accounting": result["params"].get("accounting", "coin"),
         "nav_unit": result["params"].get("nav_unit", "coin")}
    m.update(_perf_stats(daily))
    m["final_nav_coin"] = m.pop("final")
    m["alpha_an"], m["beta"], m["r2"] = _regress(nav)
    m["n_rolls"] = sum(1 for r in result["rolls"] if r["action"] == "open")

    ev = pd.DataFrame(result["events"])
    if not ev.empty:
        a = {k: float(ev[k].sum()) for k in
             ["option_pnl", "option_cost", "hedge_pnl", "funding", "rebalance_cost"]}
        a["net_pnl"] = a["option_pnl"] - a["option_cost"] + a["hedge_pnl"] + a["funding"] - a["rebalance_cost"]
        m["attribution"] = a
    return m


def _benchmark_metrics(nav: pd.DataFrame, name: str) -> dict:
    """Coin buy-and-hold row from the recorded `coin_px` path."""
    daily = _daily_last(nav, "coin_px")
    m = {"structure": name, "frequency": "-"}
    m.update(_perf_stats(daily))
    m["final_nav_coin"] = m.pop("final")
    m["alpha_an"], m["beta"], m["r2"] = 0.0, 1.0, 1.0
    m["n_rolls"] = 0
    return m


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #
_COLS = ["Total", "P.a.", "Vol", "Sharpe", "MaxDD", "Skew", "alpha_AN", "beta", "R2"]


def _row(m: dict) -> dict:
    return {"Total": m["total_return"], "P.a.": m["cagr"], "Vol": m["ann_vol"],
            "Sharpe": m["sharpe"], "MaxDD": m["max_dd"], "Skew": m["skew"],
            "alpha_AN": m["alpha_an"], "beta": m["beta"], "R2": m["r2"]}


def results_table(results: dict[str, dict], benchmark_name: str = "ETH") -> pd.DataFrame:
    """Assemble `run_all_strategies`' output into a numeric DataFrame with the
    paper's Table-1 columns, benchmark (coin buy-and-hold) as the first row.
    Rows follow catalog order. Pretty-print with `format_table`."""
    rows, index = [], []
    first_nav = next((r["nav"] for r in results.values() if not r["nav"].empty), None)
    if first_nav is not None:
        rows.append(_row(_benchmark_metrics(first_nav, benchmark_name)))
        index.append(benchmark_name)
    for name, res in results.items():
        rows.append(_row(summarize_rolls(res)))
        index.append(name)
    return pd.DataFrame(rows, index=index)[_COLS]


def format_table(df: pd.DataFrame) -> str:
    """Percent-format the return/vol/dd columns; keep Sharpe/Skew/beta/R2 raw."""
    out = df.copy()
    for c in ["Total", "P.a.", "Vol", "MaxDD", "alpha_AN"]:
        out[c] = df[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
    out["Sharpe"] = df["Sharpe"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    out["Skew"] = df["Skew"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
    out["beta"] = df["beta"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    out["R2"] = df["R2"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    return out.to_string()


def print_summary(metrics: dict) -> None:
    """Single-strategy console summary (used by the CLI's single-strategy path)."""
    print(f"{metrics['structure']} / {metrics['frequency']} rolls — {metrics.get('n_rolls', 0)} rolls")
    print(f"  Total return : {metrics['total_return']:+.1%}")
    print(f"  CAGR (p.a.)  : {metrics['cagr']:+.1%}")
    print(f"  Ann. vol     : {metrics['ann_vol']:.1%}")
    print(f"  Sharpe       : {metrics['sharpe']:.2f}")
    print(f"  Max drawdown : {metrics['max_dd']:.1%}")
    print(f"  Skew (daily) : {metrics['skew']:+.2f}")
    print(f"  alpha_AN/beta/R2 : {metrics['alpha_an']:+.1%} / {metrics['beta']:.2f} / {metrics['r2']:.0%}")
    print(f"  Final NAV    : {metrics['final_nav_coin']:.4f} {metrics.get('nav_unit', 'coin')}"
          f"   [{metrics.get('accounting', 'coin')} accounting]")
    a = metrics.get("attribution")
    if a:
        print(f"  P&L attribution ({metrics.get('nav_unit', 'coin')}):")
        print(f"    option premium : {a['option_pnl']:+.4f}")
        print(f"    option cost    : -{a['option_cost']:.4f}")
        print(f"    delta-hedge    : {a['hedge_pnl']:+.4f}")
        print(f"    funding        : {a['funding']:+.4f}")
        print(f"    rebalance cost : -{a['rebalance_cost']:.4f}")
        print(f"    NET            : {a['net_pnl']:+.4f}")
