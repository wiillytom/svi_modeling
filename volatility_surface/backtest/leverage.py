"""
The leverage effect in ETH, measured from the perp alone — no options, no implied vol.

WHAT "LEVERAGE EFFECT" MEANS HERE, and why the distinction decides the answer.

In continuous time the quantity is rho = corr(dW_S, dW_v): the correlation between
the Brownian driving the price and the one driving its variance. Every empirical
proxy below is an attempt at that number, and they are NOT interchangeable:

    CONTEMPORANEOUS   corr(r_t, dlog RV_t)
                      MECHANICALLY CONTAMINATED. RV_t is built from day t's own
                      returns, so r_t sits inside RV_t. The covariance is then
                      carried by E[r^3] — this is a realised-SKEWNESS estimator
                      wearing a correlation's clothes. Report it, never lead with
                      it, and never call it the leverage effect.

    PREDICTIVE        corr(r_t, dlog RV_{t+1})
                      CLEAN. Does a down day today raise vol TOMORROW? r_t is not
                      a component of the RV change being predicted, so no
                      mechanical link. This is the honest correlation-based proxy.

    SIGNED VARIATION  (RS+ - RS-) / RV        [Barndorff-Nielsen/Kinnebrock/Shephard]
                      BEST. No correlation coefficient at all, hence no
                      attenuation from estimator noise, and no ambiguity about
                      what is inside what. Asks directly: is the realised variance
                      coming from up moves or down moves? Zero under symmetry,
                      negative under a leverage effect. Lead with this.

    LHAR gamma^-      log RV_{t+1} = c + HAR terms + gamma^- r^-_t + gamma^+ r^+_t
                      The literature's workhorse (Corsi-Reno). Controls for vol
                      persistence, which the raw correlations do not, and
                      separates the two signs instead of averaging them.

RESOLUTION IS THE OTHER HALF OF THE PROBLEM. A daily RV built from 24 hourly bars
carries ~29% relative standard error (sqrt(2/n)); from 1-minute bars, 3.7%. That
noise is independent of the return, so it attenuates every correlation toward
zero — a real leverage effect can be invisible purely from using coarse bars. We
therefore work from 1-minute data and report the volatility signature plot, since
1-minute close-to-close also picks up bid-ask bounce which inflates RV in the
other direction. 5-minute is the headline compromise, as in the literature.

    python volatility_surface/backtest/leverage.py \
        --perp "2 - Data/parquets/eth_perp_1min_2024_2026.parquet" --out results
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

SEED = 20260810
TRADING_DAYS = 365.0          # crypto trades 24/7


# --------------------------------------------------------------------------- #
# Intraday returns
# --------------------------------------------------------------------------- #
def intraday_returns(perp_path: str, freq: str = "5min") -> pd.Series:
    """Log returns of the perp close at `freq`, UTC.

    Resampling with `.last()` and then differencing means a bar with no trades
    inherits the previous close, contributing a zero return rather than a NaN or
    a spurious jump. That is the right behaviour for realised variance: no trade
    means no price change was observed.
    """
    d = pd.read_parquet(perp_path, columns=["timestamp_ms", "close"])
    d.index = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    px = d["close"].resample(freq).last().ffill().dropna()
    r = np.log(px / px.shift(1)).dropna()
    r.attrs["freq"] = freq
    return r


# --------------------------------------------------------------------------- #
# Daily realised measures
# --------------------------------------------------------------------------- #
def _medrv(r: np.ndarray) -> float:
    """MedRV (Andersen-Dobrev-Schaumburg): jump-robust, uses rolling medians of
    three consecutive absolute returns."""
    n = len(r)
    if n < 3:
        return np.nan
    a = np.abs(r)
    med = np.median(np.column_stack([a[:-2], a[1:-1], a[2:]]), axis=1)
    scale = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
    return float(scale * (n / (n - 2)) * np.sum(med ** 2))


def _bipower(r: np.ndarray) -> float:
    """Bipower variation (Barndorff-Nielsen-Shephard): jump-robust estimate of the
    CONTINUOUS quadratic variation. mu_1^-2 = pi/2."""
    if len(r) < 2:
        return np.nan
    a = np.abs(r)
    return float((np.pi / 2.0) * np.sum(a[:-1] * a[1:]))


def daily_measures(r: pd.Series) -> pd.DataFrame:
    """One row per UTC day of realised measures built from intraday returns.

    Columns, all in VARIANCE units unless named otherwise:
      rv          realised variance,  sum r^2
      rs_plus     upside semivariance,   sum r^2 1{r>0}   [BNKS 2010]
      rs_minus    downside semivariance, sum r^2 1{r<0}
      sv_ratio    (rs_plus - rs_minus)/rv  -> the E5 estimand, in [-1, 1]
      bv          bipower variation (continuous part)
      medrv       MedRV (continuous part, more robust)
      jump        max(rv - bv, 0), the jump contribution
      rskew       realised skewness  sqrt(n) * sum r^3 / rv^1.5   [Amaya et al]
      rkurt       realised kurtosis  n * sum r^4 / rv^2
      ret         the day's own log return, sum of intraday returns
      n_obs       intraday observations that day
    """
    g = r.groupby(r.index.floor("1D"))
    rows = []
    for day, x in g:
        a = x.to_numpy()
        n = len(a)
        rv = float(np.sum(a ** 2))
        if n < 10 or rv <= 0:
            continue
        neg = a < 0
        rsm = float(np.sum(a[neg] ** 2))
        rsp = float(np.sum(a[~neg] ** 2))
        bv = _bipower(a)
        rows.append({
            "date": day, "n_obs": n, "ret": float(np.sum(a)),
            "rv": rv, "rs_plus": rsp, "rs_minus": rsm,
            "sv_ratio": (rsp - rsm) / rv,
            "bv": bv, "medrv": _medrv(a),
            "jump": max(rv - bv, 0.0) if np.isfinite(bv) else np.nan,
            "rskew": float(np.sqrt(n) * np.sum(a ** 3) / rv ** 1.5),
            "rkurt": float(n * np.sum(a ** 4) / rv ** 2),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    # annualised vol, for readability only — every test below uses the variance
    df["rv_ann"] = np.sqrt(df["rv"] * TRADING_DAYS)
    df["log_rv"] = np.log(df["rv"])
    df["d_log_rv"] = df["log_rv"].diff()
    df.attrs["freq"] = r.attrs.get("freq")
    return df


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_se(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """Newey-West standard error of the MEAN of `x`.

    Bandwidth defaults to the Newey-West plug-in floor(4*(n/100)^(2/9)), the
    standard automatic rule; it is returned so the choice is on the record.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, 0
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    s = gamma0
    for k in range(1, lags + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk      # Bartlett kernel
    se = np.sqrt(max(s, 0.0) / n)
    return float(x.mean()), float(se), lags


def n_effective(x: np.ndarray, y: np.ndarray, max_lag: int = 40) -> float:
    """Effective sample size for a correlation between two autocorrelated series.

    Uses the autocorrelation of the cross-product z = (x-xbar)(y-ybar), which is
    what actually drives the variance of rho-hat: n_eff = n / (1 + 2*sum rho_z).
    Truncated at the first lag whose |rho_z| falls inside the +/-2/sqrt(n) band,
    so a long tail of noise autocorrelations cannot inflate the correction.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 20:
        return float(n)
    z = (x - x.mean()) * (y - y.mean())
    z -= z.mean()
    band = 2.0 / np.sqrt(n)
    tot = 0.0
    denom = float(z @ z)
    for k in range(1, min(max_lag, n // 4)):
        rk = float(z[k:] @ z[:-k]) / denom
        if abs(rk) < band:
            break
        tot += rk
    return float(n / max(1.0 + 2.0 * tot, 1e-6))


def fisher_ci(rho: float, n_eff: float, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z confidence interval, using the EFFECTIVE sample size."""
    if not np.isfinite(rho) or n_eff <= 4 or abs(rho) >= 1:
        return np.nan, np.nan
    from scipy import stats
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n_eff - 3.0)
    q = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def _acf_block_length(x: np.ndarray) -> int:
    """Mean block length for the stationary bootstrap.

    Politis-White proper requires spectral estimation at zero frequency; this is
    the practical variant: 2 * (integrated autocorrelation up to the first
    insignificant lag), floored at 2. Documented rather than dressed up as PW.
    """
    n = len(x)
    e = x - x.mean()
    denom = float(e @ e)
    band = 2.0 / np.sqrt(n)
    tot = 0.0
    for k in range(1, min(50, n // 4)):
        rk = float(e[k:] @ e[:-k]) / denom
        if abs(rk) < band:
            break
        tot += abs(rk)
    return int(max(2, round(2 * (1 + 2 * tot))))


def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                         seed: int = SEED) -> tuple[float, float, float, int]:
    """Stationary (Politis-Romano) block bootstrap CI for a correlation.

    Geometric block lengths preserve stationarity of the resampled series, which
    a fixed-block bootstrap does not. Pairs (x_t, y_t) are resampled TOGETHER so
    the cross-sectional link is preserved while the serial dependence is
    replicated.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, np.nan, 0
    L = _acf_block_length(y)
    p = 1.0 / L
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)), \
        float(np.nanstd(out)), L


def corr_report(x: pd.Series, y: pd.Series, label: str, n_boot: int = 2000,
                seed: int = SEED) -> dict:
    """A correlation with everything needed to judge it: point estimate, nominal
    and effective n, Fisher CI on n_eff, bootstrap CI, and a p-value on n_eff."""
    from scipy import stats
    a, b = x.align(y, join="inner")
    m = a.notna() & b.notna()
    xa, ya = a[m].to_numpy(), b[m].to_numpy()
    n = len(xa)
    if n < 30:
        return {"measure": label, "n": n}
    rho = float(np.corrcoef(xa, ya)[0, 1])
    ne = n_effective(xa, ya)
    lo, hi = fisher_ci(rho, ne)
    blo, bhi, bse, L = block_bootstrap_corr(xa, ya, n_boot, seed)
    t = rho * np.sqrt(max(ne - 2, 1) / max(1 - rho ** 2, 1e-12))
    return {"measure": label, "rho": rho, "n": n, "n_eff": ne,
            "fisher_lo": lo, "fisher_hi": hi,
            "boot_lo": blo, "boot_hi": bhi, "boot_se": bse, "block_len": L,
            "t_neff": t, "p_neff": float(2 * (1 - stats.norm.cdf(abs(t))))}


# --------------------------------------------------------------------------- #
# E5 — signed variation (the headline, no correlation coefficient involved)
# --------------------------------------------------------------------------- #
def e5_signed_variation(daily: pd.DataFrame) -> pd.DataFrame:
    """Is realised variance coming from up moves or down moves?

    Three framings of the same question, because each has a different weakness:
      sv_ratio      per-day (RS+ - RS-)/RV, then averaged. Equal weight per day,
                    bounded in [-1,1], so a single cascade cannot dominate.
      pooled        (sum RS+ - sum RS-)/sum RV. Variance-weighted, so it answers
                    "where did the sample's total variance come from" — but one
                    huge day can carry it.
      rskew         realised skewness per day, averaged. Scale-free, third-moment.
    All standard errors are Newey-West.
    """
    from scipy import stats
    rows = []
    for col, lbl in [("sv_ratio", "E5  mean (RS+ - RS-)/RV  [per-day]"),
                     ("rskew", "     mean realised skewness")]:
        mu, se, lags = hac_se(daily[col].to_numpy())
        t = mu / se if se else np.nan
        rows.append({"measure": lbl, "estimate": mu, "hac_se": se, "nw_lags": lags,
                     "t_stat": t, "p_value": float(2 * (1 - stats.norm.cdf(abs(t)))),
                     "ci_lo": mu - 1.96 * se, "ci_hi": mu + 1.96 * se,
                     "n": int(daily[col].notna().sum())})
    tot = daily[["rs_plus", "rs_minus", "rv"]].sum()
    rows.append({"measure": "     pooled (sum RS+ - sum RS-)/sum RV",
                 "estimate": (tot["rs_plus"] - tot["rs_minus"]) / tot["rv"],
                 "hac_se": np.nan, "nw_lags": np.nan, "t_stat": np.nan,
                 "p_value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                 "n": len(daily)})
    return pd.DataFrame(rows).set_index("measure")


# --------------------------------------------------------------------------- #
# E1 — correlation proxies, contemporaneous vs predictive
# --------------------------------------------------------------------------- #
def e1_correlations(daily: pd.DataFrame, n_boot: int = 2000,
                    seed: int = SEED) -> pd.DataFrame:
    """The correlation-based proxies, with the mechanical one clearly separated."""
    r, dlv = daily["ret"], daily["d_log_rv"]
    rows = [
        corr_report(r, dlv, "CONTAMINATED corr(r_t, dlogRV_t)  [= skewness]",
                    n_boot, seed),
        corr_report(r, dlv.shift(-1), "CLEAN  corr(r_t, dlogRV_t+1)",
                    n_boot, seed),
        corr_report(r, np.log(daily["bv"]).diff().shift(-1),
                    "CLEAN  corr(r_t, dlogBV_t+1)   [jump-robust]", n_boot, seed),
        corr_report(r, np.log(daily["medrv"]).diff().shift(-1),
                    "CLEAN  corr(r_t, dlogMedRV_t+1) [jump-robust]", n_boot, seed),
        corr_report(np.log(daily["rv_ann"]), np.log(daily["rv_ann"]) * 0 +
                    np.log(daily["rv_ann"]), "(placeholder)", 10, seed),
    ][:-1]
    return pd.DataFrame(rows).set_index("measure")


def e2_level_comovement(daily: pd.DataFrame, spot: pd.Series,
                        n_boot: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """E2: is vol structurally higher when price is lower?

    Reported for completeness and flagged: both log S and log RV are highly
    persistent, so this correlation is a spurious-regression hazard. The
    stationary alternative — correlating CHANGES — is E1. An ADF test on each
    series is included so the reader can see the problem rather than take it on
    faith.
    """
    from statsmodels.tsa.stattools import adfuller
    s = np.log(spot.reindex(daily.index)).dropna()
    lv = daily["log_rv"].reindex(s.index)
    rows = [corr_report(s, lv, "E2  corr(log S_t, log RV_t)  [PERSISTENT — see ADF]",
                        n_boot, seed)]
    out = pd.DataFrame(rows).set_index("measure")
    for nm, ser in [("log S", s), ("log RV", lv.dropna())]:
        st, p, *_ = adfuller(ser.to_numpy(), autolag="AIC")
        out.loc[f"     ADF {nm}", ["rho", "t_neff", "p_neff", "n"]] = \
            [np.nan, st, p, len(ser)]
    return out


def e3_volatility_feedback(daily: pd.DataFrame, horizons=(1, 2, 3, 5, 10),
                           n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """E3: does a vol shock predict FUTURE returns? (feedback, not leverage)"""
    rows = []
    for h in horizons:
        fwd = daily["ret"].rolling(h).sum().shift(-h)
        rows.append(corr_report(daily["d_log_rv"], fwd,
                                f"E3  corr(dlogRV_t, r_t+1..t+{h})", n_boot, seed))
    return pd.DataFrame(rows).set_index("measure")


# --------------------------------------------------------------------------- #
# Asymmetric and LHAR regressions
# --------------------------------------------------------------------------- #
def asymmetric_regression(daily: pd.DataFrame) -> pd.DataFrame:
    """dlogRV_{t+1} = a + b+ * r+_t + b- * r-_t, HAC errors, Wald test b+ = -b-.

    Splitting the signs is strictly more informative than a correlation: a
    correlation forces one number onto a relationship that can be, and in
    equities is, asymmetric in magnitude as well as sign.
    """
    import statsmodels.api as sm
    d = daily.copy()
    d["r_pos"] = d["ret"].clip(lower=0)
    d["r_neg"] = d["ret"].clip(upper=0)
    d["y"] = d["d_log_rv"].shift(-1)
    d = d.dropna(subset=["y", "r_pos", "r_neg"])
    X = sm.add_constant(d[["r_pos", "r_neg"]])
    fit = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    w = fit.wald_test(np.array([[0.0, 1.0, 1.0]]), scalar=True)
    rows = []
    for nm in ["const", "r_pos", "r_neg"]:
        rows.append({"term": nm, "coef": fit.params[nm], "hac_se": fit.bse[nm],
                     "t": fit.tvalues[nm], "p": fit.pvalues[nm],
                     "ci_lo": fit.conf_int().loc[nm, 0],
                     "ci_hi": fit.conf_int().loc[nm, 1]})
    out = pd.DataFrame(rows).set_index("term")
    out.attrs["r2"] = fit.rsquared
    out.attrs["n"] = int(fit.nobs)
    out.attrs["wald_sym_stat"] = float(w.statistic)
    out.attrs["wald_sym_p"] = float(w.pvalue)
    return out


def lhar(daily: pd.DataFrame) -> pd.DataFrame:
    """LHAR (Corsi-Reno): HAR for vol persistence PLUS heterogeneous leverage.

        logRV_{t+1} = c + b_d logRV_t + b_w logRV_{t-4:t} + b_m logRV_{t-21:t}
                        + g_d r^-_t + g_w r^-_{t-4:t} + g_m r^-_{t-21:t} + g_p r^+_t

    gamma_d/w/m on the NEGATIVE returns are the leverage coefficients. This is
    the specification to quote: it controls for the vol persistence that makes
    every raw correlation on RV hard to interpret.
    """
    import statsmodels.api as sm
    d = daily.copy()
    lv = d["log_rv"]
    d["lv_d"] = lv
    d["lv_w"] = lv.rolling(5).mean()
    d["lv_m"] = lv.rolling(22).mean()
    rn = d["ret"].clip(upper=0)
    rp = d["ret"].clip(lower=0)
    d["rn_d"] = rn
    d["rn_w"] = rn.rolling(5).mean()
    d["rn_m"] = rn.rolling(22).mean()
    d["rp_d"] = rp
    d["y"] = lv.shift(-1)
    cols = ["lv_d", "lv_w", "lv_m", "rn_d", "rn_w", "rn_m", "rp_d"]
    d = d.dropna(subset=cols + ["y"])
    fit = sm.OLS(d["y"], sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 22})
    out = pd.DataFrame({"coef": fit.params, "hac_se": fit.bse,
                        "t": fit.tvalues, "p": fit.pvalues})
    out["ci_lo"], out["ci_hi"] = fit.conf_int()[0], fit.conf_int()[1]
    out.attrs["r2"] = fit.rsquared
    out.attrs["n"] = int(fit.nobs)
    return out


# --------------------------------------------------------------------------- #
# Horizon term structure
# --------------------------------------------------------------------------- #
def horizon_term_structure(perp_path: str, freq: str = "5min",
                           horizons=("1h", "4h", "12h", "1D", "3D", "7D"),
                           n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """E5 and the clean correlation at several return horizons.

    In equities the leverage effect strengthens with horizon. Whether ETH does
    the same is a genuine question, and the answer is more informative than any
    single-horizon number.
    """
    r = intraday_returns(perp_path, freq)
    rows = []
    for h in horizons:
        g = r.groupby(r.index.floor(h))
        rec = []
        for t, x in g:
            a = x.to_numpy()
            if len(a) < 10:
                continue
            rv = float(np.sum(a ** 2))
            if rv <= 0:
                continue
            neg = a < 0
            rec.append({"t": t, "ret": float(np.sum(a)), "rv": rv,
                        "sv": (float(np.sum(a[~neg] ** 2)) -
                               float(np.sum(a[neg] ** 2))) / rv})
        p = pd.DataFrame(rec).set_index("t").sort_index()
        if len(p) < 40:
            continue
        mu, se, _ = hac_se(p["sv"].to_numpy())
        dl = np.log(p["rv"]).diff()
        cr = corr_report(p["ret"], dl.shift(-1), f"h={h}", n_boot, seed)
        rows.append({"horizon": h, "n": len(p),
                     "sv_ratio": mu, "sv_se": se, "sv_t": mu / se if se else np.nan,
                     "corr_clean": cr.get("rho"), "corr_lo": cr.get("boot_lo"),
                     "corr_hi": cr.get("boot_hi"), "n_eff": cr.get("n_eff")})
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Causal regimes — replaces the hand-labelled, look-ahead R.REGIMES
# --------------------------------------------------------------------------- #
def causal_regimes(spot: pd.Series, dd_thresh: float = 0.20,
                   lookback: int = 90) -> pd.Series:
    """Trailing-drawdown regime label using ONLY past information.

    `R.REGIMES` in roll_engine is hand-drawn from the finished price chart
    (peak-to-trough by eye), so every statistic conditioned on it is
    contaminated by hindsight. This is the causal replacement: at each date,
    drawdown from the running max of the PRECEDING `lookback` days. No centred
    filter, no future prices.
    """
    s = spot.dropna()
    peak = s.rolling(lookback, min_periods=20).max()
    dd = s / peak - 1.0
    out = pd.Series("neutral", index=s.index, dtype=object)
    out[dd <= -dd_thresh] = "drawdown"
    out[(dd > -0.05)] = "near-high"
    out.name = "regime_causal"
    return out


# --------------------------------------------------------------------------- #
# Phase 2 — Monte Carlo: can the pipeline recover a KNOWN rho?
# --------------------------------------------------------------------------- #
def simulate_heston(rho: float, n_days: int, steps_per_day: int = 288,
                    v0: float = 0.42, kappa: float = 12.0, theta: float = 0.42,
                    nu: float = 2.0, n_paths: int = 8,
                    seed: int = SEED) -> pd.DataFrame:
    """Heston paths aggregated to per-day realised measures, memory-free.

    Vectorised over PATHS and looped over time (variance is path-dependent, so
    time cannot be vectorised). Daily measures are accumulated online, so no
    full price path is ever stored — otherwise 6 rho values x 8 paths x 1.1M
    steps would be gigabytes.

    Defaults are calibrated loosely to ETH: theta = 0.42 -> long-run vol ~65%,
    kappa = 12/yr -> vol half-life ~3 weeks, nu = 2.0 keeps Feller (2*k*th=10.1
    >= nu^2=4). `steps_per_day=288` matches 5-minute sampling.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / (TRADING_DAYS * steps_per_day)
    sq = np.sqrt(dt)
    v = np.full(n_paths, v0)
    logS = np.zeros(n_paths)
    rows = []
    for day in range(n_days):
        d_open = logS.copy()
        rv = np.zeros(n_paths); rsp = np.zeros(n_paths); rsm = np.zeros(n_paths)
        for _ in range(steps_per_day):
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
            vs = np.sqrt(np.maximum(v, 0.0))
            dlog = -0.5 * v * dt + vs * sq * z1
            logS += dlog
            v = np.maximum(v + kappa * (theta - v) * dt + nu * vs * sq * z2, 0.0)
            rv += dlog ** 2
            pos = dlog > 0
            rsp += np.where(pos, dlog ** 2, 0.0)
            rsm += np.where(~pos, dlog ** 2, 0.0)
        for p in range(n_paths):
            rows.append({"path": p, "day": day, "ret": logS[p] - d_open[p],
                         "rv": rv[p], "rs_plus": rsp[p], "rs_minus": rsm[p],
                         "sv_ratio": (rsp[p] - rsm[p]) / rv[p] if rv[p] > 0 else np.nan})
    return pd.DataFrame(rows)


def mc_recovery(rhos=(-0.8, -0.5, -0.2, 0.0, 0.3), n_days: int = 400,
                n_paths: int = 8, steps_per_day: int = 288,
                seed: int = SEED) -> pd.DataFrame:
    """For each true rho, what do our estimators report?

    This is the acceptance gate. If an estimator cannot recover rho = -0.8 with
    the right sign and a usable magnitude on the sample size we actually have,
    then no empirical number it produces is interpretable.
    """
    rows = []
    for i, rho in enumerate(rhos):
        sim = simulate_heston(rho, n_days, steps_per_day, n_paths=n_paths,
                              seed=seed + i)
        per = []
        for p, g in sim.groupby("path"):
            g = g.sort_values("day")
            dl = np.log(g["rv"]).diff()
            clean = g["ret"].corr(dl.shift(-1))
            contam = g["ret"].corr(dl)
            per.append({"sv": g["sv_ratio"].mean(), "clean": clean,
                        "contam": contam})
        d = pd.DataFrame(per)
        rows.append({"true_rho": rho, "n_paths": n_paths, "n_days": n_days,
                     "sv_mean": d["sv"].mean(), "sv_sd": d["sv"].std(),
                     "clean_mean": d["clean"].mean(), "clean_sd": d["clean"].std(),
                     "clean_lo": d["clean"].quantile(.05),
                     "clean_hi": d["clean"].quantile(.95),
                     "contam_mean": d["contam"].mean(),
                     "frac_clean_negative": float((d["clean"] < 0).mean())})
    return pd.DataFrame(rows).set_index("true_rho")


def placebo_null(daily: pd.DataFrame, n_rep: int = 2000,
                 seed: int = SEED) -> dict:
    """Noise floor: shuffle returns, destroying the r/RV link but preserving both
    marginals. Any observed statistic inside this band is indistinguishable from
    nothing."""
    rng = np.random.default_rng(seed)
    r = daily["ret"].to_numpy()
    dl = daily["d_log_rv"].shift(-1).to_numpy()
    m = np.isfinite(r) & np.isfinite(dl)
    r, dl = r[m], dl[m]
    out = np.array([np.corrcoef(rng.permutation(r), dl)[0, 1] for _ in range(n_rep)])
    sv = daily["sv_ratio"].to_numpy()
    sv = sv[np.isfinite(sv)]
    svn = np.array([np.mean(rng.permutation(sv) * rng.choice([-1, 1], len(sv)))
                    for _ in range(n_rep)])
    return {"corr_p2.5": float(np.percentile(out, 2.5)),
            "corr_p97.5": float(np.percentile(out, 97.5)),
            "corr_sd": float(out.std()),
            "sv_p2.5": float(np.percentile(svn, 2.5)),
            "sv_p97.5": float(np.percentile(svn, 97.5))}


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def signature_plot(perp_path: str,
                   freqs=("1min", "2min", "5min", "10min", "15min", "30min", "1h")
                   ) -> pd.DataFrame:
    """Volatility signature plot: mean annualised RV by sampling frequency.

    The classic microstructure diagnostic. RV rising as the frequency increases
    is bid-ask bounce; the plateau is where the estimate is least contaminated.
    Also reports the relative standard error sqrt(2/n) implied by each frequency,
    which is the attenuation budget for every correlation computed on it.
    """
    rows = []
    for f in freqs:
        r = intraday_returns(perp_path, f)
        d = daily_measures(r)
        n = d["n_obs"].median()
        rows.append({"freq": f, "obs_per_day": int(n),
                     "mean_rv_ann": d["rv_ann"].mean(),
                     "median_rv_ann": d["rv_ann"].median(),
                     "rel_se_pct": 100 * np.sqrt(2.0 / n),
                     "n_days": len(d)})
    return pd.DataFrame(rows).set_index("freq")


def influence_drop(daily: pd.DataFrame, drop_pct: float = 0.005) -> pd.DataFrame:
    """Recompute the headline numbers after dropping the largest |r| days.

    If a result flips, the "leverage effect" is a handful of liquidation cascades
    and must be reported as such rather than as a property of the sample.
    """
    rows = []
    k = max(1, int(len(daily) * drop_pct))
    keep = daily.drop(daily["ret"].abs().nlargest(k).index)
    for lbl, d in [("all days", daily), (f"drop top {k} |r| days", keep)]:
        mu, se, _ = hac_se(d["sv_ratio"].to_numpy())
        c = d["ret"].corr(d["d_log_rv"].shift(-1))
        rows.append({"sample": lbl, "n": len(d), "sv_ratio": mu, "sv_se": se,
                     "sv_t": mu / se if se else np.nan, "corr_clean": c})
    return pd.DataFrame(rows).set_index("sample")


def by_regime(daily: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Pooled statistics per causal regime, with n_eff and Fisher CIs."""
    rows = []
    reg = regimes.reindex(daily.index).ffill()
    for name, idx in daily.groupby(reg).groups.items():
        d = daily.loc[idx]
        if len(d) < 40:
            continue
        mu, se, _ = hac_se(d["sv_ratio"].to_numpy())
        cr = corr_report(d["ret"], d["d_log_rv"].shift(-1), str(name), n_boot=800)
        rows.append({"regime": name, "n": len(d), "mean_rv_ann": d["rv_ann"].mean(),
                     "sv_ratio": mu, "sv_se": se, "sv_t": mu / se if se else np.nan,
                     "corr_clean": cr.get("rho"), "n_eff": cr.get("n_eff"),
                     "boot_lo": cr.get("boot_lo"), "boot_hi": cr.get("boot_hi")})
    return pd.DataFrame(rows).set_index("regime")


def spike_decay_split(daily: pd.DataFrame, sigma_mult: float = 3.0,
                      window: int = 2) -> pd.DataFrame:
    """Test the spike-then-decay cancellation hypothesis.

    Prediction: strongly negative right after a big down day (vol spiking), then
    positive or zero over the following grind (vol decaying while price keeps
    falling), netting to ~zero. `sigma_mult` is in units of the TRAILING 60-day
    return sd, so the threshold uses no future information.
    """
    d = daily.copy()
    sd = d["ret"].rolling(60, min_periods=20).std().shift(1)
    shock = d["ret"] < -sigma_mult * sd
    after = shock.rolling(window + 1, min_periods=1).max().astype(bool)
    d["bucket"] = np.where(shock, "shock day",
                           np.where(after, f"within {window}d of shock", "remainder"))
    rows = []
    for b, g in d.groupby("bucket"):
        if len(g) < 20:
            rows.append({"bucket": b, "n": len(g)})
            continue
        mu, se, _ = hac_se(g["sv_ratio"].to_numpy())
        rows.append({"bucket": b, "n": len(g), "sv_ratio": mu, "sv_se": se,
                     "corr_contemp": g["ret"].corr(g["d_log_rv"]),
                     "mean_dlogrv": g["d_log_rv"].mean(),
                     "mean_rv_ann": g["rv_ann"].mean()})
    return pd.DataFrame(rows).set_index("bucket")
