"""
Gaussian hidden Markov model for estimating the market state behind the option
strategies — written out rather than pulled from a library so that the one
distinction that decides whether a regime backtest is valid stays visible in
the code:

    FILTERED  P(s_t | x_1..x_t)      uses only the past. Tradeable.
    SMOOTHED  P(s_t | x_1..x_T)      uses the whole sample. Descriptive ONLY.

`hmmlearn.predict()` (Viterbi) and every `score_samples` posterior are SMOOTHED.
Backtests built on them look excellent and mean nothing, because the state at
time t was inferred partly from what happened after t. `smoothed_states` here is
exposed for description and is named so it cannot be used by accident;
`walk_forward_states` is the one to backtest on — it also refits the parameters
on an expanding window, so not even the fitted means and covariances contain
information from the future.

Why this exists at all: the regimes used so far were drawn by hand after looking
at the price path, which makes any statement like "this strategy works in bear
markets" circular — nothing could have known the label at the time. An HMM
filtered state is available at the decision instant, so it turns a description
into a testable rule.

What it is NOT for: switching between long-call and short-call structures. Those
structures were measured to carry a residual beta of +/-0.13 to the underlying
(delta-hedging removes delta, not vanna), so alternating between them on a state
signal is directional timing wearing an options costume — and one that pays an
11% option spread to express what the perp expresses for 5bp. Use the states to
size a genuinely neutral structure, or to condition the premium measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

_LOG2PI = float(np.log(2.0 * np.pi))

#: Default feature set. All are observable AT the roll instant — no forward
#: realised vol, no regime label, nothing that needs the future. `rv_past` is the
#: vol over the week just ended, `iv_atm` what the market prices for the week
#: ahead, their difference the instantaneous premium, `rr25` the skew.
DEFAULT_FEATURES = ("log_rv_past", "iv_atm", "iv_minus_rv", "rr25")


# --------------------------------------------------------------------------- #
# Core math (log space throughout — a 100+ observation product of densities
# underflows float64 long before it becomes interesting)
# --------------------------------------------------------------------------- #
def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


def _log_gaussian(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """(T, K) log density of each observation under each state's Gaussian."""
    T, D = X.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        cov = covs[k]
        sign, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.inv(cov)
        d = X - means[k]
        maha = np.einsum("ij,jk,ik->i", d, inv, d)
        out[:, k] = -0.5 * (D * _LOG2PI + logdet + maha)
    return out


def _forward(log_b: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
    """Forward pass. Returns (log_alpha, loglik). `log_alpha[t]` normalised is
    exactly the FILTERED state probability at t."""
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    return log_alpha, _logsumexp(log_alpha[-1])


def _backward(log_b: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    T, K = log_b.shape
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
    return log_beta


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GaussianHMM:
    """Baum-Welch fitted Gaussian HMM.

    `n_states` defaults to 2 deliberately. A 3-state model on ~110 weekly
    observations spends ~15 parameters on the states alone, before any strategy
    is selected per state — with a Sharpe standard error of 0.68 on this sample
    there is nothing left to validate against.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-4,
                 reg: float = 1e-6, seed: int = 0):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg          # ridge on covariances: a state can collapse onto
        self.seed = seed        # a single point and send the likelihood to +inf

    # ---- fitting ---- #
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        K = self.n_states
        rng = np.random.default_rng(self.seed)

        # init: k-means-ish split on the first feature, which for our use is the
        # vol level — gives the states a stable, interpretable ordering to start
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, K)
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks])
        self.covs_ = np.array([np.cov(X[c].T).reshape(D, D) + self.reg * np.eye(D)
                               for c in chunks])
        self.startprob_ = np.full(K, 1.0 / K)
        self.transmat_ = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(self.transmat_, 0.9)

        prev = -np.inf
        for _ in range(self.n_iter):
            log_b = _log_gaussian(X, self.means_, self.covs_)
            log_pi, log_A = np.log(self.startprob_ + 1e-300), np.log(self.transmat_ + 1e-300)
            log_alpha, ll = _forward(log_b, log_pi, log_A)
            log_beta = _backward(log_b, log_A)

            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)

            xi = np.zeros((K, K))
            for t in range(T - 1):
                m = (log_alpha[t][:, None] + log_A
                     + (log_b[t + 1] + log_beta[t + 1])[None, :])
                xi += np.exp(m - _logsumexp(m))

            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi / xi.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            for k in range(K):
                d = X - self.means_[k]
                self.covs_[k] = (gamma[:, k][:, None] * d).T @ d / w[k] + self.reg * np.eye(D)

            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.loglik_ = float(prev)
        self.n_params_ = K * D + K * D * (D + 1) // 2 + K * (K - 1) + (K - 1)
        self.bic_ = -2 * self.loglik_ + self.n_params_ * np.log(T)
        return self

    # ---- inference ---- #
    def filtered_states(self, X: np.ndarray) -> np.ndarray:
        """P(s_t | x_1..x_t) — uses ONLY information up to t. This is the one to
        condition a trading decision on."""
        X = np.asarray(X, dtype=float)
        log_b = _log_gaussian(X, self.means_, self.covs_)
        log_alpha, _ = _forward(log_b, np.log(self.startprob_ + 1e-300),
                                np.log(self.transmat_ + 1e-300))
        return np.exp(log_alpha - _logsumexp(log_alpha, axis=1)[:, None])

    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        """P(s_t | x_1..x_T) — conditions on the WHOLE sample, including the
        future. Legitimate for describing history, never for a backtest."""
        X = np.asarray(X, dtype=float)
        log_b = _log_gaussian(X, self.means_, self.covs_)
        log_A = np.log(self.transmat_ + 1e-300)
        log_alpha, _ = _forward(log_b, np.log(self.startprob_ + 1e-300), log_A)
        log_beta = _backward(log_b, log_A)
        g = log_alpha + log_beta
        return np.exp(g - _logsumexp(g, axis=1)[:, None])


def walk_forward_states(X: np.ndarray, n_states: int = 2, min_train: int = 40,
                        refit_every: int = 4, seed: int = 0) -> np.ndarray:
    """Filtered state probabilities with NO look-ahead of any kind.

    At each step the model is refitted on x_1..x_t only and the filtered
    probability at t is taken — so neither the state inference nor the fitted
    means/covariances/transitions have seen the future. `refit_every` trades a
    little fidelity for speed; the filter itself still runs to t on every step.

    The first `min_train` rows come back as NaN: there is genuinely no estimate
    available then, and returning a default would silently create the very
    look-ahead this function exists to avoid.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    out = np.full((T, n_states), np.nan)
    model = None
    for t in range(min_train, T):
        if model is None or (t - min_train) % refit_every == 0:
            model = GaussianHMM(n_states=n_states, seed=seed).fit(X[:t + 1])
        out[t] = model.filtered_states(X[:t + 1])[-1]
    return out


# --------------------------------------------------------------------------- #
# Features, discovery, reporting
# --------------------------------------------------------------------------- #
def build_features(option_path: str, perp_path: str, frequency: str = "weekly",
                   rv_freq: str = "1h") -> pd.DataFrame:
    """Feature panel at each roll date, from the same machinery `vrp_signal`
    uses — one source of truth for how IV and RV are extracted."""
    from volatility_surface.backtest import roll_engine as R
    from volatility_surface.backtest import vrp_signal as V

    rets = V.perp_log_returns(perp_path, freq=rv_freq)
    lo = int(rets.index[0].timestamp() * 1000)
    hi = int(rets.index[-1].timestamp() * 1000)
    grid = R.roll_grid(lo, hi, frequency)
    df = V.iv_by_delta_series(option_path, grid)
    if df.empty:
        return df

    horizon = pd.Timedelta(milliseconds=int(np.median(np.diff(grid))))
    df["rv_past"] = [V.realised_vol(rets, d - horizon, d) for d in df["date"]]
    df["log_rv_past"] = np.log(df["rv_past"])
    df["iv_minus_rv"] = df["iv_atm"] - df["rv_past"]
    return df.dropna(subset=list(DEFAULT_FEATURES)).reset_index(drop=True)


def _standardise(X: np.ndarray) -> np.ndarray:
    """Features live on wildly different scales (iv_atm ~0.65 vs rr25 ~-0.02);
    without this the covariance — and therefore the state assignment — is
    decided almost entirely by the largest-scale column."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def discover(df: pd.DataFrame, n_states: int = 2,
             features: tuple[str, ...] = DEFAULT_FEATURES,
             walk_forward: bool = False, min_train: int = 40,
             seed: int = 0) -> pd.DataFrame:
    """Fit the HMM and label each roll date.

    `walk_forward=False` (default) uses SMOOTHED states: the right tool for
    "what periods are in this history", and the wrong one for any P&L claim.
    `walk_forward=True` gives what was knowable at each date.
    """
    X = _standardise(df[list(features)].to_numpy(dtype=float))
    out = df.copy()
    if walk_forward:
        probs = walk_forward_states(X, n_states=n_states, min_train=min_train, seed=seed)
        out["state"] = np.where(np.isnan(probs[:, 0]), -1, np.nanargmax(probs, axis=1))
    else:
        model = GaussianHMM(n_states=n_states, seed=seed).fit(X)
        probs = model.smoothed_states(X)
        out["state"] = probs.argmax(axis=1)
        out.attrs["model"] = model
    for k in range(n_states):
        out[f"p_state{k}"] = probs[:, k]
    return out


def segments(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of the same state — the 'periods it discovered'."""
    rows, start, cur = [], 0, df["state"].iloc[0]
    for i in range(1, len(df) + 1):
        if i == len(df) or df["state"].iloc[i] != cur:
            a, b = df["date"].iloc[start], df["date"].iloc[i - 1]
            rows.append({"state": int(cur), "start": a.date(), "end": b.date(),
                         "days": (b - a).days, "n_rolls": i - start})
            if i < len(df):
                start, cur = i, df["state"].iloc[i]
    return pd.DataFrame(rows)


def state_profile(df: pd.DataFrame, features: tuple[str, ...] = DEFAULT_FEATURES) -> pd.DataFrame:
    """Mean of each feature per state — this is what makes a state interpretable
    ('state 1 is the high-vol, negative-skew one') rather than a bare index."""
    return df.groupby("state")[list(features)].mean().round(4).join(
        df.groupby("state").size().rename("n_rolls"))


def compare_to_regimes(df: pd.DataFrame, regimes: dict) -> pd.DataFrame:
    """Share of each hand-drawn regime's rolls falling in each discovered state.

    The question this answers: did the labels drawn by looking at the price path
    correspond to anything a model could have identified without that hindsight?
    """
    rows = []
    for name, (a, b) in regimes.items():
        m = (df["date"] >= pd.Timestamp(a, tz="UTC")) & (df["date"] < pd.Timestamp(b, tz="UTC"))
        seg = df[m]
        if seg.empty:
            continue
        r = {"regime": name, "n_rolls": len(seg)}
        for k in sorted(df["state"].unique()):
            if k >= 0:
                r[f"state{k}"] = float((seg["state"] == k).mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("regime").round(3)


if __name__ == "__main__":
    import argparse
    from volatility_surface.backtest import roll_engine as R

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", required=True)
    ap.add_argument("--perp", required=True)
    ap.add_argument("--frequency", default="weekly")
    ap.add_argument("--rv-frequency", default="1h")
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--walk-forward", action="store_true",
                    help="use only information available at each date (harder, honest)")
    ap.add_argument("--min-train", type=int, default=40)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    feats = build_features(a.options, a.perp, a.frequency, a.rv_frequency)
    if feats.empty:
        raise SystemExit("no usable rolls — check the option file covers the perp range")
    print(f"\n{len(feats)} roll dates, {feats['date'].min():%Y-%m-%d} -> {feats['date'].max():%Y-%m-%d}")

    df = discover(feats, n_states=a.states, walk_forward=a.walk_forward,
                  min_train=a.min_train)
    mode = "WALK-FORWARD (only past information)" if a.walk_forward else \
           "SMOOTHED (whole sample — descriptive only, never backtest on this)"
    print(f"\n{'=' * 72}\nSTATE PROFILE   [{mode}]\n{'=' * 72}")
    print(state_profile(df).to_string())
    m = df.attrs.get("model")
    if m is not None:
        print(f"\n  persistence (diag of transition matrix): {np.diag(m.transmat_).round(3)}")
        print(f"  loglik {m.loglik_:.1f} | BIC {m.bic_:.1f} | {m.n_params_} params")

    print(f"\n{'=' * 72}\nPERIODS DISCOVERED\n{'=' * 72}")
    print(segments(df).to_string(index=False))

    print(f"\n{'=' * 72}\nOVERLAP WITH THE HAND-DRAWN REGIMES\n{'=' * 72}")
    print(compare_to_regimes(df, R.REGIMES).to_string())
    print("\n  A regime that splits across states was not one thing; two regimes")
    print("  landing in the same state were not two things.")

    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\nsaved -> {a.csv}")
