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

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


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
