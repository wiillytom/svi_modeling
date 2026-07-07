"""
Live option-chain Streamlit app.

Each tick:
  1. advance to the next cached ETH snapshot,
  2. run a fast `theta_only` eSSVI update (~30 ms),
  3. recompute IV, Black-76 prices and greeks for every listed strike,
  4. repaint the chain table.

Run with:
    streamlit run volatility_surface/utils/streamlit_chain.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` until we hit the folder that contains
    the '2 - Data' data directory. Robust to how deeply the streamlit
    script is nested inside the code tree."""
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "2 - Data").is_dir():
            return candidate
    return here.parent.parent   # fall back to the old assumption

PROJECT_ROOT = _find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from volatility_surface.core.calibration.calibrator import (
    calibrate_global_essvi, calibrate_global_essvi_update,
    calibrate_snapshot, calibrate_snapshot_sabr_update,
    crossedness, load_snapshot,
)
from volatility_surface.models.vol_models import eSSVI, SABR


MODELS = ("eSSVI", "SABR")


LIVE_DIR        = PROJECT_ROOT / "2 - Data" / "live"
ARCHIVE_DIR     = PROJECT_ROOT / "2 - Data" / "parquets"
PURPLE  = "#6A0DAD"


def _live_parquet_for(ccy: str) -> Path:
    return LIVE_DIR / f"live_{ccy}.parquet"


def _replay_parquet_for(ccy: str) -> Path:
    """Pre-recorded snapshot file used in replay mode.
    Put your file at 2 - Data/live/replay_<ccy>.parquet."""
    return LIVE_DIR / f"replay_{ccy}.parquet"


def _archive_parquet_for(ccy: str) -> Path:
    return ARCHIVE_DIR / f"{ccy}_options_data_cleaned.parquet"


def _detect_freshest_currency() -> str:
    """Return whichever live_{ccy}.parquet was modified most recently.
    Falls back to 'eth' if none exist."""
    candidates = [(c, _live_parquet_for(c)) for c in ("eth", "btc")]
    existing  = [(c, p, p.stat().st_mtime) for c, p in candidates if p.exists()]
    if not existing:
        return "eth"
    existing.sort(key=lambda x: x[2], reverse=True)
    return existing[0][0]

MATURITY_BUCKETS_DAYS = [1, 3, 7, 14, 30, 60, 90, 180]
STALE_SECONDS         = 10  # warn the user if the latest tick is older than this


# ─────────────────────────────────────────────────────────────────────────────
# Black-76 pricing & greeks (no discounting — Deribit options are perpetual-USD)
# ─────────────────────────────────────────────────────────────────────────────

def bs_pricing(F, K, T, iv, is_call):
    """Vectorised Black-76 call/put price + greeks. r = 0."""
    iv = np.maximum(iv, 1e-6)
    T_e = np.maximum(T, 1e-6)
    sqrtT = np.sqrt(T_e)
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T_e) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    pdf_d1 = norm.pdf(d1)

    call_price = F * norm.cdf(d1) - K * norm.cdf(d2)
    put_price  = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)

    delta = np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
    gamma = pdf_d1 / (F * iv * sqrtT)
    vega  = F * pdf_d1 * sqrtT * 0.01            # per 1 vol-point
    theta = -F * pdf_d1 * iv / (2 * sqrtT) / 365  # per calendar day

    return price, delta, gamma, vega, theta


# ─────────────────────────────────────────────────────────────────────────────
# Cached data + cached initial fit (one-time per snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def _which_parquet(ccy: str) -> Path:
    """
    Source-file priority:
      1. replay parquet   — if the user is in replay mode AND the file exists
      2. live parquet     — normal "gatherer is running" case
      3. archive parquet  — historical fallback
    """
    if st.session_state.get("replay_mode", False):
        rep = _replay_parquet_for(ccy)
        if rep.exists():
            return rep
    live = _live_parquet_for(ccy)
    if live.exists():
        return live
    return _archive_parquet_for(ccy)


def _latest_timestamp(ccy: str) -> str:
    p = _which_parquet(ccy)
    df = pd.read_parquet(p, columns=["file_timestamp"])
    return str(df["file_timestamp"].max())


@st.cache_data(show_spinner=False)
def _sorted_timestamps(ccy: str, mtime: float) -> list:
    """All distinct file_timestamps in the parquet, in chronological order.
    `mtime` is only passed so the cache invalidates when the file changes."""
    p = _which_parquet(ccy)
    df = pd.read_parquet(p, columns=["file_timestamp"])
    return sorted(df["file_timestamp"].unique().tolist())


def _load_snapshot_at(ccy: str, index: int | None = None) -> pd.DataFrame:
    """Load the snapshot at position `index` in chronological order (0-based).
    If `index` is None, returns the latest — the original behaviour."""
    p = _which_parquet(ccy)
    df = pd.read_parquet(p)
    if index is None:
        target_ts = df["file_timestamp"].max()
    else:
        ts_list = _sorted_timestamps(ccy, p.stat().st_mtime)
        target_ts = ts_list[index % len(ts_list)]     # loop at end
    snap = df[df["file_timestamp"] == target_ts].copy()
    snap["t"] = snap["t"].round(4)
    snap = snap[(snap["t"] > 1e-5) & (snap["mark_iv"] > 1e-4)
                & (snap["mark_iv"] < 5.0) & (snap["w"] > 0)]
    return snap.dropna(subset=["k", "w", "t", "mark_iv"]).reset_index(drop=True)


def _load_latest_snapshot(ccy: str) -> pd.DataFrame:
    """Compat shim: honours session-state replay-mode toggle."""
    if st.session_state.get("replay_mode", False):
        idx = st.session_state.get("replay_idx", 0)
        return _load_snapshot_at(ccy, idx)
    return _load_snapshot_at(ccy, None)


@st.cache_resource(show_spinner=True)
def get_initial_fit(ccy: str, model_name: str, timestamp_key: str, refit_token: int):
    """Full calibration of the chosen model on the latest snapshot.
    Cached by (ccy, model_name, ts, refit_token) — switching model never
    triggers a recalibration once each model has been fitted once."""
    snap = _load_latest_snapshot(ccy)
    otm_C = (snap["option_type"] == "C") & (snap["k"] >= 0)
    otm_P = (snap["option_type"] == "P") & (snap["k"] <= 0)
    snap_cal = snap[otm_C | otm_P].copy()

    if model_name == "eSSVI":
        m = eSSVI()
        res = calibrate_global_essvi(
            snap_cal, model=m, objective="vega_wmse",
            vega_col="vega", bid_col="bid_iv", ask_col="ask_iv",
            verbose=False,
        )
    elif model_name == "SABR":
        m = SABR()
        res = calibrate_snapshot(
            snap_cal, model=m, objective="vega_wmse",
            vega_col="vega", bid_col="bid_iv", ask_col="ask_iv",
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return snap, res, m


def _detect_calendar_arb(model, expiries, params, tol=1e-6):
    """Return (has_arb, t_short, t_long, max_violation) for the worst adjacent
    pair that violates calendar monotonicity."""
    order = list(np.argsort(expiries))
    worst_t1 = worst_t2 = None
    worst_val = 0.0
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        v = crossedness(model, params[a], params[b])
        if v > worst_val:
            worst_val = v
            worst_t1, worst_t2 = expiries[a], expiries[b]
    return (worst_val > tol, worst_t1, worst_t2, worst_val)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ETH option chain — live", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #ffffff; }}
      h1, h2, h3 {{ color: {PURPLE}; }}
      .block-container {{ padding-top: 1.2rem; }}
      .live-pulse {{
        display:inline-block; width:10px; height:10px; border-radius:50%;
        background:#3ec92e; margin-right:6px;
        animation: pulse 1s infinite;
      }}
      @keyframes pulse {{ 0%{{opacity:.4}} 50%{{opacity:1}} 100%{{opacity:.4}} }}

      /* Sticky maturity tabs — stay pinned while the chain table scrolls */
      .stTabs [data-baseweb="tab-list"] {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: #ffffff;
        padding: 0.5rem 0 0.25rem 0;
        margin-top: -0.5rem;
        border-bottom: 2px solid {PURPLE};
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
      }}
      .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 0.95rem;
      }}
      .stTabs [aria-selected="true"] {{
        color: {PURPLE};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Live chain — eSSVI")

default_ccy = _detect_freshest_currency()
ccy = st.sidebar.selectbox(
    "Currency",
    options=("eth", "btc"),
    index=("eth", "btc").index(default_ccy),
    format_func=str.upper,
)

model_name = st.sidebar.selectbox("Model", options=MODELS, index=0)

source_path = _which_parquet(ccy)
if source_path == _replay_parquet_for(ccy):
    src_label = "🔁 replay"
elif source_path == _live_parquet_for(ccy):
    src_label = "🟢 live"
else:
    src_label = "📦 archive"
st.sidebar.markdown(f"**Source**: {src_label}  \n`{source_path.name}`")

# ── Replay mode: step through a pre-recorded parquet as if live ──────────────
# Use case: work machine has no Deribit access.  Record N seconds of the live
# parquet at home, ship it to the work machine, then flip this toggle to have
# the app iterate through timestamps one per tick instead of pinning "latest".
replay_mode = st.sidebar.checkbox(
    "Replay mode",
    value=False,
    help="Iterate through every timestamp in the parquet one per tick "
         "(loops at the end). Off = pin to latest, the normal live behaviour."
)
st.session_state["replay_mode"] = replay_mode
if replay_mode:
    _ts_list = _sorted_timestamps(ccy, source_path.stat().st_mtime)
    st.sidebar.caption(
        f"Replay: {len(_ts_list)} snapshots  "
        f"(from `{_ts_list[0]}` to `{_ts_list[-1]}`)"
    )
    _idx = st.session_state.get("replay_idx", 0)
    st.sidebar.progress(
        (_idx % len(_ts_list)) / max(len(_ts_list) - 1, 1),
        text=f"snapshot {(_idx % len(_ts_list)) + 1} / {len(_ts_list)}"
    )
    if st.sidebar.button("↺ Restart replay from t=0"):
        st.session_state["replay_idx"] = 0

# Per-(ccy, model) state — cached fits + warm-start anchors live independently
if "refit_tokens"  not in st.session_state: st.session_state.refit_tokens = {}
if "prev_results"  not in st.session_state: st.session_state.prev_results = {}
if "anchor_ts"     not in st.session_state: st.session_state.anchor_ts = {}

fit_key = (ccy, model_name)
refit_token = st.session_state.refit_tokens.get(fit_key, 0)

# Skip the cold-fit path entirely when session_state already has a recent fit
# for this (ccy, model).  The per-tick fast-path has been keeping prev_results
# fresh — so switching models reuses that fit instead of re-calibrating.
if fit_key in st.session_state.prev_results:
    anchor_res  = st.session_state.prev_results[fit_key]
    model       = anchor_res["_model"]
    anchor_snap = _load_latest_snapshot(ccy)
    st.sidebar.caption(f"♻︎ reusing cached {model_name} fit")
else:
    if fit_key not in st.session_state.anchor_ts:
        st.session_state.anchor_ts[fit_key] = _latest_timestamp(ccy)
    with st.spinner(f"Cold-fitting {model_name} on {ccy.upper()} …"):
        anchor_snap, anchor_res, model = get_initial_fit(
            ccy, model_name, st.session_state.anchor_ts[fit_key], refit_token,
        )
    st.session_state.prev_results[fit_key] = anchor_res
expiries = anchor_res["expiries"]

st.sidebar.markdown("### Live loop")
live = st.sidebar.checkbox("Auto-update every tick", value=True)
tick_ms = st.sidebar.slider("Tick interval (ms)", 250, 5000, 1000, step=250)
show_smile = st.sidebar.checkbox("Show smile plot below table", value=False)

st.sidebar.markdown("### Maintenance")
if st.sidebar.button(f"Force full {model_name} refit"):
    st.session_state.refit_tokens[fit_key] = refit_token + 1
    st.session_state.anchor_ts[fit_key]    = _latest_timestamp(ccy)
    st.session_state.prev_results.pop(fit_key, None)
    st.rerun()


# ── persistent state ─────────────────────────────────────────────────────────
if "iter" not in st.session_state:
    st.session_state.iter = 0


# ── maturity tabs ────────────────────────────────────────────────────────────
def _nearest_expiry_idx(target_days: float, exp_list) -> int:
    target_T = target_days / 365.0
    return int(np.argmin(np.abs(np.asarray(exp_list) - target_T)))


available_buckets = [d for d in MATURITY_BUCKETS_DAYS
                     if min(expiries) * 365 <= d <= max(expiries) * 365 + 1]
if not available_buckets:                       # extreme edge case
    available_buckets = [int(round(expiries[len(expiries)//2] * 365))]

# Top-level banner placeholder for calendar-spread arbitrage warnings
arb_banner = st.empty()

tab_objs = st.tabs([f"{d}d" for d in available_buckets])

# Bind one set of paint-placeholders per tab so st.empty() repaints stay local
tab_boxes = []
for tab, days in zip(tab_objs, available_buckets):
    with tab:
        tab_boxes.append({
            "days":     days,
            "header":   st.empty(),
            "metrics":  st.empty(),
            "table":    st.empty(),
            "smile":    st.empty(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a single frame (re-reads the live parquet, runs theta_only)
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(frame_idx: int):
    df = _load_latest_snapshot(ccy)
    if df.empty:
        for tb in tab_boxes:
            tb["table"].warning("No data in the live parquet yet — is the gatherer running?")
        return 0.0, 0
    ts = df["file_timestamp"].iloc[0]

    # Calibration uses OTM-only — ITM marks are parity duplicates of the
    # opposite side and would double-weight dual-quoted strikes. Display still
    # uses the full df so the chain table shows real ITM quotes.
    otm_C  = (df["option_type"] == "C") & (df["k"] >= 0)
    otm_P  = (df["option_type"] == "P") & (df["k"] <= 0)
    df_cal = df[otm_C | otm_P].copy()

    # Staleness check (live parquet only)
    try:
        ts_dt = pd.to_datetime(ts)
        age = (pd.Timestamp.now() - ts_dt).total_seconds()
    except Exception:
        age = None

    # Fast-path update — dispatches by model
    t_fit_start = time.perf_counter()
    prev_result = st.session_state.prev_results[fit_key]
    try:
        if model_name == "eSSVI":
            res = calibrate_global_essvi_update(
                df_cal, prev_result=prev_result, model=model,
                objective="vega_wmse", mode="theta_only",
                vega_col="vega", bid_col="bid_iv", ask_col="ask_iv",
                verbose=False,
            )
        elif model_name == "SABR":
            res = calibrate_snapshot_sabr_update(
                df_cal, prev_result=prev_result, model=model,
                objective="vega_wmse",
                vega_col="vega", bid_col="bid_iv", ask_col="ask_iv",
                verbose=False,
            )
        else:
            raise RuntimeError(f"Unhandled model {model_name!r}")
        st.session_state.prev_results[fit_key] = res
    except Exception as e:
        res = prev_result
        print(f"  fast-path update failed at {ts} ({model_name}): {e}")
    fit_ms = (time.perf_counter() - t_fit_start) * 1000

    # Calendar-spread arbitrage check across all calibrated expiries
    has_arb, t_short, t_long, viol = _detect_calendar_arb(
        model, np.asarray(res["expiries"]), res["params"],
    )
    if has_arb:
        arb_banner.error(
            f"🚨  Calendar-spread arbitrage detected between "
            f"T = {t_short*365:.1f}d and T = {t_long*365:.1f}d  "
            f"(max w-violation = {viol:.4g})",
            icon="🚨",
        )
    else:
        arb_banner.empty()

    exp_arr = np.asarray(res["expiries"])

    # Render one tab per maturity bucket
    for tb in tab_boxes:
        _render_tab(tb, df, res, exp_arr, ts, age, fit_ms, frame_idx)

    return fit_ms, len(exp_arr)


def _render_tab(tb, df, res, exp_arr, ts, age, fit_ms, frame_idx):
    """Render one expiry-bucket tab into its placeholders."""
    days = tb["days"]
    i_match = _nearest_expiry_idx(days, exp_arr)
    T = float(exp_arr[i_match])
    params = res["params"][i_match]

    slice_df = df[np.isclose(df["t"], T)].copy()
    if slice_df.empty:
        tb["table"].warning(f"No quotes for T = {T:.4f} ({int(T*365)}d) at {ts}")
        return

    F = float(slice_df["underlying_price"].iloc[0])
    # Deribit publishes the *spot index* in estimated_delivery_price and the
    # per-expiry *forward* in underlying_price.  The implied interest-rate
    # component is r = ln(F/S)/T (see Deribit's "Inverse Options" support
    # page).  Mathematically B76(F, …) == BS(S, …, r), so the prices we
    # compute below are unchanged — we just surface the implied rate.
    if "estimated_delivery_price" in slice_df.columns:
        S_spot = float(slice_df["estimated_delivery_price"].iloc[0])
    else:
        S_spot = F
    implied_r = float(np.log(F / S_spot) / T) if S_spot > 0 and T > 0 else 0.0

    # ── Union of strikes across both option types ─────────────────────────────
    call_df = slice_df[slice_df["option_type"] == "C"].set_index("strike")
    put_df  = slice_df[slice_df["option_type"] == "P"].set_index("strike")
    strikes = np.asarray(sorted(slice_df["strike"].unique()), dtype=float)

    def _pull(idx_df, col):
        out = np.full(len(strikes), np.nan)
        for i, s in enumerate(strikes):
            if s in idx_df.index:
                v = idx_df.loc[s, col]
                out[i] = float(v if not isinstance(v, pd.Series) else v.iloc[0])
        return out

    bid_C_raw = _pull(call_df, "bid_price")
    ask_C_raw = _pull(call_df, "ask_price")
    bid_P_raw = _pull(put_df,  "bid_price")
    ask_P_raw = _pull(put_df,  "ask_price")
    oi_C_raw  = _pull(call_df, "open_interest")
    oi_P_raw  = _pull(put_df,  "open_interest")
    bsz_C_raw = _pull(call_df, "best_bid_amount") if "best_bid_amount" in call_df.columns else np.zeros(len(strikes))
    asz_C_raw = _pull(call_df, "best_ask_amount") if "best_ask_amount" in call_df.columns else np.zeros(len(strikes))
    bsz_P_raw = _pull(put_df,  "best_bid_amount") if "best_bid_amount" in put_df.columns  else np.zeros(len(strikes))
    asz_P_raw = _pull(put_df,  "best_ask_amount") if "best_ask_amount" in put_df.columns  else np.zeros(len(strikes))

    # Safeguard: missing or non-positive quotes → 0.01 placeholder
    def _safe(arr):
        return np.where(np.isnan(arr) | (arr <= 0.0), 0.01, arr)
    bid_C = _safe(bid_C_raw); ask_C = _safe(ask_C_raw)
    bid_P = _safe(bid_P_raw); ask_P = _safe(ask_P_raw)
    oi_C  = np.where(np.isnan(oi_C_raw), 0, oi_C_raw).astype(int)
    oi_P  = np.where(np.isnan(oi_P_raw), 0, oi_P_raw).astype(int)
    bsz_C = np.where(np.isnan(bsz_C_raw), 0, bsz_C_raw).astype(int)
    asz_C = np.where(np.isnan(asz_C_raw), 0, asz_C_raw).astype(int)
    bsz_P = np.where(np.isnan(bsz_P_raw), 0, bsz_P_raw).astype(int)
    asz_P = np.where(np.isnan(asz_P_raw), 0, asz_P_raw).astype(int)
    fake_C = np.isnan(bid_C_raw) | (bid_C_raw <= 0) | np.isnan(ask_C_raw) | (ask_C_raw <= 0)
    fake_P = np.isnan(bid_P_raw) | (bid_P_raw <= 0) | np.isnan(ask_P_raw) | (ask_P_raw <= 0)

    # ── IV + Black-76 ─────────────────────────────────────────────────────────
    # Deribit options are *inverse* — bid/ask are denominated in ETH per
    # contract, not USD.  Black-76 returns the USD price, so we divide by F to
    # match the venue's unit.  Vega gets the same treatment for consistency.
    k = np.log(strikes / F)
    iv = model.iv(k, params, T)
    call_price_usd, call_d, gamma, vega_usd, theta_usd = bs_pricing(
        F, strikes, T, iv, np.ones_like(strikes, bool)
    )
    put_price_usd, _put_d, _, _, _ = bs_pricing(
        F, strikes, T, iv, np.zeros_like(strikes, bool)
    )

    call_price = call_price_usd / F
    put_price  = put_price_usd  / F
    vega       = vega_usd  / F     # ETH per 1 vol-point
    theta      = theta_usd / F     # ETH per calendar day

    # ── Override greeks with Deribit's published values when available ────────
    # Vega and Γ are option-invariant in any rate regime — call-side missing →
    # put-side value is exact.  Θ and Δ are NOT call/put invariant when the
    # implied funding rate is non-zero, so they only accept a call-side value
    # (fall back to our analytic call call_d / theta at strikes without a call row).
    def _prefer_invariant(col, fallback):
        if col not in call_df.columns and col not in put_df.columns:
            return fallback
        cv = _pull(call_df, col) if col in call_df.columns else np.full(len(strikes), np.nan)
        pv = _pull(put_df,  col) if col in put_df.columns  else np.full(len(strikes), np.nan)
        merged = np.where(~np.isnan(cv), cv, pv)
        return np.where(~np.isnan(merged), merged, fallback)

    def _prefer_call_only(col, fallback):
        if col not in call_df.columns:
            return fallback
        cv = _pull(call_df, col)
        return np.where(~np.isnan(cv), cv, fallback)

    vega   = _prefer_invariant("vega_deribit",  vega)
    gamma  = _prefer_invariant("gamma_deribit", gamma)
    theta  = _prefer_call_only("theta_deribit", theta)
    call_d = _prefer_call_only("delta_deribit", call_d)

    # Trader-screen column order (one ∆ column, on the calls side; Vega on the right)
    chain = pd.DataFrame({
        "Θ":       theta,
        "Γ":       gamma,
        "Vol":     iv,
        "∆":       call_d,
        "OI_C":    oi_C,
        "BidSz_C": bsz_C,
        "Bid_C":   bid_C,
        "Theo_C":  call_price,
        "Ask_C":   ask_C,
        "AskSz_C": asz_C,
        "STRIKE":  strikes,
        "BidSz_P": bsz_P,
        "Bid_P":   bid_P,
        "Theo_P":  put_price,
        "Ask_P":   ask_P,
        "AskSz_P": asz_P,
        "OI_P":    oi_P,
        "Vega":    vega,
    })

    # ── Style matrix (matches the trader-screen colour scheme) ────────────────
    atm_idx = int(np.argmin(np.abs(strikes - F)))

    # Solid-tint columns
    col_tint = {
        "Θ":       "background-color: #E2ECF7; color: #222;",   # light blue
        "Γ":       "background-color: #FBEFE4; color: #222;",   # light peach
        "OI_C":    "background-color: #FFFFFF; color: #555;",
        "BidSz_C": "background-color: #FFFFFF; color: #444; font-size: 0.9em;",
        "Bid_C":   "background-color: #D9E6F2; color: #222;",   # call market — light blue
        "Theo_C":  "background-color: #C8DAEA; color: #222;",
        "Ask_C":   "background-color: #D9E6F2; color: #222;",
        "AskSz_C": "background-color: #FFFFFF; color: #444; font-size: 0.9em;",
        "BidSz_P": "background-color: #FFFFFF; color: #444; font-size: 0.9em;",
        "Bid_P":   "background-color: #DCD9F2; color: #222;",   # put market — light lavender
        "Theo_P":  "background-color: #CFCAEA; color: #222;",
        "Ask_P":   "background-color: #DCD9F2; color: #222;",
        "AskSz_P": "background-color: #FFFFFF; color: #444; font-size: 0.9em;",
        "OI_P":    "background-color: #FFFFFF; color: #555;",
        "Vega":    "background-color: #FFDDB0; color: #222;",   # peach
    }
    STRIKE_BLACK = "background-color: #000000; color: #FFFFFF; font-weight: 700;"
    STRIKE_ATM   = "background-color: #FFE94F; color: #000000; font-weight: 800; border: 2px solid #C70000;"
    OOS  = "background-color: #FF6961; color: white; font-weight: 700;"
    FAKE = "color: #999; font-style: italic;"

    # Manual gradients (so we don't depend on Styler.background_gradient ordering)
    def _grad(vals, lo_rgb, hi_rgb, vmin=None, vmax=None):
        v = np.asarray(vals, dtype=float)
        if vmin is None: vmin = float(np.nanmin(v))
        if vmax is None: vmax = float(np.nanmax(v))
        rng = max(vmax - vmin, 1e-9)
        t = np.clip((v - vmin) / rng, 0, 1)
        lo = np.asarray(lo_rgb); hi = np.asarray(hi_rgb)
        return [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
                for c in (lo + (hi - lo) * t[:, None])]

    vol_colors    = _grad(iv,         (245, 252, 240), (110, 180, 100), vmin=0.20, vmax=1.50)
    delta_colors  = _grad(call_d,     (255, 250, 235), (240, 175,  80), vmin=0.0,  vmax=1.0)

    def _styler(df: pd.DataFrame):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col, css in col_tint.items():
            if col in styles.columns:
                styles[col] = css
        # Per-row gradient overrides for Vol and ∆
        vol_col_idx = df.columns.get_loc("Vol")
        d_col_idx   = df.columns.get_loc("∆")
        for i in range(len(df)):
            styles.iat[i, vol_col_idx] = f"background-color: {vol_colors[i]}; color: #222;"
            styles.iat[i, d_col_idx]   = f"background-color: {delta_colors[i]}; color: #222;"
        # STRIKE column — black with white text, ATM highlighted in yellow
        strike_col = df.columns.get_loc("STRIKE")
        for i in range(len(df)):
            styles.iat[i, strike_col] = STRIKE_BLACK
        styles.iat[atm_idx, strike_col] = STRIKE_ATM
        # Out-of-spread red flags — compare at the same precision the user sees
        # in the table (4 dp on prices) so 0.0046 vs 0.004599 doesn't fire.
        DP = 4
        for theo_col, bid_col, ask_col, fake_mask in [
            ("Theo_C", "Bid_C", "Ask_C", fake_C),
            ("Theo_P", "Bid_P", "Ask_P", fake_P),
        ]:
            t = np.round(df[theo_col].to_numpy(), DP)
            b = np.round(df[bid_col].to_numpy(),  DP)
            a = np.round(df[ask_col].to_numpy(),  DP)
            for i in range(len(df)):
                if fake_mask[i]:
                    continue
                if t[i] > a[i]:
                    styles.iat[i, df.columns.get_loc(ask_col)] = OOS
                elif t[i] < b[i]:
                    styles.iat[i, df.columns.get_loc(bid_col)] = OOS
        # Dim synthetic-quote cells
        for i, fc in enumerate(fake_C):
            if fc:
                for c in ("Bid_C", "Ask_C"):
                    styles.iat[i, df.columns.get_loc(c)] += FAKE
        for i, fp in enumerate(fake_P):
            if fp:
                for c in ("Bid_P", "Ask_P"):
                    styles.iat[i, df.columns.get_loc(c)] += FAKE
        return styles

    # Render — into this tab's local placeholders
    stale_badge = ""
    if age is not None and age > STALE_SECONDS:
        stale_badge = f" <span style='color:#c0392b;font-weight:600'>· data {age:.0f}s old</span>"

    with tb["header"].container():
        st.markdown(
            f"<h3>"
            f"<span class='live-pulse'></span>"
            f"{ccy.upper()} chain · T = {T*365:.0f}d (target {days}d) · {ts}{stale_badge}"
            f"</h3>", unsafe_allow_html=True,
        )

    # Count red cells (= theo outside the *rounded* bid-ask, skipping synthetic)
    DP = 4
    red_C = int(((~fake_C) & (
        (np.round(call_price, DP) > np.round(ask_C, DP)) |
        (np.round(call_price, DP) < np.round(bid_C, DP))
    )).sum())
    red_P = int(((~fake_P) & (
        (np.round(put_price, DP) > np.round(ask_P, DP)) |
        (np.round(put_price, DP) < np.round(bid_P, DP))
    )).sum())
    red_total = red_C + red_P

    with tb["metrics"].container():
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Spot",         f"${S_spot:,.2f}")
        m2.metric("Forward",      f"${F:,.2f}",
                  delta=f"basis {(F/S_spot-1)*100:+.2f}%", delta_color="off")
        m3.metric("Implied r",    f"{implied_r*100:+.2f}%",
                  delta="ln(F/S)/T", delta_color="off")
        m4.metric("Strikes",      f"{len(strikes)}")
        m5.metric("Fast-path",    f"{fit_ms:.1f} ms")
        m6.metric("Red cells",    f"{red_total}",
                  delta=f"C {red_C} · P {red_P}", delta_color="off")

    styler = (
        chain.style
        .apply(_styler, axis=None)
        .format({
            "Θ":       "{:+.3f}", "Γ": "{:.4f}", "Vol": "{:.3f}", "∆": "{:+.2f}",
            "OI_C":    "{:,d}",
            "BidSz_C": "{:,d}",
            "Bid_C":   "{:.4f}", "Theo_C": "{:.4f}", "Ask_C": "{:.4f}",
            "AskSz_C": "{:,d}",
            "STRIKE":  "{:,.0f}",
            "BidSz_P": "{:,d}",
            "Bid_P":   "{:.4f}", "Theo_P": "{:.4f}", "Ask_P": "{:.4f}",
            "AskSz_P": "{:,d}",
            "OI_P":    "{:,d}",
            "Vega":    "{:.3f}",
        })
        .hide(axis="index")
    )
    with tb["table"].container():
        st.dataframe(styler, width="stretch", height=560)

    if show_smile:
        import matplotlib.pyplot as plt
        with tb["smile"].container():
            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="white")
            kk = np.linspace(min(k.min(), -1), max(k.max(), 1), 200)
            iv_curve = model.iv(kk, params, T)
            ax.plot(kk, iv_curve, color=PURPLE, lw=2.4, label="eSSVI fit")
            # OTM-only scatter — ITM puts/calls are redundant under put-call parity.
            # Bid and ask IV are both plotted (as crosses) at every moneyness level
            # instead of a single mark_iv point, so the quoted spread is visible.
            otm_C = (slice_df["option_type"] == "C") & (slice_df["strike"] >= F)
            otm_P = (slice_df["option_type"] == "P") & (slice_df["strike"] <= F)
            k_C = np.log(slice_df.loc[otm_C, "strike"]/F)
            k_P = np.log(slice_df.loc[otm_P, "strike"]/F)
            ax.scatter(k_C, slice_df.loc[otm_C, "bid_iv"], s=14, marker="x",
                       linewidths=0.7, color="blue", label="Bid IV")
            ax.scatter(k_P, slice_df.loc[otm_P, "bid_iv"], s=14, marker="x",
                       linewidths=0.7, color="blue")
            ax.scatter(k_C, slice_df.loc[otm_C, "ask_iv"], s=14, marker="x",
                       linewidths=0.7, color="red", label="Ask IV")
            ax.scatter(k_P, slice_df.loc[otm_P, "ask_iv"], s=14, marker="x",
                       linewidths=0.7, color="red")
            ax.set_xlabel("log-moneyness k"); ax.set_ylabel("IV")
            ax.set_title(f"Smile at T = {T:.3f}", color=PURPLE)
            ax.grid(True, alpha=0.3, color="#B284E0")
            ax.set_facecolor("white")
            ax.legend(fontsize=8)
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            st.pyplot(fig, clear_figure=True)
    else:
        tb["smile"].empty()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop — re-reads the live parquet on every tick
# ─────────────────────────────────────────────────────────────────────────────

if not live and not st.session_state.get("replay_mode", False):
    render_frame(frame_idx=0)
else:
    # Replay mode reuses the same tick loop as live mode — each tick the
    # replay cursor advances by one, and _load_latest_snapshot picks the
    # snapshot at that index instead of the latest.
    if "replay_idx" not in st.session_state:
        st.session_state["replay_idx"] = 0
    while True:
        st.session_state.iter += 1
        if st.session_state.get("replay_mode", False):
            st.session_state["replay_idx"] += 1
        render_frame(frame_idx=st.session_state.iter)
        time.sleep(tick_ms / 1000.0)
