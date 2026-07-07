"""
Live Deribit option-chain gatherer.

Loop, once per second:
  1. pull book summary + index price from Deribit (REST)
  2. clean the rows (t, k, w, vega, bid_iv/ask_iv, …) — same logic as
     `data_handling.clean_df` minus the volume / OTM filters so we keep the
     full chain for display
  3. append the cleaned snapshot to a single rolling parquet under
     `2 - Data/live/live_<ccy>.parquet` (atomic rename), trimmed to the last
     ROLLING_MINUTES of history so the file stays small.

Designed to run as its own terminal process while the Streamlit chain reads
the parquet.  Concurrent access is safe because we write to a `.tmp` and
atomically `os.replace` it into place.

    python volatility_surface/live_gather.py --currency eth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
import websocket    # websocket-client
from scipy.stats import norm

from volatility_surface.utils.option_data_gathering import (
    get_book_summary, get_index_price, process_data,
)
from volatility_surface.core.pricing.volatility_dataframe import (
    implied_volatility_dataframe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deribit websocket ticker subscriber — keeps top-of-book sizes in memory
# ─────────────────────────────────────────────────────────────────────────────

class DeribitTickerWS:
    """Background-thread websocket client maintaining live top-of-book sizes."""

    URL = "wss://www.deribit.com/ws/api/v2"

    def __init__(self, currency: str = "eth", interval: str = "100ms",
                 chunk: int = 50):
        self.currency = currency.upper()
        self.interval = interval
        self.chunk    = chunk
        self.state: dict[str, dict] = {}     # instrument -> {bid_size, ask_size, …}
        self._lock    = threading.Lock()
        self._instruments: list[str] = []
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        self._fetch_instruments()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _fetch_instruments(self):
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_instruments",
            params={"currency": self.currency, "kind": "option", "expired": "false"},
            timeout=10,
        )
        r.raise_for_status()
        result = r.json().get("result", [])
        self._instruments = [item["instrument_name"] for item in result]
        print(f"[ws] {len(self._instruments)} {self.currency} option instruments")

    def _on_open(self, ws):
        channels = [f"ticker.{ins}.{self.interval}" for ins in self._instruments]
        for i in range(0, len(channels), self.chunk):
            ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id":      i,
                "method":  "public/subscribe",
                "params":  {"channels": channels[i:i + self.chunk]},
            }))
        print(f"[ws] subscribed to {len(channels)} ticker channels")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if data.get("method") != "subscription":
            return
        td = data.get("params", {}).get("data") or {}
        ins = td.get("instrument_name")
        if not ins:
            return
        greeks = td.get("greeks") or {}
        with self._lock:
            self.state[ins] = {
                "best_bid_amount":  td.get("best_bid_amount"),
                "best_ask_amount":  td.get("best_ask_amount"),
                # Bid / ask implied vols computed by Deribit's own pricer.
                # Reported in percent — we divide by 100 in clean_live.
                # Lets us skip the per-row vollib inversion in the live path.
                "bid_iv_deribit":   td.get("bid_iv"),
                "ask_iv_deribit":   td.get("ask_iv"),
                "mark_iv_ticker":   td.get("mark_iv"),
                # Per-instrument greeks served by Deribit directly.  Convention:
                # vega/delta/gamma/theta are per-1%-vol move, scaled to coin
                # notional — our analytic vega is per-1.00-vol, unit-forward.
                # Both are stored so consumers can choose their convention.
                "vega_deribit":     greeks.get("vega"),
                "delta_deribit":    greeks.get("delta"),
                "gamma_deribit":    greeks.get("gamma"),
                "theta_deribit":    greeks.get("theta"),
                "timestamp":        td.get("timestamp"),
            }

    def _on_error(self, ws, error):
        print(f"[ws] error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[ws] closed: code={code} reason={msg}")

    def _run(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.URL,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=5)
            except Exception as e:
                print(f"[ws] run_forever crashed: {e!r}")
            if self._running:
                time.sleep(2)

    # Columns the lookup will populate.  Sizes default to 0 (no quote = no
    # depth); greeks and IVs default to NaN (a real zero is meaningful).
    _SIZE_KEYS  = ("best_bid_amount", "best_ask_amount")
    _IV_KEYS    = ("bid_iv_deribit", "ask_iv_deribit", "mark_iv_ticker")
    _GREEK_KEYS = ("vega_deribit", "delta_deribit", "gamma_deribit", "theta_deribit")
    KEYS = _SIZE_KEYS + _IV_KEYS + _GREEK_KEYS

    def lookup(self, instrument_name: str) -> dict:
        """Latest WS state for an instrument; NaN/0 placeholders if not received."""
        with self._lock:
            s = self.state.get(instrument_name) or {}
        out = {}
        for k in self._SIZE_KEYS:
            v = s.get(k)
            out[k] = float(v) if v is not None else 0.0
        for k in self._IV_KEYS + self._GREEK_KEYS:
            v = s.get(k)
            out[k] = float(v) if v is not None else float("nan")
        return out

    def coverage(self) -> int:
        with self._lock:
            return len(self.state)


LIVE_DIR  = PROJECT_ROOT / "2 - Data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_MINUTES = 60        # keep the most recent 60 minutes of ticks
POLL_SECONDS    = 1


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning — DataFrame in, DataFrame out (no CSV roundtrip)
# ─────────────────────────────────────────────────────────────────────────────

def clean_live(df: pd.DataFrame, file_ts_str: str) -> pd.DataFrame:
    """
    Apply the same field derivations as `data_handling.clean_df`, but
    operating in-memory and without the `volume>0` / OTM-only filters — we
    want the full chain visible on the trading screen.
    """
    if df.empty:
        return df

    # numeric coercion (Deribit may return strings or NaN)
    for col in ("bid_price", "ask_price", "mark_price", "mark_iv",
                "underlying_price", "strike"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["bid_price", "ask_price", "mark_iv",
                           "underlying_price", "strike", "option_type"])

    # Carry websocket-supplied fields through cleaning.
    # Sizes default to 0 if missing; IVs and greeks default to NaN.
    df = df.copy()
    for c in ("best_bid_amount", "best_ask_amount"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ("bid_iv_deribit", "ask_iv_deribit", "mark_iv_ticker",
              "vega_deribit", "delta_deribit", "gamma_deribit", "theta_deribit"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # boundary sanity (same as clean_df)
    put_mask  = (df["underlying_price"] * df["bid_price"] < df["strike"]) & (df["option_type"] == "P")
    call_mask = (df["option_type"] == "C") & (df["bid_price"] < 1)
    df = df[put_mask | call_mask].copy()
    if df.empty:
        return df

    # ── Time fields ──────────────────────────────────────────────────────────
    ts_ms = float(df["creation_timestamp_x"].iloc[0])
    df["creation_timestamp_x"] = ts_ms
    df["expiration_timestamp"] = pd.to_datetime(
        df["expiration_timestamp"], format="%d%b%y"
    ) + pd.Timedelta(hours=8)
    df["expiration_timestamp_ms"] = df["expiration_timestamp"].astype("int64") / 10**6

    # Single timestamp string for the *whole* snapshot (matches the cleaned parquet schema)
    df["file_timestamp"] = file_ts_str

    # ── Derived fields ───────────────────────────────────────────────────────
    df["mark_iv"] = df["mark_iv"] / 100.0
    df["t"] = (df["expiration_timestamp_ms"] - df["creation_timestamp_x"]) / (3.6e6 * 24 * 365)
    df = df[df["t"] > 1e-5]
    if df.empty:
        return df
    df["k"] = np.log(df["strike"] / df["underlying_price"])
    df["w"] = df["mark_iv"] ** 2 * df["t"]

    n_d1 = norm.pdf(-df["k"] / np.sqrt(df["w"]) + np.sqrt(df["w"]) / 2)
    df["vega"] = n_d1 * np.sqrt(df["t"])

    # NOTE: data_handling.clean_df applies an OTM-only filter here for the
    # offline / calibration pipeline. We keep ITM rows in the *live* parquet
    # so the trader screen can show real quotes on both sides of every strike;
    # the OTM-only filter is applied at calibration time in streamlit_chain.py.

    # ── Bid / ask implied vols ──────────────────────────────────────────────
    # Prefer Deribit's own pricer (already in the WS payload) — same
    # implied-funding convention they use elsewhere, no per-row vollib call.
    # Deribit reports IV in percent; we want decimals.
    df["bid_iv"] = pd.to_numeric(df["bid_iv_deribit"], errors="coerce") / 100.0
    df["ask_iv"] = pd.to_numeric(df["ask_iv_deribit"], errors="coerce") / 100.0

    # Treat one-sided / zero quotes as missing (Deribit reports 0 when there's
    # no quote on that side) so they don't trip the spread-weighted objectives.
    df.loc[df["bid_iv"] <= 0, "bid_iv"] = np.nan
    df.loc[df["ask_iv"] <= 0, "ask_iv"] = np.nan

    # Fallback: row-wise vollib inversion ONLY for rows the WS hasn't covered.
    missing = df["bid_iv"].isna() | df["ask_iv"].isna()
    if missing.any():
        inverted = df.loc[missing].apply(
            implied_volatility_dataframe, axis=1, result_type="expand"
        )
        if not inverted.empty:
            inverted.columns = ["bid_iv_fb", "ask_iv_fb"]
            df.loc[missing, "bid_iv"] = inverted["bid_iv_fb"].values
            df.loc[missing, "ask_iv"] = inverted["ask_iv_fb"].values

    df = df.dropna(subset=["bid_iv", "ask_iv"])
    df["half_spread_iv"] = (df["ask_iv"] - df["bid_iv"]) / 2
    df["half_spread"]    = (df["ask_price"] - df["bid_price"]) / 2

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling parquet append
# ─────────────────────────────────────────────────────────────────────────────

def append_to_parquet(parquet_path: Path, new_rows: pd.DataFrame,
                      keep_minutes: int = ROLLING_MINUTES) -> int:
    """Append rows, drop everything older than `keep_minutes`, atomic write."""
    if new_rows.empty:
        return 0

    if parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except Exception:
            combined = new_rows.copy()
    else:
        combined = new_rows.copy()

    # rolling-window trim by file_timestamp
    ts = pd.to_datetime(combined["file_timestamp"], errors="coerce")
    cutoff = ts.max() - pd.Timedelta(minutes=keep_minutes)
    mask = ts >= cutoff
    combined = combined[mask].reset_index(drop=True)

    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, parquet_path)
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(currency: str = "eth", poll_seconds: int = POLL_SECONDS) -> None:
    parquet_path = LIVE_DIR / f"live_{currency.lower()}.parquet"
    index_name = f"{currency.lower()}_usd"

    print(f"[live_gather] currency={currency}  poll={poll_seconds}s")
    print(f"[live_gather] writing to {parquet_path}")

    # Start the websocket subscriber (background thread, daemon)
    ws_client = DeribitTickerWS(currency=currency)
    ws_client.start()
    # Give the WS a couple of seconds to receive its first wave of tickers
    print("[live_gather] warming up websocket …")
    time.sleep(2.0)
    print(f"[live_gather] ws coverage: {ws_client.coverage()} instruments")
    print("[live_gather] ctrl+c to stop\n")

    n_ticks = 0
    while True:
        loop_start = time.time()
        try:
            idx_price = get_index_price(index_name)
            book      = get_book_summary(currency, "option")
            if idx_price is None or not book:
                print(f"[{datetime.now():%H:%M:%S}] empty response — skipping")
            else:
                file_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw = process_data(book, idx_price)

                # Merge websocket sizes + Deribit greeks (one lookup per instrument)
                if "instrument_id" in raw.columns:
                    states = [ws_client.lookup(ins) for ins in raw["instrument_id"]]
                    for col in DeribitTickerWS.KEYS:
                        raw[col] = [s[col] for s in states]

                cleaned = clean_live(raw, file_ts)
                wrote = append_to_parquet(parquet_path, cleaned)
                n_ticks += 1
                print(f"[{datetime.now():%H:%M:%S}] tick {n_ticks}  "
                      f"+{wrote:4d} rows  ({(time.time()-loop_start)*1000:.0f} ms)  "
                      f"ws_cov={ws_client.coverage()}  → {parquet_path.name}")
        except KeyboardInterrupt:
            print("\n[live_gather] stopped.")
            return
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] tick error: {e!r}")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, poll_seconds - elapsed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="eth", choices=["eth", "btc"])
    ap.add_argument("--poll",     type=int, default=POLL_SECONDS,
                    help="polling interval in seconds")
    ap.add_argument("--keep-minutes", type=int, default=ROLLING_MINUTES,
                    help="rolling-window size in minutes")
    args = ap.parse_args()
    ROLLING_MINUTES = args.keep_minutes
    run(args.currency, args.poll)
