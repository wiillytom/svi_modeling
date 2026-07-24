"""
Live-updating cumulative P&L chart for `engine.run_backtest`, redrawn
periodically inside a Jupyter cell.

Uses IPython's `clear_output` + a normal matplotlib static figure, redrawn in
place — NOT `%matplotlib widget` (this project tested that backend before and
reverted it for being too laggy, see CLAUDE.md). This is the same
static/inline redraw pattern, just repeated on a timer instead of once.

Incremental by design: each call only processes events added since the last
call, so redraw cost doesn't grow with total run length the way recomputing
from the full event list every time would.
"""

from __future__ import annotations

import time

import pandas as pd

try:
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


class LivePnLPlot:
    """Stateful, incremental live P&L plotter.

    Usage: create one instance before the backtest loop, call
    `maybe_render(events, status)` after each snapshot (cheap no-op unless
    `interval_secs` has elapsed since the last redraw), and call it once more
    with `force=True` at the end to show the final state.
    """

    def __init__(self, interval_secs: float = 3.0):
        self.interval_secs = interval_secs
        self._last_render_time = 0.0
        self._last_idx = 0
        self._cum_pnl = 0.0
        self._xs: list = []
        self._ys: list = []

    def maybe_render(self, events: list[dict], status: str = "", force: bool = False) -> None:
        if not _AVAILABLE:
            return
        now = time.time()
        if not force and (now - self._last_render_time) < self.interval_secs:
            return
        self._last_render_time = now

        # Incremental: only fold in events added since the last render.
        new_events = events[self._last_idx:]
        self._last_idx = len(events)
        for e in new_events:
            delta = (e.get("option_pnl") or 0.0) - (e.get("option_cost") or 0.0)
            if delta != 0.0:
                self._cum_pnl += delta
                self._xs.append(e["timestamp"])
                self._ys.append(self._cum_pnl)

        clear_output(wait=True)
        if status:
            print(status)
        if not self._xs:
            return

        fig, ax = plt.subplots(figsize=(9, 4))
        x = pd.to_datetime(pd.Series(self._xs), unit="ms")
        ax.plot(x, self._ys, linewidth=1.2, color="#6A0DAD")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Cumulative net P&L")
        ax.set_xlabel("time")
        ax.set_ylabel("P&L (coin)")
        fig.autofmt_xdate()
        fig.tight_layout()
        display(fig)
        plt.close(fig)
