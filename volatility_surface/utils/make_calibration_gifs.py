"""
make_calibration_gifs.py
=========================
Standalone script — animates a single-slice RawSVI calibration as it's
optimized, once with Nelder-Mead and once with Differential Evolution, and
saves each as a GIF (market points + fitted smile, loss shown live in the
title). Built for dropping straight into a PowerPoint slide.

Usage
-----
    python volatility_surface/utils/make_calibration_gifs.py
    python volatility_surface/utils/make_calibration_gifs.py --currency btc --expiry-index 3
    python volatility_surface/utils/make_calibration_gifs.py --outdir presentation_assets

Output
------
    <outdir>/calibration_nelder_mead.gif
    <outdir>/calibration_differential_evolution.gif
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import minimize, differential_evolution

from volatility_surface.core.calibration.calibrator import load_snapshot
from volatility_surface.core.calibration.objectives import get_objective
from volatility_surface.models.vol_models import get_model


def build_slice(currency: str, expiry_index: int):
    """Load one real Deribit snapshot and return one OTM-filtered expiry."""
    path = REPO_ROOT / "2 - Data" / "parquets" / f"{currency}_options_data_cleaned.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run volatility_surface/utils/data_handling.py "
            "first, or pass --currency for a currency that has an archive parquet."
        )
    df = load_snapshot(str(path))
    otm_C = (df["option_type"] == "C") & (df["k"] >= 0)
    otm_P = (df["option_type"] == "P") & (df["k"] <= 0)
    df = df[otm_C | otm_P].copy()

    expiries = sorted(df["t"].unique())
    idx = max(0, min(expiry_index, len(expiries) - 1))
    t_exp = expiries[idx]
    df_slice = df[df["t"] == t_exp].copy()
    print(f"[data] {currency.upper()}  expiry #{idx}  T={t_exp:.4f}  n_points={len(df_slice)}")
    return df_slice, t_exp


def make_objective(model, k_arr, w_arr, iv_arr, vega_arr, t_exp, objective_fn):
    def objective(x):
        params = model.unpack(x)
        if not model.validate(params):
            return 1e3
        w_fit = model.w(k_arr, params)
        iv_fit = model.iv(k_arr, params, t_exp)
        return objective_fn(w_fit, w_arr, iv_fit, iv_arr, k_arr, t_exp, vegas=vega_arr)
    return objective


def run_nelder_mead(model, objective, k_arr, w_arr, t_exp, maxiter, frame_stride=1):
    """
    Run Nelder-Mead for `maxiter` iterations, but only keep one frame every
    `frame_stride` iterations (e.g. maxiter=500, frame_stride=25 -> 20 frames).
    The optimizer itself still runs the full `maxiter` iterations — stride only
    thins what gets rendered into the GIF. The true final (converged) frame is
    always kept even if it doesn't land on the stride, so the GIF still ends
    on the actual fit.
    """
    x0 = model.pack(model.initial_guess(k_arr, w_arr, t_exp))
    frames = [(0, x0.copy(), objective(x0))]
    iteration = 0

    def callback(xk):
        nonlocal iteration
        iteration += 1
        if iteration % frame_stride == 0:
            frames.append((iteration, xk.copy(), objective(xk)))

    res = minimize(
        objective, x0, method="Nelder-Mead", callback=callback,
        options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-7},
    )
    if iteration % frame_stride != 0:
        frames.append((iteration, res.x.copy(), objective(res.x)))
    print(f"[Nelder-Mead] {iteration} iterations run, {len(frames)} frames kept "
          f"(1 per {frame_stride}), final loss={frames[-1][2]:.5f}")
    return frames


def run_differential_evolution(model, objective, maxiter, popsize, seed):
    frames = []

    def callback(xk, convergence):
        frames.append((len(frames) + 1, xk.copy(), objective(xk)))

    differential_evolution(
        objective, model.bounds(), seed=seed, maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.2), recombination=0.8, tol=1e-6,
        callback=callback, polish=False,
    )
    print(f"[Differential Evolution] {len(frames)} generations captured, final loss={frames[-1][2]:.5f}")
    return frames


def render_gif(frames, model, k_arr, iv_arr, t_exp, out_path, method_label, fps):
    k_lo = min(k_arr.min() - 0.15, -0.5)
    k_hi = max(k_arr.max() + 0.15, 0.5)
    k_grid = np.linspace(k_lo, k_hi, 300)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.scatter(k_arr, iv_arr * 100, color="#333333", s=25, zorder=3, label="Market mark IV")
    line, = ax.plot([], [], color="#c0392b", lw=2.2, zorder=2, label=f"RawSVI fit ({method_label})")
    ax.axvline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlim(k_grid.min(), k_grid.max())
    iv_lo = max(iv_arr.min() * 100 - 15, 0)
    iv_hi = iv_arr.max() * 100 + 15
    ax.set_ylim(iv_lo, iv_hi)
    ax.set_xlabel("Log-strike k")
    ax.set_ylabel("Implied vol (%)")
    ax.legend(loc="upper right", fontsize=9)
    title = ax.set_title("")

    n_frames = len(frames)
    max_step = max(f[0] for f in frames)

    def animate(i):
        step, x, loss = frames[i]
        params = model.unpack(x)
        iv_fit = model.iv(k_grid, params, t_exp)
        line.set_data(k_grid, iv_fit * 100)
        title.set_text(f"{method_label} — step {step}/{max_step}   loss = {loss:.5f}")
        return line, title

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 // fps,
                        blit=False, repeat_delay=1500)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", default="eth", choices=["eth", "btc"])
    parser.add_argument("--expiry-index", type=int, default=6,
                        help="Which sorted expiry to animate (default: 6)")
    parser.add_argument("--objective", default="vega_wmse")
    parser.add_argument("--nm-maxiter", type=int, default=120)
    parser.add_argument("--nm-frame-stride", type=int, default=1,
                        help="keep 1 Nelder-Mead frame every N iterations "
                             "(e.g. --nm-maxiter 500 --nm-frame-stride 25 -> 20 frames)")
    parser.add_argument("--de-maxiter", type=int, default=40)
    parser.add_argument("--de-popsize", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--outdir", default="presentation_assets")
    args = parser.parse_args()

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df_slice, t_exp = build_slice(args.currency, args.expiry_index)
    k_arr = df_slice["k"].values
    w_arr = df_slice["w"].values
    iv_arr = df_slice["mark_iv"].values
    vega_arr = df_slice["vega"].values

    model = get_model("svi")
    objective_fn = get_objective(args.objective)
    objective = make_objective(model, k_arr, w_arr, iv_arr, vega_arr, t_exp, objective_fn)

    print("\nRunning Nelder-Mead...")
    nm_frames = run_nelder_mead(model, objective, k_arr, w_arr, t_exp,
                                args.nm_maxiter, args.nm_frame_stride)
    render_gif(nm_frames, model, k_arr, iv_arr, t_exp,
              outdir / "calibration_nelder_mead.gif", "Nelder-Mead", args.fps)

    print("\nRunning Differential Evolution...")
    de_frames = run_differential_evolution(model, objective, args.de_maxiter,
                                           args.de_popsize, args.seed)
    render_gif(de_frames, model, k_arr, iv_arr, t_exp,
              outdir / "calibration_differential_evolution.gif",
              "Differential Evolution", args.fps)

    print("\nDone. GIFs are in:", outdir)


if __name__ == "__main__":
    main()
