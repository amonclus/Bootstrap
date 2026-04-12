#!/usr/bin/env python3
"""
run_facebook.py — Phase-transition analysis on the Facebook SNAP ego-network.

Usage:
    python src/experiments/run_facebook.py --model bootstrap
    python src/experiments/run_facebook.py --model all

Models: bootstrap  wtm  sis  h1  h2  h3  h4  h5  h6  all

Output: results/facebook/<model>.{png,csv}
        results/facebook/summary.png   (only when --model all)
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import warnings
import time

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool, cpu_count, freeze_support
import tqdm

# ── configuration ─────────────────────────────────────────────────────────────
GRAPH_PATH  = os.path.join(_ROOT, "data", "facebook_combined.txt")
OUTPUT_DIR  = os.path.join(_ROOT,"src","experiments", "results", "facebook")
N_REALIZ    = 10    # seed realisations per (param, strategy) point
N_REALIZ_2D = 5    # realisations for 2-D heatmap cells
N_PARAMS    = 10    # parameter points per 1-D sweep
N_GRID      = 10    # grid size per axis for 2-D heatmaps
GAMMA       = 0.1   # shared recovery rate
SEED_FRAC   = 0.01  # fraction of nodes seeded
N_WORKERS   = max(1, cpu_count() - 1)

STRATEGIES    = ["random", "high_degree", "high_kcore"]
STRAT_LABELS  = {"random": "Random", "high_degree": "High Degree", "high_kcore": "High k-core"}
STRAT_COLORS  = {"random": "#4878D0", "high_degree": "#EE854A", "high_kcore": "#6ACC65"}
STRAT_MARKERS = {"random": "o", "high_degree": "s", "high_kcore": "^"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
})

# ── graph helpers ─────────────────────────────────────────────────────────────

def load_and_describe(path: str) -> tuple[nx.Graph, float, float]:
    """Load graph, print stats, return (G, mean_degree, beta_c)."""
    from input.graph_loader import load_graph_from_edge_list
    G = load_graph_from_edge_list(path)

    n  = G.number_of_nodes()
    m  = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degs))

    A = nx.adjacency_matrix(G).astype(float)
    lam_max = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
    beta_c  = GAMMA / lam_max

    print("=" * 54)
    print("  Facebook SNAP combined ego-network")
    print("=" * 54)
    print(f"  Nodes            : {n:,}")
    print(f"  Edges            : {m:,}")
    print(f"  Mean degree      : {mean_deg:.2f}  (std {np.std(degs):.2f}, max {max(degs)})")
    print(f"  Clustering coeff : {nx.average_clustering(G):.4f}")
    print(f"  λ_max            : {lam_max:.4f}")
    print(f"  SIS threshold β_c: {beta_c:.6f}")
    print("=" * 54)
    return G, mean_deg, beta_c


# ── multiprocessing worker ────────────────────────────────────────────────────

_G: nx.Graph | None = None

def _init(src_path: str, graph_path: str) -> None:
    global _G
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from input.graph_loader import load_graph_from_edge_list
    _G = load_graph_from_edge_list(graph_path)

def _trial(task: tuple) -> tuple[int, float]:
    """(tid, model, kwargs, strategy, seed_size, seed) → (tid, fraction)"""
    global _G
    tid, model, kwargs, strategy, seed_size, seed = task
    random.seed(seed)
    np.random.seed(seed % (2**31))

    from simulation.seed_selection import select_seeds
    seed_set = set(select_seeds(_G, n=seed_size, strategy=strategy))

    if model == "bootstrap":
        from simulation.bootstrap import BootstrapPercolation as M
        sim = M(_G, **kwargs)
    elif model == "sis":
        from simulation.sis import SISModel as M
        sim = M(_G, **kwargs)
    elif model == "wtm":
        from simulation.wtm import WTMModel as M
        sim = M(_G, **kwargs)
    elif model == "h1":
        from simulation.H1 import H1Model as M
        sim = M(_G, **kwargs)
    elif model == "h2":
        from simulation.H2 import H2Model as M
        sim = M(_G, **kwargs)
    elif model == "h3":
        from simulation.H3 import H3Model as M
        sim = M(_G, **kwargs)
    elif model == "h4":
        from simulation.H4 import H4Model as M
        sim = M(_G, **kwargs)
    elif model == "h5":
        from simulation.H5 import H5Model as M
        sim = M(_G, **kwargs)
    elif model == "h6":
        from simulation.H6 import H6Model as M
        sim = M(_G, **kwargs)
    else:
        raise ValueError(model)

    try:
        result, _ = sim.run(seed_set)
        return tid, float(result.cascade_fraction)
    except Exception:
        return tid, 0.0

# ── sweep engine ──────────────────────────────────────────────────────────────

def _run_pool(tasks: list, desc: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with Pool(N_WORKERS, initializer=_init, initargs=(_SRC, GRAPH_PATH)) as pool:
        for tid, frac in tqdm.tqdm(
            pool.imap_unordered(_trial, tasks, chunksize=10),
            total=len(tasks), desc=f"  {desc}", ncols=80,
        ):
            out[tid] = frac
    return out

def sweep_1d(model: str, kwargs_fn, xvals, seed_size: int, n_realiz: int, desc: str):
    """Returns {strategy: {'mean': array, 'std': array}}."""
    tasks, index = [], []
    for pi, x in enumerate(xvals):
        kw = kwargs_fn(x)
        for si, s in enumerate(STRATEGIES):
            for ti in range(n_realiz):
                tasks.append((len(tasks), model, kw, s, seed_size, pi*10000+si*1000+ti))
                index.append((pi, si, ti))

    fracs = _run_pool(tasks, desc)
    raw = {s: np.zeros((len(xvals), n_realiz)) for s in STRATEGIES}
    for tid, (pi, si, ti) in enumerate(index):
        raw[STRATEGIES[si]][pi, ti] = fracs[tid]

    return {s: {"mean": raw[s].mean(1), "std": raw[s].std(1)} for s in STRATEGIES}

def sweep_2d(model: str, kwargs_fn, xvals, yvals, seed_size: int, n_realiz: int, desc: str):
    """Returns {strategy: 2D mean array (ny, nx)}."""
    tasks, index = [], []
    for yi, y in enumerate(yvals):
        for xi, x in enumerate(xvals):
            kw = kwargs_fn(x, y)
            for si, s in enumerate(STRATEGIES):
                for ti in range(n_realiz):
                    tasks.append((len(tasks), model, kw, s, seed_size,
                                  yi*1000000+xi*10000+si*1000+ti))
                    index.append((xi, yi, si, ti))

    fracs = _run_pool(tasks, desc)
    raw = {s: np.zeros((len(yvals), len(xvals), n_realiz)) for s in STRATEGIES}
    for tid, (xi, yi, si, ti) in enumerate(index):
        raw[STRATEGIES[si]][yi, xi, ti] = fracs[tid]

    return {s: raw[s].mean(2) for s in STRATEGIES}

# ── plotting helpers ───────────────────────────────────────────────────────────

def _curves(ax, xvals, sw, vline=None, vlabel=None):
    for s in STRATEGIES:
        m, sd = sw[s]["mean"], sw[s]["std"]
        ax.plot(xvals, m, color=STRAT_COLORS[s], marker=STRAT_MARKERS[s],
                markersize=3.5, linewidth=1.6, label=STRAT_LABELS[s])
        ax.fill_between(xvals, m - sd, m + sd, alpha=0.13, color=STRAT_COLORS[s])
    if vline is not None:
        ax.axvline(vline, color="black", ls="--", lw=1.2, label=vlabel or "Threshold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Final infected fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5)

def save_1d_fig(xvals, sw, xlabel, title, path, vline=None, vlabel=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _curves(ax, xvals, sw, vline, vlabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(path)

def save_2panel_fig(xvals, sws, sec_vals, xlabel, sec_label, title, path,
                    vline=None, vlabel=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, v in zip(axes, sec_vals):
        _curves(ax, xvals, sws[v], vline, vlabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{sec_label} = {v}")
    axes[1].set_ylabel("")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(path)

def save_heatmap_fig(xvals, yvals, heat_by_strat, xlabel, ylabel, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, s in zip(axes, STRATEGIES):
        im = ax.imshow(heat_by_strat[s], origin="lower", aspect="auto",
                       extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
                       cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Infected fraction")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_title(STRAT_LABELS[s])
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(path)

def _log(path: str):
    print(f"  saved → {os.path.relpath(path, _ROOT)}")

def csv_1d(xvals, sw, param_name, path):
    rows = [
        {param_name: x, "strategy": s, "mean": sw[s]["mean"][i], "std": sw[s]["std"][i]}
        for s in STRATEGIES for i, x in enumerate(xvals)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    _log(path)

def csv_2d(xvals, yvals, heat_by_strat, x_name, y_name, path):
    rows = [
        {x_name: x, y_name: y, "strategy": s, "mean": heat_by_strat[s][yi, xi]}
        for s in STRATEGIES
        for yi, y in enumerate(yvals)
        for xi, x in enumerate(xvals)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    _log(path)

# ── model runners ─────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, mean_deg, beta_c, seed_size):
        self.mean_deg  = mean_deg
        self.beta_c    = beta_c
        self.seed_size = seed_size

def run_bootstrap(ctx: Ctx) -> dict:
    section("Bootstrap  [sweep k = 1…15]")
    k_vals = list(range(1, 30))
    sw = sweep_1d("bootstrap", lambda k: {"threshold": k},
                  k_vals, ctx.seed_size, N_REALIZ, "bootstrap")
    save_1d_fig(k_vals, sw, "Bootstrap threshold k",
                "Bootstrap Percolation — Facebook", out("bootstrap.png"),
                vline=ctx.mean_deg, vlabel=f"⟨d⟩={ctx.mean_deg:.1f}")
    csv_1d(k_vals, sw, "k", out("bootstrap.csv"))
    return {"k_vals": k_vals, "sw": sw}

def run_wtm(ctx: Ctx) -> dict:
    section("WTM  [sweep φ = 0.01…0.99]")
    phi_vals = np.linspace(0.01, 0.99, N_PARAMS)
    sw = sweep_1d("wtm", lambda phi: {"phi": phi},
                  phi_vals, ctx.seed_size, N_REALIZ, "wtm")
    save_1d_fig(phi_vals, sw, "Fractional threshold φ",
                "Watts Threshold Model — Facebook", out("wtm.png"))
    csv_1d(phi_vals, sw, "phi", out("wtm.csv"))
    return {"phi_vals": phi_vals, "sw": sw}

def run_sis(ctx: Ctx) -> dict:
    section(f"SIS  [sweep β, γ={GAMMA}, β_c={ctx.beta_c:.4f}]")
    beta_vals = np.linspace(0, 0.2, N_PARAMS)
    sw = sweep_1d("sis", lambda b: {"beta": b, "gamma": GAMMA},
                  beta_vals, ctx.seed_size, N_REALIZ, "sis")
    save_1d_fig(beta_vals, sw, "Transmission rate β",
                f"SIS Model — Facebook  (γ={GAMMA})", out("sis.png"),
                vline=ctx.beta_c, vlabel=f"β_c={ctx.beta_c:.4f}")
    csv_1d(beta_vals, sw, "beta", out("sis.csv"))
    return {"beta_vals": beta_vals, "sw": sw}

def run_h1(ctx: Ctx) -> dict:
    section("H1 OR (SIR + Bootstrap)  [sweep k, β ∈ {0.05, 0.15}]")
    k_vals    = list(range(1, 16))
    beta_vals = [0.05, 0.15]
    sws = {}
    for b in beta_vals:
        sws[b] = sweep_1d("h1",
                           lambda k, _b=b: {"threshold": k, "beta": _b, "gamma": GAMMA},
                           k_vals, ctx.seed_size, N_REALIZ, f"h1 β={b}")
        csv_1d(k_vals, sws[b], "k", out(f"h1_beta{b}.csv"))
    save_2panel_fig(k_vals, sws, beta_vals, "Bootstrap threshold k", "β",
                    "H1 OR Hybrid (SIR + Bootstrap) — Facebook", out("h1.png"),
                    vline=ctx.mean_deg, vlabel=f"⟨d⟩={ctx.mean_deg:.1f}")
    return {"k_vals": k_vals, "sws": sws}

def run_h2(ctx: Ctx) -> dict:
    section("H2 Sequential (SIR → Bootstrap)  [sweep switch fraction f]")
    f_vals = np.linspace(0.05, 0.5, N_PARAMS)
    sw = sweep_1d("h2",
                  lambda f: {"threshold": 2, "beta": 0.1, "gamma": GAMMA,
                             "switch_fraction": f},
                  f_vals, ctx.seed_size, N_REALIZ, "h2")
    save_1d_fig(f_vals, sw, "Switch fraction f",
                "H2 Sequential (SIR→Bootstrap) — Facebook\n(k=2, β=0.1)",
                out("h2.png"))
    csv_1d(f_vals, sw, "switch_fraction", out("h2.csv"))
    return {"f_vals": f_vals, "sw": sw}

def run_h3(ctx: Ctx) -> dict:
    section("H3 Probabilistic (2-D β × γ)")
    beta_vals = np.linspace(0.01, 0.30, N_GRID)
    gam_vals  = np.linspace(0.01, 0.30, N_GRID)
    heat = sweep_2d("h3", lambda b, g: {"beta": b, "gamma": g},
                    beta_vals, gam_vals, ctx.seed_size, N_REALIZ_2D, "h3")
    save_heatmap_fig(beta_vals, gam_vals, heat,
                     "Transmission rate β", "Recovery rate γ",
                     "H3 Probabilistic Hybrid — Facebook  (β × γ)",
                     out("h3_heatmap.png"))
    csv_2d(beta_vals, gam_vals, heat, "beta", "gamma", out("h3_heatmap.csv"))
    return {}

def run_h4(ctx: Ctx) -> dict:
    section("H4 OR (SIS + WTM)  [sweep β, φ ∈ {0.1, 0.3}]")
    beta_vals = np.linspace(0, 0.2, N_PARAMS)
    phi_vals  = [0.1, 0.3]
    sws = {}
    for p in phi_vals:
        sws[p] = sweep_1d("h4",
                           lambda b, _p=p: {"phi": _p, "beta": b, "gamma": GAMMA},
                           beta_vals, ctx.seed_size, N_REALIZ, f"h4 φ={p}")
        csv_1d(beta_vals, sws[p], "beta", out(f"h4_phi{p}.csv"))
    save_2panel_fig(beta_vals, sws, phi_vals, "Transmission rate β", "φ",
                    "H4 OR Hybrid (SIS + WTM) — Facebook", out("h4.png"),
                    vline=ctx.beta_c, vlabel=f"β_c={ctx.beta_c:.4f}")
    return {"beta_vals": beta_vals, "sws": sws}

def run_h5(ctx: Ctx) -> dict:
    section("H5 Sequential (SIS → WTM)  [sweep switch fraction f]")
    f_vals = np.linspace(0.05, 0.5, N_PARAMS)
    sw = sweep_1d("h5",
                  lambda f: {"phi": 0.2, "beta": 0.1, "gamma": GAMMA,
                             "switch_fraction": f},
                  f_vals, ctx.seed_size, N_REALIZ, "h5")
    save_1d_fig(f_vals, sw, "Switch fraction f",
                "H5 Sequential (SIS→WTM) — Facebook\n(φ=0.2, β=0.1)",
                out("h5.png"))
    csv_1d(f_vals, sw, "switch_fraction", out("h5.csv"))
    return {"f_vals": f_vals, "sw": sw}

def run_h6(ctx: Ctx) -> dict:
    section("H6 Probabilistic (2-D φ × γ)")
    phi_vals = np.linspace(0.05, 0.95, N_GRID)
    gam_vals = np.linspace(0.01, 0.30, N_GRID)
    heat = sweep_2d("h6", lambda p, g: {"phi": p, "gamma": g},
                    phi_vals, gam_vals, ctx.seed_size, N_REALIZ_2D, "h6")
    save_heatmap_fig(phi_vals, gam_vals, heat,
                     "Fractional threshold φ", "Recovery rate γ",
                     "H6 Probabilistic Hybrid — Facebook  (φ × γ)",
                     out("h6_heatmap.png"))
    csv_2d(phi_vals, gam_vals, heat, "phi", "gamma", out("h6_heatmap.csv"))
    return {}

RUNNERS = {
    "bootstrap": run_bootstrap,
    "wtm":       run_wtm,
    "sis":       run_sis,
    "h1":        run_h1,
    "h2":        run_h2,
    "h3":        run_h3,
    "h4":        run_h4,
    "h5":        run_h5,
    "h6":        run_h6,
}

# ── summary figure ─────────────────────────────────────────────────────────────

def build_summary(results: dict, ctx: Ctx):
    section("Summary figure (2×3)")
    boot = results["bootstrap"]
    wtm  = results["wtm"]
    sis  = results["sis"]
    h1   = results["h1"]
    h2   = results["h2"]
    h4   = results["h4"]

    panels = [
        (boot["k_vals"],    boot["sw"],       "Bootstrap threshold k",
         "Bootstrap Percolation", ctx.mean_deg, f"⟨d⟩={ctx.mean_deg:.1f}"),
        (wtm["phi_vals"],   wtm["sw"],        "Fractional threshold φ",
         "Watts Threshold Model", None, None),
        (sis["beta_vals"],  sis["sw"],        "Transmission rate β",
         f"SIS  (γ={GAMMA})",    ctx.beta_c, f"β_c={ctx.beta_c:.4f}"),
        (h1["k_vals"],      h1["sws"][0.15],  "Bootstrap threshold k",
         "H1 OR  (β=0.15)",     ctx.mean_deg, f"⟨d⟩={ctx.mean_deg:.1f}"),
        (h4["beta_vals"],   h4["sws"][0.1],   "Transmission rate β",
         "H4 OR  (φ=0.1)",      ctx.beta_c, f"β_c={ctx.beta_c:.4f}"),
        (h2["f_vals"],      h2["sw"],         "Switch fraction f",
         "H2 Sequential (k=2, β=0.1)", None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (xv, sw, xl, title, vl, vll) in zip(axes.flatten(), panels):
        _curves(ax, xv, sw, vl, vll)
        ax.set_xlabel(xl)
        ax.set_title(title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flatten():
        if ax.get_legend():
            ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=10, framealpha=0.9)
    fig.suptitle("Phase Transitions on Facebook SNAP Graph — All Models",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out("summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(out("summary.png"))

# ── main ──────────────────────────────────────────────────────────────────────

def out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)

def section(title: str):
    print(f"\n{'━'*54}\n  {title}\n{'━'*54}")

def main():
    freeze_support()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default="all",
        choices=[*RUNNERS.keys(), "all"],
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading graph and computing stats …")
    G, mean_deg, beta_c = load_and_describe(GRAPH_PATH)
    seed_size = max(1, int(SEED_FRAC * G.number_of_nodes()))
    ctx = Ctx(mean_deg, beta_c, seed_size)

    print(f"\n  Seed size : {seed_size}  |  Workers: {N_WORKERS}"
          f"  |  Realisations: {N_REALIZ} (2D: {N_REALIZ_2D})")

    t0 = time.time()

    to_run = list(RUNNERS.keys()) if args.model == "all" else [args.model]
    results: dict[str, dict] = {}

    for name in to_run:
        results[name] = RUNNERS[name](ctx)

    if args.model == "all" and all(k in results for k in ("bootstrap","wtm","sis","h1","h2","h4")):
        build_summary(results, ctx)

    elapsed = time.time() - t0
    print(f"\n{'='*54}")
    print(f"  Done in {elapsed:.1f}s  |  output: results/facebook/")
    print("=" * 54)


if __name__ == "__main__":
    main()
