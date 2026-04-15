"""
Generate Figure 2 from the paper: initial simulation states for three configurations.
Produces PDF (vector) and PNG versions suitable for LaTeX inclusion.

Configurations:
  (a) 20×20 grid, avg sugar = 0.5  → G20_S0.50_R0
  (b) 30×30 grid, avg sugar = 1.5  → G30_S1.50_R0
  (c) 50×50 grid, avg sugar = 3.0  → batch_20260211_130117/G50_S3.00_R0

Usage:
  python generate_initial_states.py
  → saves: static_comparison20x20.pdf/.png
           static_comparison30x30.pdf/.png
           static_comparison50x50.pdf/.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Data sources ──────────────────────────────────────────────────────────────

REPO_DATA = Path(__file__).parent / "results/simulation_data"
ALT_DATA  = Path("/workspaces/Coding/Tesi/Batch_Simulations/batch_20260211_130117")

CONFIGS = [
    {
        "label":    "20x20",
        "run_dir":  REPO_DATA / "G20_S0.50_R0",
        "title":    r"$20 \times 20$ grid, $\bar{s} = 0.5$",
        "subfig":   "(a)",
    },
    {
        "label":    "30x30",
        "run_dir":  REPO_DATA / "G30_S1.50_R0",
        "title":    r"$30 \times 30$ grid, $\bar{s} = 1.5$",
        "subfig":   "(b)",
    },
    {
        "label":    "50x50",
        "run_dir":  ALT_DATA / "G50_S3.00_R0",
        "title":    r"$50 \times 50$ grid, $\bar{s} = 3.0$",
        "subfig":   "(c)",
    },
]

OUTPUT_DIR = Path(__file__).parent


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    run_dir = Path(run_dir)

    # Config
    cfg = dict(zip(*[pd.read_csv(run_dir / "config.csv")[c]
                     for c in ["parameter", "value"]]))
    grid_size = int(cfg.get("grid_size", 20))

    # Sugar capacity (static map)
    cap_df = pd.read_csv(run_dir / "sugar_capacity.csv")
    sugar_cap = np.zeros((grid_size, grid_size))
    for _, row in cap_df.iterrows():
        sugar_cap[int(row["y"]), int(row["x"])] = float(row["capacity"])

    # Map state at step 0
    map_df = pd.read_csv(run_dir / "map_state.csv")
    step0 = map_df[map_df["step"] == 0]
    sugar_map = np.zeros((grid_size, grid_size))
    for _, row in step0.iterrows():
        sugar_map[int(row["y"]), int(row["x"])] = float(row["sugar"])

    # Initial agents (step 0 from agent_moves)
    moves_df = pd.read_csv(run_dir / "agent_moves.csv")
    agents0 = moves_df[moves_df["step"] == 0].copy()

    return grid_size, sugar_cap, sugar_map, agents0


# ── Single-panel plot ─────────────────────────────────────────────────────────

def plot_initial_state(ax, grid_size, sugar_map, agents0, title, subfig):
    """Draw one panel: sugar heatmap + agent scatter."""

    sugar_vmax = max(5.0, sugar_map.max())

    # --- Sugar heatmap ---
    im = ax.imshow(
        sugar_map,
        extent=[-0.5, grid_size - 0.5, -0.5, grid_size - 0.5],
        origin="lower",
        cmap="YlGn",
        vmin=0,
        vmax=sugar_vmax,
        alpha=0.85,
        zorder=1,
        interpolation="nearest",
    )

    # --- Agents ---
    alive = agents0[agents0["alive"] == True].copy()

    if len(alive):
        sugar_vals = alive["sugar"].values.astype(float)
        alpha_vals = alive["alpha"].values.astype(float)

        # Size proportional to sugar (15–180 pt²)
        s_min, s_max = sugar_vals.min(), sugar_vals.max()
        if s_max > s_min:
            sizes = 15 + (sugar_vals - s_min) / (s_max - s_min) * 165
        else:
            sizes = np.full(len(alive), 60.0)

        sc = ax.scatter(
            alive["x"].values,
            alive["y"].values,
            c=alpha_vals,
            s=sizes,
            cmap="RdYlBu_r",
            vmin=-1, vmax=1,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
    else:
        sc = None

    # --- Labels & formatting ---
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{subfig}  {title}", fontsize=9, pad=5)

    # Agent count annotation
    n_alive = int(alive["alive"].sum()) if len(alive) else 0
    ax.text(
        0.03, 0.97, f"$n = {n_alive}$",
        transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )

    return im, sc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── individual figures (one per config, for \includegraphics) ──────────────
    for cfg in CONFIGS:
        print(f"Processing {cfg['label']} …")
        grid_size, sugar_cap, sugar_map, agents0 = load_run(cfg["run_dir"])

        fig, ax = plt.subplots(figsize=(4.0, 4.0), facecolor="white")
        fig.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.10)

        im, sc = plot_initial_state(
            ax, grid_size, sugar_map, agents0, cfg["title"], cfg["subfig"]
        )

        # Colorbars
        cb_s = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, location="right")
        cb_s.set_label("Sugar", fontsize=7)
        cb_s.ax.tick_params(labelsize=6)

        if sc is not None:
            cb_a = fig.colorbar(
                ScalarMappable(norm=Normalize(-1, 1), cmap="RdYlBu_r"),
                ax=ax, fraction=0.038, pad=0.10, location="right"
            )
            cb_a.set_label(r"$\alpha$", fontsize=7)
            cb_a.ax.tick_params(labelsize=6)

        for ext in ("pdf", "png"):
            out = OUTPUT_DIR / f"static_comparison{cfg['label']}.{ext}"
            fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {out.name}")

        plt.close(fig)

    # ── combined 3-panel figure (optional, for reference) ──────────────────────
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.4), facecolor="white",
        gridspec_kw={"wspace": 0.35}
    )

    for ax, cfg in zip(axes, CONFIGS):
        grid_size, sugar_cap, sugar_map, agents0 = load_run(cfg["run_dir"])
        im, sc = plot_initial_state(
            ax, grid_size, sugar_map, agents0, cfg["title"], cfg["subfig"]
        )
        cb_s = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
        cb_s.set_label("Sugar", fontsize=7)
        cb_s.ax.tick_params(labelsize=6)
        if sc is not None:
            cb_a = fig.colorbar(
                ScalarMappable(norm=Normalize(-1, 1), cmap="RdYlBu_r"),
                ax=ax, fraction=0.038, pad=0.10
            )
            cb_a.set_label(r"$\alpha$", fontsize=7)
            cb_a.ax.tick_params(labelsize=6)

    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"static_comparison_combined.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out.name}")
    plt.close(fig)

    print("\nDone. LaTeX usage:")
    print(r"  \includegraphics[width=0.32\textwidth]{static_comparison20x20.pdf}")
    print(r"  \includegraphics[width=0.32\textwidth]{static_comparison30x30.pdf}")
    print(r"  \includegraphics[width=0.32\textwidth]{static_comparison50x50.pdf}")


if __name__ == "__main__":
    main()
