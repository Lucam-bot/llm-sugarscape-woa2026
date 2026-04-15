"""
Aggregated Survival Curves with Fitted Exponential Decay
=========================================================
For each (grid, sugar) configuration:
  - Average the survival curves across all repeats (different v, m)
  - Fit N(t) = N_inf + (N0 - N_inf) * exp(-lambda * t)
  - Show data (mean +/- std) and fit on a single plot

Produces:
  1. One big figure with all configs overlaid
  2. Grid of subplots: one per config with fit details
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — CHANGE THESE PATHS
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N0 = 50
MAX_STEPS = 200

# ============================================================
# HELPERS
# ============================================================

def exp_decay(t, N_inf, lam):
    return N_inf + (N0 - N_inf) * np.exp(-lam * t)


def _read_config(config_file):
    config = {}
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    config[row[0].strip()] = float(row[1].strip())
                except ValueError:
                    config[row[0].strip()] = row[1].strip()
    return config


def load_all_curves(batch_folder):
    """Load survival curves from all runs, grouped by (grid, sugar)."""
    curves = {}  # (grid, sugar) -> list of arrays
    
    # Subfolders
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    # Prefixed files fallback
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefixes.add(f.name.replace("config.csv", ""))
        for prefix in sorted(prefixes):
            config_file = batch_folder / f"{prefix}config.csv"
            stats_file = batch_folder / f"{prefix}step_stats.csv"
            if config_file.exists() and stats_file.exists():
                config = _read_config(config_file)
                G = int(config.get('grid_size', 0))
                S = float(config.get('avg_sugar_target', 0))
                stats = pd.read_csv(stats_file, encoding='utf-8-sig')
                key = (G, S)
                if key not in curves:
                    curves[key] = []
                curves[key].append(stats[['step', 'agents_alive']].values)
        return curves
    
    for i, folder in enumerate(folders):
        config_file = folder / "config.csv"
        stats_file = folder / "step_stats.csv"
        if not config_file.exists() or not stats_file.exists():
            continue
        config = _read_config(config_file)
        G = int(config.get('grid_size', 0))
        S = float(config.get('avg_sugar_target', 0))
        stats = pd.read_csv(stats_file, encoding='utf-8-sig')
        key = (G, S)
        if key not in curves:
            curves[key] = []
        curves[key].append(stats[['step', 'agents_alive']].values)
    
    return curves


def aggregate_curves(curve_list, max_steps=MAX_STEPS):
    """Compute mean and std of survival curves, padding shorter runs with 0."""
    t = np.arange(max_steps)
    all_N = np.zeros((len(curve_list), max_steps))
    
    for i, data in enumerate(curve_list):
        steps = data[:, 0].astype(int)
        alive = data[:, 1].astype(float)
        for s, a in zip(steps, alive):
            if s < max_steps:
                all_N[i, s] = a
        # Pad: if run ended early, agents stayed at last value (usually 0)
        last_step = int(steps[-1])
        last_alive = alive[-1]
        if last_step < max_steps - 1:
            all_N[i, last_step + 1:] = last_alive
    
    mean_N = np.mean(all_N, axis=0)
    std_N = np.std(all_N, axis=0)
    return t, mean_N, std_N


def fit_mean_curve(t, mean_N):
    """Fit exponential decay to the mean curve."""
    N_inf_guess = max(mean_N[-1], 0.5)
    try:
        popt, _ = curve_fit(
            exp_decay, t, mean_N,
            p0=[N_inf_guess, 0.02],
            bounds=([0, 0.001], [N0, 1.0]),
            maxfev=10000
        )
        N_inf, lam = popt
        N_pred = exp_decay(t, *popt)
        ss_res = np.sum((mean_N - N_pred) ** 2)
        ss_tot = np.sum((mean_N - np.mean(mean_N)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return N_inf, lam, r2
    except:
        return None, None, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  AGGREGATED SURVIVAL CURVES + FIT")
    print("=" * 65)
    
    curves = load_all_curves(BATCH_FOLDER)
    if not curves:
        print("  No data found!")
        return
    
    configs = sorted(curves.keys())
    grids = sorted(set(g for g, s in configs))
    sugars = sorted(set(s for g, s in configs))
    
    print(f"  Configurations: {len(configs)}")
    print(f"  Grids: {grids}")
    print(f"  Sugars: {sugars}")
    
    # Compute aggregated curves and fits
    results = {}
    for key in configs:
        G, S = key
        t, mean_N, std_N = aggregate_curves(curves[key])
        N_inf, lam, r2 = fit_mean_curve(t, mean_N)
        results[key] = {
            't': t, 'mean': mean_N, 'std': std_N,
            'N_inf': N_inf, 'lambda': lam, 'R2': r2,
            'n_runs': len(curves[key])
        }
        if N_inf is not None:
            print(f"  G={G:2d} S={S:.1f}: N_inf={N_inf:.1f}, lambda={lam:.4f}, R2={r2:.3f} ({len(curves[key])} runs)")
    
    # Color scheme: YlGnBu colormap for sugar levels
    cmap = plt.cm.YlGnBu
    sugar_colors = {}
    for i, s in enumerate(sugars):
        sugar_colors[s] = cmap(0.25 + 0.7 * i / max(len(sugars) - 1, 1))
    
    # Fallback for unexpected sugar values
    _extra_colors = ['#1ABC9C', '#E91E63', '#795548', '#607D8B']
    for i, s in enumerate(sugars):
        if s not in sugar_colors:
            sugar_colors[s] = _extra_colors[i % len(_extra_colors)]
    
    # All solid lines, distinguished by width
    grid_styles = {g: '-' for g in grids}
    grid_widths = {g: w for g, w in zip(grids, [3.0, 2.0, 1.2, 0.8][:len(grids)])}
    
    # ============================================================
    # PLOT 1: All configs on one figure (data + fit)
    # ============================================================
    print("\n  Generating plots...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for key in configs:
        G, S = key
        r = results[key]
        
        color = sugar_colors.get(S, '#333333')
        style = grid_styles.get(G, '-')
        width = grid_widths.get(G, 1.5)
        
        # Data: mean with shaded std
        ax.plot(r['t'], r['mean'], color=color, linestyle=style, linewidth=width, alpha=0.85,
                label=f'G={G} S={S}')
        ax.fill_between(r['t'], r['mean'] - r['std'], r['mean'] + r['std'],
                        color=color, alpha=0.08)
        
        # Fit: dashed, darker version of same color
        if r['N_inf'] is not None:
            t_fit = np.linspace(0, MAX_STEPS, 300)
            N_fit = exp_decay(t_fit, r['N_inf'], r['lambda'])
            # Darken color for fit
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(color)
            dark_color = tuple(c * 0.90 for c in rgb)
            ax.plot(t_fit, N_fit, '--', color=dark_color, linewidth=width * 0.6, alpha=0.7)
    
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('Agents Alive', fontsize=13)
    ax.set_title('Survival Curves: Data (solid, mean +/- std) and Fit (dashed)\n'
                 r'$N(t) = N_\infty + (N_0 - N_\infty) \cdot e^{-\lambda t}$',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(bottom=0)
    
    # Build legend: grid widths + sugar colors + fit explanation
    from matplotlib.lines import Line2D
    
    grid_legend = [Line2D([0], [0], color='gray', linestyle='-',
                          linewidth=grid_widths[G], label=f'Grid {G}x{G}') for G in grids]
    sugar_legend = [Line2D([0], [0], color=sugar_colors[S], linestyle='-',
                           linewidth=2.5, label=f'S={S}') for S in sugars]
    fit_legend = [Line2D([0], [0], color='black', linestyle='--',
                         linewidth=1.2, label='Exp. fit')]
    
    leg1 = ax.legend(handles=grid_legend, loc='upper right', fontsize=9, title='Grid (line width)')
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=sugar_legend, loc='center right', fontsize=9, title='Sugar (color)')
    ax.add_artist(leg2)
    leg3 = ax.legend(handles=fit_legend, loc='center left', bbox_to_anchor=(0.922, 0.75), fontsize=9)


    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_curves_aggregated.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    survival_curves_aggregated.png")
    
    # ============================================================
    # PLOT 2: Grid of subplots — one per (grid, sugar)
    # ============================================================
    n_rows = len(grids)
    n_cols = len(sugars)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              sharex=True, sharey=True, squeeze=False)
    
    for i, G in enumerate(grids):
        for j, S in enumerate(sugars):
            ax = axes[i, j]
            key = (G, S)
            
            if key not in results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=11)
                ax.set_facecolor('#f5f5f5')
            else:
                r = results[key]
                
                # Data
                ax.plot(r['t'], r['mean'], 'b-', linewidth=1.5, label='Data (mean)')
                ax.fill_between(r['t'], r['mean'] - r['std'], r['mean'] + r['std'],
                                color='blue', alpha=0.15, label='+/- std')
                
                # Fit
                if r['N_inf'] is not None:
                    t_fit = np.linspace(0, MAX_STEPS, 300)
                    N_fit = exp_decay(t_fit, r['N_inf'], r['lambda'])
                    ax.plot(t_fit, N_fit, 'r--', linewidth=1.5, label='Fit')
                    
                    # Annotation
                    ax.text(0.97, 0.97,
                            f"$N_\\infty$={r['N_inf']:.1f}\n"
                            f"$\\lambda$={r['lambda']:.4f}\n"
                            f"$R^2$={r['R2']:.3f}\n"
                            f"n={r['n_runs']}",
                            transform=ax.transAxes, fontsize=7,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
                
                ax.set_ylim(bottom=0, top=N0 + 3)
            
            # Labels
            if i == 0:
                ax.set_title(f'S={S}', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Grid {G}x{G}\nAgents Alive', fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel('Step', fontsize=10)
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=6, loc='center right')
    
    fig.suptitle(r'Survival Curves per Configuration: $N(t) = N_\infty + (N_0 - N_\infty) \cdot e^{-\lambda t}$',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_curves_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    survival_curves_grid.png")
    
    # ============================================================
    # PLOT 3: Fit parameters summary
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    grid_colors = {g: c for g, c in zip(grids, ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'][:len(grids)])}
    
    # Lambda vs sugar
    ax = axes[0]
    for G in grids:
        s_vals, l_vals = [], []
        for S in sugars:
            if (G, S) in results and results[(G, S)]['lambda'] is not None:
                s_vals.append(S)
                l_vals.append(results[(G, S)]['lambda'])
        ax.plot(s_vals, l_vals, 'o-', color=grid_colors[G], label=f'Grid {G}x{G}',
                linewidth=2, markersize=8)
    ax.set_xlabel('Average Sugar per Cell', fontsize=12)
    ax.set_ylabel(r'$\lambda$ (decay rate)', fontsize=12)
    ax.set_title(r'$\lambda$ vs Sugar Density', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # N_inf vs sugar
    ax = axes[1]
    for G in grids:
        s_vals, n_vals = [], []
        for S in sugars:
            if (G, S) in results and results[(G, S)]['N_inf'] is not None:
                s_vals.append(S)
                n_vals.append(results[(G, S)]['N_inf'])
        ax.plot(s_vals, n_vals, 'o-', color=grid_colors[G], label=f'Grid {G}x{G}',
                linewidth=2, markersize=8)
    ax.set_xlabel('Average Sugar per Cell', fontsize=12)
    ax.set_ylabel(r'$N_\infty$ (carrying capacity)', fontsize=12)
    ax.set_title(r'$N_\infty$ vs Sugar Density', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # R2 vs sugar
    ax = axes[2]
    for G in grids:
        s_vals, r_vals = [], []
        for S in sugars:
            if (G, S) in results and results[(G, S)]['R2'] is not None:
                s_vals.append(S)
                r_vals.append(results[(G, S)]['R2'])
        ax.plot(s_vals, r_vals, 'o-', color=grid_colors[G], label=f'Grid {G}x{G}',
                linewidth=2, markersize=8)
    ax.set_xlabel('Average Sugar per Cell', fontsize=12)
    ax.set_ylabel(r'$R^2$', fontsize=12)
    ax.set_title(r'Goodness of Fit ($R^2$) vs Sugar', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_fit_parameters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    survival_fit_parameters.png")
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n  {'='*65}")
    print(f"  FIT SUMMARY")
    print(f"  {'='*65}")
    print(f"  {'Grid':>5s}  {'Sugar':>6s}  {'N_inf':>6s}  {'lambda':>8s}  {'R2':>6s}  {'Runs':>5s}")
    for key in configs:
        G, S = key
        r = results[key]
        if r['N_inf'] is not None:
            print(f"  {G:5d}  {S:6.1f}  {r['N_inf']:6.1f}  {r['lambda']:8.4f}  {r['R2']:6.3f}  {r['n_runs']:5d}")
        else:
            print(f"  {G:5d}  {S:6.1f}  {'FAIL':>6s}  {'':>8s}  {'':>6s}  {r['n_runs']:5d}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()