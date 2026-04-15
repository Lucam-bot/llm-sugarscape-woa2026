"""
Alpha & Neighbors Interaction Analysis
========================================
Analyzes how agent identity (alpha) interacts with neighbor context
to influence decisions. Computes per-decision:
  - mean |delta alpha| with all agents in vision
  - agent's own alpha
  - number of neighbors

Produces:
  1. Attack rate vs mean alpha gap (does identity difference trigger attacks?)
  2. Action distribution by agent alpha value  
  3. 2D heatmap: n_neighbors x alpha_gap → attack rate
  4. Comparison: alpha gap of attackers vs non-attackers (given neighbors present)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import hypot
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — CHANGE THESE PATHS
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def classify_action(dec_str):
    s = str(dec_str).strip()
    if s.startswith('1,'):   return 'Move'
    elif s.startswith('2') or s == '2-STAY': return 'Stay'
    elif s.startswith('3,'): return 'Reproduce'
    elif s.startswith('4,'): return 'Attack'
    else: return None


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


def _process_run(folder, prefix=""):
    """Load a run and compute neighbor count + mean alpha gap per decision."""
    config_file = folder / f"{prefix}config.csv"
    moves_file = folder / f"{prefix}agent_moves.csv"
    
    if not config_file.exists() or not moves_file.exists():
        return None
    
    config = _read_config(config_file)
    grid_size = int(config.get('grid_size', 20))
    vision = int(config.get('global_vision', 4))
    
    full_moves = pd.read_csv(moves_file, encoding='utf-8-sig')
    
    # --- Per step: compute neighbors + mean alpha gap (vision AND adjacent) ---
    neighbor_count_map = {}       # vision range
    alpha_gap_map = {}            # vision range
    adj_count_map = {}            # adjacent only (dist <= 1)
    adj_alpha_gap_map = {}        # adjacent only (dist <= 1)
    
    for step, group in full_moves.groupby('step'):
        agents = group[['agent_id', 'x', 'y', 'alpha']].drop_duplicates('agent_id').values
        n = len(agents)
        
        for i in range(n):
            aid_i = int(agents[i][0])
            ax, ay, alpha_i = agents[i][1], agents[i][2], agents[i][3]
            
            vision_alphas = []
            adjacent_alphas = []
            
            for j in range(n):
                if i == j:
                    continue
                bx, by, alpha_j = agents[j][1], agents[j][2], agents[j][3]
                dist = hypot(ax - bx, ay - by)
                
                if dist <= vision:
                    vision_alphas.append(alpha_j)
                if dist <= 1.0001:  # adjacent (includes diagonals)
                    adjacent_alphas.append(alpha_j)
            
            # Vision-range metrics
            neighbor_count_map[(step, aid_i)] = len(vision_alphas)
            if vision_alphas:
                alpha_gap_map[(step, aid_i)] = np.mean([abs(alpha_i - a) for a in vision_alphas])
            else:
                alpha_gap_map[(step, aid_i)] = np.nan
            
            # Adjacent metrics
            adj_count_map[(step, aid_i)] = len(adjacent_alphas)
            if adjacent_alphas:
                adj_alpha_gap_map[(step, aid_i)] = np.mean([abs(alpha_i - a) for a in adjacent_alphas])
            else:
                adj_alpha_gap_map[(step, aid_i)] = np.nan
    
    # Filter to LLM decisions
    full_moves['action'] = full_moves['decision'].apply(classify_action)
    moves = full_moves[full_moves['action'].notna()].copy()
    
    if moves.empty:
        return None
    
    moves['n_neighbors'] = moves.apply(
        lambda r: neighbor_count_map.get((r['step'], r['agent_id']), 0), axis=1)
    moves['mean_alpha_gap'] = moves.apply(
        lambda r: alpha_gap_map.get((r['step'], r['agent_id']), np.nan), axis=1)
    moves['n_adjacent'] = moves.apply(
        lambda r: adj_count_map.get((r['step'], r['agent_id']), 0), axis=1)
    moves['adj_alpha_gap'] = moves.apply(
        lambda r: adj_alpha_gap_map.get((r['step'], r['agent_id']), np.nan), axis=1)
    moves['abs_alpha'] = moves['alpha'].abs()
    
    moves['grid_size'] = grid_size
    moves['run_id'] = prefix.rstrip('_') if prefix else folder.name
    
    return moves


def load_all(batch_folder):
    all_data = []
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefixes.add(f.name.replace("config.csv", ""))
        for prefix in sorted(prefixes):
            result = _process_run(batch_folder, prefix=prefix)
            if result is not None:
                all_data.append(result)
    else:
        for i, folder in enumerate(folders):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    Processing {i+1}/{len(folders)}: {folder.name}")
            result = _process_run(folder)
            if result is not None:
                all_data.append(result)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  ALPHA & NEIGHBORS INTERACTION ANALYSIS")
    print("=" * 65)
    
    df = load_all(BATCH_FOLDER)
    if df.empty:
        print("  No data found!")
        return
    
    print(f"\n  Total LLM decisions: {len(df):,}")
    
    # Only decisions where agent had at least 1 neighbor (alpha gap is meaningful)
    df_with_neighbors = df[df['n_neighbors'] > 0].copy()
    print(f"  Decisions with neighbors: {len(df_with_neighbors):,}")
    
    actions = ['Move', 'Stay', 'Attack', 'Reproduce']
    colors = {'Move': '#0072B2', 'Stay': '#D55E00', 'Attack': '#009E73', 'Reproduce': '#CC79A7'}
    
    # ============================================================
    # PLOT 1: Attack rate vs mean alpha gap with neighbors
    # ============================================================
    print("\n  Plot 1: Actions vs Alpha Gap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    gap_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    gap_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.5', '1.5+']
    df_with_neighbors['alpha_gap_bin'] = pd.cut(
        df_with_neighbors['mean_alpha_gap'], bins=gap_bins, labels=gap_labels
    )
    
    # Left: stacked bar
    ax = axes[0]
    grouped = df_with_neighbors.groupby(['alpha_gap_bin', 'action']).size().unstack(fill_value=0)
    grouped = grouped[grouped.sum(axis=1) >= 10]
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(grouped_pct))
    x_pos = range(len(grouped_pct))
    
    for action in actions:
        vals = grouped_pct[action].values if action in grouped_pct.columns else np.zeros(len(grouped_pct))
        ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.7)
        for j, v in enumerate(vals):
            if v > 5:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=8)
        bottom += vals
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped_pct.index, fontsize=9)
    ax.set_xlabel('Mean |alpha_self - alpha_neighbor|', fontsize=12)
    ax.set_ylabel('Action Frequency (%)', fontsize=12)
    ax.set_title('Action Distribution by Alpha Gap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    for j, idx in enumerate(grouped_pct.index):
        n = int(grouped.loc[idx].sum())
        ax.text(j, 102, f'n={n:,}', ha='center', fontsize=7, color='gray')
    
    # Right: line plot — attack rate by alpha gap
    ax = axes[1]
    for action in actions:
        rates = df_with_neighbors.groupby('alpha_gap_bin').apply(
            lambda g: (g['action'] == action).mean() * 100 if len(g) >= 10 else np.nan
        ).dropna()
        ax.plot(range(len(rates)), rates.values, 'o-', label=action, color=colors[action],
                linewidth=2, markersize=7)
    
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=9)
    ax.set_xlabel('Mean |alpha_self - alpha_neighbor|', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Action Rates by Alpha Gap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    
    fig.suptitle('Does Identity Difference With Neighbors Affect Decisions?',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_gap_actions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    alpha_gap_actions.png")
    
    # ============================================================
    # PLOT 2: Actions by agent's own alpha value
    # ============================================================
    print("  Plot 2: Actions vs Agent Alpha...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    alpha_bins = [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01]
    alpha_labels = ['-1 to -0.6', '-0.6 to -0.2', '-0.2 to 0.2', '0.2 to 0.6', '0.6 to 1']
    df['alpha_bin'] = pd.cut(df['alpha'], bins=alpha_bins, labels=alpha_labels)
    
    # Left: stacked bar
    ax = axes[0]
    grouped = df.groupby(['alpha_bin', 'action']).size().unstack(fill_value=0)
    grouped = grouped[grouped.sum(axis=1) >= 10]
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(grouped_pct))
    x_pos = range(len(grouped_pct))
    
    for action in actions:
        vals = grouped_pct[action].values if action in grouped_pct.columns else np.zeros(len(grouped_pct))
        ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.7)
        for j, v in enumerate(vals):
            if v > 5:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=8)
        bottom += vals
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped_pct.index, fontsize=9)
    ax.set_xlabel('Agent Alpha Value', fontsize=12)
    ax.set_ylabel('Action Frequency (%)', fontsize=12)
    ax.set_title('Action Distribution by Own Alpha', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    for j, idx in enumerate(grouped_pct.index):
        n = int(grouped.loc[idx].sum())
        ax.text(j, 102, f'n={n:,}', ha='center', fontsize=7, color='gray')
    
    # Right: attack rate by alpha (line)
    ax = axes[1]
    for action in actions:
        rates = df.groupby('alpha_bin').apply(
            lambda g: (g['action'] == action).mean() * 100 if len(g) >= 10 else np.nan
        ).dropna()
        ax.plot(range(len(rates)), rates.values, 'o-', label=action, color=colors[action],
                linewidth=2, markersize=7)
    
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=9)
    ax.set_xlabel('Agent Alpha Value', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Action Rates by Own Alpha', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    
    fig.suptitle('Does Agent Identity (Alpha) Affect Its Decisions?',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_own_actions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    alpha_own_actions.png")
    
    # ============================================================
    # PLOT 3: 2D heatmap — n_neighbors x alpha_gap → attack rate
    # ============================================================
    print("  Plot 3: 2D Heatmap neighbors x alpha gap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bin both dimensions
    df_wn = df_with_neighbors.copy()
    df_wn['n_neigh_bin'] = df_wn['n_neighbors'].clip(upper=6)
    df_wn['gap_bin'] = pd.cut(df_wn['mean_alpha_gap'], bins=[0, 0.3, 0.6, 0.9, 1.2, 2.0],
                               labels=['0-0.3', '0.3-0.6', '0.6-0.9', '0.9-1.2', '1.2+'])
    
    # Attack rate heatmap
    ax = axes[0]
    pivot = df_wn.groupby(['n_neigh_bin', 'gap_bin']).apply(
        lambda g: (g['action'] == 'Attack').mean() * 100 if len(g) >= 5 else np.nan
    ).unstack()
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    neigh_labels = [str(int(x)) if x < 6 else '6+' for x in pivot.index]
    ax.set_yticklabels(neigh_labels, fontsize=10)
    ax.set_xlabel('Mean Alpha Gap with Neighbors', fontsize=12)
    ax.set_ylabel('Number of Neighbors in Vision', fontsize=12)
    ax.set_title('Attack Rate (%)', fontsize=13, fontweight='bold')
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmax(pivot.values) * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)
            else:
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8, color='gray')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Attack Rate (%)')
    
    # Sample count heatmap
    ax = axes[1]
    count_pivot = df_wn.groupby(['n_neigh_bin', 'gap_bin']).size().unstack(fill_value=0)
    
    im = ax.imshow(count_pivot.values, cmap='Blues', aspect='auto', origin='lower')
    ax.set_xticks(range(len(count_pivot.columns)))
    ax.set_xticklabels(count_pivot.columns, fontsize=9)
    ax.set_yticks(range(len(count_pivot.index)))
    ax.set_yticklabels(neigh_labels, fontsize=10)
    ax.set_xlabel('Mean Alpha Gap with Neighbors', fontsize=12)
    ax.set_ylabel('Number of Neighbors in Vision', fontsize=12)
    ax.set_title('Sample Count', fontsize=13, fontweight='bold')
    
    for i in range(len(count_pivot.index)):
        for j in range(len(count_pivot.columns)):
            val = count_pivot.values[i, j]
            color = 'white' if val > np.max(count_pivot.values) * 0.6 else 'black'
            ax.text(j, i, f'{int(val):,}', ha='center', va='center', fontsize=9, color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
    
    fig.suptitle('Attack Rate by Neighbors x Alpha Gap\n(2D interaction)',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_2d_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    alpha_2d_heatmap.png")
    
    # ============================================================
    # PLOT 4: Alpha gap distribution — attackers vs non-attackers
    # ============================================================
    print("  Plot 4: Attackers vs Non-attackers alpha gap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Only rows with neighbors (gap is meaningful)
    has_gap = df_with_neighbors['mean_alpha_gap'].notna()
    attackers = df_with_neighbors[has_gap & (df_with_neighbors['action'] == 'Attack')]
    non_attackers = df_with_neighbors[has_gap & (df_with_neighbors['action'] != 'Attack')]
    
    # Left: overlapping histograms
    ax = axes[0]
    bins_hist = np.linspace(0, 2, 25)
    ax.hist(non_attackers['mean_alpha_gap'], bins=bins_hist, alpha=0.6, density=True,
            label=f'Non-Attack (n={len(non_attackers):,})', color='#5B8DB8')
    ax.hist(attackers['mean_alpha_gap'], bins=bins_hist, alpha=0.7, density=True,
            label=f'Attack (n={len(attackers):,})', color='#6BBF8A')
    
    # Means
    if len(attackers) > 0:
        ax.axvline(attackers['mean_alpha_gap'].mean(), color='#2ECC71', linewidth=2, linestyle='--',
                   label=f'Attack mean: {attackers["mean_alpha_gap"].mean():.3f}')
    ax.axvline(non_attackers['mean_alpha_gap'].mean(), color='#2980B9', linewidth=2, linestyle='--',
               label=f'Non-attack mean: {non_attackers["mean_alpha_gap"].mean():.3f}')
    
    ax.set_xlabel('Mean |alpha gap| with neighbors', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Alpha Gap: Attackers vs Others', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Right: own alpha distribution — attackers vs non-attackers
    ax = axes[1]
    bins_alpha = np.linspace(-1, 1, 25)
    ax.hist(non_attackers['alpha'], bins=bins_alpha, alpha=0.6, density=True,
            label=f'Non-Attack (n={len(non_attackers):,})', color='#5B8DB8')
    ax.hist(attackers['alpha'], bins=bins_alpha, alpha=0.7, density=True,
            label=f'Attack (n={len(attackers):,})', color='#6BBF8A')
    
    ax.set_xlabel('Agent Alpha', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Own Alpha: Attackers vs Others', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    fig.suptitle('Are Attackers Different? Alpha Gap & Identity Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_attackers_vs_others.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    alpha_attackers_vs_others.png")
    
    # ============================================================
    # PLOT 5: VISION vs ADJACENT — side-by-side comparison
    # ============================================================
    print("  Plot 5: Vision-range vs Adjacent alpha gap comparison...")
    
    df_with_adj = df[df['n_adjacent'] > 0].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Top-left: Attack rate by VISION alpha gap ---
    ax = axes[0, 0]
    gap_bins_v = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    gap_labels_v = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.5', '1.5+']
    df_with_neighbors['vision_gap_bin'] = pd.cut(
        df_with_neighbors['mean_alpha_gap'], bins=gap_bins_v, labels=gap_labels_v)
    
    for action in actions:
        rates = df_with_neighbors.groupby('vision_gap_bin').apply(
            lambda g: (g['action'] == action).mean() if len(g) >= 10 else np.nan
        ).dropna()
        ax.plot(range(len(rates)), rates.values, 'o-', label=action, color=colors[action],
                linewidth=2, markersize=6)
    
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=8)
    ax.set_xlabel('Mean Alpha Gap (VISION range)', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Action Rates by Vision Alpha Gap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    
    # --- Top-right: Attack rate by ADJACENT alpha gap ---
    ax = axes[0, 1]
    df_with_adj['adj_gap_bin'] = pd.cut(
        df_with_adj['adj_alpha_gap'], bins=gap_bins_v, labels=gap_labels_v)
    
    for action in actions:
        rates = df_with_adj.groupby('adj_gap_bin').apply(
            lambda g: (g['action'] == action).mean() if len(g) >= 10 else np.nan
        ).dropna()
        if len(rates) > 0:
            ax.plot(range(len(rates)), rates.values, 'o-', label=action, color=colors[action],
                    linewidth=2, markersize=6)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels(rates.index, fontsize=8)
    
    ax.set_xlabel('Mean Alpha Gap (ADJACENT only, dist <= 1)', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Action Rates by Adjacent Alpha Gap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    
    # --- Bottom-left: Attackers vs Others — VISION gap ---
    ax = axes[1, 0]
    bins_hist = np.linspace(0, 2, 25)
    att_v = df_with_neighbors[df_with_neighbors['action'] == 'Attack']['mean_alpha_gap'].dropna()
    non_v = df_with_neighbors[df_with_neighbors['action'] != 'Attack']['mean_alpha_gap'].dropna()
    
    ax.hist(non_v, bins=bins_hist, alpha=0.6, density=True, label=f'Non-Attack (n={len(non_v):,})', color='#5B8DB8')
    ax.hist(att_v, bins=bins_hist, alpha=0.7, density=True, label=f'Attack (n={len(att_v):,})', color='#6BBF8A')
    if len(att_v) > 0:
        ax.axvline(att_v.mean(), color='#2ECC71', linewidth=2, linestyle='--',
                   label=f'Attack mean: {att_v.mean():.3f}')
    ax.axvline(non_v.mean(), color='#2980B9', linewidth=2, linestyle='--',
               label=f'Non-attack mean: {non_v.mean():.3f}')
    ax.set_xlabel('Mean Alpha Gap (VISION range)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Vision Gap: Attackers vs Others', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    
    # --- Bottom-right: Attackers vs Others — ADJACENT gap ---
    ax = axes[1, 1]
    att_a = df_with_adj[df_with_adj['action'] == 'Attack']['adj_alpha_gap'].dropna()
    non_a = df_with_adj[df_with_adj['action'] != 'Attack']['adj_alpha_gap'].dropna()
    
    ax.hist(non_a, bins=bins_hist, alpha=0.6, density=True, label=f'Non-Attack (n={len(non_a):,})', color='#5B8DB8')
    if len(att_a) > 0:
        ax.hist(att_a, bins=bins_hist, alpha=0.7, density=True, label=f'Attack (n={len(att_a):,})', color='#6BBF8A')
        ax.axvline(att_a.mean(), color='#2ECC71', linewidth=2, linestyle='--',
                   label=f'Attack mean: {att_a.mean():.3f}')
    ax.axvline(non_a.mean(), color='#2980B9', linewidth=2, linestyle='--',
               label=f'Non-attack mean: {non_a.mean():.3f}')
    ax.set_xlabel('Mean Alpha Gap (ADJACENT only, dist <= 1)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Adjacent Gap: Attackers vs Others', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    
    fig.suptitle('Vision-Range vs Adjacent Alpha Gap — Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_vision_vs_adjacent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    alpha_vision_vs_adjacent.png")
    
    # ============================================================
    # STATISTICS
    # ============================================================
    print(f"\n  {'='*55}")
    print(f"  STATISTICS")
    print(f"  {'='*55}")
    
    from scipy.stats import pearsonr, mannwhitneyu
    
    # --- VISION-RANGE alpha gap ---
    valid = df_with_neighbors[df_with_neighbors['mean_alpha_gap'].notna()]
    print(f"\n  === VISION-RANGE alpha gap (all agents within vision) ===")
    print(f"  Pearson correlations (n={len(valid):,}):")
    print(f"  {'Action':>12s}  {'r':>8s}  {'p-value':>10s}  {'Sig':>5s}")
    for action in actions:
        binary = (valid['action'] == action).astype(int)
        r, p = pearsonr(valid['mean_alpha_gap'], binary)
        sig = 'YES' if p < 0.05 else 'no'
        print(f"  {action:>12s}  {r:+.4f}  {p:10.6f}  {sig:>5s}")
    
    if len(attackers) >= 5 and len(non_attackers) >= 5:
        stat, p_mw = mannwhitneyu(
            attackers['mean_alpha_gap'].dropna(),
            non_attackers['mean_alpha_gap'].dropna(),
            alternative='two-sided')
        print(f"\n  Mann-Whitney U (vision gap: attackers vs others):")
        print(f"    Attacker mean:     {attackers['mean_alpha_gap'].mean():.4f}")
        print(f"    Non-attacker mean: {non_attackers['mean_alpha_gap'].mean():.4f}")
        print(f"    p-value: {p_mw:.6f}  {'SIGNIFICANT' if p_mw < 0.05 else 'not significant'}")
    
    # --- ADJACENT alpha gap ---
    valid_adj = df_with_adj[df_with_adj['adj_alpha_gap'].notna()]
    print(f"\n  === ADJACENT alpha gap (only agents at dist <= 1) ===")
    print(f"  Pearson correlations (n={len(valid_adj):,}):")
    print(f"  {'Action':>12s}  {'r':>8s}  {'p-value':>10s}  {'Sig':>5s}")
    for action in actions:
        binary = (valid_adj['action'] == action).astype(int)
        if valid_adj['adj_alpha_gap'].std() > 0:
            r, p = pearsonr(valid_adj['adj_alpha_gap'], binary)
            sig = 'YES' if p < 0.05 else 'no'
            print(f"  {action:>12s}  {r:+.4f}  {p:10.6f}  {sig:>5s}")
        else:
            print(f"  {action:>12s}  N/A (no variance)")
    
    if len(att_a) >= 5 and len(non_a) >= 5:
        stat_a, p_mw_a = mannwhitneyu(att_a, non_a, alternative='two-sided')
        print(f"\n  Mann-Whitney U (adjacent gap: attackers vs others):")
        print(f"    Attacker mean:     {att_a.mean():.4f}")
        print(f"    Non-attacker mean: {non_a.mean():.4f}")
        print(f"    p-value: {p_mw_a:.6f}  {'SIGNIFICANT' if p_mw_a < 0.05 else 'not significant'}")
    
    # --- OWN ALPHA ---
    print(f"\n  === OWN ALPHA ===")
    print(f"  Pearson correlations (n={len(df):,}):")
    print(f"  {'Action':>12s}  {'r(alpha)':>10s}  {'r(|alpha|)':>12s}")
    for action in actions:
        binary = (df['action'] == action).astype(int)
        r1, p1 = pearsonr(df['alpha'], binary)
        r2, p2 = pearsonr(df['abs_alpha'], binary)
        s1 = '*' if p1 < 0.05 else ' '
        s2 = '*' if p2 < 0.05 else ' '
        print(f"  {action:>12s}  {r1:+.4f}{s1}    {r2:+.4f}{s2}")
    print("  * = p < 0.05")
    
    # --- Reference values ---
    print(f"\n  === REFERENCE ===")
    print(f"  E[|a-b|] for Uniform(-1,1) = 2/3 = 0.667")
    print(f"  Vision gap — all decisions: {valid['mean_alpha_gap'].mean():.4f}")
    if len(att_v) > 0:
        print(f"  Vision gap — attackers:     {att_v.mean():.4f}")
    print(f"  Adjacent gap — all decisions: {valid_adj['adj_alpha_gap'].mean():.4f}")
    if len(att_a) > 0:
        print(f"  Adjacent gap — attackers:     {att_a.mean():.4f}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
