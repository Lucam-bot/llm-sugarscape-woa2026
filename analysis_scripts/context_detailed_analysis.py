"""
Action Probability by Context — 3 Plots
=========================================
1. P(action) vs number of nearby agents
2. P(action) vs average map sugar per cell
3. P(action) vs agent age

Reads agent_moves.csv + config.csv + step_stats.csv from each run folder.
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
OUTPUT_DIR = Path(r"./results\Analysis_Results2_3")
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
    else: return None  # skip Traveling/Other


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


def count_neighbors(moves_step, vision):
    """For each agent in a step, count how many other agents are within vision range."""
    positions = moves_step[['agent_id', 'x', 'y']].values  # id, x, y
    n = len(positions)
    neighbor_counts = {}
    
    for i in range(n):
        aid, ax, ay = positions[i]
        count = 0
        for j in range(n):
            if i == j:
                continue
            _, bx, by = positions[j]
            if hypot(ax - bx, ay - by) <= vision:
                count += 1
        neighbor_counts[int(aid)] = count
    
    return neighbor_counts


def load_and_enrich(batch_folder):
    """Load all runs, compute neighbors count and avg map sugar per cell."""
    all_data = []
    
    # Find run folders
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    # Fallback: prefixed files
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
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def _process_run(folder, prefix=""):
    """Process a single run: load moves, count neighbors, merge map sugar."""
    config_file = folder / f"{prefix}config.csv"
    moves_file = folder / f"{prefix}agent_moves.csv"
    stats_file = folder / f"{prefix}step_stats.csv"
    
    if not all(f.exists() for f in [config_file, moves_file, stats_file]):
        return None
    
    config = _read_config(config_file)
    grid_size = int(config.get('grid_size', 20))
    target_sugar = config.get('avg_sugar_target', 1.0)
    vision = int(config.get('global_vision', 4))
    total_cells = grid_size * grid_size
    
    full_moves = pd.read_csv(moves_file, encoding='utf-8-sig')
    stats = pd.read_csv(stats_file, encoding='utf-8-sig')
    
    # --- Count neighbors per agent per step (vision + adjacent) ---
    neighbor_map = {}
    adjacent_map = {}
    for step, group in full_moves.groupby('step'):
        positions = group[['agent_id', 'x', 'y']].drop_duplicates('agent_id').values
        n = len(positions)
        for i in range(n):
            aid, ax, ay = int(positions[i][0]), positions[i][1], positions[i][2]
            count_vision = 0
            count_adj = 0
            for j in range(n):
                if i == j:
                    continue
                _, bx, by = positions[j]
                dist = hypot(ax - bx, ay - by)
                if dist <= vision:
                    count_vision += 1
                if dist <= 1.0001:
                    count_adj += 1
            neighbor_map[(step, aid)] = count_vision
            adjacent_map[(step, aid)] = count_adj
    
    # Now filter to LLM decisions only
    full_moves['action'] = full_moves['decision'].apply(classify_action)
    moves = full_moves[full_moves['action'].notna()].copy()
    
    if moves.empty:
        return None
    
    moves['n_neighbors'] = moves.apply(
        lambda r: neighbor_map.get((r['step'], r['agent_id']), 0), axis=1
    )
    moves['n_adjacent'] = moves.apply(
        lambda r: adjacent_map.get((r['step'], r['agent_id']), 0), axis=1
    )
    
    # --- Average map sugar per cell (from step_stats) ---
    sugar_per_cell = stats[['step', 'total_map_sugar']].copy()
    sugar_per_cell['avg_cell_sugar'] = sugar_per_cell['total_map_sugar'] / total_cells
    moves = moves.merge(sugar_per_cell[['step', 'avg_cell_sugar']], on='step', how='left')
    
    moves['grid_size'] = grid_size
    moves['target_sugar'] = target_sugar
    moves['run_id'] = prefix.rstrip('_') if prefix else folder.name
    
    return moves


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_stacked_bar(ax, data, group_col, actions, colors, xlabel, min_n=10):
    """Stacked bar of action distribution grouped by group_col."""
    grouped = data.groupby([group_col, 'action']).size().unstack(fill_value=0)
    grouped = grouped[grouped.sum(axis=1) >= min_n]
    if grouped.empty:
        ax.text(0.5, 0.5, 'Too few\nsamples', ha='center', va='center', transform=ax.transAxes, color='gray')
        return
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(grouped_pct))
    x_pos = range(len(grouped_pct))
    
    for action in actions:
        vals = grouped_pct[action].values if action in grouped_pct.columns else np.zeros(len(grouped_pct))
        ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.7)
        for j, v in enumerate(vals):
            if v > 7:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=7)
        bottom += vals
    
    ax.set_xticks(x_pos)
    labels = [str(x) for x in grouped_pct.index]
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Action Frequency (%)', fontsize=10)
    
    # Sample counts on top
    for j, idx in enumerate(grouped_pct.index):
        n = int(grouped.loc[idx].sum())
        ax.text(j, 102, f'n={n:,}', ha='center', fontsize=6, color='gray')


def plot_line_rates(ax, data, group_col, actions, colors, xlabel, min_n=10):
    """Line plot of action rates by group_col."""
    grouped = data.groupby(group_col)
    valid_groups = [g for g, d in grouped if len(d) >= min_n]
    
    for action in actions:
        rates = []
        for g in valid_groups:
            d = grouped.get_group(g)
            rates.append((d['action'] == action).mean() * 100)
        ax.plot(range(len(valid_groups)), rates, 'o-', label=action, color=colors[action],
                linewidth=2, markersize=6)
    
    ax.set_xticks(range(len(valid_groups)))
    ax.set_xticklabels([str(g) for g in valid_groups], fontsize=8, rotation=30)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Probability (%)', fontsize=10)
    ax.set_ylim(bottom=0)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  ACTION PROBABILITY BY CONTEXT")
    print("=" * 65)
    
    df = load_and_enrich(BATCH_FOLDER)
    if df.empty:
        print("  No data found! Check BATCH_FOLDER.")
        return
    
    print(f"\n  Total LLM decisions: {len(df):,} from {df['run_id'].nunique()} runs")
    
    actions = ['Move', 'Stay', 'Attack', 'Reproduce']
    colors = {'Move': '#5B8DB8', 'Stay': '#A8A8A8', 'Attack': '#6BBF8A', 'Reproduce': '#F0A050'}
    grids = sorted(df['grid_size'].unique())
    n_grids = len(grids)
    
    # --- Prepare bins ---
    max_neighbors = 8
    df['n_neighbors_bin'] = df['n_neighbors'].clip(upper=max_neighbors)
    
    #df['sugar_density_bin'] = pd.cut(df['avg_cell_sugar'], bins=8)
    
    age_bins = [-1, 5, 10, 20, 30, 50, 75, 100, 150, 500]
    age_labels = ['0-5', '6-10', '11-20', '21-30', '31-50', '51-75', '76-100', '101-150', '150+']
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    max_adjacent = 4
    df['n_adjacent_bin'] = df['n_adjacent'].clip(upper=max_adjacent)
    
    # ===========================================================================
    # A) AGGREGATED PLOTS (all configs together)
    # ===========================================================================
    print("\n  --- AGGREGATED PLOTS (all configs) ---")
    
    # --- A1: Neighbors (aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_stacked_bar(axes[0], df, 'n_neighbors_bin', actions, colors, 'Agents in Vision Range')
    axes[0].set_title('Action Distribution', fontsize=12, fontweight='bold')
    plot_line_rates(axes[1], df, 'n_neighbors_bin', actions, colors, 'Agents in Vision Range')
    axes[1].set_title('Action Rates', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    fig.suptitle('How Nearby Agents Affect Decisions (all configs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_neighbors_aggregated.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_neighbors_aggregated.png")
    
    # --- A2: Sugar density (aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_stacked_bar(axes[0], df, 'target_sugar', actions, colors, 'Avg Sugar per Cell (map-wide)')
    axes[0].set_title('Action Distribution', fontsize=12, fontweight='bold')
    plot_line_rates(axes[1], df, 'target_sugar', actions, colors, 'Avg Sugar per Cell (map-wide)')
    axes[1].set_title('Action Rates', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    fig.suptitle('How Map Sugar Density Affects Decisions (all configs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_sugar_density_aggregated.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_sugar_density_aggregated.png")
    
    # --- A3: Age (aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_stacked_bar(axes[0], df, 'age_bin', actions, colors, 'Agent Age (steps survived)')
    axes[0].set_title('Action Distribution', fontsize=12, fontweight='bold')
    plot_line_rates(axes[1], df, 'age_bin', actions, colors, 'Agent Age (steps survived)')
    axes[1].set_title('Action Rates', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    fig.suptitle('How Agent Age Affects Decisions (all configs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_age_aggregated.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_age_aggregated.png")
    
    # --- A4: Adjacent neighbors (aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_stacked_bar(axes[0], df, 'n_adjacent_bin', actions, colors, 'Adjacent Agents (dist <= 1)')
    axes[0].set_title('Action Distribution', fontsize=12, fontweight='bold')
    plot_line_rates(axes[1], df, 'n_adjacent_bin', actions, colors, 'Adjacent Agents (dist <= 1)')
    axes[1].set_title('Action Rates', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    fig.suptitle('How Adjacent Agents Affect Decisions (all configs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_adjacent_aggregated.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_adjacent_aggregated.png")
    
    # ===========================================================================
    # B) PER-GRID PLOTS — one column per grid size
    # ===========================================================================
    print("\n  --- PER-GRID PLOTS ---")
    
    # --- B1: Neighbors by grid ---
    fig, axes = plt.subplots(2, n_grids, figsize=(6 * n_grids, 11), squeeze=False)
    for col, G in enumerate(grids):
        sub = df[df['grid_size'] == G]
        plot_stacked_bar(axes[0, col], sub, 'n_neighbors_bin', actions, colors, 'Agents in Vision Range')
        axes[0, col].set_title(f'Grid {G}x{G} — Distribution', fontsize=11, fontweight='bold')
        plot_line_rates(axes[1, col], sub, 'n_neighbors_bin', actions, colors, 'Agents in Vision Range')
        axes[1, col].set_title(f'Grid {G}x{G} — Rates', fontsize=11, fontweight='bold')
        if col == n_grids - 1:
            axes[0, col].legend(fontsize=8, loc='upper right')
            axes[1, col].legend(fontsize=8, loc='upper right')
    fig.suptitle('How Nearby Agents Affect Decisions — by Grid Size', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_neighbors_by_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_neighbors_by_grid.png")
    
    # --- B2: Sugar density by grid ---
    fig, axes = plt.subplots(2, n_grids, figsize=(6 * n_grids, 11), squeeze=False)
    for col, G in enumerate(grids):
        sub = df[df['grid_size'] == G]
        # Rebin per grid (different sugar ranges per grid)
        sub = sub.copy()
        sub['sugar_density_bin_local'] = pd.cut(sub['avg_cell_sugar'], bins=6)
        plot_stacked_bar(axes[0, col], sub, 'sugar_density_bin_local', actions, colors, 'Avg Sugar per Cell')
        axes[0, col].set_title(f'Grid {G}x{G} — Distribution', fontsize=11, fontweight='bold')
        plot_line_rates(axes[1, col], sub, 'sugar_density_bin_local', actions, colors, 'Avg Sugar per Cell')
        axes[1, col].set_title(f'Grid {G}x{G} — Rates', fontsize=11, fontweight='bold')
        if col == n_grids - 1:
            axes[0, col].legend(fontsize=8, loc='upper right')
            axes[1, col].legend(fontsize=8, loc='upper right')
    fig.suptitle('How Map Sugar Density Affects Decisions — by Grid Size', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_sugar_density_by_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_sugar_density_by_grid.png")
    
    # --- B3: Age by grid ---
    fig, axes = plt.subplots(2, n_grids, figsize=(6 * n_grids, 11), squeeze=False)
    for col, G in enumerate(grids):
        sub = df[df['grid_size'] == G]
        plot_stacked_bar(axes[0, col], sub, 'age_bin', actions, colors, 'Agent Age')
        axes[0, col].set_title(f'Grid {G}x{G} — Distribution', fontsize=11, fontweight='bold')
        plot_line_rates(axes[1, col], sub, 'age_bin', actions, colors, 'Agent Age')
        axes[1, col].set_title(f'Grid {G}x{G} — Rates', fontsize=11, fontweight='bold')
        if col == n_grids - 1:
            axes[0, col].legend(fontsize=8, loc='upper right')
            axes[1, col].legend(fontsize=8, loc='upper right')
    fig.suptitle('How Agent Age Affects Decisions — by Grid Size', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_age_by_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_age_by_grid.png")
    
    # --- B4: Adjacent neighbors by grid ---
    fig, axes = plt.subplots(2, n_grids, figsize=(6 * n_grids, 11), squeeze=False)
    for col, G in enumerate(grids):
        sub = df[df['grid_size'] == G]
        plot_stacked_bar(axes[0, col], sub, 'n_adjacent_bin', actions, colors, 'Adjacent Agents (dist <= 1)')
        axes[0, col].set_title(f'Grid {G}x{G} — Distribution', fontsize=11, fontweight='bold')
        plot_line_rates(axes[1, col], sub, 'n_adjacent_bin', actions, colors, 'Adjacent Agents (dist <= 1)')
        axes[1, col].set_title(f'Grid {G}x{G} — Rates', fontsize=11, fontweight='bold')
        if col == n_grids - 1:
            axes[0, col].legend(fontsize=8, loc='upper right')
            axes[1, col].legend(fontsize=8, loc='upper right')
    fig.suptitle('How Adjacent Agents Affect Decisions — by Grid Size', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_adjacent_by_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_adjacent_by_grid.png")
    
    # ===========================================================================
    # C) COMPARISON PLOT — overlay grids on same axes to see differences
    # ===========================================================================
    print("\n  --- COMPARISON ACROSS GRIDS ---")
    
    grid_colors = {g: c for g, c in zip(grids, ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'][:n_grids])}
    grid_styles = {g: s for g, s in zip(grids, ['-', '--', '-.', ':'][:n_grids])}
    
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    
    contexts = [
        ('n_neighbors_bin', 'Agents in Vision Range', 'Vision\nNeighbors'),
        ('n_adjacent_bin', 'Adjacent Agents (dist <= 1)', 'Adjacent\nNeighbors'),
        ('target_sugar', 'Target Sugar Density', 'Sugar\nDensity'),
        ('age_bin', 'Agent Age', 'Age'),
    ]
    
    # Valori fissi che desideri su ogni asse X
    fixed_x_axes = {
        'target_sugar': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'n_neighbors_bin': list(range(9)), # 0 a 8
        'n_adjacent_bin': list(range(5)),  # 0 a 4
        'age_bin': ['0-5', '6-10', '11-20', '21-30', '31-50', '51-75', '76-100', '101-150', '150+']
    }

    for row, (ctx_col, xlabel, ctx_label) in enumerate(contexts):
        x_labels = fixed_x_axes[ctx_col]
        for col, action in enumerate(actions):
            ax = axes[row, col]
            
            for G in grids:
                sub = df[df['grid_size'] == G]
                grouped = sub.groupby(ctx_col)
                
                valid_groups = [g for g, d in grouped if len(d) >= 10]
                rates = []
                for val in x_labels:
                    if val in grouped.groups:
                        group_data = grouped.get_group(val)
                        # Qui puoi ancora tenere un controllo minimo (es. se n > 0)
                        if len(group_data) >= 10: 
                            rates.append((group_data['action'] == action).mean())
                        else:
                            # Se sono meno di 10, non disegnare il punto (evita picchi falsi)
                            rates.append(np.nan)
                    else:
                        rates.append(np.nan) # nan se il valore non esiste proprio (es. 50x50)
                
                # DISEGNA: l'asse X ora è sempre lungo quanto x_labels
                ax.plot(range(len(x_labels)), rates, 'o-', 
                    color=grid_colors[G], linestyle=grid_styles[G],
                    label=f'Grid {G}x{G}', linewidth=2, markersize=5)
                
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels([str(l) for l in x_labels], fontsize=9, rotation=45)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_xlim(-0.5, len(x_labels) - 0.5) # Blocca l'ampiezza del grafico
            ax.set_ylabel(f'{action} Rate', fontsize=9)
            ax.set_ylim(bottom=0)
            
            if row == 0:
                ax.set_title(action, fontsize=13, fontweight='bold')
            if col == 3:
                ax.legend(fontsize=8, loc='best')
    
    # Row labels on left
    for row, (_, _, ctx_label) in enumerate(contexts):
        axes[row, 0].annotate(ctx_label, xy=(-0.35, 0.5), xycoords='axes fraction',
                              fontsize=13, fontweight='bold', rotation=90,
                              ha='center', va='center')
    
    fig.suptitle('Action Rates Compared Across Grid Sizes\n(rows = context variable, columns = action type)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_comparison_grids.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_comparison_grids.png")
    
    # ============================================================
    # SUMMARY STATS & CORRELATIONS
    # ============================================================
    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")
    print(f"  Total decisions analyzed: {len(df):,}")
    print(f"  Grids: {grids}")
    print(f"  Neighbor range (vision): {df['n_neighbors'].min()} - {df['n_neighbors'].max()}")
    print(f"  Neighbor range (adjacent): {df['n_adjacent'].min()} - {df['n_adjacent'].max()}")
    print(f"  Sugar density range: {df['avg_cell_sugar'].min():.3f} - {df['avg_cell_sugar'].max():.3f}")
    print(f"  Age range: {df['age'].min()} - {df['age'].max()}")
    
    from scipy.stats import pearsonr
    
    # Global correlations
    print(f"\n  Pearson correlations — ALL CONFIGS:")
    print(f"  {'Context':>20s}  {'Move':>8s}  {'Stay':>8s}  {'Attack':>8s}  {'Reprod':>8s}")
    for ctx in ['n_neighbors', 'n_adjacent', 'avg_cell_sugar', 'age']:
        print(f"  {ctx:>20s}", end="")
        for action in actions:
            binary = (df['action'] == action).astype(int)
            r, p = pearsonr(df[ctx], binary)
            sig = '*' if p < 0.05 else ' '
            print(f"  {r:+.4f}{sig}", end="")
        print()
    print("  * = p < 0.05")
    
    # Per-grid correlations
    for G in grids:
        sub = df[df['grid_size'] == G]
        print(f"\n  Pearson correlations — Grid {G}x{G} (n={len(sub):,}):")
        print(f"  {'Context':>20s}  {'Move':>8s}  {'Stay':>8s}  {'Attack':>8s}  {'Reprod':>8s}")
        for ctx in ['n_neighbors', 'n_adjacent', 'avg_cell_sugar', 'age']:
            print(f"  {ctx:>20s}", end="")
            for action in actions:
                binary = (sub['action'] == action).astype(int)
                if sub[ctx].std() > 0:
                    r, p = pearsonr(sub[ctx], binary)
                    sig = '*' if p < 0.05 else ' '
                    print(f"  {r:+.4f}{sig}", end="")
                else:
                    print(f"  {'N/A':>7s}", end="")
            print()
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()