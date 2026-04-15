"""
Context Comparison — Combined actions per context variable
===========================================================
4 subplots (2x2): one per context variable (neighbors, adjacent, sugar, age).
Each subplot shows all 4 action rates on the same axes.
Grid sizes differentiated by line style, actions by color.
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
# CONFIG
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results2_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS (same as original)

def classify_action(dec_str):
    s = str(dec_str).strip()
    if s.startswith('1,'):   return 'Move'
    elif s.startswith('2') or s == '2-STAY': return 'Stay'
    elif s.startswith('3,'): return 'Reproduce'
    elif s.startswith('4,'): return 'Attack'
    return None

def _read_config(config_file):
    config = {}
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try: config[row[0].strip()] = float(row[1].strip())
                except ValueError: config[row[0].strip()] = row[1].strip()
    return config

def _process_run(folder, prefix=""):
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
    neighbor_map = {}
    adjacent_map = {}
    for step, group in full_moves.groupby('step'):
        positions = group[['agent_id', 'x', 'y']].drop_duplicates('agent_id').values
        n = len(positions)
        for i in range(n):
            aid, ax, ay = int(positions[i][0]), positions[i][1], positions[i][2]
            cv, ca = 0, 0
            for j in range(n):
                if i == j: continue
                _, bx, by = positions[j]
                dist = hypot(ax - bx, ay - by)
                if dist <= vision: cv += 1
                if dist <= 1.0001: ca += 1
            neighbor_map[(step, aid)] = cv
            adjacent_map[(step, aid)] = ca
    full_moves['action'] = full_moves['decision'].apply(classify_action)
    moves = full_moves[full_moves['action'].notna()].copy()
    if moves.empty: return None
    moves['n_neighbors'] = moves.apply(lambda r: neighbor_map.get((r['step'], r['agent_id']), 0), axis=1)
    moves['n_adjacent'] = moves.apply(lambda r: adjacent_map.get((r['step'], r['agent_id']), 0), axis=1)
    sugar_pc = stats[['step', 'total_map_sugar']].copy()
    sugar_pc['avg_cell_sugar'] = sugar_pc['total_map_sugar'] / total_cells
    moves = moves.merge(sugar_pc[['step', 'avg_cell_sugar']], on='step', how='left')
    moves['grid_size'] = grid_size
    moves['target_sugar'] = target_sugar
    moves['run_id'] = prefix.rstrip('_') if prefix else folder.name
    return moves

def load_and_enrich(batch_folder):
    all_data = []
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefixes.add(f.name.replace("config.csv", ""))
        for prefix in sorted(prefixes):
            result = _process_run(batch_folder, prefix=prefix)
            if result is not None: all_data.append(result)
    else:
        for i, folder in enumerate(folders):
            if (i+1) % 20 == 0 or i == 0:
                print(f"    Processing {i+1}/{len(folders)}: {folder.name}")
            result = _process_run(folder)
            if result is not None: all_data.append(result)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# ============================================================
# MAIN

def main():
    print("=" * 65)
    print("  CONTEXT COMPARISON — COMBINED ACTIONS")
    print("=" * 65)
    
    df = load_and_enrich(BATCH_FOLDER)
    if df.empty:
        print("  No data found!")
        return
    
    print(f"\n  Total LLM decisions: {len(df):,}")
    
    actions = ['Move', 'Stay', 'Attack', 'Reproduce']
    action_colors = {
        'Move': '#0072B2',
        'Stay': '#D55E00',
        'Attack': '#009E73',
        'Reproduce': '#CC79A7'
    }
    
    grids = sorted(df['grid_size'].unique())
    grid_styles = {g: s for g, s in zip(grids, ['-', '--', '-.', ':'][:len(grids)])}
    grid_widths = {g: w for g, w in zip(grids, [2.5, 2.0, 1.5, 1.0][:len(grids)])}
    
    # Bins
    df['n_neighbors_bin'] = df['n_neighbors'].clip(upper=8)
    df['n_adjacent_bin'] = df['n_adjacent'].clip(upper=4)
    age_bins = [-1, 5, 10, 20, 30, 50, 75, 100, 150, 500]
    age_labels = ['0-5', '6-10', '11-20', '21-30', '31-50', '51-75', '76-100', '101-150', '150+']
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    contexts = [
        ('n_neighbors_bin', 'Agents in Vision Range', list(range(9))),
        ('n_adjacent_bin', 'Adjacent Agents (dist <= 1)', list(range(5))),
        ('target_sugar', 'Avg Sugar per Cell', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        ('age_bin', 'Agent Age', age_labels),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()
    
    for idx, (ctx_col, xlabel, x_labels) in enumerate(contexts):
        ax = axes_flat[idx]
        
        for action in actions:
            for G in grids:
                sub = df[df['grid_size'] == G]
                grouped = sub.groupby(ctx_col)
                
                rates = []
                for val in x_labels:
                    if val in grouped.groups:
                        grp = grouped.get_group(val)
                        if len(grp) >= 5:
                            rates.append((grp['action'] == action).mean())
                        else:
                            rates.append(np.nan)
                    else:
                        rates.append(np.nan)
                
                ax.plot(range(len(x_labels)), rates,
                        color=action_colors[action],
                        linestyle=grid_styles[G],
                        linewidth=grid_widths[G],
                        alpha=0.85)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels([str(l) for l in x_labels], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Action Rate', fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, len(x_labels) - 0.7)
    
    # Build combined legend: action colors + grid styles
    from matplotlib.lines import Line2D
    legend_elements = []
    for action in actions:
        legend_elements.append(
            Line2D([0], [0], color=action_colors[action], linewidth=2.5,
                   label=action))
    legend_elements.append(Line2D([0], [0], color='none', label=''))  # spacer
    for G in grids:
        legend_elements.append(
            Line2D([0], [0], color='black', linestyle=grid_styles[G],
                   linewidth=grid_widths[G], label=f'Grid {G}x{G}'))
    
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(actions) + 1 + len(grids), fontsize=14,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "context_comparison_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    context_comparison_combined.png")
    print("  Done!")

if __name__ == "__main__":
    main()