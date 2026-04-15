"""
Agent Behavior Analysis — Decision Patterns by Configuration
=============================================================
Analyzes how agent LLM decisions vary with:
  - Grid size and sugar density (configuration)
  - Time (early vs late simulation)
  - Agent state (sugar owned, age)
  - Local context (neighbors in vision)

Reads agent_moves.csv + config.csv from each run folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import csv
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — CHANGE THESE PATHS
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

def classify_action(dec_str):
    """Classify decision string into action category."""
    s = str(dec_str).strip()
    if s.startswith('1,'):
        return 'Move'
    elif s.startswith('2') or s == '2-STAY':
        return 'Stay'
    elif s.startswith('3,'):
        return 'Reproduce'
    elif s.startswith('4,'):
        return 'Attack'
    elif s == 'TRAVELING':
        return 'Traveling'
    else:
        return 'Other'


def classify_action_core(dec_str):
    """Classify only LLM decisions (exclude Traveling/Other)."""
    s = str(dec_str).strip()
    if s.startswith('1,'):
        return 'Move'
    elif s.startswith('2') or s == '2-STAY':
        return 'Stay'
    elif s.startswith('3,'):
        return 'Reproduce'
    elif s.startswith('4,'):
        return 'Attack'
    else:
        return None  # skip non-LLM actions


def load_all_runs(batch_folder):
    """Load agent_moves + config from all run subfolders."""
    all_data = []
    
    # Pattern 1: subfolders
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    # Pattern 2: prefixed files in same folder
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefix = f.name.replace("config.csv", "")
            prefixes.add(prefix)
        
        for prefix in sorted(prefixes):
            config_file = batch_folder / f"{prefix}config.csv"
            moves_file = batch_folder / f"{prefix}agent_moves.csv"
            if config_file.exists() and moves_file.exists():
                config = _read_config(config_file)
                moves = pd.read_csv(moves_file, encoding='utf-8-sig')
                moves['grid_size'] = int(config.get('grid_size', 0))
                moves['avg_sugar'] = float(config.get('avg_sugar_target', 0))
                moves['vision'] = float(config.get('global_vision', 0))
                moves['metabolism'] = float(config.get('global_metabolism', 0))
                moves['run_id'] = prefix.rstrip('_')
                all_data.append(moves)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    for folder in folders:
        config_file = folder / "config.csv"
        moves_file = folder / "agent_moves.csv"
        
        if not config_file.exists() or not moves_file.exists():
            continue
        
        config = _read_config(config_file)
        moves = pd.read_csv(moves_file, encoding='utf-8-sig')
        moves['grid_size'] = int(config.get('grid_size', 0))
        moves['avg_sugar'] = float(config.get('avg_sugar_target', 0))
        moves['vision'] = float(config.get('global_vision', 0))
        moves['metabolism'] = float(config.get('global_metabolism', 0))
        moves['run_id'] = folder.name
        all_data.append(moves)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


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


# ============================================================
# 2. MAIN ANALYSIS
# ============================================================

def main():
    print("=" * 65)
    print("  AGENT BEHAVIOR ANALYSIS")
    print("=" * 65)
    
    # Load data
    df = load_all_runs(BATCH_FOLDER)
    if df.empty:
        print("No data found! Check BATCH_FOLDER path.")
        return
    
    print(f"  Loaded {len(df):,} action records from {df['run_id'].nunique()} runs")
    
    # Classify actions
    df['action'] = df['decision'].apply(classify_action)
    df['action_core'] = df['decision'].apply(classify_action_core)
    
    # Config label
    df['config'] = 'G=' + df['grid_size'].astype(str) + ', S=' + df['avg_sugar'].astype(str)
    
    # Time phase: early (0-50), mid (50-100), late (100+)
    df['phase'] = pd.cut(df['step'], bins=[-1, 50, 100, 300], labels=['Early\n(0-50)', 'Mid\n(51-100)', 'Late\n(100+)'])
    
    # Sugar brackets for the agent
    df['sugar_bracket'] = pd.cut(df['sugar'], bins=[-1, 5, 15, 50, 150, 10000],
                                  labels=['0-5', '5-15', '15-50', '50-150', '150+'])
    
    # Only LLM decisions (no Traveling/Other)
    df_llm = df[df['action_core'].notna()].copy()
    
    print(f"  LLM decisions: {len(df_llm):,} ({len(df_llm)/len(df)*100:.1f}% of all records)")
    print(f"  Configurations: {sorted(df['config'].unique())}")
    
    # ============================================================
    # PLOT 1: Action distribution by (grid, sugar) — LLM only
    # ============================================================
    print("\n  Generating plots...")
    
    grids = sorted(df['grid_size'].unique())
    sugars = sorted(df['avg_sugar'].unique())
    actions = ['Move', 'Stay', 'Attack', 'Reproduce']
    colors = {'Move': '#5B8DB8', 'Stay': '#A8A8A8', 'Attack': '#6BBF8A', 'Reproduce': '#F0A050'}
    
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 6), sharey=True)
    if len(grids) == 1:
        axes = [axes]
    
    for ax, G in zip(axes, grids):
        subset = df_llm[df_llm['grid_size'] == G]
        
        # Per sugar level, compute action proportions
        props = subset.groupby(['avg_sugar', 'action_core']).size().unstack(fill_value=0)
        props = props.div(props.sum(axis=1), axis=0) * 100
        
        # Stacked bar
        bottom = np.zeros(len(props))
        x_pos = range(len(props))
        
        for action in actions:
            if action in props.columns:
                vals = props[action].values
            else:
                vals = np.zeros(len(props))
            ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.7)
            
            # Label percentages > 5%
            for j, v in enumerate(vals):
                if v > 5:
                    ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=8)
            bottom += vals
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'S={s}' for s in props.index], fontsize=10)
        ax.set_title(f'Grid {G}x{G}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Average Sugar per Cell', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Action Frequency (%)', fontsize=11)
    
    axes[-1].legend(loc='upper right', fontsize=10)
    fig.suptitle('LLM Decision Distribution by Configuration', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_actions_by_config.png", dpi=150, bbox_inches='tight')
    print("    behavior_actions_by_config.png")
    
    # ============================================================
    # PLOT 2: Action distribution over TIME (by phase)
    # ============================================================
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 6), sharey=True)
    if len(grids) == 1:
        axes = [axes]
    
    for ax, G in zip(axes, grids):
        subset = df_llm[df_llm['grid_size'] == G]
        
        phases = ['Early\n(0-50)', 'Mid\n(51-100)', 'Late\n(100+)']
        props = subset.groupby(['phase', 'action_core']).size().unstack(fill_value=0)
        props = props.div(props.sum(axis=1), axis=0) * 100
        
        phases_present = [p for p in phases if p in props.index]
        props_filtered = props.loc[phases_present]
        
        bottom = np.zeros(len(props_filtered))
        x_pos = range(len(props_filtered))
        
        for action in actions:
            if action in props_filtered.columns:
                vals = props_filtered[action].values
            else:
                vals = np.zeros(len(props_filtered))
            if len(vals) == len(props_filtered):
                ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.6)
                for j, v in enumerate(vals):
                    if v > 5:
                        ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=9)
                bottom += vals
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phases_present, fontsize=10)
        ax.set_title(f'Grid {G}x{G}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Simulation Phase', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Action Frequency (%)', fontsize=11)
    
    axes[-1].legend(loc='upper right', fontsize=10)
    fig.suptitle('How Agent Decisions Change Over Time', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_actions_over_time.png", dpi=150, bbox_inches='tight')
    print("    behavior_actions_over_time.png")
    
    # ============================================================
    # PLOT 3: Action distribution by AGENT SUGAR LEVEL
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    props = df_llm.groupby(['sugar_bracket', 'action_core']).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(props))
    x_pos = range(len(props))
    
    for action in actions:
        if action in props.columns:
            vals = props[action].values
        else:
            vals = np.zeros(len(props))
        ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.6)
        for j, v in enumerate(vals):
            if v > 3:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=9)
        bottom += vals
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(props.index, fontsize=11)
    ax.set_xlabel('Agent Sugar Owned', fontsize=13)
    ax.set_ylabel('Action Frequency (%)', fontsize=13)
    ax.set_title('How Sugar Wealth Affects Agent Decisions', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_actions_by_sugar.png", dpi=150, bbox_inches='tight')
    print("    behavior_actions_by_sugar.png")
    
    # ============================================================
    # PLOT 4: Attack rate vs sugar owned (scatter/line)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 4a: Attack rate by agent sugar bracket
    ax = axes[0]
    attack_rate = df_llm.groupby('sugar_bracket').apply(
        lambda g: (g['action_core'] == 'Attack').mean() * 100
    )
    ax.bar(range(len(attack_rate)), attack_rate.values, color='#6BBF8A', width=0.6)
    ax.set_xticks(range(len(attack_rate)))
    ax.set_xticklabels(attack_rate.index, fontsize=11)
    ax.set_xlabel('Agent Sugar Owned', fontsize=12)
    ax.set_ylabel('Attack Rate (%)', fontsize=12)
    ax.set_title('Attack Probability by Wealth', fontsize=13, fontweight='bold')
    for i, v in enumerate(attack_rate.values):
        ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # 4b: Reproduce rate by agent sugar bracket
    ax = axes[1]
    repr_rate = df_llm.groupby('sugar_bracket').apply(
        lambda g: (g['action_core'] == 'Reproduce').mean() * 100
    )
    ax.bar(range(len(repr_rate)), repr_rate.values, color='#F0A050', width=0.6)
    ax.set_xticks(range(len(repr_rate)))
    ax.set_xticklabels(repr_rate.index, fontsize=11)
    ax.set_xlabel('Agent Sugar Owned', fontsize=12)
    ax.set_ylabel('Reproduce Rate (%)', fontsize=12)
    ax.set_title('Reproduce Probability by Wealth', fontsize=13, fontweight='bold')
    for i, v in enumerate(repr_rate.values):
        ax.text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_attack_reproduce_by_sugar.png", dpi=150, bbox_inches='tight')
    print("    behavior_attack_reproduce_by_sugar.png")
    
    # ============================================================
    # PLOT 5: Heatmaps — Attack rate & Reproduce rate by (grid, sugar)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    
    for ax, action_name, cmap, title in [
        (axes[0], 'Attack', 'Reds', 'Attack Rate (%) by Configuration'),
        (axes[1], 'Reproduce', 'Oranges', 'Reproduce Rate (%) by Configuration'),
    ]:
        pivot = df_llm.groupby(['grid_size', 'avg_sugar']).apply(
            lambda g: (g['action_core'] == action_name).mean() * 100
        ).unstack(fill_value=0)
        
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', origin='upper')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{c:.1f}' for c in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(g) for g in pivot.index], fontsize=10)
        ax.set_xlabel('Average Sugar per Cell', fontsize=12)
        ax.set_ylabel('Grid Size', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = 'white' if val > np.nanmax(pivot.values) * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=11,
                        fontweight='bold', color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_heatmaps.png", dpi=150, bbox_inches='tight')
    print("    behavior_heatmaps.png")
    
    # ============================================================
    # PLOT 6: Move rate over time (line plot per config)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax_idx, (ax, action_name) in enumerate(zip(axes.flat, actions)):
        for G in grids:
            subset = df_llm[df_llm['grid_size'] == G]
            # Group by step, compute rate, smooth with rolling window
            rate_by_step = subset.groupby('step').apply(
                lambda g: (g['action_core'] == action_name).mean() * 100
            )
            # Smooth
            smoothed = rate_by_step.rolling(window=10, min_periods=1, center=True).mean()
            ax.plot(smoothed.index, smoothed.values, label=f'Grid {G}x{G}', linewidth=1.5)
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(f'{action_name} Rate (%)', fontsize=11)
        ax.set_title(f'{action_name} Rate Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_actions_over_steps.png", dpi=150, bbox_inches='tight')
    print("    behavior_actions_over_steps.png")
    
    # ============================================================
    # PLOT 7: Decision by age of agent
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_llm['age_bracket'] = pd.cut(df_llm['age'], bins=[-1, 10, 30, 60, 100, 500],
                                    labels=['0-10', '11-30', '31-60', '61-100', '100+'])
    
    props = df_llm.groupby(['age_bracket', 'action_core']).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(props))
    x_pos = range(len(props))
    
    for action in actions:
        if action in props.columns:
            vals = props[action].values
        else:
            vals = np.zeros(len(props))
        ax.bar(x_pos, vals, bottom=bottom, label=action, color=colors[action], width=0.6)
        for j, v in enumerate(vals):
            if v > 3:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=10)
        bottom += vals
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(props.index, fontsize=11)
    ax.set_xlabel('Agent Age (steps survived)', fontsize=13)
    ax.set_ylabel('Action Frequency (%)', fontsize=13)
    ax.set_title('How Agent Age Affects Decision Making', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Add count per bracket
    counts = df_llm['age_bracket'].value_counts().sort_index()
    for i, (bracket, count) in enumerate(counts.items()):
        ax.text(i, 102, f'n={count:,}', ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_actions_by_age.png", dpi=150, bbox_inches='tight')
    print("    behavior_actions_by_age.png")
    
    # ============================================================
    # NUMERICAL SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 65)
    print("  NUMERICAL SUMMARY")
    print("=" * 65)
    
    summary_rows = []
    for (G, S), group in df_llm.groupby(['grid_size', 'avg_sugar']):
        n = len(group)
        row = {
            'Grid': int(G),
            'Sugar': S,
            'N_decisions': n,
            'Move%': round((group['action_core'] == 'Move').mean() * 100, 1),
            'Stay%': round((group['action_core'] == 'Stay').mean() * 100, 1),
            'Attack%': round((group['action_core'] == 'Attack').mean() * 100, 1),
            'Reproduce%': round((group['action_core'] == 'Reproduce').mean() * 100, 1),
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{summary_df.to_string(index=False)}")
    
    # Save summary
    summary_df.to_csv(OUTPUT_DIR / "behavior_summary.csv", index=False)
    print(f"\n  Saved behavior_summary.csv")
    
    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    print("\n" + "=" * 65)
    print("  CORRELATION: Config → Action Rates")
    print("=" * 65)
    
    # Per-run action rates
    run_stats = []
    for run_id, group in df_llm.groupby('run_id'):
        n = len(group)
        if n < 10:
            continue
        row = {
            'grid_size': group['grid_size'].iloc[0],
            'avg_sugar': group['avg_sugar'].iloc[0],
            'vision': group['vision'].iloc[0],
            'metabolism': group['metabolism'].iloc[0],
            'move_rate': (group['action_core'] == 'Move').mean(),
            'stay_rate': (group['action_core'] == 'Stay').mean(),
            'attack_rate': (group['action_core'] == 'Attack').mean(),
            'reproduce_rate': (group['action_core'] == 'Reproduce').mean(),
        }
        run_stats.append(row)
    
    run_df = pd.DataFrame(run_stats)
    
    if len(run_df) >= 5:
        from scipy.stats import pearsonr
        
        predictors = ['grid_size', 'avg_sugar', 'vision', 'metabolism']
        outcomes = ['move_rate', 'stay_rate', 'attack_rate', 'reproduce_rate']
        
        print(f"\n  Pearson correlations (n={len(run_df)} runs):\n")
        print(f"  {'':15s}", end="")
        for o in outcomes:
            print(f"  {o:>14s}", end="")
        print()
        
        for p in predictors:
            print(f"  {p:15s}", end="")
            for o in outcomes:
                r, pval = pearsonr(run_df[p], run_df[o])
                sig = '*' if pval < 0.05 else ' '
                print(f"  {r:+.3f}{sig}        ", end="")
            print()
        
        print("\n  * = p < 0.05")
        
        # Save
        run_df.to_csv(OUTPUT_DIR / "behavior_per_run.csv", index=False)
        print(f"  Saved behavior_per_run.csv")
    
    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
