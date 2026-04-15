"""
Combat Intelligence Analysis
==============================
Analyzes whether LLM agents estimate target strength before attacking.
Since agents cannot see target sugar, this reveals implicit decision-making.

Produces:
  1. Attacker vs Target pre-combat sugar (scatter, colored by outcome)
  2. Win rate by sugar ratio (attacker/target)
  3. Sugar ratio distribution: attackers vs all adjacent opportunities
  4. Attack decision analysis: sugar of agents who attack vs who don't (given opportunity)
  5. Combined: sugar ratio x alpha gap → win rate heatmap

Pre-combat sugar is estimated from the previous step's logged sugar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def classify_outcome(row):
    dec = str(row['decision'])
    if 'WIN' in dec:
        return 'WIN'
    elif 'LOSE' in dec:
        return 'LOSE'
    elif dec.startswith('4,') and row['alive'] == False:
        return 'LOSE'
    elif dec.startswith('4,') and row['alive'] == True:
        return 'WIN'
    elif 'FAIL' in dec:
        return 'FAIL'
    return 'FAIL'


def load_combat_data(batch_folder):
    """Load attacks with pre-combat sugar for both attacker and target."""
    all_combats = []
    
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_agent_moves.csv"):
            prefixes.add(f.name.replace("agent_moves.csv", ""))
        sources = [(batch_folder, p) for p in sorted(prefixes)]
    else:
        sources = [(f, "") for f in folders]
    
    for i, (folder, prefix) in enumerate(sources):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Processing {i+1}/{len(sources)}")
        
        moves_file = folder / f"{prefix}agent_moves.csv"
        if not moves_file.exists():
            continue
        
        df = pd.read_csv(moves_file, encoding='utf-8-sig')
        result = _process_combats(df)
        if result is not None:
            result['run_id'] = prefix.rstrip('_') if prefix else folder.name
            all_combats.append(result)
    
    if all_combats:
        return pd.concat(all_combats, ignore_index=True)
    return pd.DataFrame()


def _process_combats(df):
    """Extract combats with pre-combat sugar from previous step."""
    attacks = df[df['decision'].astype(str).str.startswith('4,') & df['target_alpha'].notna()].copy()
    
    if attacks.empty:
        return None
    
    attacks['outcome'] = attacks.apply(classify_outcome, axis=1)
    attacks = attacks[attacks['outcome'] != 'FAIL'].copy()
    
    if attacks.empty:
        return None
    
    # Build lookup: (step, agent_id) -> sugar
    sugar_lookup = df.set_index(['step', 'agent_id'])['sugar'].to_dict()
    
    # Build lookup: (step, alpha_rounded) -> (agent_id, sugar)
    # Used to find target by alpha
    alpha_lookup = {}
    for _, row in df.iterrows():
        key = (row['step'], round(row['alpha'], 3))
        alpha_lookup[key] = (row['agent_id'], row['sugar'])
    
    records = []
    for _, atk in attacks.iterrows():
        step = atk['step']
        aid = atk['agent_id']
        t_alpha = round(atk['target_alpha'], 3)
        
        # Pre-combat sugar: use previous step
        att_pre = sugar_lookup.get((step - 1, aid))
        
        # Find target agent by alpha at previous step
        tgt_info = alpha_lookup.get((step - 1, t_alpha))
        tgt_pre = tgt_info[1] if tgt_info else None
        
        # If step 0, try same step (less reliable but better than nothing)
        if att_pre is None:
            att_pre = sugar_lookup.get((step, aid))
        if tgt_pre is None:
            tgt_info_same = alpha_lookup.get((step, t_alpha))
            tgt_pre = tgt_info_same[1] if tgt_info_same else None
        
        if att_pre is not None and tgt_pre is not None and tgt_pre > 0:
            records.append({
                'step': step,
                'attacker_id': aid,
                'attacker_alpha': atk['alpha'],
                'target_alpha': atk['target_alpha'],
                'attacker_sugar': att_pre,
                'target_sugar': tgt_pre,
                'sugar_ratio': att_pre / tgt_pre,
                'alpha_gap': abs(atk['alpha'] - atk['target_alpha']),
                'outcome': atk['outcome'],
                'attacker_stronger': att_pre > tgt_pre,
            })
    
    if records:
        return pd.DataFrame(records)
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  COMBAT INTELLIGENCE ANALYSIS")
    print("=" * 60)
    
    df = load_combat_data(BATCH_FOLDER)
    if df.empty:
        print("  No combat data found!")
        return
    
    print(f"\n  Combats with pre-combat sugar: {len(df):,}")
    print(f"  WIN: {(df['outcome']=='WIN').sum():,}  LOSE: {(df['outcome']=='LOSE').sum():,}")
    
    # ============================================================
    # PLOT 1: Attacker vs Target sugar (scatter)
    # ============================================================
    print("\n  Generating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    wins = df[df['outcome'] == 'WIN']
    losses = df[df['outcome'] == 'LOSE']
    
    ax.scatter(losses['attacker_sugar'], losses['target_sugar'],
               c='#E24B4A', alpha=0.4, s=15, label=f'Lose (n={len(losses):,})', edgecolors='none')
    ax.scatter(wins['attacker_sugar'], wins['target_sugar'],
               c='#2ECC71', alpha=0.4, s=15, label=f'Win (n={len(wins):,})', edgecolors='none')
    
    max_sugar = max(df['attacker_sugar'].quantile(0.98), df['target_sugar'].quantile(0.98))
    ax.plot([0, max_sugar], [0, max_sugar], 'k--', linewidth=1, alpha=0.4, label='Equal sugar')
    
    ax.set_xlabel('Attacker pre-combat sugar', fontsize=12)
    ax.set_ylabel('Target pre-combat sugar', fontsize=12)
    ax.set_title('Pre-combat sugar: attacker vs target', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    
    # Right: histogram of sugar ratio
    ax = axes[1]
    ax.hist(wins['sugar_ratio'].clip(upper=5), bins=30, alpha=0.6, color='#2ECC71',
            density=True, label=f'Win (n={len(wins):,})')
    ax.hist(losses['sugar_ratio'].clip(upper=5), bins=30, alpha=0.6, color='#E24B4A',
            density=True, label=f'Lose (n={len(losses):,})')
    
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Ratio = 1')
    ax.axvline(wins['sugar_ratio'].median(), color='#27AE60', linestyle='--', linewidth=1.5,
               label=f'Win median: {wins["sugar_ratio"].median():.2f}')
    ax.axvline(losses['sugar_ratio'].median(), color='#C0392B', linestyle='--', linewidth=1.5,
               label=f'Lose median: {losses["sugar_ratio"].median():.2f}')
    
    ax.set_xlabel('Sugar ratio (attacker / target)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sugar ratio distribution by outcome', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_sugar_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_sugar_scatter.png")
    
    # ============================================================
    # PLOT 2: Win rate by sugar ratio bin
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: win rate by sugar ratio
    ax = axes[0]
    ratio_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
    ratio_labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1', '1-1.5', '1.5-2', '2-3', '3-5', '5+']
    df['ratio_bin'] = pd.cut(df['sugar_ratio'], bins=ratio_bins, labels=ratio_labels)
    
    grouped = df.groupby('ratio_bin')
    win_rates = grouped.apply(lambda g: (g['outcome'] == 'WIN').mean() * 100)
    counts = grouped.size()
    
    valid = counts >= 5
    x_pos = range(valid.sum())
    
    bars = ax.bar(x_pos, win_rates[valid].values, color='#5B8DB8', width=0.7)
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4, label='50% baseline')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(win_rates[valid].index, fontsize=9, rotation=30)
    ax.set_xlabel('Sugar ratio (attacker / target)', fontsize=12)
    ax.set_ylabel('Attacker win rate (%)', fontsize=12)
    ax.set_title('Win rate by sugar advantage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    
    for i, (v, n) in enumerate(zip(win_rates[valid].values, counts[valid].values)):
        ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i, -5, f'n={n}', ha='center', fontsize=7, color='gray')
    
    # Right: % of attackers who were stronger
    ax = axes[1]
    pct_stronger = df['attacker_stronger'].mean() * 100
    pct_stronger_win = wins['attacker_stronger'].mean() * 100
    pct_stronger_lose = losses['attacker_stronger'].mean() * 100
    
    categories = ['All attacks', 'Won attacks', 'Lost attacks']
    values = [pct_stronger, pct_stronger_win, pct_stronger_lose]
    colors_bar = ['#5B8DB8', '#2ECC71', '#E24B4A']
    
    bars = ax.bar(categories, values, color=colors_bar, width=0.5)
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_ylabel('% where attacker had more sugar', fontsize=12)
    ax.set_title('Did attackers pick weaker targets?', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.1f}%',
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_win_rate_by_ratio.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_win_rate_by_ratio.png")
    
    # ============================================================
    # PLOT 3: Sugar ratio x alpha gap → win rate (2D heatmap)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    n_bins = 6
    ratio_edges = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 10]
    ratio_labs = ['<0.5', '0.5-0.75', '0.75-1', '1-1.5', '1.5-2', '>2']
    gap_edges = [0, 0.3, 0.6, 0.9, 1.2, 2.0]
    gap_labs = ['0-0.3', '0.3-0.6', '0.6-0.9', '0.9-1.2', '1.2+']
    
    df['ratio_bin2'] = pd.cut(df['sugar_ratio'], bins=ratio_edges, labels=ratio_labs)
    df['gap_bin2'] = pd.cut(df['alpha_gap'], bins=gap_edges, labels=gap_labs)
    
    # Win rate heatmap
    ax = axes[0]
    pivot_wr = df.groupby(['ratio_bin2', 'gap_bin2']).apply(
        lambda g: (g['outcome'] == 'WIN').mean() * 100 if len(g) >= 5 else np.nan
    ).unstack()
    # Reindex to ensure all bins present
    pivot_wr = pivot_wr.reindex(index=ratio_labs, columns=gap_labs)
    
    im = ax.imshow(pivot_wr.values, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=100)
    ax.set_xticks(range(len(gap_labs)))
    ax.set_xticklabels(gap_labs, fontsize=9)
    ax.set_yticks(range(len(ratio_labs)))
    ax.set_yticklabels(ratio_labs, fontsize=9)
    ax.set_xlabel(r'$|\alpha_{attacker} - \alpha_{target}|$', fontsize=12)
    ax.set_ylabel('Sugar ratio (attacker / target)', fontsize=12)
    ax.set_title('Win rate (%)\nby sugar ratio and alpha gap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Win rate (%)')
    
    for i in range(pivot_wr.shape[0]):
        for j in range(pivot_wr.shape[1]):
            val = pivot_wr.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > 65 or val < 35 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)
    
    # Count heatmap
    ax = axes[1]
    pivot_n = df.groupby(['ratio_bin2', 'gap_bin2']).size().unstack(fill_value=0)
    pivot_n = pivot_n.reindex(index=ratio_labs, columns=gap_labs, fill_value=0)
    
    im = ax.imshow(pivot_n.values, cmap='Blues', aspect='auto', origin='lower')
    ax.set_xticks(range(len(gap_labs)))
    ax.set_xticklabels(gap_labs, fontsize=9)
    ax.set_yticks(range(len(ratio_labs)))
    ax.set_yticklabels(ratio_labs, fontsize=9)
    ax.set_xlabel(r'$|\alpha_{attacker} - \alpha_{target}|$', fontsize=12)
    ax.set_ylabel('Sugar ratio (attacker / target)', fontsize=12)
    ax.set_title('Sample count\nby sugar ratio and alpha gap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
    
    for i in range(pivot_n.shape[0]):
        for j in range(pivot_n.shape[1]):
            val = pivot_n.values[i, j]
            color = 'white' if val > np.max(pivot_n.values) * 0.6 else 'black'
            ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=9, color=color)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_2d_ratio_alpha.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_2d_ratio_alpha.png")
    
    # ============================================================
    # PLOT 4: Do agents attack when they think they're stronger?
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sugar_bins = [0, 5, 15, 30, 60, 100, 500, 10000]
    sugar_labels = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-500', '500+']
    df['att_sugar_bin'] = pd.cut(df['attacker_sugar'], bins=sugar_bins, labels=sugar_labels)
    
    grouped = df.groupby('att_sugar_bin')
    
    pct_stronger_by_bin = grouped.apply(
        lambda g: g['attacker_stronger'].mean() * 100 if len(g) >= 5 else np.nan
    ).dropna()
    
    win_rate_by_bin = grouped.apply(
        lambda g: (g['outcome'] == 'WIN').mean() * 100 if len(g) >= 5 else np.nan
    ).dropna()
    
    x = range(len(pct_stronger_by_bin))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], pct_stronger_by_bin.values,
                    width, label='% stronger than target', color='#5B8DB8')
    bars2 = ax.bar([i + width/2 for i in x], win_rate_by_bin.loc[pct_stronger_by_bin.index].values,
                    width, label='% win rate', color='#2ECC71')
    
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(pct_stronger_by_bin.index, fontsize=10)
    ax.set_xlabel('Attacker sugar (pre-combat)', fontsize=12)
    ax.set_ylabel('%', fontsize=12)
    ax.set_title('Do richer agents pick weaker targets?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                    ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_attacker_strength.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_attacker_strength.png")
    
    # ============================================================
    # STATISTICS
    # ============================================================
    print(f"\n  {'='*55}")
    print(f"  STATISTICS")
    print(f"  {'='*55}")
    
    print(f"\n  Total valid combats: {len(df):,}")
    print(f"  Win rate: {(df['outcome']=='WIN').mean()*100:.1f}%")
    
    print(f"\n  --- Sugar ratio ---")
    print(f"  Mean ratio (all):  {df['sugar_ratio'].mean():.2f}")
    print(f"  Mean ratio (WIN):  {wins['sugar_ratio'].mean():.2f}")
    print(f"  Mean ratio (LOSE): {losses['sugar_ratio'].mean():.2f}")
    print(f"  Median ratio (WIN):  {wins['sugar_ratio'].median():.2f}")
    print(f"  Median ratio (LOSE): {losses['sugar_ratio'].median():.2f}")
    
    print(f"\n  --- Attacker advantage ---")
    print(f"  Attacker had more sugar: {df['attacker_stronger'].mean()*100:.1f}% of attacks")
    print(f"  Among wins:  {wins['attacker_stronger'].mean()*100:.1f}%")
    print(f"  Among losses: {losses['attacker_stronger'].mean()*100:.1f}%")
    print(f"  If random: 50% expected")
    
    print(f"\n  --- Alpha gap ---")
    print(f"  Mean gap (all):  {df['alpha_gap'].mean():.3f}")
    print(f"  Mean gap (WIN):  {wins['alpha_gap'].mean():.3f}")
    print(f"  Mean gap (LOSE): {losses['alpha_gap'].mean():.3f}")
    
    from scipy.stats import pearsonr
    print(f"\n  --- Correlations ---")
    r1, p1 = pearsonr(df['sugar_ratio'], (df['outcome']=='WIN').astype(int))
    r2, p2 = pearsonr(df['alpha_gap'], (df['outcome']=='WIN').astype(int))
    r3, p3 = pearsonr(df['attacker_sugar'], (df['outcome']=='WIN').astype(int))
    print(f"  Sugar ratio → win:    r={r1:+.4f}, p={p1:.6f}")
    print(f"  Alpha gap → win:      r={r2:+.4f}, p={p2:.6f}")
    print(f"  Attacker sugar → win: r={r3:+.4f}, p={p3:.6f}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
