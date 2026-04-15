"""
Attack Heatmaps by Alpha — Attacker vs Target
================================================
1. Density heatmap: number of attacks per (α_attacker, α_target) bin
2. Win rate heatmap: % of attacker victories per bin (excluding FAIL)

Reads agent_moves.csv from each run folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — CHANGE THESE PATHS
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of bins per axis
N_BINS = 10

# ============================================================
# LOAD & CLASSIFY
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


def load_attacks(batch_folder):
    all_attacks = []
    
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_agent_moves.csv"):
            prefixes.add(f.name.replace("agent_moves.csv", ""))
        for prefix in sorted(prefixes):
            moves_file = batch_folder / f"{prefix}agent_moves.csv"
            if moves_file.exists():
                df = _extract_attacks(moves_file)
                if df is not None:
                    all_attacks.append(df)
    else:
        for i, folder in enumerate(folders):
            moves_file = folder / "agent_moves.csv"
            if moves_file.exists():
                df = _extract_attacks(moves_file)
                if df is not None:
                    all_attacks.append(df)
    
    if all_attacks:
        return pd.concat(all_attacks, ignore_index=True)
    return pd.DataFrame()


def _extract_attacks(moves_file):
    df = pd.read_csv(moves_file, encoding='utf-8-sig')
    attacks = df[df['decision'].astype(str).str.startswith('4,')].copy()
    if attacks.empty:
        return None
    attacks['outcome'] = attacks.apply(classify_outcome, axis=1)
    attacks = attacks[['alpha', 'target_alpha', 'outcome']].dropna(subset=['target_alpha'])
    return attacks


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  ATTACK HEATMAPS BY ALPHA")
    print("=" * 60)
    
    df = load_attacks(BATCH_FOLDER)
    if df.empty:
        print("  No attack data found!")
        return
    
    print(f"  Total attacks loaded: {len(df):,}")
    print(f"  Outcomes: {df['outcome'].value_counts().to_dict()}")
    
    # Exclude FAIL
    df_valid = df[df['outcome'] != 'FAIL'].copy()
    print(f"  Valid attacks (WIN + LOSE): {len(df_valid):,}")
    
    # Bins
    bins = np.linspace(-1, 1, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    df_valid['att_bin'] = pd.cut(df_valid['alpha'], bins=bins, labels=False, include_lowest=True)
    df_valid['tgt_bin'] = pd.cut(df_valid['target_alpha'], bins=bins, labels=False, include_lowest=True)
    
    # ============================================================
    # Heatmap 1: Attack density
    # ============================================================
    density = np.zeros((N_BINS, N_BINS))
    for _, row in df_valid.iterrows():
        a, t = int(row['att_bin']), int(row['tgt_bin'])
        density[t, a] += 1  # y=target, x=attacker
    
    # ============================================================
    # Heatmap 2: Win rate (% attacker wins)
    # ============================================================
    wins = np.zeros((N_BINS, N_BINS))
    total = np.zeros((N_BINS, N_BINS))
    
    for _, row in df_valid.iterrows():
        a, t = int(row['att_bin']), int(row['tgt_bin'])
        total[t, a] += 1
        if row['outcome'] == 'WIN':
            wins[t, a] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        win_rate = np.where(total >= 5, wins / total * 100, np.nan)
    
    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    tick_labels = [f'{c:.1f}' for c in bin_centers]
    
    # --- Density ---
    ax = axes[0]
    im1 = ax.imshow(density, origin='lower', cmap='YlOrRd', aspect='equal',
                     extent=[-1, 1, -1, 1])
    ax.set_xlabel(r'$\alpha_{attacker}$', fontsize=14)
    ax.set_ylabel(r'$\alpha_{target}$', fontsize=14)
    ax.set_title('Attack density', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
    cbar1.set_label('Number of attacks', fontsize=11)
    ax.plot([-1, 1], [-1, 1], 'k--', linewidth=0.8, alpha=0.4)
    
    # Annotate cells with counts (only if > 0)
    step = 2.0 / N_BINS
    for i in range(N_BINS):
        for j in range(N_BINS):
            val = int(density[i, j])
            if val > 0:
                x_pos = -1 + (j + 0.5) * step
                y_pos = -1 + (i + 0.5) * step
                color = 'white' if val > np.max(density) * 0.6 else 'black'
                ax.text(x_pos, y_pos, str(val), ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
    
    # --- Win rate ---
    ax = axes[1]
    im2 = ax.imshow(win_rate, origin='lower', cmap='RdYlGn', aspect='equal',
                     extent=[-1, 1, -1, 1], vmin=0, vmax=100)
    ax.set_xlabel(r'$\alpha_{attacker}$', fontsize=14)
    ax.set_ylabel(r'$\alpha_{target}$', fontsize=14)
    ax.set_title('Attacker win rate (%)', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label('Win rate (%)', fontsize=11)
    ax.plot([-1, 1], [-1, 1], 'k--', linewidth=0.8, alpha=0.4)
    
    # Annotate cells with percentages
    step = 2.0 / N_BINS
    for i in range(N_BINS):
        for j in range(N_BINS):
            val = win_rate[i, j]
            n = int(total[i, j])
            if not np.isnan(val):
                x_pos = -1 + (j + 0.5) * step
                y_pos = -1 + (i + 0.5) * step
                color = 'white' if val > 65 or val < 35 else 'black'
                ax.text(x_pos, y_pos, f'{val:.0f}%', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
            elif n > 0:
                x_pos = -1 + (j + 0.5) * step
                y_pos = -1 + (i + 0.5) * step
                ax.text(x_pos, y_pos, f'n={n}', ha='center', va='center',
                        fontsize=6, color='gray')
    
    fig.suptitle(r'Attack patterns by $\alpha_{attacker}$ vs $\alpha_{target}$'
                 f'\n(n={len(df_valid):,} valid attacks, excluding FAIL)',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_attacks_alpha.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: heatmap_attacks_alpha.png")
    
    # ============================================================
    # STATS
    # ============================================================
    print(f"\n  {'='*50}")
    print(f"  SUMMARY")
    print(f"  {'='*50}")
    print(f"  Valid attacks: {len(df_valid):,}")
    print(f"  WIN: {(df_valid['outcome']=='WIN').sum():,}  "
          f"({(df_valid['outcome']=='WIN').mean()*100:.1f}%)")
    print(f"  LOSE: {(df_valid['outcome']=='LOSE').sum():,}  "
          f"({(df_valid['outcome']=='LOSE').mean()*100:.1f}%)")
    
    print(f"\n  Diagonal tendency (same α attacks same α):")
    df_valid['alpha_gap'] = (df_valid['alpha'] - df_valid['target_alpha']).abs()
    print(f"  Mean |gap|: {df_valid['alpha_gap'].mean():.3f}")
    print(f"  Median |gap|: {df_valid['alpha_gap'].median():.3f}")
    print(f"  Reference (random): 0.667")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
