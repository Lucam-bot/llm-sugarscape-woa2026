"""
Attack Heatmaps by Alpha — with Attack Rate
=============================================
1. Attack rate heatmap: % of adjacency opportunities that resulted in attack
2. Win rate heatmap: % of attacker victories (excluding FAIL)

For each step, reconstructs all adjacent pairs and their alphas.
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
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 10

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


def process_run(folder, prefix=""):
    """For one run: count all adjacent pairs and actual attacks, binned by alpha."""
    config_file = folder / f"{prefix}config.csv"
    moves_file = folder / f"{prefix}agent_moves.csv"
    
    if not config_file.exists() or not moves_file.exists():
        return None, None
    
    config = _read_config(config_file)
    vision = int(config.get('global_vision', 4))
    
    df = pd.read_csv(moves_file, encoding='utf-8-sig')
    
    bins = np.linspace(-1, 1, N_BINS + 1)
    
    # Matrices to accumulate across steps
    opportunities = np.zeros((N_BINS, N_BINS))  # (target_bin, attacker_bin)
    attacks_count = np.zeros((N_BINS, N_BINS))
    wins_count = np.zeros((N_BINS, N_BINS))
    total_attacks = np.zeros((N_BINS, N_BINS))  # for win rate (attacks that went through)
    
    # Identify attack rows
    df['is_attack'] = df['decision'].astype(str).str.startswith('4,')
    df['outcome'] = df.apply(classify_outcome, axis=1)
    
    for step, group in df.groupby('step'):
        agents = group[['agent_id', 'x', 'y', 'alpha']].drop_duplicates('agent_id').values
        n = len(agents)
        
        # Find all adjacent pairs (directed: each agent can attack each neighbor)
        for i in range(n):
            aid_i, xi, yi, alpha_i = int(agents[i][0]), agents[i][1], agents[i][2], agents[i][3]
            bin_i = min(N_BINS - 1, max(0, int(np.digitize(alpha_i, bins) - 1)))
            
            for j in range(n):
                if i == j:
                    continue
                aid_j, xj, yj, alpha_j = int(agents[j][0]), agents[j][1], agents[j][2], agents[j][3]
                
                if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                    bin_j = min(N_BINS - 1, max(0, int(np.digitize(alpha_j, bins) - 1)))
                    # Agent i could attack agent j
                    opportunities[bin_j, bin_i] += 1  # y=target, x=attacker
        
        # Count actual attacks in this step
        step_attacks = group[group['is_attack'] & group['target_alpha'].notna()]
        for _, atk in step_attacks.iterrows():
            att_bin = min(N_BINS - 1, max(0, int(np.digitize(atk['alpha'], bins) - 1)))
            tgt_bin = min(N_BINS - 1, max(0, int(np.digitize(atk['target_alpha'], bins) - 1)))
            attacks_count[tgt_bin, att_bin] += 1
            
            if atk['outcome'] in ('WIN', 'LOSE'):
                total_attacks[tgt_bin, att_bin] += 1
                if atk['outcome'] == 'WIN':
                    wins_count[tgt_bin, att_bin] += 1
    
    return {
        'opportunities': opportunities,
        'attacks': attacks_count,
        'wins': wins_count,
        'total_valid': total_attacks
    }


def load_all(batch_folder):
    """Aggregate across all runs."""
    opp_total = np.zeros((N_BINS, N_BINS))
    atk_total = np.zeros((N_BINS, N_BINS))
    win_total = np.zeros((N_BINS, N_BINS))
    valid_total = np.zeros((N_BINS, N_BINS))
    
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefixes.add(f.name.replace("config.csv", ""))
        sources = [(batch_folder, p) for p in sorted(prefixes)]
    else:
        sources = [(f, "") for f in folders]
    
    for i, (folder, prefix) in enumerate(sources):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Processing {i+1}/{len(sources)}")
        
        result = process_run(folder, prefix)
        if result is not None:
            opp_total += result['opportunities']
            atk_total += result['attacks']
            win_total += result['wins']
            valid_total += result['total_valid']
    
    return opp_total, atk_total, win_total, valid_total


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  ATTACK RATE HEATMAPS BY ALPHA")
    print("=" * 60)
    
    opportunities, attacks, wins, valid_attacks = load_all(BATCH_FOLDER)
    
    total_opp = int(opportunities.sum())
    total_atk = int(attacks.sum())
    total_valid = int(valid_attacks.sum())
    total_wins = int(wins.sum())
    
    print(f"\n  Total adjacency opportunities: {total_opp:,}")
    print(f"  Total attacks: {total_atk:,}")
    print(f"  Valid attacks (WIN+LOSE): {total_valid:,}")
    print(f"  Global attack rate: {total_atk/max(total_opp,1)*100:.2f}%")
    print(f"  Global win rate: {total_wins/max(total_valid,1)*100:.1f}%")
    
    bins = np.linspace(-1, 1, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    step = 2.0 / N_BINS
    
    # Attack rate
    with np.errstate(divide='ignore', invalid='ignore'):
        attack_rate = np.where(opportunities >= 10, attacks / opportunities * 100, np.nan)
        win_rate = np.where(valid_attacks >= 5, wins / valid_attacks * 100, np.nan)
    
    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Tick setup: labels at cell edges ---
    edge_positions = np.arange(N_BINS + 1) - 0.5  # -0.5, 0.5, ..., 9.5
    edge_labels = [f'{v:.1f}' for v in bins]  

    # --- Attack rate ---
    ax = axes[0]
    # Clip colorscale to 5th-95th percentile
    ar_valid = attack_rate[~np.isnan(attack_rate)]
    ar_vmin = np.percentile(ar_valid, 5) if len(ar_valid) > 0 else 0
    ar_vmax = np.percentile(ar_valid, 95) if len(ar_valid) > 0 else 1
    
    im1 = ax.imshow(attack_rate, origin='lower', cmap='YlOrBr', aspect='equal',
                     vmin=ar_vmin, vmax=ar_vmax)
    ax.set_xticks(edge_positions)
    ax.set_xticklabels(edge_labels, fontsize=8)
    ax.set_yticks(edge_positions)
    ax.set_yticklabels(edge_labels, fontsize=8)
    ax.set_xlabel(r'$\alpha_{attacker}$', fontsize=14)
    ax.set_ylabel(r'$\alpha_{target}$', fontsize=14)
    #ax.set_title('Attack rate\n(attacks / adjacency opportunities)', fontsize=13, fontweight='bold')#ax.set_title('Attack rate\n(attacks / adjacency opportunities)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
    cbar1.set_label('Attack rate (%)', fontsize=11)
    cbar1.ax.yaxis.set_label_position('left')
    ax.plot([-0.5, N_BINS - 0.5], [-0.5, N_BINS - 0.5], 'k--', linewidth=0.8, alpha=0.4)
    for i in range(N_BINS):
        for j in range(N_BINS):
            val = attack_rate[i, j]
            opp = int(opportunities[i, j])
            if not np.isnan(val):
                color = 'white' if val > (ar_vmin + ar_vmax) / 2 + (ar_vmax - ar_vmin) * 0.15 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
            elif opp > 0:
                ax.text(j, i, f'n={opp}', ha='center', va='center',
                        fontsize=5, color='gray')
    
    # --- Win rate ---
    ax = axes[1]
    # Clip colorscale to 5th-95th percentile
    wr_valid = win_rate[~np.isnan(win_rate)]
    wr_vmin = np.percentile(wr_valid, 5) if len(wr_valid) > 0 else 0
    wr_vmax = np.percentile(wr_valid, 95) if len(wr_valid) > 0 else 100
    
    im2 = ax.imshow(win_rate, origin='lower', cmap='Blues', aspect='equal',
                     vmin=wr_vmin, vmax=wr_vmax)
    ax.set_xticks(edge_positions)
    ax.set_xticklabels(edge_labels, fontsize=8)
    ax.set_yticks(edge_positions)
    ax.set_yticklabels(edge_labels, fontsize=8)
    ax.set_xlabel(r'$\alpha_{attacker}$', fontsize=14)
    ax.set_ylabel(r'$\alpha_{target}$', fontsize=14)
    #ax.set_title('Attacker win rate (%)\n(excluding failed attacks)', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label('Win rate (%)', fontsize=11)
    cbar2.ax.yaxis.set_label_position('left')
    ax.plot([-0.5, N_BINS - 0.5], [-0.5, N_BINS - 0.5], 'k--', linewidth=0.8, alpha=0.4)
    
    for i in range(N_BINS):
        for j in range(N_BINS):
            val = win_rate[i, j]
            n = int(valid_attacks[i, j])
            if not np.isnan(val):
                mid = (wr_vmin + wr_vmax) / 2
                color = 'white' if val > mid + (wr_vmax - wr_vmin) * 0.2 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
            elif n > 0:
                ax.text(j, i, f'n={n}', ha='center', va='center',
                        fontsize=5, color='gray')
    
    #fig.suptitle(r'Attack patterns by $\alpha_{attacker}$ vs $\alpha_{target}$'
                # f'\n({total_opp:,} adjacency opportunities, {total_atk:,} attacks, '
                 #f'global attack rate: {total_atk/max(total_opp,1)*100:.1f}%)',
                 #fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_attacks_alpha.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: heatmap_attacks_alpha.png")
    
    # ============================================================
    # STATS
    # ============================================================
    print(f"\n  {'='*50}")
    print(f"  ANALYSIS")
    print(f"  {'='*50}")
    
    # Is attack rate higher off-diagonal (different alpha)?
    diag_mask = np.zeros((N_BINS, N_BINS), dtype=bool)
    for k in range(N_BINS):
        diag_mask[k, k] = True
    # Expand diagonal to ±1 bin
    for k in range(N_BINS - 1):
        diag_mask[k, k+1] = True
        diag_mask[k+1, k] = True
    
    opp_diag = opportunities[diag_mask].sum()
    atk_diag = attacks[diag_mask].sum()
    opp_off = opportunities[~diag_mask].sum()
    atk_off = attacks[~diag_mask].sum()
    
    rate_diag = atk_diag / max(opp_diag, 1) * 100
    rate_off = atk_off / max(opp_off, 1) * 100
    
    print(f"\n  Attack rate — similar alpha (diagonal ±1 bin):")
    print(f"    Opportunities: {int(opp_diag):,}, Attacks: {int(atk_diag):,}, Rate: {rate_diag:.2f}%")
    print(f"  Attack rate — different alpha (off-diagonal):")
    print(f"    Opportunities: {int(opp_off):,}, Attacks: {int(atk_off):,}, Rate: {rate_off:.2f}%")
    print(f"  Ratio (off/diag): {rate_off/max(rate_diag,0.001):.2f}x")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()