"""
Combat Estimation Analysis
============================
Do LLM agents behave as if they estimate the target's sugar?

Compares sugar ratio (attacker/target) in two populations:
  - ALL adjacency opportunities (baseline: what ratio would random attacks have)
  - ACTUAL attacks (what ratio do agents choose to attack at)

If agents attack randomly → same distribution.
If agents pick weaker targets → attack distribution shifts right (ratio > 1).
If agents attack from desperation → attack distribution shifts left (ratio < 1).

Also analyzes proxies the agent might use to estimate target strength.
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
OUTPUT_DIR = Path(r"./results\Analysis_Results11")
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
    """For one run: collect all adjacency opportunities and actual attacks with sugar."""
    config_file = folder / f"{prefix}config.csv"
    moves_file = folder / f"{prefix}agent_moves.csv"
    map_file = folder / f"{prefix}map_state.csv"
    
    if not config_file.exists() or not moves_file.exists():
        return None
    
    config = _read_config(config_file)
    grid_size = int(config.get('grid_size', 20))
    vision = int(config.get('global_vision', 4))
    
    df = pd.read_csv(moves_file, encoding='utf-8-sig')
    
    # Load map state: step -> 2D sugar array
    sugar_maps = {}
    if map_file.exists():
        map_df = pd.read_csv(map_file, encoding='utf-8-sig')
        for step, grp in map_df.groupby('step'):
            grid = np.zeros((grid_size, grid_size), dtype=int)
            for _, r in grp.iterrows():
                grid[int(r['y']), int(r['x'])] = int(r['sugar'])
            sugar_maps[step] = grid
    
    # Build previous-step sugar lookup for pre-combat estimation
    prev_sugar = {}
    for _, row in df.iterrows():
        prev_sugar[(row['step'] + 1, row['agent_id'])] = row['sugar']
    
    opportunities = []
    attacks = []
    
    for step in df['step'].unique():
        step_agents = df[df['step'] == step][['agent_id', 'x', 'y', 'alpha', 'sugar', 'decision', 'alive']].drop_duplicates('agent_id')
        agents = step_agents.values
        n = len(agents)
        
        # Occupied positions this step
        occupied = set()
        for k in range(n):
            occupied.add((int(agents[k][1]), int(agents[k][2])))
        
        # Get sugar map for this step (or previous)
        smap = sugar_maps.get(step, sugar_maps.get(step - 1, None))
        
        # Precompute z (max visible free cell sugar) for each agent
        z_cache = {}
        if smap is not None:
            for k in range(n):
                aid_k = int(agents[k][0])
                xk, yk = int(agents[k][1]), int(agents[k][2])
                max_s = 0
                for dx in range(-vision, vision + 1):
                    for dy in range(-vision, vision + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if dx*dx + dy*dy <= vision*vision:
                            nx, ny = xk + dx, yk + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                if (nx, ny) not in occupied:
                                    max_s = max(max_s, smap[ny, nx])
                z_cache[aid_k] = max_s
        
        for i in range(n):
            aid_i, xi, yi, alpha_i, sugar_i, dec_i, alive_i = agents[i]
            aid_i = int(aid_i)
            pre_sugar_i = prev_sugar.get((step, aid_i), sugar_i)
            z_i = z_cache.get(aid_i, np.nan)
            
            for j in range(n):
                if i == j:
                    continue
                aid_j, xj, yj, alpha_j, sugar_j, dec_j, alive_j = agents[j]
                aid_j = int(aid_j)
                
                if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                    pre_sugar_j = prev_sugar.get((step, aid_j), sugar_j)
                    
                    opp = {
                        'step': step,
                        'agent_sugar': pre_sugar_i,
                        'target_sugar': pre_sugar_j,
                        'agent_alpha': alpha_i,
                        'target_alpha': alpha_j,
                        'alpha_gap': abs(alpha_i - alpha_j),
                        'z': z_i,
                    }
                    
                    if pre_sugar_j > 0:
                        opp['sugar_ratio'] = pre_sugar_i / pre_sugar_j
                    else:
                        opp['sugar_ratio'] = np.nan
                    
                    opp['agent_stronger'] = pre_sugar_i > pre_sugar_j
                    opportunities.append(opp)
    
    # Separately extract actual attacks more reliably
    attack_rows = df[df['decision'].astype(str).str.startswith('4,') & df['target_alpha'].notna()].copy()
    attack_rows['outcome'] = attack_rows.apply(classify_outcome, axis=1)
    attack_rows = attack_rows[attack_rows['outcome'] != 'FAIL']
    
    for _, atk in attack_rows.iterrows():
        step = atk['step']
        aid = atk['agent_id']
        t_alpha = round(atk['target_alpha'], 3)
        
        pre_att = prev_sugar.get((step, aid), atk['sugar'])
        
        target_candidates = df[(df['step'] == step - 1) & (abs(df['alpha'] - t_alpha) < 0.005)]
        if target_candidates.empty:
            target_candidates = df[(df['step'] == step) & (abs(df['alpha'] - t_alpha) < 0.005) & (df['agent_id'] != aid)]
        
        if not target_candidates.empty:
            pre_tgt = target_candidates.iloc[0]['sugar']
        else:
            continue
        
        if pre_tgt > 0:
            ratio = pre_att / pre_tgt
        else:
            ratio = np.nan
        
        # Get z for this attacker at this step
        # Reconstruct from opportunities if possible
        z_att = np.nan
        smap = sugar_maps.get(step, sugar_maps.get(step - 1, None))
        if smap is not None:
            ax_pos, ay_pos = int(atk['x']), int(atk['y'])
            occ_step = set()
            step_df = df[df['step'] == step]
            for _, r in step_df.iterrows():
                occ_step.add((int(r['x']), int(r['y'])))
            max_s = 0
            for dx in range(-vision, vision + 1):
                for dy in range(-vision, vision + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if dx*dx + dy*dy <= vision*vision:
                        nx, ny = ax_pos + dx, ay_pos + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            if (nx, ny) not in occ_step:
                                max_s = max(max_s, smap[ny, nx])
            z_att = max_s
        
        attacks.append({
            'step': step,
            'attacker_sugar': pre_att,
            'target_sugar': pre_tgt,
            'sugar_ratio': ratio,
            'attacker_alpha': atk['alpha'],
            'target_alpha': atk['target_alpha'],
            'alpha_gap': abs(atk['alpha'] - atk['target_alpha']),
            'outcome': atk['outcome'],
            'attacker_stronger': pre_att > pre_tgt,
            'z': z_att,
        })
    
    return pd.DataFrame(opportunities), pd.DataFrame(attacks)


def load_all(batch_folder):
    all_opp = []
    all_atk = []
    
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    
    if not folders:
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefixes.add(f.name.replace("agent_moves.csv", "").replace("config.csv", ""))
        sources = [(batch_folder, p) for p in sorted(prefixes)]
    else:
        sources = [(f, "") for f in folders]
    
    for i, (folder, prefix) in enumerate(sources):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Processing {i+1}/{len(sources)}")
        
        result = process_run(folder, prefix)
        if result is not None:
            opp_df, atk_df = result
            if not opp_df.empty:
                all_opp.append(opp_df)
            if not atk_df.empty:
                all_atk.append(atk_df)
    
    opp = pd.concat(all_opp, ignore_index=True) if all_opp else pd.DataFrame()
    atk = pd.concat(all_atk, ignore_index=True) if all_atk else pd.DataFrame()
    return opp, atk


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  COMBAT ESTIMATION BY DELTA ALPHA")
    print("=" * 60)
    
    opp, atk = load_all(BATCH_FOLDER)
    
    if opp.empty or atk.empty:
        print("  No data found!")
        return
    
    opp_valid = opp[opp['sugar_ratio'].notna() & (opp['sugar_ratio'] < 50)].copy()
    atk_valid = atk[atk['sugar_ratio'].notna() & (atk['sugar_ratio'] < 50)].copy()
    
    print(f"\n  Adjacency opportunities (with valid ratio): {len(opp_valid):,}")
    print(f"  Actual attacks (with valid ratio): {len(atk_valid):,}")
    
    # Delta alpha bins — same for opportunities and attacks
    gap_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 2.0]
    gap_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.3', '1.3+']
    
    opp_valid['gap_bin'] = pd.cut(opp_valid['alpha_gap'], bins=gap_edges, labels=gap_labels)
    atk_valid['gap_bin'] = pd.cut(atk_valid['alpha_gap'], bins=gap_edges, labels=gap_labels)
    
    # ============================================================
    # PLOT 1: By delta alpha — % attacker stronger, attack rate,
    #         median sugar ratio, win rate
    # ============================================================
    print("\n  Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- 1a: Attack rate by delta alpha ---
    ax = axes[0, 0]
    opp_by_gap = opp_valid['gap_bin'].value_counts().sort_index()
    atk_by_gap = atk_valid['gap_bin'].value_counts().sort_index()
    rate_by_gap = (atk_by_gap / opp_by_gap * 100).fillna(0)
    valid_gaps = [g for g in gap_labels if g in opp_by_gap.index and opp_by_gap[g] >= 20]
    
    bars = ax.bar(range(len(valid_gaps)), [rate_by_gap.get(g, 0) for g in valid_gaps],
                  color='#E74C3C', width=0.6)
    ax.set_xticks(range(len(valid_gaps)))
    ax.set_xticklabels(valid_gaps, fontsize=9)
    ax.set_xlabel(r'$|\Delta\alpha|$ (attacker - target)', fontsize=12)
    ax.set_ylabel('Attack rate (%)', fontsize=12)
    ax.set_title('Attack rate by identity gap\n(attacks / adjacency opportunities)', fontsize=12, fontweight='bold')
    
    for i, g in enumerate(valid_gaps):
        v = rate_by_gap.get(g, 0)
        n = int(opp_by_gap.get(g, 0))
        ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i, -max(rate_by_gap.max()*0.04, 0.02), f'n={n:,}', ha='center', fontsize=7, color='gray')
    
    # --- 1b: % attacker stronger by delta alpha ---
    ax = axes[0, 1]
    
    # Non-attack opportunities: all adjacent pairs MINUS actual attacks
    # For each gap bin, compute % stronger among those who did NOT attack
    pct_stronger_noatk_by_gap = {}
    for g in valid_gaps:
        opp_sub = opp_valid[opp_valid['gap_bin'] == g]
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        
        # Remove attack pairs from opportunities (approximate: same gap bin)
        n_opp = len(opp_sub)
        n_atk = len(atk_sub)
        n_noatk = n_opp - n_atk
        
        if n_noatk > 0 and n_opp > 0:
            # stronger_in_opp_total = opp_sub['agent_stronger'].sum()
            # stronger_in_atk = atk_sub['attacker_stronger'].sum() if n_atk > 0 else 0
            # stronger_in_noatk = stronger_in_opp_total - stronger_in_atk
            # Use fraction (0-1) not percentage
            stronger_opp = opp_sub['agent_stronger'].sum()
            stronger_atk = atk_sub['attacker_stronger'].sum() if n_atk > 0 else 0
            pct_stronger_noatk_by_gap[g] = (stronger_opp - stronger_atk) / n_noatk
        else:
            pct_stronger_noatk_by_gap[g] = np.nan
    
    pct_stronger_atk_by_gap = atk_valid.groupby('gap_bin')['attacker_stronger'].apply(
        lambda g: g.mean() if len(g) >= 5 else np.nan
    )
    
    x = range(len(valid_gaps))
    width = 0.35
    
    vals_noatk = [pct_stronger_noatk_by_gap.get(g, np.nan) for g in valid_gaps]
    vals_atk = [pct_stronger_atk_by_gap.get(g, np.nan) for g in valid_gaps]
    
    bars1 = ax.bar([i - width/2 for i in x], vals_noatk, width,
                    label='No attack', color='#A8A8A8')
    bars2 = ax.bar([i + width/2 for i in x], vals_atk, width,
                    label='Attack', color='#E74C3C')
    
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_gaps, fontsize=9)
    ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
    ax.set_ylabel('P(attacker had more sugar)', fontsize=12)
    ax.set_title('Sugar advantage by identity gap\n(no-attack vs attack)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    
    # --- 1c: Median sugar ratio by delta alpha ---
    ax = axes[1, 0]
    
    med_ratio_opp = opp_valid.groupby('gap_bin')['sugar_ratio'].median()
    med_ratio_atk = atk_valid.groupby('gap_bin').apply(
        lambda g: g['sugar_ratio'].median() if len(g) >= 5 else np.nan
    )
    
    vals_opp_r = [med_ratio_opp.get(g, np.nan) for g in valid_gaps]
    vals_atk_r = [med_ratio_atk.get(g, np.nan) for g in valid_gaps]
    
    bars1 = ax.bar([i - width/2 for i in x], vals_opp_r, width,
                    label='All adjacent pairs', color='#A8A8A8')
    bars2 = ax.bar([i + width/2 for i in x], vals_atk_r, width,
                    label='Actual attacks', color='#E74C3C')
    
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.4, label='Equal sugar')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_gaps, fontsize=9)
    ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
    ax.set_ylabel('Median sugar ratio (attacker / target)', fontsize=12)
    ax.set_title('Median sugar ratio by identity gap\n(< 1 = attacker weaker)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    
    for bars, vals in [(bars1, vals_opp_r), (bars2, vals_atk_r)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.2f}',
                        ha='center', fontsize=8, fontweight='bold')
    
    # --- 1d: Win rate by delta alpha ---
    ax = axes[1, 1]
    
    win_rate_by_gap = atk_valid.groupby('gap_bin').apply(
        lambda g: (g['outcome'] == 'WIN').mean() * 100 if len(g) >= 5 else np.nan
    )
    
    vals_wr = [win_rate_by_gap.get(g, np.nan) for g in valid_gaps]
    valid_wr = [(i, g, v) for i, (g, v) in enumerate(zip(valid_gaps, vals_wr)) if not np.isnan(v)]
    
    if valid_wr:
        bars = ax.bar([x[0] for x in valid_wr], [x[2] for x in valid_wr], color='#2ECC71', width=0.6)
        ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xticks([x[0] for x in valid_wr])
        ax.set_xticklabels([x[1] for x in valid_wr], fontsize=9)
        ax.set_ylim(0, 105)
        
        for bar, (_, g, v) in zip(bars, valid_wr):
            n = len(atk_valid[atk_valid['gap_bin'] == g])
            ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.0f}%',
                    ha='center', fontsize=10, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, -4, f'n={n}',
                    ha='center', fontsize=7, color='gray')
    
    ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
    ax.set_ylabel('Attacker win rate (%)', fontsize=12)
    ax.set_title('Win rate by identity gap', fontsize=12, fontweight='bold')
    
    fig.suptitle(r'Combat analysis as a function of $|\Delta\alpha|$'
                 f'\n({len(opp_valid):,} opportunities, {len(atk_valid):,} attacks)',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_estimation_by_delta_alpha.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_estimation_by_delta_alpha.png")
    
    # ============================================================
    # PLOT 2: Sugar distributions per delta alpha bin (small multiples)
    # ============================================================
    n_valid = len(valid_gaps)
    fig, axes = plt.subplots(2, min(n_valid, 4), figsize=(5 * min(n_valid, 4), 10), squeeze=False)
    
    if n_valid > 4:
        fig2, axes2 = plt.subplots(2, n_valid - 4, figsize=(5 * (n_valid - 4), 10), squeeze=False)
    
    bins_hist = np.linspace(0, 5, 30)
    
    for idx, g in enumerate(valid_gaps):
        if idx < 4:
            ax_top = axes[0, idx]
            ax_bot = axes[1, idx]
        elif n_valid > 4:
            ax_top = axes2[0, idx - 4]
            ax_bot = axes2[1, idx - 4]
        else:
            continue
        
        opp_sub = opp_valid[opp_valid['gap_bin'] == g]
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        
        # Top: sugar ratio distributions
        if len(opp_sub) > 0:
            ax_top.hist(opp_sub['sugar_ratio'].clip(upper=5), bins=bins_hist, alpha=0.5,
                        density=True, color='#A8A8A8', label='Opportunities')
        if len(atk_sub) > 0:
            ax_top.hist(atk_sub['sugar_ratio'].clip(upper=5), bins=bins_hist, alpha=0.7,
                        density=True, color='#E74C3C', label='Attacks')
        
        ax_top.axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax_top.set_title(rf'$|\Delta\alpha|$ = {g}', fontsize=11, fontweight='bold')
        ax_top.set_xlabel('Sugar ratio', fontsize=9)
        ax_top.set_ylabel('Density', fontsize=9)
        ax_top.set_xlim(0, 5)
        if idx == 0:
            ax_top.legend(fontsize=8)
        
        n_opp = len(opp_sub)
        n_atk = len(atk_sub)
        ax_top.text(0.97, 0.95, f'opp={n_opp:,}\natk={n_atk}',
                    transform=ax_top.transAxes, fontsize=7, va='top', ha='right', color='gray')
        
        # Bottom: attacker sugar distributions
        if len(opp_sub) > 0:
            ax_bot.hist(opp_sub['agent_sugar'].clip(upper=150), bins=30, alpha=0.5,
                        density=True, color='#A8A8A8', label='All agents')
        if len(atk_sub) > 0:
            ax_bot.hist(atk_sub['attacker_sugar'].clip(upper=150), bins=30, alpha=0.7,
                        density=True, color='#E74C3C', label='Attackers')
        
        ax_bot.set_xlabel('Agent sugar', fontsize=9)
        ax_bot.set_ylabel('Density', fontsize=9)
        if idx == 0:
            ax_bot.legend(fontsize=8)
    
    # Hide unused axes
    for idx in range(n_valid, 4):
        axes[0, idx].set_visible(False)
        axes[1, idx].set_visible(False)
    
    fig.suptitle(r'Sugar distributions by $|\Delta\alpha|$ bin'
                 '\n(top: sugar ratio, bottom: attacker sugar)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_estimation_distributions_by_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_estimation_distributions_by_gap.png")
    
    if n_valid > 4:
        fig2.suptitle(r'Sugar distributions by $|\Delta\alpha|$ bin (continued)',
                      fontsize=14, fontweight='bold', y=1.02)
        plt.figure(fig2.number)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "combat_estimation_distributions_by_gap_2.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    combat_estimation_distributions_by_gap_2.png")
    
    # ============================================================
    # PLOT 3: Who attacks? Attack rate by agent sugar bracket
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sugar_bins = [0, 3, 8, 15, 30, 60, 150, 10000]
    sugar_labels = ['0-3', '3-8', '8-15', '15-30', '30-60', '60-150', '150+']
    opp_valid['sugar_bin'] = pd.cut(opp_valid['agent_sugar'], bins=sugar_bins, labels=sugar_labels)
    atk_valid['sugar_bin'] = pd.cut(atk_valid['attacker_sugar'], bins=sugar_bins, labels=sugar_labels)
    
    # Left: attack rate by agent sugar
    ax = axes[0]
    opp_counts = opp_valid['sugar_bin'].value_counts().sort_index()
    atk_counts = atk_valid['sugar_bin'].value_counts().sort_index()
    attack_rate_sugar = (atk_counts / opp_counts * 100).fillna(0)
    valid_s_bins = [b for b in sugar_labels if b in opp_counts.index and opp_counts[b] >= 50]
    
    bars = ax.bar(range(len(valid_s_bins)), [attack_rate_sugar.get(b, 0) for b in valid_s_bins],
                  color='#E74C3C', width=0.6)
    ax.set_xticks(range(len(valid_s_bins)))
    ax.set_xticklabels(valid_s_bins, fontsize=9, rotation=20)
    ax.set_xlabel('Agent sugar (pre-combat)', fontsize=12)
    ax.set_ylabel('Attack rate (%)', fontsize=12)
    ax.set_title('Who attacks?\n(attack rate by agent wealth)', fontsize=12, fontweight='bold')
    
    for i, b in enumerate(valid_s_bins):
        v = attack_rate_sugar.get(b, 0)
        n = int(opp_counts.get(b, 0))
        ax.text(i, v + max(attack_rate_sugar.max() * 0.02, 0.02), f'{v:.2f}%',
                ha='center', fontsize=9, fontweight='bold')
        ax.text(i, -max(attack_rate_sugar.max() * 0.04, 0.02), f'n={n:,}',
                ha='center', fontsize=7, color='gray')
    
    # Right: sugar distribution of attackers vs all agents with adjacency
    ax = axes[1]
    ax.hist(opp_valid['agent_sugar'].clip(upper=200), bins=40, alpha=0.5, density=True,
            color='#A8A8A8', label=f'All agents with adjacent neighbor (n={len(opp_valid):,})')
    ax.hist(atk_valid['attacker_sugar'].clip(upper=200), bins=40, alpha=0.7, density=True,
            color='#E74C3C', label=f'Actual attackers (n={len(atk_valid):,})')
    ax.axvline(opp_valid['agent_sugar'].median(), color='#666666', linestyle=':', linewidth=1.5,
               label=f'All agents median: {opp_valid["agent_sugar"].median():.1f}')
    ax.axvline(atk_valid['attacker_sugar'].median(), color='#C0392B', linestyle=':', linewidth=1.5,
               label=f'Attackers median: {atk_valid["attacker_sugar"].median():.1f}')
    ax.set_xlabel('Agent sugar (pre-combat)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sugar distribution:\nattackers vs all agents', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    fig.suptitle('Who initiates attacks? Agent wealth analysis',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_who_attacks.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_who_attacks.png")
    
    # ============================================================
    # PLOT 4: Do attackers pick weaker targets? (aggregated)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    wins = atk_valid[atk_valid['outcome'] == 'WIN']
    losses = atk_valid[atk_valid['outcome'] == 'LOSE']
    
    # Left: sugar ratio distributions
    ax = axes[0]
    bins_hist = np.linspace(0, 5, 40)
    ax.hist(opp_valid['sugar_ratio'].clip(upper=5), bins=bins_hist, alpha=0.5, density=True,
            color='#A8A8A8', label=f'All opportunities (n={len(opp_valid):,})')
    ax.hist(atk_valid['sugar_ratio'].clip(upper=5), bins=bins_hist, alpha=0.7, density=True,
            color='#E74C3C', label=f'Actual attacks (n={len(atk_valid):,})')
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(opp_valid['sugar_ratio'].median(), color='#666666', linestyle=':', linewidth=1.5,
               label=f'Opp. median: {opp_valid["sugar_ratio"].median():.2f}')
    ax.axvline(atk_valid['sugar_ratio'].median(), color='#C0392B', linestyle=':', linewidth=1.5,
               label=f'Attack median: {atk_valid["sugar_ratio"].median():.2f}')
    ax.set_xlabel('Sugar ratio (agent / target)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sugar ratio distribution\n(> 1 = attacker stronger)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    
    # Center: % stronger comparison
    ax = axes[1]
    pct_stronger_opp = opp_valid['agent_stronger'].mean() * 100
    pct_stronger_atk = atk_valid['attacker_stronger'].mean() * 100
    pct_stronger_win = wins['attacker_stronger'].mean() * 100 if len(wins) > 0 else 0
    pct_stronger_lose = losses['attacker_stronger'].mean() * 100 if len(losses) > 0 else 0
    
    categories = ['All adjacent\npairs', 'Actual\nattacks', 'Won\nattacks', 'Lost\nattacks']
    values = [pct_stronger_opp, pct_stronger_atk, pct_stronger_win, pct_stronger_lose]
    colors_bar = ['#A8A8A8', '#E74C3C', '#2ECC71', '#E24B4A']
    
    bars = ax.bar(categories, values, color=colors_bar, width=0.6)
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_ylabel('% where agent had more sugar', fontsize=12)
    ax.set_title('Sugar advantage:\nbaseline vs attacks', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.1f}%',
                ha='center', fontsize=11, fontweight='bold')
    
    # Right: target sugar — attacked vs not attacked
    ax = axes[2]
    ax.hist(opp_valid['target_sugar'].clip(upper=200), bins=40, alpha=0.5, density=True,
            color='#A8A8A8', label=f'All adjacent targets (n={len(opp_valid):,})')
    ax.hist(atk_valid['target_sugar'].clip(upper=200), bins=40, alpha=0.7, density=True,
            color='#E74C3C', label=f'Attacked targets (n={len(atk_valid):,})')
    ax.axvline(opp_valid['target_sugar'].median(), color='#666666', linestyle=':', linewidth=1.5,
               label=f'All targets median: {opp_valid["target_sugar"].median():.1f}')
    ax.axvline(atk_valid['target_sugar'].median(), color='#C0392B', linestyle=':', linewidth=1.5,
               label=f'Attacked median: {atk_valid["target_sugar"].median():.1f}')
    ax.set_xlabel('Target sugar (pre-combat)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Target selection:\nattacked vs all adjacent', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    fig.suptitle('Do LLM agents estimate target strength before attacking?',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_target_estimation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_target_estimation.png")
    
    # ============================================================
    # PLOT 5: Delta alpha modulates strategy — the key insight
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Left: shift (% stronger in attacks - % stronger in baseline) by gap bin
    ax = axes[0]
    shifts = []
    for g in valid_gaps:
        opp_sub = opp_valid[opp_valid['gap_bin'] == g]
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        if len(atk_sub) >= 5:
            s = atk_sub['attacker_stronger'].mean() * 100 - opp_sub['agent_stronger'].mean() * 100
            shifts.append((g, s, len(atk_sub)))
        else:
            shifts.append((g, np.nan, len(atk_sub)))
    
    valid_shifts = [(g, s, n) for g, s, n in shifts if not np.isnan(s)]
    
    if valid_shifts:
        x_pos = range(len(valid_shifts))
        colors_shift = ['#2ECC71' if s > 0 else '#E24B4A' for _, s, _ in valid_shifts]
        bars = ax.bar(x_pos, [s for _, s, _ in valid_shifts], color=colors_shift, width=0.6)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([g for g, _, _ in valid_shifts], fontsize=9, rotation=20)
        ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
        ax.set_ylabel('Shift (pp)', fontsize=12)
        ax.set_title('Strategic advantage shift\n(+ = pick weaker, - = desperation)',
                      fontsize=12, fontweight='bold')
        
        for i, (g, s, n) in enumerate(valid_shifts):
            ax.text(i, s + (1 if s >= 0 else -2), f'{s:+.1f}pp',
                    ha='center', fontsize=9, fontweight='bold')
            ax.text(i, ax.get_ylim()[0] + 0.5, f'n={n}',
                    ha='center', fontsize=7, color='gray')
    
    # Center: win rate by gap bin
    ax = axes[1]
    wr_data = []
    for g in valid_gaps:
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        if len(atk_sub) >= 5:
            wr = (atk_sub['outcome'] == 'WIN').mean() * 100
            wr_data.append((g, wr, len(atk_sub)))
    
    if wr_data:
        x_pos = range(len(wr_data))
        bars = ax.bar(x_pos, [wr for _, wr, _ in wr_data], color='#2ECC71', width=0.6)
        ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([g for g, _, _ in wr_data], fontsize=9, rotation=20)
        ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
        ax.set_ylabel('Win rate (%)', fontsize=12)
        ax.set_title('Win rate by identity gap', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        
        for i, (g, wr, n) in enumerate(wr_data):
            ax.text(i, wr + 2, f'{wr:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Right: median attacker sugar by gap bin (attacks vs opportunities)
    ax = axes[2]
    med_opp = []
    med_atk = []
    gap_x = []
    for g in valid_gaps:
        opp_sub = opp_valid[opp_valid['gap_bin'] == g]
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        if len(atk_sub) >= 5:
            gap_x.append(g)
            med_opp.append(opp_sub['agent_sugar'].median())
            med_atk.append(atk_sub['attacker_sugar'].median())
    
    if gap_x:
        x_pos = range(len(gap_x))
        width = 0.35
        ax.bar([i - width/2 for i in x_pos], med_opp, width,
               label='All adjacent agents', color='#A8A8A8')
        ax.bar([i + width/2 for i in x_pos], med_atk, width,
               label='Actual attackers', color='#E74C3C')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gap_x, fontsize=9, rotation=20)
        ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
        ax.set_ylabel('Median sugar', fontsize=12)
        ax.set_title('Attacker wealth by identity gap\n(who attacks at each gap level?)',
                      fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
    
    fig.suptitle(r'Does $|\Delta\alpha|$ modulate combat strategy?',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_delta_alpha_strategy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_delta_alpha_strategy.png")
    
    # ============================================================
    # PLOT 6: Attack probability vs ΔS (sugar difference)
    # ============================================================
    
    # Compute ΔS for all opportunities and flag which ones became attacks
    opp_valid['delta_sugar'] = opp_valid['agent_sugar'] - opp_valid['target_sugar']
    atk_valid['delta_sugar'] = atk_valid['attacker_sugar'] - atk_valid['target_sugar']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: attack rate as function of ΔS
    ax = axes[0]
    ds_edges = [-500, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500]
    ds_labels = ['<-100', '-100:-50', '-50:-20', '-20:-10', '-10:-5', '-5:0',
                 '0:5', '5:10', '10:20', '20:50', '50:100', '>100']
    
    opp_valid['ds_bin'] = pd.cut(opp_valid['delta_sugar'], bins=ds_edges, labels=ds_labels)
    atk_valid['ds_bin'] = pd.cut(atk_valid['delta_sugar'], bins=ds_edges, labels=ds_labels)
    
    opp_ds_counts = opp_valid['ds_bin'].value_counts().sort_index()
    atk_ds_counts = atk_valid['ds_bin'].value_counts().sort_index()
    ds_attack_rate = (atk_ds_counts / opp_ds_counts * 100).fillna(0)
    
    valid_ds = [b for b in ds_labels if b in opp_ds_counts.index and opp_ds_counts[b] >= 20]
    
    x_pos = range(len(valid_ds))
    vals = [ds_attack_rate.get(b, 0) for b in valid_ds]
    
    # Color: red for negative ΔS (attacker weaker), green for positive (attacker stronger)
    colors_ds = []
    for b in valid_ds:
        idx = ds_labels.index(b)
        if idx < 6:  # negative ΔS bins
            colors_ds.append('#E24B4A')
        elif idx == 6:  # near zero
            colors_ds.append('#F0A050')
        else:
            colors_ds.append('#2ECC71')
    
    bars = ax.bar(x_pos, vals, color=colors_ds, width=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_ds, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel(r'$\Delta S = S_{attacker} - S_{target}$', fontsize=12)
    ax.set_ylabel('Attack rate (%)', fontsize=12)
    ax.set_title('Attack probability vs sugar difference\n'
                 r'(red = attacker weaker, green = stronger)', fontsize=12, fontweight='bold')
    
    for i, (bar, v) in enumerate(zip(bars, vals)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + max(max(vals)*0.02, 0.01),
                    f'{v:.2f}%', ha='center', fontsize=7, fontweight='bold')
    
    # Right: same but as a smooth line with confidence
    ax = axes[1]
    
    # Finer bins for smooth curve
    fine_edges = np.arange(-150, 160, 10)
    opp_valid['ds_fine'] = pd.cut(opp_valid['delta_sugar'], bins=fine_edges)
    atk_valid['ds_fine'] = pd.cut(atk_valid['delta_sugar'], bins=fine_edges)
    
    opp_fine = opp_valid['ds_fine'].value_counts().sort_index()
    atk_fine = atk_valid['ds_fine'].value_counts().sort_index()
    rate_fine = (atk_fine / opp_fine * 100).fillna(0)
    
    # Filter bins with enough data and plot
    midpoints = [(e.left + e.right) / 2 for e in rate_fine.index if opp_fine.get(e, 0) >= 30]
    rates = [rate_fine[e] for e in rate_fine.index if opp_fine.get(e, 0) >= 30]
    
    if midpoints:
        ax.plot(midpoints, rates, 'o-', color='#E74C3C', linewidth=2, markersize=5, alpha=0.8)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.fill_between(midpoints, 0, rates, alpha=0.1, color='#E74C3C')
    
    ax.set_xlabel(r'$\Delta S = S_{attacker} - S_{target}$', fontsize=12)
    ax.set_ylabel('Attack rate (%)', fontsize=12)
    ax.set_title('Attack probability curve\n(smoothed, bins with n >= 30)', fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    
    fig.suptitle('Aggression curve: does attack probability increase with sugar advantage?',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_aggression_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_aggression_curve.png")
    
    # ============================================================
    # PLOT 7: Calibration plot — P(win) real vs P(win) theoretical
    # ============================================================
    
    # Theoretical P(win) given ΔS:
    # power = sugar + Uniform(0,1), so P(A wins) = P(U_A - U_B > -ΔS)
    # where U_A - U_B ~ Triangular(-1, 1) with peak at 0
    # CDF: F(x) = (1+x)²/2 for -1<x<=0, F(x) = 1-(1-x)²/2 for 0<x<1
    # P(win) = 1 - F(-ΔS)
    
    def theoretical_pwin(delta_s):
        """P(attacker wins) given delta_s = sugar_attacker - sugar_target."""
        ds = np.asarray(delta_s, dtype=float)
        p = np.zeros_like(ds)
        
        # ΔS >= 1: always win
        p[ds >= 1] = 1.0
        # ΔS <= -1: always lose
        p[ds <= -1] = 0.0
        # 0 <= ΔS < 1
        mask1 = (ds >= 0) & (ds < 1)
        p[mask1] = 1 - (1 - ds[mask1])**2 / 2
        # -1 < ΔS < 0
        mask2 = (ds > -1) & (ds < 0)
        p[mask2] = (1 + ds[mask2])**2 / 2
        
        return p
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: calibration plot
    ax = axes[0]
    
    # Bin attacks by sugar ratio for calibration
    ratio_bins_cal = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 100]
    atk_valid['ratio_bin_cal'] = pd.cut(atk_valid['sugar_ratio'], bins=ratio_bins_cal)
    
    cal_data = []
    for bin_label, group in atk_valid.groupby('ratio_bin_cal'):
        if len(group) < 5:
            continue
        
        # Actual win rate
        actual_wr = (group['outcome'] == 'WIN').mean()
        
        # Theoretical win rate (mean of P(win) for each attack in this bin)
        theoretical_wr = theoretical_pwin(group['delta_sugar'].values).mean()
        
        # Median delta sugar in this bin
        med_ds = group['delta_sugar'].median()
        med_ratio = group['sugar_ratio'].median()
        
        cal_data.append({
            'bin': str(bin_label),
            'actual': actual_wr * 100,
            'theoretical': theoretical_wr * 100,
            'n': len(group),
            'med_ratio': med_ratio,
            'med_ds': med_ds,
        })
    
    if cal_data:
        cal_df = pd.DataFrame(cal_data)
        
        # Perfect calibration line
        ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.4, label='Perfect calibration')
        
        # Points: actual vs theoretical
        sc = ax.scatter(cal_df['theoretical'], cal_df['actual'],
                       s=cal_df['n'] * 3, c=cal_df['med_ratio'], cmap='RdYlGn',
                       vmin=0, vmax=3, edgecolors='black', linewidth=0.5, zorder=5)
        
        plt.colorbar(sc, ax=ax, shrink=0.8, label='Median sugar ratio')
        
        # Label each point
        for _, row in cal_df.iterrows():
            ax.annotate(f"n={row['n']}", (row['theoretical'], row['actual']),
                       fontsize=7, textcoords='offset points', xytext=(5, 5), color='gray')
        
        ax.set_xlabel('Theoretical P(win) %', fontsize=12)
        ax.set_ylabel('Actual win rate %', fontsize=12)
        ax.set_title('Calibration: actual vs theoretical win rate\n'
                     '(point size = sample count, color = sugar ratio)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    
    # Right: where do agents choose to attack on the P(win) curve?
    ax = axes[1]
    
    # Theoretical P(win) for all opportunities vs actual attacks
    opp_pwin = theoretical_pwin(opp_valid['delta_sugar'].values) * 100
    atk_pwin = theoretical_pwin(atk_valid['delta_sugar'].values) * 100
    
    bins_pw = np.linspace(0, 100, 25)
    
    ax.hist(opp_pwin, bins=bins_pw, alpha=0.5, density=True,
            color='#A8A8A8', label=f'All opportunities (n={len(opp_valid):,})')
    ax.hist(atk_pwin, bins=bins_pw, alpha=0.7, density=True,
            color='#E74C3C', label=f'Actual attacks (n={len(atk_valid):,})')
    
    ax.axvline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(np.median(opp_pwin), color='#666666', linestyle=':', linewidth=1.5,
               label=f'Opp. median: {np.median(opp_pwin):.0f}%')
    ax.axvline(np.median(atk_pwin), color='#C0392B', linestyle=':', linewidth=1.5,
               label=f'Attack median: {np.median(atk_pwin):.0f}%')
    
    # % of attacks where theoretical P(win) > 50%
    pct_favorable = (atk_pwin > 50).mean() * 100
    pct_favorable_opp = (opp_pwin > 50).mean() * 100
    
    ax.text(0.97, 0.95,
            f'Attacks with P(win) > 50%: {pct_favorable:.1f}%\n'
            f'Baseline (all opp.): {pct_favorable_opp:.1f}%',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Theoretical P(win) %', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('At what P(win) do agents choose to attack?\n'
                 '(right-shifted = rational/conservative)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    
    fig.suptitle('Combat rationality: do agents attack when they are likely to win?',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_calibration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    combat_calibration.png")
    
    # ============================================================
    # PLOT 8: Attack rate vs z (best visible cell sugar)
    # ============================================================
    
    opp_z = opp_valid[opp_valid['z'].notna() & (opp_valid['z'] >= 0)].copy()
    atk_z = atk_valid[atk_valid['z'].notna() & (atk_valid['z'] >= 0)].copy()
    
    if not opp_z.empty and not atk_z.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: attack rate vs z
        ax = axes[0]
        z_edges = [0, 1, 2, 3, 4, 5, 100]
        z_labels = ['0', '1', '2', '3', '4', '5+']
        
        opp_z['z_bin'] = pd.cut(opp_z['z'], bins=z_edges, labels=z_labels, include_lowest=True)
        atk_z['z_bin'] = pd.cut(atk_z['z'], bins=z_edges, labels=z_labels, include_lowest=True)
        
        opp_z_counts = opp_z['z_bin'].value_counts().sort_index()
        atk_z_counts = atk_z['z_bin'].value_counts().sort_index()
        rate_z = (atk_z_counts / opp_z_counts * 100).fillna(0)
        
        valid_z_bins = [b for b in z_labels if b in opp_z_counts.index and opp_z_counts[b] >= 20]
        
        bars = ax.bar(range(len(valid_z_bins)), [rate_z.get(b, 0) for b in valid_z_bins],
                      color='#5B8DB8', width=0.6)
        ax.set_xticks(range(len(valid_z_bins)))
        ax.set_xticklabels(valid_z_bins, fontsize=10)
        ax.set_xlabel('Best visible cell sugar ($z$)', fontsize=12)
        ax.set_ylabel('Attack rate (%)', fontsize=12)
        ax.set_title('Attack rate vs foraging alternative\n'
                     '(higher $z$ = better non-attack option)', fontsize=12, fontweight='bold')
        
        for i, b in enumerate(valid_z_bins):
            v = rate_z.get(b, 0)
            n = int(opp_z_counts.get(b, 0))
            ax.text(i, v + max(rate_z.max() * 0.02, 0.01), f'{v:.2f}%',
                    ha='center', fontsize=9, fontweight='bold')
            ax.text(i, -max(rate_z.max() * 0.05, 0.02), f'n={n:,}',
                    ha='center', fontsize=7, color='gray')
        
        # Right: attack rate vs (S_target - z), "excess value" of attack
        ax = axes[1]
        opp_z['excess'] = opp_z['target_sugar'] - opp_z['z']
        atk_z['excess'] = atk_z['target_sugar'] - atk_z['z']
        
        ex_edges = [-500, -50, -20, -5, 0, 5, 20, 50, 500]
        ex_labels = ['<-50', '-50:-20', '-20:-5', '-5:0', '0:5', '5:20', '20:50', '>50']
        
        opp_z['ex_bin'] = pd.cut(opp_z['excess'], bins=ex_edges, labels=ex_labels)
        atk_z['ex_bin'] = pd.cut(atk_z['excess'], bins=ex_edges, labels=ex_labels)
        
        opp_ex = opp_z['ex_bin'].value_counts().sort_index()
        atk_ex = atk_z['ex_bin'].value_counts().sort_index()
        rate_ex = (atk_ex / opp_ex * 100).fillna(0)
        
        valid_ex = [b for b in ex_labels if b in opp_ex.index and opp_ex[b] >= 20]
        
        colors_ex = ['#E24B4A' if ex_labels.index(b) < 4 else '#2ECC71' for b in valid_ex]
        
        bars = ax.bar(range(len(valid_ex)), [rate_ex.get(b, 0) for b in valid_ex],
                      color=colors_ex, width=0.6)
        ax.set_xticks(range(len(valid_ex)))
        ax.set_xticklabels(valid_ex, fontsize=8, rotation=30, ha='right')
        ax.set_xlabel(r'$S_{target} - z$ (attack gain vs foraging gain)', fontsize=12)
        ax.set_ylabel('Attack rate (%)', fontsize=12)
        ax.set_title('Attack rate vs excess value\n'
                     r'(red: $S_j < z$, green: $S_j > z$)', fontsize=12, fontweight='bold')
        
        for i, b in enumerate(valid_ex):
            v = rate_ex.get(b, 0)
            if v > 0:
                ax.text(i, v + max(rate_ex.max() * 0.02, 0.01), f'{v:.2f}%',
                        ha='center', fontsize=8, fontweight='bold')
        
        fig.suptitle('Does the foraging alternative influence the attack decision?',
                     fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "combat_vs_foraging.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    combat_vs_foraging.png")
        
        # ============================================================
        # PLOT 9: E[attack] vs E[no attack] scatter
        # ============================================================
        
        # E[attack] = P(win) * S_j - (1-P(win)) * S_i
        # E[no attack] = z
        
        opp_z['delta_sugar'] = opp_z['agent_sugar'] - opp_z['target_sugar']
        atk_z['delta_sugar'] = atk_z['attacker_sugar'] - atk_z['target_sugar']
        
        opp_z['pwin'] = theoretical_pwin(opp_z['delta_sugar'].values)
        atk_z['pwin'] = theoretical_pwin(atk_z['delta_sugar'].values)
        
        opp_z['ev_attack'] = opp_z['pwin'] * opp_z['target_sugar'] - (1 - opp_z['pwin']) * opp_z['agent_sugar']
        atk_z['ev_attack'] = atk_z['pwin'] * atk_z['target_sugar'] - (1 - atk_z['pwin']) * atk_z['attacker_sugar']
        
        opp_z['ev_forage'] = opp_z['z']
        atk_z['ev_forage'] = atk_z['z']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: scatter all opportunities + attacks highlighted
        ax = axes[0]
        
        # Subsample opportunities for visibility
        opp_sample = opp_z.sample(n=min(5000, len(opp_z)), random_state=42)
        
        ax.scatter(opp_sample['ev_forage'], opp_sample['ev_attack'],
                   c='#A8A8A8', alpha=0.15, s=8, edgecolors='none', label='No attack')
        ax.scatter(atk_z['ev_forage'], atk_z['ev_attack'],
                   c='#E74C3C', alpha=0.6, s=20, edgecolors='none', label='Attack')
        
        # Decision boundary: E[attack] = E[no attack]
        lim = max(opp_z['ev_forage'].quantile(0.95), opp_z['ev_attack'].quantile(0.95), 20)
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.4, label='E[attack] = E[forage]')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('E[no attack] = $z$ (best visible sugar)', fontsize=12)
        ax.set_ylabel(r'E[attack] = $P_{win} \cdot S_j - P_{lose} \cdot S_i$', fontsize=12)
        ax.set_title('Expected value: attack vs forage\n(rational = attack above diagonal)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(-1, lim * 0.8)
        ax.set_ylim(-lim * 0.5, lim * 0.8)
        
        # Right: % of attacks that were "rational" (E[attack] > E[forage])
        ax = axes[1]
        
        opp_z['rational'] = opp_z['ev_attack'] > opp_z['ev_forage']
        atk_z['rational'] = atk_z['ev_attack'] > atk_z['ev_forage']
        
        # By delta alpha bin
        atk_z['gap_bin'] = pd.cut(atk_z['alpha_gap'], bins=gap_edges, labels=gap_labels)
        opp_z['gap_bin'] = pd.cut(opp_z['alpha_gap'], bins=gap_edges, labels=gap_labels)
        
        rat_opp = opp_z.groupby('gap_bin')['rational'].mean()
        rat_atk = atk_z.groupby('gap_bin').apply(
            lambda g: g['rational'].mean() if len(g) >= 5 else np.nan
        )
        
        valid_rat = [g for g in gap_labels if g in rat_opp.index and not np.isnan(rat_atk.get(g, np.nan))]
        
        if valid_rat:
            x_r = range(len(valid_rat))
            width = 0.35
            
            ax.bar([i - width/2 for i in x_r],
                   [rat_opp.get(g, 0) for g in valid_rat],
                   width, label='All opportunities', color='#A8A8A8')
            ax.bar([i + width/2 for i in x_r],
                   [rat_atk.get(g, 0) for g in valid_rat],
                   width, label='Actual attacks', color='#E74C3C')
            
            ax.set_xticks(x_r)
            ax.set_xticklabels(valid_rat, fontsize=9, rotation=20)
            ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
            ax.set_ylabel('P(E[attack] > E[forage])', fontsize=12)
            ax.set_title('Fraction of "rational" decisions\nby identity gap',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1.05)
        
        fig.suptitle('Decision-theoretic analysis: is attacking worth more than foraging?',
                     fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "combat_decision_theory.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    combat_decision_theory.png")
        
        # ============================================================
        # PLOT 10: Decision tree rationality — confusion matrix + bars
        # ============================================================
        from scipy.stats import chi2_contingency
        
        # Classify all opportunities
        # rational = True means E[attack] > E[forage] for this pair
        # opp_z['rational'] already computed above
        
        # For opportunities, we need to know which ones resulted in attack
        # We flag opportunities as "attacked" if there's a matching attack
        # Simple approach: compute attack rate in rational vs irrational groups
        
        # Among ALL opportunities
        n_rational = opp_z['rational'].sum()
        n_irrational = (~opp_z['rational']).sum()
        
        # Attack rate = attacks / opportunities in each group
        # We can approximate: overall attack rate * ratio of attacks in each group
        # Better: use the z-binned attack rates we already have
        
        # Since we can't directly link opportunities to attacks 1:1 across dataframes,
        # use the rational flag on attacks (already computed as atk_z['rational'])
        
        n_atk_rational = atk_z['rational'].sum()
        n_atk_irrational = (~atk_z['rational']).sum()
        
        # Attack rates
        rate_rational = n_atk_rational / max(n_rational, 1) * 100
        rate_irrational = n_atk_irrational / max(n_irrational, 1) * 100
        
        # Confusion matrix
        # Rows: Rational / Irrational opportunity
        # Cols: Attacked / Not attacked
        cm = np.array([
            [int(n_atk_rational), int(n_rational - n_atk_rational)],
            [int(n_atk_irrational), int(n_irrational - n_atk_irrational)]
        ])
        
        # Chi-squared
        if cm.min() > 0:
            chi2, p_chi2, _, _ = chi2_contingency(cm)
        else:
            chi2, p_chi2 = 0, 1.0
        
        # Precision and recall
        total_attacks_z = int(n_atk_rational + n_atk_irrational)
        precision = n_atk_rational / max(total_attacks_z, 1)
        recall = n_atk_rational / max(n_rational, 1)
        ratio_rates = rate_rational / max(rate_irrational, 0.001)
        
        # --- Figure: 3 panels ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Panel 1: Decision tree diagram (simplified visual)
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Decision framework', fontsize=13, fontweight='bold')
        
        # Root
        ax.annotate('Agent $i$ adjacent\nto agent $j$',
                    xy=(5, 9.2), fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8', edgecolor='black'))
        
        # Branch A: Attack
        ax.annotate('', xy=(2.5, 7.5), xytext=(5, 8.6),
                    arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5))
        ax.text(3.2, 8.3, 'Attack', fontsize=10, color='#DC2626', fontweight='bold')
        
        ax.annotate('$S_i > S_j$?',
                    xy=(2.5, 7.2), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FECACA', edgecolor='#DC2626'))
        
        # Win
        ax.annotate('', xy=(1, 5.5), xytext=(2, 6.5),
                    arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.2))
        ax.text(1.0, 6.6, 'Yes', fontsize=9, color='#16A34A')
        ax.annotate('Win: gain $S_j$\n$P \\approx 1$',
                    xy=(1, 5.2), fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#DCFCE7', edgecolor='#16A34A'))
        
        # Lose
        ax.annotate('', xy=(4, 5.5), xytext=(3, 6.5),
                    arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.2))
        ax.text(3.7, 6.6, 'No', fontsize=9, color='#DC2626')
        ax.annotate('Lose: die\n$P \\approx 1$',
                    xy=(4, 5.2), fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#DC2626'))
        
        # Branch B: Don't attack
        ax.annotate('', xy=(7.5, 7.5), xytext=(5, 8.6),
                    arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5))
        ax.text(6.5, 8.3, 'Move/Stay', fontsize=10, color='#2563EB', fontweight='bold')
        
        ax.annotate('Gain $z$\n(best visible cell)\n$P = 1$',
                    xy=(7.5, 6.8), fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor='#2563EB'))
        
        # Decision rule
        ax.text(5, 4.0,
                'Rational to attack when:\n'
                r'$E[\mathrm{attack}] > E[\mathrm{forage}]$' '\n'
                r'$P_{\mathrm{win}} \cdot S_j - P_{\mathrm{lose}} \cdot S_i > z$',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9C3', edgecolor='#CA8A04'))
        
        # Panel 2: Confusion matrix
        ax = axes[1]
        cm_pct = np.array([
            [rate_rational, 100 - rate_rational],
            [rate_irrational, 100 - rate_irrational]
        ])
        
        im = ax.imshow(cm_pct, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Attacked', 'Did not attack'], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['E[atk] > E[forage]\n(rational)', 'E[atk] ≤ E[forage]\n(irrational)'], fontsize=10)
        #ax.set_title('Attack rate by rationality of opportunity', fontsize=12, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_pct[i, j]
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{pct:.2f}%\n(n={count:,})',
                        ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.7, label='Rate (%)')
        
        # Panel 3: Attack rate rational vs irrational, by delta alpha
        ax = axes[2]
        
        atk_z_copy = atk_z.copy()
        opp_z_copy = opp_z.copy()
        atk_z_copy['gap_bin_dt'] = pd.cut(atk_z_copy['alpha_gap'], bins=gap_edges, labels=gap_labels)
        opp_z_copy['gap_bin_dt'] = pd.cut(opp_z_copy['alpha_gap'], bins=gap_edges, labels=gap_labels)
        
        rat_rates_by_gap = []
        irr_rates_by_gap = []
        valid_gap_dt = []
        
        for g in gap_labels:
            opp_g = opp_z_copy[opp_z_copy['gap_bin_dt'] == g]
            atk_g = atk_z_copy[atk_z_copy['gap_bin_dt'] == g]
            
            n_rat_opp = opp_g['rational'].sum()
            n_irr_opp = (~opp_g['rational']).sum()
            n_rat_atk = atk_g['rational'].sum() if len(atk_g) > 0 else 0
            n_irr_atk = (~atk_g['rational']).sum() if len(atk_g) > 0 else 0
            
            if n_rat_opp >= 20 and n_irr_opp >= 20:
                valid_gap_dt.append(g)
                rat_rates_by_gap.append(n_rat_atk / n_rat_opp * 100)
                irr_rates_by_gap.append(n_irr_atk / n_irr_opp * 100)
        
        if valid_gap_dt:
            x_dt = range(len(valid_gap_dt))
            width = 0.35
            ax.bar([i - width/2 for i in x_dt], rat_rates_by_gap, width,
                   label='Rational opportunities', color="#3866E4F0")
            ax.bar([i + width/2 for i in x_dt], irr_rates_by_gap, width,
                   label='Irrational opportunities', color="#DECC32")
            ax.set_xticks(x_dt)
            ax.set_xticklabels(valid_gap_dt, fontsize=9, rotation=20)
            ax.set_xlabel(r'$|\Delta\alpha|$', fontsize=12)
            ax.set_ylabel('Attack rate (%)', fontsize=12)
            #ax.set_title('Attack rate: rational vs irrational\nby identity gap', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
        
        fig.suptitle(f'Decision tree rationality analysis\n'
                     f'(precision: {precision*100:.1f}% of attacks were rational, '
                     f'ratio: {ratio_rates:.1f}×, χ²={chi2:.1f}, p={p_chi2:.2e})',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "combat_decision_tree.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    combat_decision_tree.png")
        
        # Print decision tree stats
        print(f"\n  --- DECISION TREE RATIONALITY ---")
        print(f"  Rational opportunities: {int(n_rational):,} / {len(opp_z):,} "
              f"({n_rational/len(opp_z)*100:.1f}%)")
        print(f"  Attack rate when rational:     {rate_rational:.3f}%")
        print(f"  Attack rate when irrational:   {rate_irrational:.3f}%")
        print(f"  Ratio: {ratio_rates:.1f}x")
        print(f"  Precision (% attacks that were rational): {precision*100:.1f}%")
        print(f"  Recall (% rational opp exploited):        {recall*100:.2f}%")
        print(f"  Chi-squared: {chi2:.1f}, p = {p_chi2:.2e}")
    else:
        print("    (skipped decision-theory plots: no z data)")
    
    # ============================================================
    # STATISTICS
    # ============================================================
    print(f"\n  {'='*65}")
    print(f"  STATISTICS BY DELTA ALPHA")
    print(f"  {'='*65}")
    
    from scipy.stats import mannwhitneyu
    
    print(f"\n  {'Gap bin':>10s}  {'Opp n':>8s}  {'Atk n':>7s}  {'Rate%':>7s}  "
          f"{'%Str Opp':>8s}  {'%Str Atk':>8s}  {'Shift':>7s}  {'WinR%':>7s}  {'p(MWU)':>10s}")
    print("  " + "-" * 85)
    
    for g in valid_gaps:
        opp_sub = opp_valid[opp_valid['gap_bin'] == g]
        atk_sub = atk_valid[atk_valid['gap_bin'] == g]
        
        n_opp = len(opp_sub)
        n_atk = len(atk_sub)
        rate = n_atk / n_opp * 100 if n_opp > 0 else 0
        
        pct_str_opp = opp_sub['agent_stronger'].mean() * 100 if n_opp > 0 else np.nan
        pct_str_atk = atk_sub['attacker_stronger'].mean() * 100 if n_atk > 0 else np.nan
        shift = pct_str_atk - pct_str_opp if not np.isnan(pct_str_atk) else np.nan
        
        win_r = (atk_sub['outcome'] == 'WIN').mean() * 100 if n_atk >= 5 else np.nan
        
        # Mann-Whitney U on sugar ratio
        if n_atk >= 5 and n_opp >= 10:
            _, p_mwu = mannwhitneyu(opp_sub['sugar_ratio'].dropna(),
                                     atk_sub['sugar_ratio'].dropna(),
                                     alternative='two-sided')
            p_str = f'{p_mwu:.6f}'
        else:
            p_str = 'n/a'
        
        shift_str = f'{shift:+.1f}pp' if not np.isnan(shift) else 'n/a'
        wr_str = f'{win_r:.1f}' if not np.isnan(win_r) else 'n/a'
        
        print(f"  {g:>10s}  {n_opp:>8,}  {n_atk:>7}  {rate:>6.2f}%  "
              f"{pct_str_opp:>7.1f}%  {pct_str_atk if not np.isnan(pct_str_atk) else 0:>7.1f}%  "
              f"{shift_str:>7s}  {wr_str:>7s}  {p_str:>10s}")
    
    print(f"\n  Shift = (% attacker stronger in attacks) - (% stronger in all opportunities)")
    print(f"  Positive shift → agents select weaker targets at that gap level")
    print(f"  Negative shift → attacks driven by desperation at that gap level")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()