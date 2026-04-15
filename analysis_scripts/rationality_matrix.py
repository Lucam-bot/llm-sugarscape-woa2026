"""
Combat Rationality Decision Tree Analysis
==========================================
Evaluates whether LLM agents attack rationally by comparing the 
Expected Value of attacking vs the Expected Value of foraging (z).
Generates the Decision Matrix and rationality statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from pathlib import Path
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BATCH_FOLDER = Path(r"./results")
OUTPUT_DIR = Path(r"./results\Analysis_Results_Rationality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

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

def compute_p_win(sugar_a, sugar_b):
    """
    Combat mechanic: power = sugar + U(0,1)
    Calculates the exact theoretical probability of A winning.
    """
    delta = sugar_a - sugar_b
    if delta >= 1:
        return 1.0
    elif delta <= -1:
        return 0.0
    else:
        x = -delta 
        if x <= 0:
            return 1.0 - 0.5 * (1 + x) ** 2
        else:
            return 0.5 * (1 - x) ** 2

def process_run(folder, prefix=""):
    """Extracts all adjacency opportunities and marks actual attacks."""
    config_file = folder / f"{prefix}config.csv"
    moves_file = folder / f"{prefix}agent_moves.csv"
    map_file = folder / f"{prefix}map_state.csv"
    
    if not config_file.exists() or not moves_file.exists():
        return None
        
    config = _read_config(config_file)
    grid_size = int(config.get('grid_size', 20))
    vision = int(config.get('global_vision', 4))
    
    df = pd.read_csv(moves_file, encoding='utf-8-sig')
    
    sugar_maps = {}
    if map_file.exists():
        map_df = pd.read_csv(map_file, encoding='utf-8-sig')
        for step, grp in map_df.groupby('step'):
            grid = np.zeros((grid_size, grid_size), dtype=int)
            for _, r in grp.iterrows():
                grid[int(r['y']), int(r['x'])] = int(r['sugar'])
            sugar_maps[step] = grid

    prev_sugar = {}
    for _, row in df.iterrows():
        prev_sugar[(row['step'] + 1, row['agent_id'])] = row['sugar']
        
    opportunities = []
    
    for step in df['step'].unique():
        step_agents = df[df['step'] == step].drop_duplicates('agent_id')
        agents = step_agents[['agent_id', 'x', 'y', 'alpha', 'sugar', 'decision', 'target_alpha']].values
        n = len(agents)
        
        occupied = set((int(a[1]), int(a[2])) for a in agents)
        smap = sugar_maps.get(step, sugar_maps.get(step - 1, None))
        
        z_cache = {}
        if smap is not None:
            for k in range(n):
                aid_k, xk, yk = int(agents[k][0]), int(agents[k][1]), int(agents[k][2])
                max_s = 0
                for dx in range(-vision, vision + 1):
                    for dy in range(-vision, vision + 1):
                        if dx == 0 and dy == 0: continue
                        if dx*dx + dy*dy <= vision*vision:
                            nx, ny = xk + dx, yk + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                if (nx, ny) not in occupied:
                                    max_s = max(max_s, smap[ny, nx])
                z_cache[aid_k] = max_s

        for i in range(n):
            aid_i, xi, yi, alpha_i, sugar_i, dec_i, target_alpha_i = agents[i]
            aid_i = int(aid_i)
            pre_sugar_i = prev_sugar.get((step, aid_i), sugar_i)
            z_i = z_cache.get(aid_i, np.nan)
            
            # Check if agent attacked this turn
            attacked_this_turn = str(dec_i).startswith('4,')
            
            for j in range(n):
                if i == j: continue
                aid_j, xj, yj, alpha_j, sugar_j, _, _ = agents[j]
                
                if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                    pre_sugar_j = prev_sugar.get((step, int(aid_j)), sugar_j)
                    
                    # Did agent i attack agent j specifically?
                    did_attack_j = False
                    if attacked_this_turn and pd.notna(target_alpha_i):
                        if abs(alpha_j - target_alpha_i) < 0.005:
                            did_attack_j = True
                            
                    opportunities.append({
                        'step': step,
                        'agent_sugar': pre_sugar_i,
                        'target_sugar': pre_sugar_j,
                        'alpha_gap': abs(alpha_i - alpha_j),
                        'z': z_i,
                        'did_attack': did_attack_j
                    })
                    
    return pd.DataFrame(opportunities)

def load_all(batch_folder):
    all_opp = []
    folders = sorted([f for f in batch_folder.iterdir() if f.is_dir() and f.name.startswith('G')])
    if not folders:
        prefixes = sorted(list(set(f.name.replace("agent_moves.csv", "").replace("config.csv", "") for f in batch_folder.glob("G*_config.csv"))))
        sources = [(batch_folder, p) for p in prefixes]
    else:
        sources = [(f, "") for f in folders]
        
    for i, (folder, prefix) in enumerate(sources):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Processing run {i+1}/{len(sources)}")
        df = process_run(folder, prefix)
        if df is not None and not df.empty:
            all_opp.append(df)
            
    return pd.concat(all_opp, ignore_index=True) if all_opp else pd.DataFrame()

# ============================================================
# MAIN ANALYSIS & PLOTTING
# ============================================================

def main():
    print("=" * 60)
    print("  DECISION TREE & COMBAT RATIONALITY")
    print("=" * 60)
    
    opp = load_all(BATCH_FOLDER)
    if opp.empty:
        print("  No data found!")
        return
        
    # Filter valid opportunities (need both z and valid sugar)
    df = opp[opp['z'].notna() & opp['agent_sugar'].notna() & opp['target_sugar'].notna()].copy()
    
    # 1. Compute Expected Values
    df['p_win'] = df.apply(lambda r: compute_p_win(r['agent_sugar'], r['target_sugar']), axis=1)
    df['ev_attack'] = df['p_win'] * df['target_sugar'] - (1 - df['p_win']) * df['agent_sugar']
    df['ev_forage'] = df['z']
    
    # 2. Determine Rationality
    df['is_rational'] = df['ev_attack'] > df['ev_forage']
    
    # 3. Calculate Confusion Matrix Stats
    rat_atk = len(df[(df['is_rational'] == True) & (df['did_attack'] == True)])
    rat_no = len(df[(df['is_rational'] == True) & (df['did_attack'] == False)])
    irr_atk = len(df[(df['is_rational'] == False) & (df['did_attack'] == True)])
    irr_no = len(df[(df['is_rational'] == False) & (df['did_attack'] == False)])
    
    total_rat = rat_atk + rat_no
    total_irr = irr_atk + irr_no
    total_atk = rat_atk + irr_atk
    
    rate_when_rat = rat_atk / total_rat if total_rat > 0 else 0
    rate_when_irr = irr_atk / total_irr if total_irr > 0 else 0
    precision = rat_atk / total_atk if total_atk > 0 else 0
    
    # Chi-Squared Test
    contingency = np.array([[rat_atk, rat_no], [irr_atk, irr_no]])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    
    # Print Console Report
    print(f"\n  Total evaluated opportunities: {len(df):,}")
    print(f"  Rational opportunities (EV_atk > z): {total_rat:,} ({(total_rat/len(df)*100):.1f}%)")
    print(f"\n  Attack rate when RATIONAL:     {rate_when_rat*100:.2f}%")
    print(f"  Attack rate when IRRATIONAL:   {rate_when_irr*100:.2f}%")
    print(f"  Ratio (Rat / Irr):             {rate_when_rat/rate_when_irr if rate_when_irr>0 else float('inf'):.1f}x")
    print(f"  Chi-squared: {chi2:.1f}, p-value = {p_value:.2e}")
    print(f"\n  Among {total_atk:,} actual attacks, {precision*100:.1f}% were rational.")
    
    # ============================================================
    # PLOT 1: Confusion Matrix
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    matrix = np.array([[rat_atk, rat_no], [irr_atk, irr_no]])
    
    # Custom color mapping for the matrix
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', alpha=0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Attacked\n(Action Taken)', 'Stayed/Moved\n(Action Avoided)'], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Rational\n($EV_{attack} > EV_{forage}$)', 'Irrational\n($EV_{attack} < EV_{forage}$)'], fontsize=11)
    
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            tot_row = sum(matrix[i])
            pct = (val / tot_row * 100) if tot_row > 0 else 0
            text_color = 'white' if val > matrix.max()/2 else 'black'
            ax.text(j, i, f'{val:,}\n({pct:.1f}% of row)', ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=12)

    ax.set_title('Combat Decision Rationality Matrix', fontsize=13, fontweight='bold')
    
    # ============================================================
    # PLOT 2: Attack Rate by Rationality and Alpha Gap
    # ============================================================
    ax = axes[1]
    
    gap_edges = [0, 0.4, 0.8, 1.3, 2.0]
    gap_labels = ['0-0.4', '0.4-0.8', '0.8-1.3', '1.3+']
    df['gap_bin'] = pd.cut(df['alpha_gap'], bins=gap_edges, labels=gap_labels)
    
    rat_rates = []
    irr_rates = []
    
    for g in gap_labels:
        sub = df[df['gap_bin'] == g]
        if len(sub) > 0:
            rat_sub = sub[sub['is_rational'] == True]
            irr_sub = sub[sub['is_rational'] == False]
            
            r_rate = rat_sub['did_attack'].mean() if len(rat_sub) > 0 else 0
            i_rate = irr_sub['did_attack'].mean() if len(irr_sub) > 0 else 0
            
            rat_rates.append(r_rate)
            irr_rates.append(i_rate)
        else:
            rat_rates.append(0)
            irr_rates.append(0)
            
    x = np.arange(len(gap_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rat_rates, width, label='When Rational ($EV_{atk} > z$)', color='#2ECC71')
    bars2 = ax.bar(x + width/2, irr_rates, width, label='When Irrational ($EV_{atk} < z$)', color='#E74C3C')
    
    ax.set_xticks(x)
    ax.set_xticklabels(gap_labels, fontsize=10)
    ax.set_xlabel(r'Identity Gap ($|\Delta\alpha|$)', fontsize=12)
    ax.set_ylabel('Attack Rate (probability)', fontsize=12)
    ax.set_title('Attack Probability: Rational vs Irrational Scenarios', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    fig.suptitle('Expected Value Analysis: Do LLM agents weigh risk against foraging?', 
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_rationality_decision_tree.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to: {OUTPUT_DIR / 'combat_rationality_decision_tree.png'}")

    # ============================================================
    # PLOT 3: Target Estimation Error (S_j vs z) by Delta Alpha
    # ============================================================
    # Definizione matematica dell'errore suggerita dal relatore:
    # S_effettivo = target_sugar
    # S_stimato = z (la soglia minima per cui l'attacco ha senso economico)
    # Errore (Delta S_j) = S_effettivo - S_stimato
    
    atk_df = df[df['did_attack'] == True].copy()
    atk_df['estimation_error'] = atk_df['target_sugar'] - atk_df['z']
    atk_df['abs_estimation_error'] = atk_df['estimation_error'].abs()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Pannello 1: S_stimato vs S_effettivo (Scatter) ---
    ax = axes[0]
    # Usiamo un sample per evitare overplotting
    sample_df = atk_df.sample(min(2000, len(atk_df))) if len(atk_df) > 2000 else atk_df
    
    ax.scatter(sample_df['z'], sample_df['target_sugar'], 
               c='#E74C3C', alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Calcoliamo i limiti effettivi separati per i due assi
    z_max = sample_df['z'].max()
    # Usiamo il 95° percentile per Y per evitare che un singolo outlier schiacci il grafico
    y_max = sample_df['target_sugar'].quantile(0.95) 
    
    # Linea di "Stima Perfetta" (S_effettivo = S_stimato = z)
    # La disegniamo solo fino a z_max
    ax.plot([0, z_max], [0, z_max], 'k--', linewidth=1.5, label='Perfect Estimation Boundary ($S_j = z$)')
    
    # Evidenziamo le zone
    # Zona Rossa: S_j < z (Il bersaglio aveva meno zucchero della cella vuota = Errore fatale)
    ax.fill_between([0, z_max], 0, [0, z_max], color='red', alpha=0.1, label='Overestimated Target ($S_j < z$)')
    # Zona Verde: S_j > z (Il bersaglio era più ricco della cella vuota)
    ax.fill_between([0, z_max], [0, z_max], y_max * 1.2, color='green', alpha=0.1, label='Underestimated Target ($S_j > z$)')
    
    ax.set_xlim(-0.2, z_max + 0.5)
    ax.set_ylim(-0.5, y_max)
    
    ax.set_xlabel('Estimated Minimum Target Sugar ($\hat{S}_j = z$)', fontsize=12)
    ax.set_ylabel('Actual Target Sugar ($S_j$)', fontsize=12)
    ax.set_title('Target Wealth Estimation\n(Actual vs Implicit Estimate)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
        
    plt.suptitle('Target Sugar Estimation Analysis', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combat_estimation_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to: {OUTPUT_DIR / 'combat_estimation_error.png'}")

if __name__ == "__main__":
    main()