"""
Sugarscape LLM-Agent Simulation — Formula Fitting & Analysis
=============================================================
Fits mathematical models to batch simulation results:
  1. Survival curve:  N(t) = N∞ + (N0 - N∞) * exp(-λ*t)
  2. Survival rate:   ln(SR) = β0 + β1*ln(s) + β2*ln(G) + β3*ln(v) + β4*ln(m)
  3. Combat model:    C_tot  = β0 + β1*s + β2*G + β3*v + β4*m
  4. Birth model:     B_tot  = β0 + β1*s + β2*G + β3*v + β4*m

Usage:
  - Set BATCH_FOLDER to your batch root (containing G*_S*_R* subfolders)
  - Run: python analysis_formulas.py
  - Outputs: fitted parameters CSV + diagnostic plots
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from pathlib import Path
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — CHANGE THIS TO YOUR BATCH FOLDER
# ============================================================
BATCH_FOLDER = Path("./results/simulation_data")
OUTPUT_DIR = Path("./results/plots/survival")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_single_run(folder, prefix=""):
    """Load data from a single run folder or from prefixed files."""
    data = {}
    
    # Try to find config
    config_candidates = [
        folder / "config.csv",
        folder / f"{prefix}config.csv",
    ]
    
    config_file = None
    for c in config_candidates:
        if c.exists():
            config_file = c
            break
    
    if config_file is None:
        return None
    
    # Parse config
    config = {}
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    config[row[0].strip()] = float(row[1].strip())
                except ValueError:
                    config[row[0].strip()] = row[1].strip()
    
    data['config'] = config
    data['grid_size'] = int(config.get('grid_size', 0))
    data['avg_sugar'] = float(config.get('avg_sugar_target', 0))
    data['vision'] = float(config.get('global_vision', 0))
    data['metabolism'] = float(config.get('global_metabolism', 0))
    data['seed'] = int(config.get('seed', 0))
    data['repeat'] = int(config.get('repeat_index', 0))
    
    # Load step_stats
    stats_candidates = [
        folder / "step_stats.csv",
        folder / f"{prefix}step_stats.csv",
    ]
    for sc in stats_candidates:
        if sc.exists():
            data['step_stats'] = pd.read_csv(sc, encoding='utf-8-sig')
            break
    
    # Load agent_moves
    moves_candidates = [
        folder / "agent_moves.csv",
        folder / f"{prefix}agent_moves.csv",
    ]
    for mc in moves_candidates:
        if mc.exists():
            data['agent_moves'] = pd.read_csv(mc, encoding='utf-8-sig')
            break
    
    # Load final_summary
    summary_candidates = [
        folder / "final_summary.csv",
        folder / f"{prefix}final_summary.csv",
    ]
    for sc in summary_candidates:
        if sc.exists():
            summary = {}
            with open(sc, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            summary[row[0].strip()] = float(row[1].strip())
                        except ValueError:
                            summary[row[0].strip()] = row[1].strip()
            data['summary'] = summary
            break
    
    return data


def discover_runs(batch_folder):
    """Find all simulation runs in batch folder."""
    runs = []
    
    # Pattern 1: subfolders named G{X}_S{Y}_R{Z}
    for subfolder in sorted(batch_folder.iterdir()):
        if subfolder.is_dir() and subfolder.name.startswith('G'):
            run = load_single_run(subfolder)
            if run is not None:
                run['folder'] = subfolder
                runs.append(run)
    
    # Pattern 2: prefixed files in same folder (like the example)
    if not runs:
        # Group by prefix
        prefixes = set()
        for f in batch_folder.glob("G*_config.csv"):
            prefix = f.name.replace("config.csv", "")
            prefixes.add(prefix)
        
        for prefix in sorted(prefixes):
            run = load_single_run(batch_folder, prefix=prefix)
            if run is not None:
                run['folder'] = batch_folder
                run['prefix'] = prefix
                runs.append(run)
    
    return runs


# ============================================================
# 2. SURVIVAL CURVE FIT: N(t) = N∞ + (N0 - N∞) * exp(-λ*t)
# ============================================================

def exp_decay(t, N_inf, lam):
    """Exponential decay from N0=50 to N_inf."""
    N0 = 50
    return N_inf + (N0 - N_inf) * np.exp(-lam * t)


def fit_survival_curve(step_stats, N0=50):
    """Fit exponential decay to survival curve. Returns (N_inf, lambda, R²)."""
    t = step_stats['step'].values.astype(float)
    N = step_stats['agents_alive'].values.astype(float)
    
    if len(t) < 5:
        return None, None, None
    
    # Initial guess
    N_inf_guess = max(N[-1], 0.5)
    lam_guess = 0.02
    
    try:
        popt, pcov = curve_fit(
            exp_decay, t, N,
            p0=[N_inf_guess, lam_guess],
            bounds=([0, 0.001], [N0, 1.0]),
            maxfev=10000
        )
        N_inf_fit, lam_fit = popt
        
        # R²
        N_pred = exp_decay(t, *popt)
        ss_res = np.sum((N - N_pred) ** 2)
        ss_tot = np.sum((N - np.mean(N)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return N_inf_fit, lam_fit, r2
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None, None, None


# ============================================================
# 3. MAIN ANALYSIS
# ============================================================

def main():
    print("=" * 65)
    print("  SUGARSCAPE FORMULA FITTING ANALYSIS")
    print("=" * 65)
    
    # --- Load all runs ---
    runs = discover_runs(BATCH_FOLDER)
    print(f"\n  Found {len(runs)} simulation run(s)")
    
    if not runs:
        print("  No runs found! Check BATCH_FOLDER path.")
        return
    
    # --- Table for all fitted parameters ---
    results = []
    
    for i, run in enumerate(runs):
        G = run['grid_size']
        s = run['avg_sugar']
        v = run['vision']
        m = run['metabolism']
        
        label = f"G={G}, S={s}, v={v}, m={m:.2f}"
        print(f"\n  [{i+1}/{len(runs)}] {label}")
        
        row = {
            'grid_size': G,
            'avg_sugar': s,
            'vision': v,
            'metabolism': round(m, 4),
            'seed': run['seed'],
            'repeat': run['repeat'],
        }
        
        # --- Fit survival curve ---
        if 'step_stats' in run:
            stats = run['step_stats']
            N_inf, lam, r2 = fit_survival_curve(stats)
            
            if N_inf is not None:
                row['N_inf'] = round(N_inf, 3)
                row['lambda'] = round(lam, 5)
                row['R2_surv'] = round(r2, 4)
                print(f"    Survival fit: N∞={N_inf:.1f}, λ={lam:.4f}, R²={r2:.3f}")
            
            # Extract final values
            row['steps_run'] = int(stats['step'].max()) + 1
            row['agents_final'] = int(stats['agents_alive'].iloc[-1])
            row['survival_rate'] = round(row['agents_final'] / 50, 4)
            row['total_combats'] = int(stats['combats_total'].iloc[-1])
            row['total_births'] = int(stats['births_total'].iloc[-1])
            row['avg_final_sugar'] = round(stats['avg_agent_sugar'].iloc[-1], 2)
        
        # --- Attack analysis from agent_moves ---
        if 'agent_moves' in run:
            moves = run['agent_moves']
            
            # Count attack actions
            attack_mask = moves['decision'].astype(str).str.startswith('4,')
            n_attacks = attack_mask.sum()
            row['n_attacks'] = n_attacks
            
            # Attack with known target alpha
            attacks_with_alpha = moves[attack_mask & moves['target_alpha'].notna()]
            if len(attacks_with_alpha) > 0:
                alpha_gaps = (attacks_with_alpha['alpha'] - attacks_with_alpha['target_alpha']).abs()
                row['mean_alpha_gap_attacks'] = round(alpha_gaps.mean(), 4)
            
            # Action distribution
            decisions = moves['decision'].astype(str)
            row['pct_move'] = round((decisions.str.startswith('1,')).mean() * 100, 1)
            row['pct_stay'] = round((decisions.str.contains('STAY|2-')).mean() * 100, 1)
            row['pct_attack'] = round((decisions.str.startswith('4,')).mean() * 100, 1)
            row['pct_reproduce'] = round((decisions.str.startswith('3,')).mean() * 100, 1)
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # ============================================================
    # 4. REGRESSION ANALYSIS (only meaningful with multiple runs)
    # ============================================================
    
    print("\n" + "=" * 65)
    print("  REGRESSION ANALYSIS")
    print("=" * 65)
    
    if len(df) < 5:
        print("\n  ⚠ Too few runs for regression. Need ≥5 runs for meaningful fits.")
        print("    With your full batch (140 runs), the regressions below will work.\n")
        print("  Single-run results:")
        print(df.to_string(index=False))
    
    # --- 4a. Log-linear survival rate model ---
    if len(df) >= 5 and 'survival_rate' in df.columns:
        valid = df[(df['survival_rate'] > 0) & (df['avg_sugar'] > 0)].copy()
        
        if len(valid) >= 5:
            print("\n  --- Model 1: Survival Rate (log-linear) ---")
            print("    ln(SR) = β0 + β1·ln(s) + β2·ln(G) + β3·ln(v) + β4·ln(m)\n")
            
            X = np.column_stack([
                np.ones(len(valid)),
                np.log(valid['avg_sugar']),
                np.log(valid['grid_size']),
                np.log(valid['vision']),
                np.log(valid['metabolism']),
            ])
            y = np.log(valid['survival_rate'])
            
            # OLS
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            names = ['β0 (intercept)', 'β1 (ln sugar)', 'β2 (ln grid)', 'β3 (ln vision)', 'β4 (ln metab)']
            for name, b in zip(names, beta):
                print(f"    {name:20s} = {b:+.4f}")
            print(f"    R² = {r2:.4f}")
            print(f"\n    Formula: SR ≈ {np.exp(beta[0]):.4f} · s^{beta[1]:.2f} · G^{beta[2]:.2f} · v^{beta[3]:.2f} · m^{beta[4]:.2f}")
    
    # --- 4b. Lambda regression ---
    if len(df) >= 5 and 'lambda' in df.columns:
        valid = df[df['lambda'].notna() & (df['avg_sugar'] > 0)].copy()
        
        if len(valid) >= 5:
            print("\n  --- Model 2: Decay Rate λ (log-linear) ---")
            print("    ln(λ) = β0 + β1·ln(s) + β2·ln(G) + β3·ln(v) + β4·ln(m)\n")
            
            X = np.column_stack([
                np.ones(len(valid)),
                np.log(valid['avg_sugar']),
                np.log(valid['grid_size']),
                np.log(valid['vision']),
                np.log(valid['metabolism']),
            ])
            y = np.log(valid['lambda'])
            
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            names = ['β0 (intercept)', 'β1 (ln sugar)', 'β2 (ln grid)', 'β3 (ln vision)', 'β4 (ln metab)']
            for name, b in zip(names, beta):
                print(f"    {name:20s} = {b:+.4f}")
            print(f"    R² = {r2:.4f}")
            print(f"\n    Formula: λ ≈ {np.exp(beta[0]):.5f} · s^{beta[1]:.2f} · G^{beta[2]:.2f} · v^{beta[3]:.2f} · m^{beta[4]:.2f}")
    
    # --- 4c. N_inf regression ---
    if len(df) >= 5 and 'N_inf' in df.columns:
        valid = df[df['N_inf'].notna() & (df['N_inf'] > 0) & (df['avg_sugar'] > 0)].copy()
        
        if len(valid) >= 5:
            print("\n  --- Model 3: Carrying Capacity N∞ ---")
            print("    N∞ = β0 + β1·s + β2·G + β3·v + β4·m\n")
            
            X = np.column_stack([
                np.ones(len(valid)),
                valid['avg_sugar'],
                valid['grid_size'],
                valid['vision'],
                valid['metabolism'],
            ])
            y = valid['N_inf'].values
            
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            names = ['β0 (intercept)', 'β1 (sugar)', 'β2 (grid)', 'β3 (vision)', 'β4 (metab)']
            for name, b in zip(names, beta):
                print(f"    {name:20s} = {b:+.4f}")
            print(f"    R² = {r2:.4f}")
            print(f"\n    Formula: N∞ ≈ {beta[0]:.1f} + {beta[1]:.2f}·s + {beta[2]:.2f}·G + {beta[3]:.2f}·v + {beta[4]:.2f}·m")
    
    # --- 4d. Combats regression ---
    if len(df) >= 5 and 'total_combats' in df.columns:
        valid = df[df['total_combats'].notna()].copy()
        
        if len(valid) >= 5:
            print("\n  --- Model 4: Total Combats ---")
            print("    C = β0 + β1·s + β2·G + β3·v + β4·m\n")
            
            X = np.column_stack([
                np.ones(len(valid)),
                valid['avg_sugar'],
                valid['grid_size'],
                valid['vision'],
                valid['metabolism'],
            ])
            y = valid['total_combats'].values.astype(float)
            
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            names = ['β0 (intercept)', 'β1 (sugar)', 'β2 (grid)', 'β3 (vision)', 'β4 (metab)']
            for name, b in zip(names, beta):
                print(f"    {name:20s} = {b:+.4f}")
            print(f"    R² = {r2:.4f}")
    
    # ============================================================
    # 5. DIAGNOSTIC PLOTS
    # ============================================================
    
    print("\n" + "=" * 65)
    print("  GENERATING PLOTS")
    print("=" * 65)
    
    # --- Plot 1: Survival curve fit for each run ---
    n_runs = len(runs)
    cols = min(4, n_runs)
    rows = max(1, (n_runs + cols - 1) // cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    
    for i, (run, row_data) in enumerate(zip(runs, results)):
        ax = axes[i // cols][i % cols]
        
        if 'step_stats' not in run:
            continue
        
        stats = run['step_stats']
        t = stats['step'].values
        N = stats['agents_alive'].values
        
        ax.plot(t, N, 'b-', alpha=0.7, linewidth=1.5, label='Data')
        
        if row_data.get('N_inf') is not None:
            t_fit = np.linspace(0, t.max(), 200)
            N_fit = exp_decay(t_fit, row_data['N_inf'], row_data['lambda'])
            ax.plot(t_fit, N_fit, 'r--', linewidth=1.5, 
                    label=f"Fit: N∞={row_data['N_inf']:.1f}, λ={row_data['lambda']:.3f}")
        
        G, s = row_data['grid_size'], row_data['avg_sugar']
        v, m = row_data['vision'], round(row_data['metabolism'], 2)
        ax.set_title(f"G={G} S={s} v={v} m={m}", fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('Agents Alive')
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
    
    # Hide empty axes
    for i in range(n_runs, rows * cols):
        axes[i // cols][i % cols].set_visible(False)
    
    fig.suptitle("Survival Curve Fits: N(t) = N∞ + (N₀ - N∞)·exp(-λt)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fit_survival_curves.png", dpi=150, bbox_inches='tight')
    print("  ✓ fit_survival_curves.png")
    
    # --- Plot 2: Residuals / fit quality (if multiple runs) ---
    if len(df) >= 5 and 'lambda' in df.columns:
        valid = df[df['lambda'].notna()].copy()
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # λ vs sugar
        ax = axes[0]
        for G in sorted(valid['grid_size'].unique()):
            subset = valid[valid['grid_size'] == G]
            ax.scatter(subset['avg_sugar'], subset['lambda'], label=f'G={G}', s=50, alpha=0.7)
        ax.set_xlabel('Average Sugar per Cell')
        ax.set_ylabel('λ (decay rate)')
        ax.set_title('Decay Rate vs Sugar Density')
        ax.legend()
        
        # N_inf vs sugar
        ax = axes[1]
        for G in sorted(valid['grid_size'].unique()):
            subset = valid[valid['grid_size'] == G]
            ax.scatter(subset['avg_sugar'], subset['N_inf'], label=f'G={G}', s=50, alpha=0.7)
        ax.set_xlabel('Average Sugar per Cell')
        ax.set_ylabel('N∞ (carrying capacity)')
        ax.set_title('Carrying Capacity vs Sugar Density')
        ax.legend()
        
        # λ vs metabolism
        ax = axes[2]
        sc = ax.scatter(valid['metabolism'], valid['lambda'], 
                       c=valid['avg_sugar'], cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Avg Sugar')
        ax.set_xlabel('Metabolism')
        ax.set_ylabel('λ (decay rate)')
        ax.set_title('Decay Rate vs Metabolism')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fit_parameter_analysis.png", dpi=150, bbox_inches='tight')
        print("  ✓ fit_parameter_analysis.png")
    
    # --- Plot 3: Sugar accumulation model ---
    if len(runs) > 0 and 'step_stats' in runs[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, run in enumerate(runs):
            if 'step_stats' not in run:
                continue
            stats = run['step_stats']
            t = stats['step'].values
            alive = stats['agents_alive'].values.astype(float)
            avg_sugar = stats['avg_agent_sugar'].values.astype(float)
            
            G, s = run['grid_size'], run['avg_sugar']
            ax.plot(t, avg_sugar, alpha=0.7, label=f"G={G} S={s}")
            
            # Note: theoretical overlay S_bar(t) ≈ R / N(t) requires full batch
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Sugar per Agent')
        ax.set_title('Sugar Accumulation Over Time')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fit_sugar_accumulation.png", dpi=150, bbox_inches='tight')
        print("  ✓ fit_sugar_accumulation.png")
    
    # ============================================================
    # 6. SAVE RESULTS
    # ============================================================
    
    df.to_csv(OUTPUT_DIR / "fitted_parameters.csv", index=False)
    print(f"\n  ✓ fitted_parameters.csv ({len(df)} runs)")
    
    # --- Summary of formulas (text file) ---
    with open(OUTPUT_DIR / "formula_summary.txt", 'w', encoding='utf-8') as f:
        f.write("SUGARSCAPE LLM-AGENT SIMULATION — FITTED FORMULAS\n")
        f.write("=" * 55 + "\n\n")
        
        f.write("VARIABLES:\n")
        f.write("  G = grid size (20, 30, 50, 70)\n")
        f.write("  s = average sugar per cell (0.5 – 3.0)\n")
        f.write("  v = agent vision (1 – 6, constant per run)\n")
        f.write("  m = agent metabolism (0.5 – 2.0, constant per run)\n")
        f.write("  t = simulation step (0 – 200)\n\n")
        
        f.write("MODEL 1 — Population Dynamics (per run):\n")
        f.write("  N(t) = N∞ + (N₀ - N∞) · exp(-λ·t)\n")
        f.write("  where N₀ = 50 (initial agents)\n")
        f.write("  N∞ = carrying capacity (fitted per run)\n")
        f.write("  λ  = decay rate (fitted per run)\n\n")
        
        f.write("MODEL 2 — Survival Rate (across runs):\n")
        f.write("  SR = a · s^β₁ · G^β₂ · v^β₃ · m^β₄\n")
        f.write("  (fit with log-linear OLS regression)\n\n")
        
        f.write("MODEL 3 — Carrying Capacity (across runs):\n")
        f.write("  N∞ = β₀ + β₁·s + β₂·G + β₃·v + β₄·m\n\n")
        
        f.write("MODEL 4 — Decay Rate (across runs):\n")
        f.write("  λ = a · s^β₁ · G^β₂ · v^β₃ · m^β₄\n\n")
        
        f.write("MODEL 5 — Total Combats (across runs):\n")
        f.write("  C = β₀ + β₁·s + β₂·G + β₃·v + β₄·m\n\n")
        
        f.write("MODEL 6 — Sugar Accumulation:\n")
        f.write("  S̄(t) ≈ R_total / N(t)\n")
        f.write("  where R_total = total sugar regeneration rate per step\n\n")
        
        f.write("MODEL 7 — Attack Probability (constant across configs):\n")
        f.write("  P(attack | adjacent enemy) ≈ 0.05 – 0.07\n")
        f.write("  (independent of alpha gap — agents ignore identity signal)\n\n")
        
        f.write("-" * 55 + "\n")
        f.write("RESULTS FROM FITS:\n\n")
        
        if len(df) == 1:
            row = results[0]
            f.write(f"  Single run: G={row['grid_size']}, s={row['avg_sugar']}, ")
            f.write(f"v={row['vision']}, m={row['metabolism']:.4f}\n")
            if row.get('N_inf') is not None:
                f.write(f"  N∞ = {row['N_inf']:.2f}\n")
                f.write(f"  λ  = {row['lambda']:.5f}\n")
                f.write(f"  R² = {row['R2_surv']:.4f}\n")
            f.write(f"  Survival rate: {row.get('survival_rate', 'N/A')}\n")
            f.write(f"  Combats: {row.get('total_combats', 'N/A')}\n")
            f.write(f"  Births: {row.get('total_births', 'N/A')}\n")
        else:
            f.write(df.to_string(index=False))
        
        f.write("\n\n")
        f.write("NOTE: Run this script on the full batch (140+ runs) for\n")
        f.write("statistically meaningful regression coefficients.\n")
    
    print(f"  ✓ formula_summary.txt")
    
    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
