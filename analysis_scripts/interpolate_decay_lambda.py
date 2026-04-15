"""
Interpolate missing λ values and generate complete heatmap.
Uses 2D interpolation from existing (grid, sugar) → λ data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = Path(r"./results\analysis_output\fitted_parameters.csv")
OUTPUT_DIR = Path(r"./results\analysis_output")

# ============================================================
# LOAD DATA & COMPUTE MEAN λ PER (grid, sugar)
# ============================================================
import pandas as pd

df = pd.read_csv(INPUT_CSV)
df = df[df['lambda'].notna()]  # drop failed fits

# Mean lambda per (grid_size, avg_sugar)
grouped = df.groupby(['grid_size', 'avg_sugar'])['lambda'].mean()

observed = {(int(g), float(s)): v for (g, s), v in grouped.items()}

print(f"Loaded {len(df)} runs from {INPUT_CSV.name}")
print(f"Computed mean λ for {len(observed)} (grid, sugar) combinations\n")

# Separate into arrays
points = np.array(list(observed.keys()), dtype=float)  # (N, 2)
values = np.array(list(observed.values()), dtype=float)  # (N,)

# ============================================================
# GRID TO FILL (auto-detect from data)
# ============================================================
grid_sizes = sorted(df['grid_size'].unique().astype(int).tolist())
sugar_values = sorted(df['avg_sugar'].unique().tolist())

# All combinations we want
target_points = np.array([(g, s) for g in grid_sizes for s in sugar_values], dtype=float)

# ============================================================
# METHOD 1: RBF Interpolation (smooth, handles extrapolation)
# ============================================================
# Work in log-space since λ decays roughly exponentially
log_values = np.log(values)

rbf = RBFInterpolator(points, log_values, kernel='thin_plate_spline', smoothing=0.1)
log_predicted = rbf(target_points)
predicted_rbf = np.exp(log_predicted)

# ============================================================
# METHOD 2: Linear griddata (for comparison, NaN outside convex hull)
# ============================================================
predicted_linear = griddata(points, values, target_points, method='linear')

# ============================================================
# BUILD RESULT TABLE
# ============================================================
print("=" * 70)
print("  LAMBDA INTERPOLATION RESULTS")
print("=" * 70)
print(f"\n  {'Grid':>5s}  {'Sugar':>6s}  {'Observed':>10s}  {'RBF Interp':>10s}  {'Linear':>10s}  {'Source':>12s}")
print("  " + "-" * 60)

result_matrix = np.full((len(grid_sizes), len(sugar_values)), np.nan)

for idx, (g, s) in enumerate(target_points):
    g_int, s_float = int(g), float(s)
    gi = grid_sizes.index(g_int)
    si = sugar_values.index(s_float)
    
    obs = observed.get((g_int, s_float))
    rbf_val = predicted_rbf[idx]
    lin_val = predicted_linear[idx] if not np.isnan(predicted_linear[idx]) else None
    
    if obs is not None:
        source = "OBSERVED"
        result_matrix[gi, si] = obs
        # Show how close interpolation is to observed (validation)
        err_pct = abs(rbf_val - obs) / obs * 100
        lin_str = f"{lin_val:.4f}" if lin_val is not None else "N/A"
        print(f"  {g_int:5d}  {s_float:6.1f}  {obs:10.4f}  {rbf_val:10.4f}  "
              f"{lin_str:>10s}  {source} (err={err_pct:.1f}%)")
    else:
        source = "INTERPOLATED"
        result_matrix[gi, si] = rbf_val
        lin_str = f"{lin_val:.4f}" if lin_val else "N/A"
        print(f"  {g_int:5d}  {s_float:6.1f}  {'---':>10s}  {rbf_val:10.4f}  "
              f"{lin_str:>10s}  {source}")

# ============================================================
# HEATMAP WITH INTERPOLATED VALUES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

# --- Left: heatmap with observed + interpolated ---
ax = axes[0]
im = ax.imshow(result_matrix, cmap='YlOrRd', aspect='auto', origin='upper')

ax.set_xticks(range(len(sugar_values)))
ax.set_xticklabels([f"{s:.1f}" for s in sugar_values], fontsize=11)
ax.set_yticks(range(len(grid_sizes)))
ax.set_yticklabels([str(g) for g in grid_sizes], fontsize=11)
ax.set_xlabel('Average Sugar per Cell', fontsize=13)
ax.set_ylabel('Grid Size', fontsize=13)
ax.set_title('Mean Decay Rate λ\n(observed + interpolated)', fontsize=14, fontweight='bold')

for i in range(len(grid_sizes)):
    for j in range(len(sugar_values)):
        val = result_matrix[i, j]
        is_observed = (grid_sizes[i], sugar_values[j]) in observed
        
        color = 'white' if val > (np.nanmax(result_matrix) * 0.6) else 'black'
        label = f'{val:.3f}'
        if not is_observed:
            label += '*'  # Mark interpolated values
        
        ax.text(j, i, label, ha='center', va='center', fontsize=12,
                fontweight='bold', color=color,
                fontstyle='italic' if not is_observed else 'normal')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('λ (decay rate)', fontsize=12)
ax.text(0.02, -0.15, '* = interpolated (RBF, thin-plate spline in log-space)',
        transform=ax.transAxes, fontsize=9, color='gray')

# --- Right: 2D view showing interpolation surface ---
ax = axes[1]

# Fine grid for smooth surface
g_fine = np.linspace(18, 52, 100)
s_fine = np.linspace(0.3, 3.2, 100)
G_mesh, S_mesh = np.meshgrid(g_fine, s_fine)
query = np.column_stack([G_mesh.ravel(), S_mesh.ravel()])
Z = np.exp(rbf(query)).reshape(G_mesh.shape)

contour = ax.contourf(S_mesh, G_mesh, Z, levels=20, cmap='YlOrRd')
cbar2 = plt.colorbar(contour, ax=ax, shrink=0.8)
cbar2.set_label('λ (decay rate)', fontsize=12)

# Overlay observed points
obs_sugar = [k[1] for k in observed]
obs_grid = [k[0] for k in observed]
obs_lam = list(observed.values())
ax.scatter(obs_sugar, obs_grid, c='black', s=80, zorder=5, label='Observed')

# Overlay interpolated points
for idx, (g, s) in enumerate(target_points):
    if (int(g), float(s)) not in observed:
        ax.scatter(s, g, c='white', s=80, edgecolors='black', linewidth=2, zorder=5)

ax.set_xlabel('Average Sugar per Cell', fontsize=13)
ax.set_ylabel('Grid Size', fontsize=13)
ax.set_title('Interpolation Surface\n(RBF thin-plate spline)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_lambda_interpolated.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'heatmap_lambda_interpolated.png'}")

# ============================================================
# FINAL COMPLETE TABLE
# ============================================================
print("\n\nFinal λ matrix (observed + interpolated):")
print(f"\n       ", end="")
for s in sugar_values:
    print(f"  S={s:.1f}", end="")
print()
for i, g in enumerate(grid_sizes):
    print(f"  G={g:2d}  ", end="")
    for j in range(len(sugar_values)):
        marker = " " if (g, sugar_values[j]) in observed else "*"
        print(f"  {result_matrix[i,j]:.4f}{marker}", end="")
    print()
print("\n  * = interpolated")
