"""
Heatmap of λ (decay rate) by Grid Size × Average Sugar per Cell
Reads fitted_parameters.csv and averages λ across repeats (different v, m).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG — same as your analysis script
# ============================================================
INPUT_CSV = Path(r"./results\analysis_output\fitted_parameters.csv")
OUTPUT_DIR = Path(r"./results\analysis_output")

# ============================================================
# LOAD & AGGREGATE
# ============================================================
df = pd.read_csv(INPUT_CSV)

# Average lambda across repeats for each (grid, sugar) combo
pivot = df.pivot_table(
    values='lambda',
    index='grid_size',
    columns='avg_sugar',
    aggfunc='mean'
)

# Sort axes
pivot = pivot.sort_index(ascending=True)
pivot = pivot[sorted(pivot.columns)]

# ============================================================
# HEATMAP
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='upper')

# Axis labels
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(g) for g in pivot.index], fontsize=11)

ax.set_xlabel('Average Sugar per Cell', fontsize=13)
ax.set_ylabel('Grid Size', fontsize=13)
ax.set_title('Mean Decay Rate λ by Configuration', fontsize=15, fontweight='bold')

# Annotate cells with values
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if np.isnan(val):
            ax.text(j, i, 'N/A', ha='center', va='center', fontsize=11, color='gray')
        else:
            # Choose text color based on background brightness
            color = 'white' if val > (pivot.values[~np.isnan(pivot.values)].max() * 0.6) else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=12, 
                    fontweight='bold', color=color)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('λ (decay rate)', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_lambda.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / 'heatmap_lambda.png'}")

# ============================================================
# PRINT TABLE
# ============================================================
print("\nMean λ by (Grid, Sugar):\n")
print(pivot.round(4).to_string())

# Also show count per cell (to verify all 10 repeats are there)
count_pivot = df.pivot_table(values='lambda', index='grid_size', columns='avg_sugar', aggfunc='count')
print("\nCount of runs per cell:\n")
print(count_pivot.to_string())
