"""
Generate simulation flowchart with paper variable notation.
Produces: flowchart.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── helpers ───────────────────────────────────────────────────────────────────

def rect(ax, cx, cy, w, h, text, fs=7.2, fc='white'):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.02',
                                facecolor=fc, edgecolor='black', linewidth=0.8, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            multialignment='center', zorder=4)

def diamond(ax, cx, cy, w, h, text, fs=7.2):
    pts = [[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]]
    ax.add_patch(plt.Polygon(pts, facecolor='white', edgecolor='black',
                             linewidth=0.8, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            multialignment='center', zorder=4)

def circle(ax, cx, cy, r=0.22):
    ax.add_patch(plt.Circle((cx, cy), r, facecolor='white',
                             edgecolor='black', linewidth=0.9, zorder=3))

def arr(ax, x1, y1, x2, y2, lw=0.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=lw),
                zorder=5)

def line(ax, xs, ys, lw=0.8):
    ax.plot(xs, ys, color='black', lw=lw, zorder=2)

def label(ax, x, y, text, fs=7.0, ha='center', va='center', style='normal'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs,
            fontstyle=style, zorder=6)

# ── figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.8, 12))
ax.set_xlim(0, 6.8)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_aspect('equal')

# ── dashed bounding box (main loop) ──────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.18, 0.55), 6.2, 10.55,
                             boxstyle='square,pad=0',
                             facecolor='none', edgecolor='#555555',
                             linewidth=0.9, linestyle=(0, (6, 4)), zorder=1))

# ── START circle ─────────────────────────────────────────────────────────────
circle(ax, 5.6, 11.6)
# ── INITIALIZATION ────────────────────────────────────────────────────────────
rect(ax, 3.4, 11.6, 3.6, 0.52,
     "Initialization: $l^2$ grid, $n_0$ agents $A_k$\n"
     "parameters: $v$, $m$, $s_0$, $\\alpha_{min}$, $\\alpha_{max}$", fs=7.0)

arr(ax, 5.38, 11.6, 5.20, 11.6)   # start → init
arr(ax, 3.4, 11.34, 3.4, 11.05)   # init → dashed box / shuffle label

# "for each agent" label
label(ax, 3.55, 10.95, "for each agent $A_k$", fs=7.0, style='italic', ha='center')

# ── SHUFFLE ───────────────────────────────────────────────────────────────────
arr(ax, 3.4, 10.82, 3.4, 10.60)
rect(ax, 3.4, 10.40, 2.4, 0.38, "Shuffle agent order", fs=7.2)

# ── PERCEPTION ────────────────────────────────────────────────────────────────
arr(ax, 3.4, 10.21, 3.4, 9.96)
rect(ax, 3.4, 9.72, 3.7, 0.44,
     "Perception: scan cells $C_i$ ($s_{i,t}$, $o_{i,t}$)\n"
     "and agents $A_j$ ($\\alpha_j$) in vision range $v$", fs=7.0)

# ── PENDING TARGET diamond ────────────────────────────────────────────────────
arr(ax, 3.4, 9.50, 3.4, 9.22)
diamond(ax, 3.4, 8.90, 2.1, 0.60,
        "$A_k$ has\npending target?", fs=7.0)

# YES → left → one step
arr(ax, 2.35, 8.90, 1.50, 8.90)
label(ax, 1.90, 8.99, "Yes", fs=6.8)
rect(ax, 0.88, 8.90, 1.30, 0.42, "One step toward\ntarget cell", fs=7.0)

# NO → down → Query LLM
arr(ax, 3.4, 8.60, 3.4, 8.30)
label(ax, 3.55, 8.47, "No", fs=6.8, ha='left')

# ── QUERY LLM ─────────────────────────────────────────────────────────────────
rect(ax, 3.4, 7.98, 3.8, 0.56,
     "Query LLM: system prompt +\n"
     "state ($s_k$, $m$, $\\alpha_k$) + memory ($l_m$ events)\n"
     "+ visible data ($s_{i,t}$, $o_{i,t}$, $\\alpha_j$)", fs=6.9)

# ── ACTION TYPE diamond ───────────────────────────────────────────────────────
arr(ax, 3.4, 7.70, 3.4, 7.40)
diamond(ax, 3.4, 7.08, 2.0, 0.58, "Action\ntype?", fs=7.2)

# branch labels
label(ax, 2.25, 7.30, "Stay", fs=6.8, ha='center')
label(ax, 3.4,  6.72, "Move", fs=6.8, ha='center')
label(ax, 4.65, 7.30, "Attack", fs=6.8, ha='center')
label(ax, 5.80, 7.08, "Reproduce", fs=6.8, ha='center')

# STAY box (left)
line(ax, [2.40, 1.65, 1.65], [7.08, 7.08, 6.55])
arr(ax, 1.65, 6.55, 1.65, 6.18)
rect(ax, 1.65, 5.98, 1.50, 0.37, "Stay", fs=7.2)

# MOVE box (down from diamond)
arr(ax, 3.4, 6.79, 3.4, 6.52)
rect(ax, 3.4, 6.28, 1.8, 0.44,
     "Move to target cell\n(1 cell per step $t$)", fs=7.0)

# ATTACK box (right)
line(ax, [4.40, 5.20, 5.20], [7.08, 7.08, 7.42])
arr(ax, 5.20, 7.42, 5.20, 7.62)
rect(ax, 5.20, 7.88, 1.80, 0.52,
     "Attack $A_j$:\nwin: $s_k {+}= s_j$, $A_j$ dies\nlose: $A_k$ dies", fs=6.8)

# REPRODUCE box (far right)
line(ax, [4.40, 6.40, 6.40], [7.08, 7.08, 7.28])
arr(ax, 6.40, 7.28, 6.40, 7.54)
rect(ax, 6.40, 7.84, 1.55, 0.56,
     "Reproduce:\n$s_k \\geq s_r$\n$\\alpha_{k'}{=}\\mathrm{clip}(\\alpha_k{+}\\mathcal{N}(0,\\epsilon))$\n$s_{k'}{=}s_k/2$",
     fs=6.5)

# ── converge all branches → COLLECT SUGAR ────────────────────────────────────
# Stay → down
line(ax, [1.65, 1.65], [5.79, 5.55])
line(ax, [1.65, 3.4], [5.55, 5.55])
# Move → down
line(ax, [3.4, 3.4], [6.06, 5.55])
# Attack → down
line(ax, [5.20, 5.20], [7.62, 5.55])
line(ax, [5.20, 3.4], [5.55, 5.55])
# Reproduce → down
line(ax, [6.40, 6.40], [7.56, 5.55])
line(ax, [6.40, 3.4], [5.55, 5.55])
# One step toward target → down to collect
line(ax, [0.88, 0.32, 0.32], [8.69, 8.69, 5.55])
line(ax, [0.32, 3.4], [5.55, 5.55])
arr(ax, 3.4, 5.55, 3.4, 5.36)

rect(ax, 3.4, 5.18, 2.8, 0.36,
     "Collect $s_{i,t}$ from cell $C_i$", fs=7.1)

# ── CONSUME METABOLISM ────────────────────────────────────────────────────────
arr(ax, 3.4, 5.00, 3.4, 4.72)
rect(ax, 3.4, 4.54, 2.8, 0.36,
     "Consume $m$ sugar:  $s_k \\leftarrow s_k - m$", fs=7.1)

# ── SUGAR ≤ 0 diamond ─────────────────────────────────────────────────────────
arr(ax, 3.4, 4.36, 3.4, 4.06)
diamond(ax, 3.4, 3.74, 1.9, 0.56, "$s_k \\leq 0$?", fs=7.2)

# YES → right → Die
arr(ax, 4.35, 3.74, 5.00, 3.74)
label(ax, 4.65, 3.84, "Yes", fs=6.8)
rect(ax, 5.55, 3.74, 1.00, 0.36, "Die", fs=7.2)

# NO → left → back up to Shuffle (inner agent loop)
line(ax, [2.45, 0.50, 0.50], [3.74, 3.74, 10.40])
arr(ax, 0.50, 10.40, 2.20, 10.40)
label(ax, 0.34, 7.00, "NO", fs=7.5, ha='center', va='center')

# ── REMOVE DEAD AGENTS ───────────────────────────────────────────────────────
arr(ax, 3.4, 3.46, 3.4, 3.18)
rect(ax, 3.4, 3.00, 2.6, 0.36, "Remove dead agents", fs=7.2)

# ── SUGAR REGENERATION ────────────────────────────────────────────────────────
arr(ax, 3.4, 2.82, 3.4, 2.54)
rect(ax, 3.4, 2.34, 3.2, 0.38,
     "Sugar regeneration: $s_{i,t} {+}= 1$ every 2 steps (up to $s_{c,\\max}$)", fs=6.8)

# ── STEP CHECK diamond ────────────────────────────────────────────────────────
arr(ax, 3.4, 2.15, 3.4, 1.85)
diamond(ax, 3.4, 1.52, 2.8, 0.56,
        "$t = t_{\\max}$ or all $A_k$ dead?", fs=7.0)

# YES → down → End
arr(ax, 3.4, 1.24, 3.4, 0.92)
label(ax, 3.55, 1.10, "Yes", fs=6.8, ha='left')
circle(ax, 3.4, 0.72)

# NO → (handled by the outer "NO" label on the left — same line already drawn)
label(ax, 1.80, 1.52, "No", fs=6.8, ha='center')
line(ax, [2.00, 0.50], [1.52, 1.52])  # NO branch joins the left spine already drawn

plt.tight_layout(pad=0)
plt.savefig('/workspaces/Coding/paper_repo/flowchart.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print("Saved: flowchart.png")
