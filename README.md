# Emergent Combat Rationality and Kin-Based Aggression in LLM-Powered Artificial Societies

> **Luca Moroni, Matteo Prefumo, Samuele Astuti, Leonardo Mascagni, Francesco Bertolotti**
> Intelligence, Complexity, and Technology Lab (ICT Lab) & School of Industrial Engineering, University Cattaneo (LIUC), Italy
> *WOA 2026 — 26th Workshop "From Objects to Agents", Italy*

Corresponding authors: [lu15.moroni@stud.liuc.it], [fbertolotti@liuc.it]

---

## Overview

This repository contains the simulation code, analysis scripts, and results for the paper **"Emergent Combat Rationality and Kin-Based Aggression in LLM-Powered Artificial Societies"**.

We study the emergent behavior of **LLM-driven agents** (GPT-4o-mini) in a modified Sugarscape environment, investigating whether agents reproduce **kin-discriminated aggression** when provided with a heritable phenotypic marker (α gene) and explicitly informed that genetic dissimilarity increases aggression propensity. The analysis spans three grid sizes and six resource density levels.

Key findings:
- The alpha gap $\Delta\alpha_{jk}$ is statistically detectable but **operationally negligible**, explaining less than 0.1% of variance in attack decisions (point-biserial $r = 0.019$–$0.024$ vs $r = 0.46$ for neighbor count).
- Agents exhibit **emergent combat rationality**: despite lacking access to neighbors' sugar reserves, they attack preferentially when holding a resource advantage. Attack rate is 1.41% in rational opportunities vs 1.03% in irrational ones ($\chi^2 = 116.5$, $p = 3.64 \times 10^{-27}$).
- Genetic dissimilarity modulates **not the frequency but the strategic quality** of aggression, activating a risk-assessment process absent when attacking similar neighbors.
- Population dynamics follow an exponential decay model $n(t) = n_\infty + (n_0 - n_\infty)\,e^{-\lambda t}$, with $R^2 > 0.90$ in 127 / 135 runs (mean $R^2 = 0.958$).

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── generate_flowchart.py              # Generates flowchart figure (Fig. 1)
├── generate_initial_states.py         # Generates static_comparison*.pdf/png (Fig. 2)
├── flowchart.png                      # Simulation flowchart figure
│
├── notebooks/
│   ├── SugLLM.ipynb                   # Main simulation runner (batch experiments)
│   └── GraphSugLLM.ipynb              # Overview plots from batch results
│
├── analysis_scripts/
│   ├── alpha_gap_analysis.py          # Alpha gap distribution & attacker vs non-attacker
│   ├── behavior_analysis.py           # Action frequencies by config, sugar, age
│   ├── combat_estimation.py           # Combat outcomes, decision theory, rationality
│   ├── combat_intelligence.py         # Sugar ratio & win-rate analysis
│   ├── context_detailed_analysis.py   # Context variables (neighbors, sugar, age) — detailed
│   ├── context_combined_analysis.py   # Context variables — combined single plot
│   ├── heatmap_attack_density.py      # 2D heatmap of attack counts by alpha bin
│   ├── heatmap_attack_rate.py         # 2D heatmap of attack rate (attacks / opportunities)
│   ├── heatmap_decay_lambda.py        # Heatmap of λ decay rate across configurations
│   ├── interpolate_decay_lambda.py    # RBF interpolation of λ over grid/sugar space
│   ├── rationality_matrix.py          # Decision-theoretic rationality classification
│   ├── survival_aggregated_curves.py  # Aggregated survival curves across runs
│   └── survival_curve_fitting.py      # Exponential fit of population dynamics
│
└── results/
    ├── logs/                          # Notes and fitted formula summary
    ├── simulation_data/               # Raw CSV output — one folder per run (G×S×repeat)
    └── plots/
        ├── alpha/                     # Alpha gap & attacker analysis
        ├── behavior/                  # Behavioral action distributions
        ├── combat/                    # Combat estimation & rationality
        ├── context/                   # Contextual decision patterns
        ├── heatmaps/                  # Attack density/rate & λ heatmaps
        ├── survival/                  # Survival curves & parameter fits (incl. fitted_parameters.csv)
        └── overview/                  # General batch summary plots
```

---

## Experimental setup

| Parameter | Values |
|---|---|
| Grid sizes | 20 × 20, 30 × 30, 50 × 50 |
| Average sugar per cell | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| Repeats per configuration | 10 |
| Agents | 50 (initial) |
| Max steps | 200 |
| Agent vision range | 1 – 6 (random per agent) |
| Agent metabolism | 0.5 – 2.0 (random per agent) |
| LLM model | GPT-4o-mini |
| Base seed | 42 |

Total runs in this dataset: **135 runs** across 15 configurations (all six sugar values for 20×20 and 30×30; partial for 50×50), 10 repeats each.

---

## Reproducing the results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the project root (never commit this file):

```
OPENAI_API_KEY=your-key-here
```

### 3. Run the simulation *(optional — data already included)*

Open and run `notebooks/SugLLM.ipynb`. Results will be saved to `results/simulation_data/`.

### 4. Reproduce all plots

Each analysis script reads from `results/simulation_data/` and writes to the corresponding subfolder under `results/plots/`. Run them independently:

```bash
python analysis_scripts/survival_curve_fitting.py      # produces results/plots/survival/
python analysis_scripts/alpha_gap_analysis.py           # produces results/plots/alpha/
python analysis_scripts/behavior_analysis.py            # produces results/plots/behavior/
python analysis_scripts/combat_estimation.py            # produces results/plots/combat/
python analysis_scripts/combat_intelligence.py          # produces results/plots/combat/
python analysis_scripts/context_detailed_analysis.py    # produces results/plots/context/
python analysis_scripts/context_combined_analysis.py    # produces results/plots/context/
python analysis_scripts/heatmap_attack_density.py       # produces results/plots/heatmaps/
python analysis_scripts/heatmap_attack_rate.py          # produces results/plots/heatmaps/
python analysis_scripts/heatmap_decay_lambda.py         # requires survival_curve_fitting.py first
python analysis_scripts/interpolate_decay_lambda.py     # requires heatmap_decay_lambda.py first
python analysis_scripts/rationality_matrix.py           # produces results/plots/combat/
python analysis_scripts/survival_aggregated_curves.py   # produces results/plots/survival/
```

> **Note on paths:** scripts use relative paths (`./results`). Run them from the repository root.

---

## Data format

Each run folder under `results/simulation_data/` (e.g. `G20_S1.00_R3/`) contains:

| File | Description |
|---|---|
| `config.csv` | Run parameters (grid size, sugar, vision, metabolism, seed) |
| `agent_moves.csv` | Per-step agent decisions (action, alpha, sugar, neighbors, …) |
| `step_stats.csv` | Aggregate statistics per simulation step |
| `map_state.csv` | Sugar map state at each step |
| `initial_agents.csv` | Agent attributes at step 0 |
| `survivors.csv` | Agents still alive at the final step |
| `houses.csv` | House positions (empty — houses disabled in this experiment) |
| `sugar_capacity.csv` | Maximum sugar capacity per cell |

---

## Keywords

artificial societies · large language models · agent-based modeling · kin selection · Sugarscape · emergent behavior · aggressive behavior · social simulation

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{moroni2026emergent,
  title     = {Emergent Combat Rationality and Kin-Based Aggression in LLM-Powered Artificial Societies},
  author    = {Moroni, Luca and Prefumo, Matteo and Astuti, Samuele and Mascagni, Leonardo and Bertolotti, Francesco},
  booktitle = {Proceedings of the 26th Workshop ``From Objects to Agents'' (WOA 2026)},
  year      = {2026},
  address   = {Italy},
  series    = {<series — to be confirmed>},
  publisher = {<publisher — to be confirmed>},
  note      = {To appear}
}
```

---

## License

Code, data and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Copyright © 2026 the authors. Use permitted under Creative Commons Attribution 4.0 International (CC BY 4.0).
