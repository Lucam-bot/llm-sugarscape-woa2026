Results

Population Dynamics and Survival
<p align="center">
  <img src="plots/curves_survival.png" width="30%" />
  <img src="Analysis_output/heatmap_lambda_interpolated.png" width="60%" />
</p>

Cosa mostra: Le curve di sopravvivenza seguono un decadimento esponenziale in tutte le configurazioni. λ decresce con la densità di zucchero e con la dimensione della griglia.

Perché mostrarlo: Valida che il modello funziona come atteso — l'ambiente è selettivo e risponde ai parametri in modo coerente con il Sugarscape classico. È la baseline necessaria per interpretare i risultati successivi.

Cosa dimostra: La mortalità è determinata principalmente dalle condizioni ambientali (risorse, spazio), non dal comportamento strategico degli agenti.


Agent Behavior and Decision Patterns
![Decisions per age](Analysis_Results1/context_age_by_grid.png)

![Decisions per age](Analysis_Results1/context_comparison_grids.png)

Cosa mostra: Come la distribuzione delle azioni (move, stay, attack, reproduce) varia in funzione del numero di vicini, della densità di zucchero e dell'età dell'agente, confrontando le tre dimensioni di griglia.

Perché mostrarlo: Caratterizza il comportamento degli agenti LLM in modo sistematico e mostra che rispondono razionalmente al contesto locale — più vicini → più movimento e più attacchi, più zucchero → più stabilità.

Cosa dimostra: Gli agenti LLM sviluppano pattern comportamentali contestualmente sensibili senza che questi siano esplicitamente programmati. L'aggressività è guidata dalla densità di popolazione più che dalla scarsità di risorse.

Genetic Similarity and Aggressive Behavior}
![Alpha Gap Attck](plots/attack_alpha_gap.png)

Cosa mostra: La distribuzione di |α_attaccante - α_bersaglio| per ogni attacco, confrontata con il valore atteso casuale (0.667). Il pannello WIN/LOSE mostra se l'esito è correlato al gap.

Perché mostrarlo: È il test diretto della domanda di ricerca principale.

Cosa dimostra: Il mean alpha gap negli attacchi (0.66) è indistinguibile dal caso. Gli agenti non selezionano i bersagli in base alla dissimilarità genetica nonostante la disposizione sia esplicitamente encodata nel prompt. L'esito del combattimento non è correlato al gap.

![Decisions per age](Analysis_Results3/alpha_2d_heatmap.png)
![Decisions per age](Analysis_Results3/alpha_vision_vs_adjacent.png)

Cosa mostra: Le action rates per bin di alpha gap (vision range e adiacente) e la distribuzione del gap per attaccanti vs non-attaccanti, nei due contesti a confronto.

Perché mostrarlo: Verifica se la disposizione encodata nel prompt si traduce in comportamento differenziato — non sull'esito degli attacchi, ma sulla decisione di attaccare.

Cosa dimostra: L'attack rate adiacente sale da ~7% a ~12% al crescere della dissimilarità, e gli attaccanti hanno gap medio leggermente più alto dei non-attaccanti (0.645 vs 0.606). La disposizione è quindi recepita, ma l'effect size è trascurabile: le distribuzioni sono quasi completamente sovrapposte e la differenza nelle medie è ~0.04. Alta dissimilarità nel campo visivo (≥1.5) produce inazione anziché aggressione (Stay sale al 23%), coerentemente con il fatto che l'attacco è disponibile solo in adiacenza. La disposizione genetica emerge solo quando l'opportunità è immediata, e resta sistematicamente dominata da fattori contestuali.
