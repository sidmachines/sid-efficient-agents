# Research guide

## From the original idea to the expanded study

The original question was whether a learned model could reduce trial-and-error learning and how much experience it needed to retain. The expansion asks what the agent should retain and how much computation it should spend using that evidence.

The contribution is an auditable comparison under fixed interaction budgets. It distinguishes sample efficiency, memory, computation, and adaptation instead of treating them as one score.

## What was compared

Q learning updates action values directly. Q replay reuses retained transitions. The full-model agent aggregates all observed outcomes and plans on those statistics. Bounded model agents maintain statistics for observations selected by recent/FIFO, reservoir, or coverage retention.

Coverage retention removes records from the most represented state-action group. It aims to preserve represented groups under a capacity limit; it is not guaranteed to maximize performance. It adds bookkeeping and selection work.

All control methods receive 3,000 interactions. Main model agents use 20 planning sweeps every 50 interactions. Ablations use five sweeps every 50 interactions or 20 sweeps every 200 interactions at capacity 128. Q learning performs one update per interaction; Q replay adds four replay updates.

## Questions and evidence

| Question | Evidence | Supported interpretation |
|---|---|---|
| Does learned-model planning help? | Stationary curves and per-run summaries | It improves learning progress over plain Q learning in these tasks |
| Is coverage better than recent retention? | Primary paired difference and interval | No clear average advantage at capacity 128 |
| Does forgetting help when goals change? | Changed-goal post-shift area | Recent evidence avoids some interference from old rewards |
| Does a small buffer save storage? | Corrected retained-state estimates | Full aggregated statistics can be smaller |
| Does total update count characterize planning? | Equal-backup ablations | Depth and frequency can produce different results |
| Can diverse replay preserve old classes? | Neural balanced accuracy | Yes in this small synthetic task, with added training work |

## Reading the numbers

Success is the exact probability of reaching a terminal goal within 100 steps under the learned greedy policy. The main endpoint integrates success over checkpoints and normalizes by the 3,000-interaction budget. A high final score can coexist with poor early learning.

The primary comparison averages coverage-minus-recent areas over four fixed stationary tasks within each training seed, then bootstraps the 20 seed blocks. Its 95% interval spans zero. This does not establish equivalence or identical behavior.

The changed-goal task is analyzed separately after the objective changes. The neural extension is also separate: balanced classification accuracy is not the same endpoint as simulator success.

## Evidence inventory

| File | Unit of observation |
|---|---|
| [runs.csv](../research/03_Results/runs.csv) | Control evaluation checkpoint; 21,000 rows |
| [run_metrics.csv](../research/03_Results/run_metrics.csv) | Completed control run; 2,100 rows |
| [summary_by_environment.csv](../research/03_Results/summary_by_environment.csv) | Environment/configuration aggregate; 105 rows |
| [paired_effects.csv](../research/03_Results/paired_effects.csv) | Exploratory within-environment paired contrast |
| [neural_runs.csv](../research/03_Results/neural_runs.csv) | Neural checkpoint; 560 rows from 140 runs |
| [findings.json](../research/03_Results/findings.json) | Main report aggregates and primary contrast |
| [neural_findings.json](../research/03_Results/neural_findings.json) | Neural summaries and paired comparison |

The paper has nine primary references. [references.json](../research/05_Research_Notes/references.json) records citations, URLs, and their roles. [analysis_decisions.txt](../research/05_Research_Notes/analysis_decisions.txt) records analysis refinements.

## Practical interpretation

When tool behavior or objectives change, recent observations may be more useful than obsolete outcomes. When earlier skills must be preserved, diverse replay can help. These objectives can conflict.

Real systems need end-to-end measurements of success and total cost, including retrieval, planning, inference, retries, storage, and memory selection. This evidence does not determine the best memory design for an arbitrary LLM agent.
