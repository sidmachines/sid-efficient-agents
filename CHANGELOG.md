# Research change record

## Expanded empirical study

- Extended the original question from model size in one gridworld to memory selection and planning budgets across five environments.
- Replaced episode-budget comparisons with matched budgets of 3,000 simulator interactions per control run.
- Added Q learning with replay, reservoir retention, and state-action coverage retention.
- Completed 1,500 main runs and 600 planning-ablation runs using 20 seeds per configuration.
- Added exact finite-horizon evaluation and saved all 21,000 control checkpoints.
- Added changed-goal adaptation and computation, storage, and timing measurements.
- Documented the array-base memory correction and paired seed-block bootstrap specification.
- Added 140 exploratory neural-classifier runs and saved 560 checkpoints.
- Produced the paper, supplement, five figure sets, raw data, analysis code, protocols, references, and integrity manifest.
- Preserved inconclusive and weak results, including the primary coverage-versus-recent comparison.
- Specified future local LLM-agent validation without representing it as executed.
- Added repository documentation, evidence navigation, and reproduction instructions.

## Original investigation

The initial project compared tabular Q learning with planning on a learned model in a stochastic 5x5 gridworld and explored recent-transition memory caps. Its question, implementation, write-up, and figures are preserved under `research/06_Original_Project/`. The original implementation also remains at the repository root.

The source snapshot matched the files at repository commit `6653bd372df44b75ca98a26e73c71f7f1e3004d7` before this expansion was prepared. Historical results were preserved, not rerun as part of the expanded study.
