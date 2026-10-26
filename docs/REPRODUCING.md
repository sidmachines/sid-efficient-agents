# Reproducing the research

## Choose the operation

| Operation | Changes saved outputs? |
|---|---|
| Verify archive | No |
| Run unit tests | No research data changes |
| Verify saved results | Rewrites `verification.json` |
| Rebuild analysis | Regenerates summaries and figures |
| Rerun experiments | Overwrites raw data and metadata |
| Rebuild reports | Regenerates PDFs and HTML |

Keep the committed archive intact. Use a separate copy or Git branch for reruns. The manifest describes the archived execution; regenerated timing, runtime metadata, figures, or PDFs can legitimately differ.

## Environment and checks

Use Python 3.12 or later. Training records Python 3.12.13 and NumPy 2.3.5; analysis records Python 3.13.5, NumPy 2.3.3, and Matplotlib 3.10.6. The pinned requirements specify NumPy 2.3.5 and Matplotlib 3.10.6 for a practical reproduction environment.

Create and activate a virtual environment using your normal Python setup, then run from the repository root:

```bash
python -m pip install -r research/02_Experiments/requirements.txt
python tools/verify_archive.py
cd research/02_Experiments
python -m unittest test_study.py -v
python verify_results.py
```

No API key, paid model, GPU, downloaded language model, or external dataset is needed.

The six tests cover buffer/model counts, rare state-action coverage, terminal handling, exact evaluation, environment distributions, and seeded reproducibility. The result verifier checks the full configuration matrix, 21,000 control checkpoints, source/protocol hashes, capacity, computation counts, memory correction, 560 neural checkpoints, and the primary effect.

The deep memory correction assumes the recorded CPython/NumPy object layout. Its exact byte-size assertion may fail on another architecture or interpreter. This is a limitation of object-size accounting; do not silently alter the archived correction.

## Rebuild analysis

From `research/02_Experiments`:

```bash
python analyze.py
python analyze_neural.py
```

These consume saved CSV checkpoints, produce CSV/JSON summaries, and regenerate figures. Analysis uses 10,000 bootstrap replicates. Secondary comparisons remain exploratory.

## Rerun training

In a working copy, from `research/02_Experiments`:

```bash
python study.py --smoke --workers 1
python study.py --workers 1
python neural_transfer.py
python analyze.py
python analyze_neural.py
python verify_results.py
```

The smoke run writes separate `smoke*` files and uses seed 943, outside the main seeds. The full control run overwrites `runs.csv` and its metadata; the neural run overwrites its own outputs. There are 2,100 control runs, each with 3,000 interactions. Runtime depends on hardware. Optional parallel execution uses `--workers`; the archived execution used one worker after a process-pool permission error.

Seeds 0 through 19 are shared across methods. The true transition kernel is used only inside evaluation. The changed-goal checkpoint at interaction 1,500 still uses the old objective, so post-change area uses checkpoints 1,750 through 3,000.

## Rebuild documents

From `research/02_Experiments`:

```bash
python -m pip install -r requirements_documents.txt
python build_documents.py
```

The builder reads saved findings, execution metadata, references, figures, and `../01_Paper/branding.json`. Keep the logo asset with the reports. The PDFs contain 10 and five pages respectively, including covers.

## Original project

From the repository root:

```bash
python -m experiments.compare
python -m experiments.memory_sweep
```

These regenerate historical-style figures under root `results/`. They do not reproduce the new paper. The unchanged source snapshot is also under `research/06_Original_Project/`.

## Evidence definitions

- [Main protocol](../research/02_Experiments/protocol.json) and [neural protocol](../research/02_Experiments/neural_protocol.json).
- [Execution metadata](../research/03_Results/runs_metadata.json) and [neural metadata](../research/03_Results/neural_metadata.json).
- [Analysis decisions](../research/05_Research_Notes/analysis_decisions.txt).
- [Research manifest](../research/MANIFEST_SHA256.txt) and [provenance](../research/05_Research_Notes/provenance.txt).

Preserve the recorded protocols and hashes when describing the archived execution. Algorithm, evaluation, task, or threshold changes constitute a new experiment and should be reported separately.
