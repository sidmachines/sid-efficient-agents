MEMORY SELECTION AND PLANNING BUDGETS IN LEARNING AGENTS
SID Machines | Private research package

PRIVATE RESEARCH - FOR AUTHORIZED RECIPIENTS ONLY
Research directed by the company owner.
This report is privately circulated by the owner of SID Machines for authorized recipients only. It is not intended for public distribution. Do not reproduce, publish, or forward it without the owner's permission.

Open 00_START_HERE.html after extracting the ZIP. No internet is needed to
read the documents or figures. External bibliography links require internet.

Suggested reading order
1. 01_Paper/Research_Paper.pdf -- private cover plus nine-page research paper.
2. 01_Paper/Technical_Supplement.pdf -- methods, reproduction, and proposed LLM validation.
3. 03_Results/summary_by_environment.csv -- every main configuration.
4. 02_Experiments/ -- executable code, tests, and prospective protocols.

Completed evidence
- 2,100 seeded control-agent training runs across five simulated environments.
- 6,300,000 real simulator interactions, with exact policy evaluation.
- 140 additional small neural-classifier runs on synthetic sequential data.
- 21,000 control checkpoints and 560 neural checkpoints.
- Bootstrap uncertainty, computation and memory accounting, and all outcomes.

The run count is experimental repetitions across configurations, not thousands
of different agent architectures or foundation models. The primary comparison
does not establish an average advantage for coverage memory at capacity 128.
Different memory rules behave differently under changed goals and retention
of older classes. These qualified results are the central findings.

LLM applicability is a proposed, specified next experiment. No LLM inference,
GPU use, energy savings, or real external-tool deployment was measured.

The paper and supplement also have editable HTML versions generated from the
same content. The original supplied repository is preserved in folder 06;
its old figures are not the new study's evidence.

Reproduction instructions are in the technical supplement. Running study.py
again overwrites the included main run data; copy the package first if needed.
MANIFEST_SHA256.txt records the packaged files' hashes.
