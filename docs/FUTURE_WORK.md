# Future work and application boundaries

These are proposed experiments, not completed benchmarks or claims of production reliability.

## Local LLM-agent validation

The supplement proposes two frozen local instruction models, 30 held-out tool tasks, matched prompts and tool contracts, and four memory conditions: none, recent, reservoir, and coverage-based retention.

Control stored and retrieved memory in tokens. Keep within-task history identical. Separate calibration from evaluation and reset cross-task memory between independently seeded task sequences. Store only observable feedback; evaluator solutions must not enter memory. Include version-tagged tool changes.

Measure success, inference tokens, tool calls, invalid actions, retries, latency, storage, and retrieval cost. The prospective target is a lower 95% confidence bound above -3 percentage points on success difference plus at least 20% lower inference tokens than recent memory. These are design targets, not achieved results. Thirty tasks may be underpowered.

Begin with a free local feasibility pilot. Choose models that fit the hardware and record checkpoint hashes, quantization, decoding, and runtime. Use fresh held-out tasks after pilot-driven tuning.

## Stronger control and neural evaluations

1. Test new task instances rather than only new seeds in the same environments.
2. Compare mixtures of recent and diverse memory, including change detection.
3. Measure peak process memory and temporary allocations.
4. Compare methods under matched wall-time or measured hardware-work budgets.
5. Test partial observability, noisy abstraction, longer horizons, and larger spaces.
6. Extend neural experiments to multiple architectures and datasets with independently generated test sets.

## Claims requiring separate evidence

Reduced LLM pretraining data or GPU cost requires language-model training experiments. Energy savings require measured energy. Production reliability requires real workloads, failure analysis, deployment constraints, and appropriate validation. Current simulations do not establish these outcomes.

See the [technical supplement](../research/01_Paper/Technical_Supplement.pdf) for the full proposed local LLM protocol.
