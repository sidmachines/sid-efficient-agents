# How Little Model Is Enough? A Small Empirical Study of Model-Based vs. Model-Free Control in a Gridworld

**SID Machines** · An independent study · 2026

---

## Abstract

Model-based reinforcement learning is known to be more sample-efficient than
model-free learning, and it is known that planning does not require a perfect
model. This report is a clean, reproducible reproduction of both effects in a
small stochastic gridworld, plus a small empirical characterization of *how
complete* a learned model must be before planning helps at all. We compare
tabular Q-learning (model-free) against a Dyna-style agent that estimates the
environment's transitions and rewards by counting and plans on them with value
iteration (model-based). The model-based agent reaches near-optimal performance
in roughly an order of magnitude fewer episodes. We then progressively starve
the learned model using a fixed-size memory of recent transitions, and find a
sharp threshold: below roughly 32 remembered transitions the agent cannot plan
usefully, and by roughly 128 it recovers essentially full performance, with a
high-variance transition region in between. We make no claim of novelty; the
underlying phenomena are well established. The value of this work is a careful,
seed-averaged, honestly-reported reproduction, and the code to run it.

---

## 1. Motivation

An agent dropped into an unfamiliar environment can behave in two broad ways.
It can **react** — remember, per situation, how good each action felt, and
slowly refine those estimates through trial and error (model-free control).
Or it can **understand** — use its experience to build an internal model of
how the environment behaves, then plan on that model (model-based control).

Intuitively, understanding should pay off: each observed transition teaches the
agent something reusable about the world's structure, rather than updating a
single value. This report asks two concrete questions in a setting small enough
to measure everything cleanly:

1. **Does the model-based agent actually learn from less experience** than the
   model-free agent, and by how much?
2. **How good does the learned model have to be** before that advantage
   appears — i.e., how much can we starve the model before planning stops
   helping?

Neither question is novel. The point is to answer both rigorously and
reproducibly, and to see the tradeoffs first-hand.

---

## 2. The Environment

We use a 5x5 gridworld with four walls, giving **21 reachable states**. The
agent starts in one corner and must reach a goal in the opposite corner.

- **States:** the agent's `(row, col)` cell.
- **Actions:** up, down, left, right.
- **Transitions:** the intended move succeeds with probability `1 - slip`;
  with probability `slip = 0.1` a uniformly random action is taken instead.
  Moves into walls or off the edge leave the agent in place. This stochasticity
  is what makes a model worth learning rather than a single memorized path.
- **Rewards:** `+1.0` on reaching the goal, `-0.01` on every other step
  (a small penalty encouraging short paths).
- **Discount:** `gamma = 0.95`.

Crucially, the environment never exposes its transition or reward functions to
the agents. They observe only `(state, action, reward, next_state)` tuples.
Any agent that wants a model must build one from that stream.

---

## 3. The Two Agents

Both agents share the same discount, the same epsilon-greedy exploration
(`epsilon = 0.1`), and the same experience stream. They differ in one respect:
whether they build a model of the world.

### 3.1 Model-free baseline (Q-learning)

The baseline maintains action-values `Q(s, a)` and updates them from single
transitions:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s,a) ]
```

with learning rate `alpha = 0.1`. It never records where actions lead; it only
records how good they felt. When an episode terminates, the future term is
dropped (a terminal state has no successor value).

### 3.2 Model-based agent (learned model + value iteration)

The model-based agent estimates the environment from counts:

- **Reward model:** `R_hat(s,a)` = the average reward observed for `(s,a)`.
- **Transition model:** `T_hat(s,a,s')` = the fraction of times `(s,a)` was
  observed to lead to `s'`.

It then runs **value iteration** on its estimated model to compute a policy:

```
V(s) <- max_a [ R_hat(s,a) + gamma * sum_s' T_hat(s,a,s') * V(s') ]
```

This is the same Bellman backup used in textbook planning; the only difference
is that `T_hat` and `R_hat` are *learned from experience* rather than given.
Planning is re-run periodically (`replan_every = 50` steps, `vi_sweeps = 20`).

---

## 4. Experiment 1 — Sample Efficiency

**Setup.** We train both agents on the same environment and, at a schedule of
episode counts, freeze learning and exploration and evaluate the greedy policy
on 20 fixed test episodes. We report the mean and standard deviation across
**20 random seeds**.

**Result.** The model-based agent reaches near-optimal greedy return
(about `+0.91`) within roughly **5 episodes**, while the model-free agent is
still failing (negative return) at that point and does not reach the same level
until roughly **60 episodes** — about an order of magnitude more experience.
Both agents converge to the same final performance; the difference is entirely
in *how much experience it takes to get there*. The seed-averaged confidence
bands are clearly separated through the transition region, so the effect is not
an artifact of a lucky seed.

*(See `results/sample_efficiency.png`.)*

---

## 5. Experiment 2 — How Little Model Is Enough?

**Setup.** We take the model-based agent and cap its model to the most recent
`M` transitions using a rolling buffer; older experiences are forgotten and
subtracted from the counts. Everything else is identical, so any difference is
attributable only to `M`. We sweep `M` over
`{2, 4, 8, 16, 32, 64, 128, 256, unlimited}`, train each for a fixed budget of
80 episodes, and evaluate greedily across **20 seeds**. The model-free baseline
at the same budget is drawn as a reference.

**Result.** Performance is flat and useless (about `-2.0`) for `M <= 16`, rises
steeply between `M = 32` and `M = 128`, and plateaus at full performance
(about `+0.88`) by `M = 128`; beyond that, additional memory yields negligible
benefit. The transition region (`M = 32` to `64`) shows large variance across
seeds — near the threshold the outcome depends sensitively on which transitions
happen to be retained, giving the knee a phase-transition-like character.

**Interpretation.** In this environment the agent needs only on the order of
**~128 remembered transitions** — a small fraction of a full history — to
recover essentially all of the planning benefit. "Enough" is much less than
"everything."

*(See `results/memory_sweep.png`.)*

---

## 6. An Honest Caveat

At the 80-episode budget used in Experiment 2, the model-free baseline has
*already* fully solved the environment (about `+0.92`), matching the
unlimited-memory model-based agent. This is expected: 80 episodes is ample for
even slow trial-and-error learning in a 21-state world, so the
sample-efficiency advantage has washed out by then. Experiment 2 therefore
measures *how much model is needed to match full planning at a fixed budget*,
which is a different question from the *sample-efficiency* advantage measured in
Experiment 1. Both figures are needed; neither alone tells the whole story. We
flag this explicitly rather than presenting the two results as if they measured
the same thing.

---

## 7. Limitations and Non-Claims

- **No novelty is claimed.** Model-based sample efficiency (Dyna, Sutton 1990)
  and planning with imperfect models are long-established. This is a
  reproduction plus a small empirical characterization on a toy problem.
- **Single tiny environment.** Results are shown for one 21-state gridworld.
  Whether the ~128-transition threshold means anything beyond this specific
  setup is untested, and almost certainly it does not transfer as a number.
- **Tabular only.** Both the model and the values are exact tables. Nothing
  here speaks to function approximation, large state spaces, or deep RL.
- **Planning cost not deeply analyzed.** The model-based agent's compute cost
  (repeated value iteration) is real and, in the starved-but-failing regime,
  dominant; we note this but do not characterize the full compute/sample
  tradeoff curve.

---

## 8. Reproducing This Work

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy matplotlib

# sanity checks
python -m gridworld.environment          # watch the world step
python -m agents.q_learning              # train the baseline
python -m agents.model_based             # train the model-based agent

# the two figures
python -m experiments.compare            # -> results/sample_efficiency.png
python -m experiments.memory_sweep       # -> results/memory_sweep.png
```

All randomness is seeded; results are averaged over 20 seeds.

---

## 9. What I Took From This

This was a first, deliberately-scoped research exercise. It is not a
contribution to the field, and it was not meant to be. It was meant to build
and verify — end to end — the loop that real research runs on: define a
question precisely, implement a fair experiment, measure across seeds, plot
honestly, and report limitations without inflating the result. The most useful
thing it taught me is *why* novelty is hard: you cannot tell whether a question
is open until you know the surrounding work, and earning that knowledge is the
actual prerequisite for a real contribution. This is the first brick.

---

*Reference: Sutton, R. S. (1990). Integrated architectures for learning,
planning, and reacting based on approximating dynamic programming. (Dyna.)*
