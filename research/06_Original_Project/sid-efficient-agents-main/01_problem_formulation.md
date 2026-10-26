# The Problem, Formally

*This is the math section of the paper. It defines the world both agents operate in, so that every later claim ("the model-based agent needs less experience") is stated precisely rather than hand-waved.*

---

## 1. The environment as an MDP

We study a finite, episodic **Markov Decision Process** defined by the tuple

**M = (S, A, T, R, γ)**

where:

- **S** — the finite set of states. In our gridworld, a state *s* is the robot's cell, i.e. a coordinate `(row, col)`.
- **A** — the finite set of actions. Here **A = { up, down, left, right }**.
- **T(s, a, s′) = P(s′ | s, a)** — the transition function: the probability of landing in state *s′* after taking action *a* in state *s*. This encodes both the grid's walls (you can't move into a wall) and its **stochasticity** (a "slip" probability *p* that the robot moves in an unintended direction).
- **R(s, a, s′)** — the reward function: a scalar received on each transition. We use a small step penalty (to encourage short paths) and a large terminal reward at the goal.
- **γ ∈ [0, 1)** — the discount factor, making future reward worth less than immediate reward and keeping the infinite-horizon return finite.

An episode starts at a fixed start state and ends when the robot reaches the goal (a terminal state).

## 2. What the agent wants

A **policy** π maps each state to an action, **π : S → A**.

The **value** of a state under a policy is the expected discounted return from that state:

**Vπ(s) = E[ Σₜ γᵗ · rₜ  |  s₀ = s, π ]**

The agent seeks the **optimal policy π\*** that maximizes this everywhere. Its value function **V\*** satisfies the **Bellman optimality equation**:

**V\*(s) = maxₐ  Σₛ′  T(s, a, s′) · [ R(s, a, s′) + γ · V\*(s′) ]**

*(In words: the value of a state is the best action's immediate reward plus the discounted value of where that action lands you, averaged over the uncertainty.)*

**Value iteration** computes V\* by applying the right-hand side as an update rule repeatedly until the values converge; the optimal policy is then the action that achieves the max in each state.

## 3. The two settings we compare — this is the whole paper

The distinction our study rests on is **what the agent knows about T and R.**

**Setting A — model-free (the baseline).**
The agent is *not* given T or R and does *not* try to reconstruct them. It only ever observes samples: "I was in *s*, did *a*, got reward *r*, landed in *s′*." It learns action-values Q(s, a) directly from these samples by trial and error (tabular Q-learning). It never builds a map; it memorizes what felt good.

**Setting B — model-based (the star).**
The agent is *also* not given T or R, but it *estimates* them from the same stream of experience:

- **T̂(s, a, s′)** — estimated by counting observed transitions: how often did *a* in *s* actually lead to *s′*.
- **R̂(s, a, s′)** — estimated by averaging observed rewards.

It then runs **value iteration on its estimated model (T̂, R̂)** to plan, acts, gathers more experience, and refines the model. Learn the map, then plan on it.

## 4. The claim we will test

Both agents converge to (near-)optimal behavior. The hypothesis is about **efficiency**, not final score:

> **The model-based agent reaches near-optimal performance using substantially less environment experience than the model-free agent**, because each observed transition updates a reusable model of the world rather than a single value estimate.

We measure this with two quantities, reported across many random seeds:

1. **Sample efficiency** — environment steps (or episodes) needed to reach a performance threshold.
2. **Compute** — wall-clock / operation cost, to check the model-based agent's planning overhead doesn't erase its sample-efficiency win.

The gap between the two curves *is* the result: a concrete, measured instance of **performance over power.**
