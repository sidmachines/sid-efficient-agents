"""
experiments/compare.py

Day 5 -- the head-to-head experiment.

Turns "the model-based agent seemed faster" into a measured, defensible
result. The design has three rigor pieces:

  1. SAME harness for both agents (fair comparison).
  2. MANY seeds, averaged, with a spread band (result, not luck).
  3. Performance measured vs AMOUNT OF EXPERIENCE (the sample-efficiency
     curve) -- we periodically freeze learning and run a clean greedy test,
     so the y-axis is "how good is the policy" and the x-axis is "how much
     experience did it take to get there." That plot IS the claim.

Run from the project root:
    python -m experiments.compare
"""

import numpy as np
import matplotlib.pyplot as plt

from gridworld.environment import default_world
from agents.q_learning import QLearningAgent
from agents.model_based import ModelBasedAgent


# --- clean evaluation: no exploration, no learning, just measure ---------

def evaluate(agent, make_env, n_eval=20, max_steps=200):
    """
    Run the agent GREEDILY (no exploration, no learning) for a few episodes
    and return the average return. This measures how good its current policy
    actually is, separate from the noise of training-time exploration.
    """
    total = 0.0
    for k in range(n_eval):
        env = make_env(seed=1000 + k)      # fixed eval seeds, same for everyone
        s = env.reset()
        for _ in range(max_steps):
            a = greedy_action(agent, s)
            s, r, done = env.step(a)
            total += r
            if done:
                break
    return total / n_eval


def greedy_action(agent, state):
    """Ask either agent for its best action, no randomness."""
    if isinstance(agent, QLearningAgent):
        row = agent.Q.get(state)
        if row is None:
            return 0
        return int(np.argmax(row))
    else:  # ModelBasedAgent
        return agent.policy.get(state, 0)


# --- train one agent for N episodes, measuring along the way -------------

def train_agent(agent_kind, seed, eval_points, max_steps=200):
    """
    Train a fresh agent of the given kind, and at each episode number in
    `eval_points`, pause and record its greedy performance. Returns a list
    of scores aligned with eval_points.
    """
    env = default_world(seed=seed)
    if agent_kind == "model_free":
        agent = QLearningAgent(n_actions=env.n_actions, seed=seed)
    else:
        agent = ModelBasedAgent(n_actions=env.n_actions, seed=seed)

    scores = []
    max_ep = max(eval_points)
    for ep in range(1, max_ep + 1):
        s = env.reset()
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, done = env.step(a)
            if agent_kind == "model_free":
                agent.learn(s, a, r, s2, done)
            else:
                agent.observe(s, a, r, s2)
            s = s2
            if done:
                break
        if ep in eval_points:
            scores.append(evaluate(agent, default_world))
    return scores


# --- run both agents across many seeds and plot ---------------------------

if __name__ == "__main__":
    N_SEEDS = 20
    EVAL_POINTS = [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 60, 80, 120, 160, 200]

    results = {}
    for kind in ["model_free", "model_based"]:
        print(f"running {kind} across {N_SEEDS} seeds...")
        all_runs = []
        for seed in range(N_SEEDS):
            all_runs.append(train_agent(kind, seed, EVAL_POINTS))
        results[kind] = np.array(all_runs)   # shape: (N_SEEDS, len(EVAL_POINTS))

    # mean and standard deviation across seeds, at each eval point
    x = np.array(EVAL_POINTS)
    plt.figure(figsize=(8, 5))
    for kind, color, label in [
        ("model_free", "tab:red", "Model-free (Q-learning) — the baseline"),
        ("model_based", "tab:blue", "Model-based (learn model + plan) — the star"),
    ]:
        data = results[kind]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        plt.plot(x, mean, color=color, label=label, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("Training episodes (log scale)  —  amount of experience")
    plt.ylabel("Greedy return (higher = better policy)")
    plt.title("Sample efficiency: performance vs. experience\n"
              "(mean ± std over 20 seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = "results/sample_efficiency.png"
    plt.savefig(out, dpi=150)
    print(f"\nsaved plot to {out}")

    # also print a small table so the numbers are in the terminal too
    print("\nepisodes |  model-free  | model-based")
    for i, ep in enumerate(EVAL_POINTS):
        mf = results["model_free"].mean(axis=0)[i]
        mb = results["model_based"].mean(axis=0)[i]
        print(f"{ep:8d} |   {mf:+.3f}    |   {mb:+.3f}")
