"""
experiments/memory_sweep.py

Day 6 -- the experiment that IS the contribution.

Question: how little world-model does the agent need before planning still
beats plain model-free learning? We sweep the model-based agent's memory
cap from tiny to unlimited and measure the policy it reaches, across many
seeds. The model-free baseline is drawn as a flat reference floor.

The picture: performance stays down near the baseline for tiny memories,
then rises through a KNEE to full performance and plateaus. The knee is the
finding -- "about N remembered experiences unlocks most of the advantage."

NOTE ON SPEED / FAIRNESS:
The model-based agent re-plans with value iteration, which is the expensive
part. We use replan_every=50 and vi_sweeps=20 -- plenty for a 21-state world
(values converge fast) and much faster than the defaults. These settings are
applied consistently to EVERY model-based run so all figures are comparable.

Run from the project root:
    python -m experiments.memory_sweep
"""

import numpy as np
import matplotlib.pyplot as plt

from gridworld.environment import default_world
from agents.q_learning import QLearningAgent
from agents.starved_model_based import StarvedModelBasedAgent


# --- consistent model-based planning settings (used everywhere) ----------
REPLAN_EVERY = 50
VI_SWEEPS = 20


def evaluate_greedy(agent, n_eval=20, max_steps=200):
    """Greedy, no-learning evaluation of the current policy (fixed test seeds)."""
    total = 0.0
    for k in range(n_eval):
        env = default_world(seed=1000 + k)
        s = env.reset()
        for _ in range(max_steps):
            if isinstance(agent, QLearningAgent):
                row = agent.Q.get(s)
                a = 0 if row is None else int(np.argmax(row))
            else:
                a = agent.policy.get(s, 0)
            s, r, done = env.step(a)
            total += r
            if done:
                break
    return total / n_eval


def train_starved(memory_size, seed, n_episodes=80, max_steps=200):
    env = default_world(seed=seed)
    agent = StarvedModelBasedAgent(n_actions=env.n_actions,
                                   memory_size=memory_size, seed=seed,
                                   replan_every=REPLAN_EVERY, vi_sweeps=VI_SWEEPS)
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.observe(s, a, r, s2)
            s = s2
            if done:
                break
    return evaluate_greedy(agent)


def train_baseline(seed, n_episodes=80, max_steps=200):
    env = default_world(seed=seed)
    agent = QLearningAgent(n_actions=env.n_actions, seed=seed)
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2
            if done:
                break
    return evaluate_greedy(agent)


if __name__ == "__main__":
    N_SEEDS = 20
    # memory sizes to sweep; None = unlimited (plotted at the far right)
    MEMORIES = [2, 4, 8, 16, 32, 64, 128, 256, None]
    N_EPISODES = 80        # fixed experience budget, same for every setting

    # --- sweep the starved model-based agent ---
    means, stds = [], []
    for mem in MEMORIES:
        scores = [train_starved(mem, seed, N_EPISODES) for seed in range(N_SEEDS)]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        label = "unlimited" if mem is None else str(mem)
        print(f"memory={label:>9} | mean {means[-1]:+.3f} ± {stds[-1]:.3f}")

    # --- baseline floor (model-free, same budget) ---
    base_scores = [train_baseline(seed, N_EPISODES) for seed in range(N_SEEDS)]
    base_mean, base_std = np.mean(base_scores), np.std(base_scores)
    print(f"\nmodel-free baseline (same {N_EPISODES} eps): "
          f"{base_mean:+.3f} ± {base_std:.3f}")

    # --- plot: performance vs memory size ---
    # represent 'unlimited' as one step past the largest finite memory on a log axis
    finite = [m for m in MEMORIES if m is not None]
    xs = finite + [finite[-1] * 2]        # place 'unlimited' at 2x the max
    xticklabels = [str(m) for m in finite] + ["\u221e"]

    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(8, 5))
    plt.axhline(base_mean, color="tab:red", linestyle="--", linewidth=1.5,
                label=f"model-free baseline ({base_mean:+.2f})")
    plt.fill_between([xs[0], xs[-1]], base_mean - base_std, base_mean + base_std,
                     color="tab:red", alpha=0.10)
    plt.plot(xs, means, "o-", color="tab:blue", linewidth=2,
             label="model-based, memory-capped")
    plt.fill_between(xs, means - stds, means + stds, color="tab:blue", alpha=0.15)

    plt.xscale("log")
    plt.xticks(xs, xticklabels)
    plt.xlabel("Model memory size (transitions remembered)  \u2014  log scale")
    plt.ylabel("Greedy return after fixed experience budget")
    plt.title("How little model is enough?\n"
              "Planning advantage vs. memory size (mean \u00b1 std, 20 seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = "results/memory_sweep.png"
    plt.savefig(out, dpi=150)
    print(f"\nsaved plot to {out}")
