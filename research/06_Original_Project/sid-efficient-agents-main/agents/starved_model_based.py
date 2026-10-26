"""
agents/starved_model_based.py

The model-based agent from Day 4, with ONE new knob: a memory cap.

Normally the agent remembers every transition it has ever seen, building an
ever-more-complete map. Here we cap it: it can only keep the most recent
`memory_size` experiences. Older ones are forgotten and their counts removed.

This lets us STARVE the model on purpose and ask the paper's real question:
    How complete does the map have to be before planning still beats
    plain model-free learning? How little "understanding" is enough?

memory_size = infinite  -> the full Day-4 agent (rich map)
memory_size = tiny       -> a forgetful, starved map (should behave more
                            like the model-free baseline)

Sweeping memory_size from tiny to large traces out exactly where the
model-based advantage switches on. The knee in that curve is the finding.
"""

import numpy as np
from collections import deque


class StarvedModelBasedAgent:
    def __init__(self, n_actions, gamma=0.95, epsilon=0.1,
                 vi_sweeps=50, replan_every=10, memory_size=None, seed=None):
        """
        memory_size : max number of recent transitions to remember.
                      None = unlimited (the full Day-4 agent).
        Everything else is identical to the Day-4 ModelBasedAgent, so any
        performance difference is attributable ONLY to the memory cap.
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.vi_sweeps = vi_sweeps
        self.replan_every = replan_every
        self.memory_size = memory_size
        self.rng = np.random.default_rng(seed)

        # A rolling buffer of the most recent experiences. When it overflows,
        # the oldest experience is evicted and we subtract it from the counts.
        self.memory = deque(maxlen=memory_size)   # maxlen=None means unlimited

        # the learned model, rebuilt/maintained from whatever is in memory
        self.trans_counts = {}   # {(s, a): {s': count}}
        self.reward_sum = {}     # {(s, a): summed reward}
        self.sa_count = {}       # {(s, a): count}
        self.states = set()

        self.V = {}
        self.policy = {}
        self._steps_since_replan = 0

    # --- adding and removing a single experience from the counts ---------

    def _add_counts(self, s, a, r, s2):
        key = (s, a)
        self.sa_count[key] = self.sa_count.get(key, 0) + 1
        self.reward_sum[key] = self.reward_sum.get(key, 0.0) + r
        self.trans_counts.setdefault(key, {})
        self.trans_counts[key][s2] = self.trans_counts[key].get(s2, 0) + 1
        self.states.add(s)
        self.states.add(s2)

    def _remove_counts(self, s, a, r, s2):
        """Undo one experience's contribution when it's evicted from memory."""
        key = (s, a)
        self.sa_count[key] -= 1
        self.reward_sum[key] -= r
        self.trans_counts[key][s2] -= 1
        # clean up empties so estimates stay well-defined
        if self.trans_counts[key][s2] <= 0:
            del self.trans_counts[key][s2]
        if self.sa_count[key] <= 0:
            self.sa_count.pop(key, None)
            self.reward_sum.pop(key, None)
            self.trans_counts.pop(key, None)

    def observe(self, state, action, reward, next_state):
        # If memory is full, the deque will drop the oldest on append; capture
        # it first so we can subtract it from the counts.
        evicted = None
        if self.memory_size is not None and len(self.memory) == self.memory_size:
            evicted = self.memory[0]

        self.memory.append((state, action, reward, next_state))
        self._add_counts(state, action, reward, next_state)
        if evicted is not None:
            self._remove_counts(*evicted)

        self._steps_since_replan += 1
        if self._steps_since_replan >= self.replan_every:
            self.plan()
            self._steps_since_replan = 0

    # --- model read-out (identical to Day 4) ----------------------------

    def r_hat(self, s, a):
        n = self.sa_count.get((s, a), 0)
        return self.reward_sum[(s, a)] / n if n > 0 else 0.0

    def t_hat(self, s, a):
        counts = self.trans_counts.get((s, a))
        if not counts:
            return {}
        total = self.sa_count[(s, a)]
        return {s2: c / total for s2, c in counts.items()}

    # --- planning (identical to Day 4: value iteration on the map) -------

    def plan(self):
        V = {s: self.V.get(s, 0.0) for s in self.states}
        for _ in range(self.vi_sweeps):
            new_V = {}
            for s in self.states:
                best = -np.inf
                for a in range(self.n_actions):
                    q = self.r_hat(s, a)
                    for s2, p in self.t_hat(s, a).items():
                        q += self.gamma * p * V.get(s2, 0.0)
                    best = max(best, q)
                new_V[s] = best if best != -np.inf else 0.0
            V = new_V
        policy = {}
        for s in self.states:
            best_a, best_q = 0, -np.inf
            for a in range(self.n_actions):
                q = self.r_hat(s, a)
                for s2, p in self.t_hat(s, a).items():
                    q += self.gamma * p * V.get(s2, 0.0)
                if q > best_q:
                    best_q, best_a = q, a
            policy[s] = best_a
        self.V, self.policy = V, policy

    def act(self, state):
        if self.rng.random() < self.epsilon or state not in self.policy:
            return self.rng.integers(self.n_actions)
        return self.policy[state]


if __name__ == "__main__":
    from gridworld.environment import default_world

    # quick sanity check: a tiny memory should clearly hurt, a big one shouldn't
    for mem in [5, 25, 100, None]:
        env = default_world(seed=0)
        agent = StarvedModelBasedAgent(n_actions=env.n_actions,
                                       memory_size=mem, seed=0)
        returns = []
        for ep in range(60):
            s = env.reset()
            total = 0.0
            for _ in range(200):
                a = agent.act(s)
                s2, r, done = env.step(a)
                agent.observe(s, a, r, s2)
                s = s2
                total += r
                if done:
                    break
            returns.append(total)
        label = "unlimited" if mem is None else f"{mem:>4}"
        print(f"memory={label} | avg return over 60 eps: {np.mean(returns):+.3f}")
