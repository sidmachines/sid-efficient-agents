"""
agents/model_based.py

The STAR agent — model-based control.

Same experience stream as the baseline, but instead of only updating a
scorecard, it *remembers* what it sees and builds a model of the world:

    R_hat(s, a)      = average reward observed for (s, a)
    T_hat(s, a, s')  = fraction of times (s, a) led to s'   (from counts)

Then it runs VALUE ITERATION on that self-built model to compute the best
action in every state, and acts on that plan. Every real step improves the
*model*, which improves planning *everywhere* at once -- not just one value.
That reuse is why it should need far less experience than Q-learning.

    explore a little -> update counts -> re-plan (value iteration) -> act
"""

import numpy as np


class ModelBasedAgent:
    def __init__(self, n_actions, gamma=0.95, epsilon=0.1,
                 vi_sweeps=50, replan_every=10, seed=None):
        """
        n_actions    : number of actions (4).
        gamma        : discount factor (same as baseline, for fair comparison).
        epsilon      : exploration rate (same as baseline, for fair comparison).
        vi_sweeps    : how many Bellman sweeps per planning pass.
        replan_every : re-run value iteration every N environment steps
                       (planning is the expensive bit, so we batch it).
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.vi_sweeps = vi_sweeps
        self.replan_every = replan_every
        self.rng = np.random.default_rng(seed)

        # --- the learned model, built from raw counts ---
        # counts of (s, a) -> s' transitions, and summed rewards for (s, a)
        self.trans_counts = {}   # {(s, a): {s': count}}
        self.reward_sum = {}     # {(s, a): total reward seen}
        self.sa_count = {}       # {(s, a): times taken}
        self.states = set()      # every state we've encountered

        # the current plan: value of each state, and best action per state
        self.V = {}
        self.policy = {}

        self._steps_since_replan = 0

    # --- 1. observe: fold one real experience into the model -------------

    def observe(self, state, action, reward, next_state):
        """Update the counts that define T_hat and R_hat. Pure bookkeeping."""
        self.states.add(state)
        self.states.add(next_state)
        key = (state, action)

        self.sa_count[key] = self.sa_count.get(key, 0) + 1
        self.reward_sum[key] = self.reward_sum.get(key, 0.0) + reward

        if key not in self.trans_counts:
            self.trans_counts[key] = {}
        self.trans_counts[key][next_state] = \
            self.trans_counts[key].get(next_state, 0) + 1

        self._steps_since_replan += 1
        if self._steps_since_replan >= self.replan_every:
            self.plan()
            self._steps_since_replan = 0

    # --- 2. the learned model, read out from the counts -----------------

    def r_hat(self, s, a):
        """Estimated reward for (s, a) = average of rewards seen."""
        key = (s, a)
        n = self.sa_count.get(key, 0)
        if n == 0:
            return 0.0
        return self.reward_sum[key] / n

    def t_hat(self, s, a):
        """Estimated transitions for (s, a) = {s': probability}."""
        key = (s, a)
        counts = self.trans_counts.get(key)
        if not counts:
            return {}
        total = self.sa_count[key]
        return {s2: c / total for s2, c in counts.items()}

    # --- 3. plan: value iteration on the learned model ------------------

    def plan(self):
        """
        Run value iteration using T_hat and R_hat. This is the SAME Bellman
        update from the Day 1 lecture -- the only difference is we run it on
        the model we *built*, not one we were handed.
        """
        V = {s: self.V.get(s, 0.0) for s in self.states}

        for _ in range(self.vi_sweeps):
            new_V = {}
            for s in self.states:
                best = -np.inf
                for a in range(self.n_actions):
                    # Bellman backup for this action, under the learned model
                    q = self.r_hat(s, a)
                    for s2, p in self.t_hat(s, a).items():
                        q += self.gamma * p * V.get(s2, 0.0)
                    if q > best:
                        best = q
                # states we've never acted from have no info yet -> leave at 0
                new_V[s] = best if best != -np.inf else 0.0
            V = new_V

        # read out the greedy policy from the converged values
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

    # --- 4. act: follow the plan, explore occasionally ------------------

    def act(self, state):
        """Epsilon-greedy over the current plan (same exploration as baseline)."""
        if self.rng.random() < self.epsilon or state not in self.policy:
            return self.rng.integers(self.n_actions)
        return self.policy[state]


# --- train it on the same world and compare to the baseline's ~100 eps --

if __name__ == "__main__":
    from gridworld.environment import default_world

    env = default_world(seed=0)
    agent = ModelBasedAgent(n_actions=env.n_actions, seed=0)

    n_episodes = 500
    max_steps = 200
    returns = []

    for ep in range(n_episodes):
        s = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.observe(s, a, r, s2)     # <-- build the model
            s = s2
            total += r
            if done:
                break
        returns.append(total)

        if (ep + 1) % 10 == 0:            # print MORE often: watch it rocket up
            avg = np.mean(returns[-10:])
            print(f"episode {ep+1:4d} | avg return (last 10): {avg:+.3f}")

    print("\nCompare the FIRST few lines to the baseline: this agent should")
    print("reach high return in a small fraction of the episodes Q-learning needed.")
