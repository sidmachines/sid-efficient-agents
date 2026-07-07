"""
agents/q_learning.py

The BASELINE agent — model-free Q-learning.

It never builds a model of the world. It keeps one scorecard:
    Q[state][action] = "expected total reward if I take this action here,
                        then act well afterwards."

Every real step it takes, it nudges the relevant Q value a little closer
to reality using the reward it just saw plus the value of where it landed.
That's it. No T, no R, no planning — just memorizing what felt good, one
experience at a time. This is the "blunders around" agent our model-based
agent has to beat on sample efficiency.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, seed=None):
        """
        n_actions : how many actions exist (4 here).
        alpha     : learning rate — how big a nudge each experience gives.
        gamma     : discount factor — how much future reward counts.
        epsilon   : exploration rate — how often to act randomly instead
                    of greedily (so it doesn't get stuck on a bad habit).
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # The scorecard. We use a dict keyed by state so we don't need to
        # know the grid's size up front: Q[state] is an array of n_actions
        # values. Unseen states default to all-zeros.
        self.Q = {}

    def _row(self, state):
        """Return the action-value array for a state, creating it if new."""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def act(self, state):
        """
        Choose an action using epsilon-greedy:
          - with prob epsilon: explore (random action)
          - otherwise:         exploit (best action so far)
        Exploration matters because early on the scorecard is basically
        blank, so always trusting it would trap the agent in bad habits.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        row = self._row(state)
        # argmax, but break ties randomly so we don't always pick action 0
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def learn(self, state, action, reward, next_state, done):
        """
        The one update rule. Nudge Q[state][action] toward:
            reward + gamma * (value of where we landed)
        If the episode ended, there is no 'next', so that future term is 0.
        """
        q_sa = self._row(state)[action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._row(next_state).max()
        # move the old estimate a fraction (alpha) of the way to the target
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)

    def greedy_policy(self):
        """The current best action in each known state (for evaluation)."""
        return {s: int(np.argmax(row)) for s, row in self.Q.items()}


# --- train it on the gridworld and watch it improve --------------------

if __name__ == "__main__":
    from gridworld.environment import default_world

    env = default_world(seed=0)
    agent = QLearningAgent(n_actions=env.n_actions, seed=0)

    n_episodes = 500
    max_steps = 200           # give up an episode after this many steps
    returns = []              # total reward per episode

    for ep in range(n_episodes):
        s = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2
            total += r
            if done:
                break
        returns.append(total)

        # print progress every 50 episodes: average return over last 50
        if (ep + 1) % 50 == 0:
            avg = np.mean(returns[-50:])
            print(f"episode {ep+1:4d} | avg return (last 50): {avg:+.3f}")

    print("\nDone. Early episodes should be very negative (lots of wandering),")
    print("later ones closer to 0 or positive as it learns the route.")
