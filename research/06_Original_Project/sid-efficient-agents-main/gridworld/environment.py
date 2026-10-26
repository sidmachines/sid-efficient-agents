"""
gridworld/environment.py

The world both agents live in — a small stochastic gridworld.

This is the MDP from the math section, made executable:
  - state      -> (row, col) cell the robot is on
  - actions    -> 0=up, 1=down, 2=left, 3=right
  - transition -> intended move, but with slip probability the robot
                  moves in a random unintended direction (stochasticity)
  - reward     -> small step penalty everywhere, big reward at the goal
  - walls      -> cells the robot cannot enter (bumping keeps it in place)

The environment does NOT hand its rules to the agent. The agent only
ever sees (state, action, reward, next_state) — it must learn T and R
itself if it wants a model. That restriction is the whole point.
"""

import numpy as np


class GridWorld:
    # action index -> (d_row, d_col)
    MOVES = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(self, grid, start, goal, slip=0.1,
                 step_reward=-0.01, goal_reward=1.0, seed=None):
        """
        grid        : 2D list/array. 0 = free cell, 1 = wall.
        start, goal : (row, col) tuples.
        slip        : probability the robot moves in a *random* direction
                      instead of the intended one (stochastic transitions).
        step_reward : reward for any non-goal transition (small penalty).
        goal_reward : reward for reaching the goal.
        seed        : RNG seed for reproducible episodes.
        """
        self.grid = np.array(grid)
        self.n_rows, self.n_cols = self.grid.shape
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)
        self.state = self.start

    # --- helpers ---------------------------------------------------------

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_wall(self, cell):
        return self.grid[cell] == 1

    def free_cells(self):
        """All non-wall cells — the actual state space S."""
        return [tuple(c) for c in np.argwhere(self.grid == 0)]

    def _attempt_move(self, cell, action):
        """Where you'd land taking `action` from `cell`, respecting walls."""
        dr, dc = self.MOVES[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        if not self.in_bounds(nxt) or self.is_wall(nxt):
            return cell          # bumped a wall / edge -> stay put
        return nxt

    # --- the MDP interface ----------------------------------------------

    def reset(self):
        """Start a new episode. Returns the start state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take an action. Returns (next_state, reward, done).

        With prob (1 - slip) the intended action happens; with prob slip
        a uniformly random action happens instead. This is T(s, a, s').
        """
        if self.rng.random() < self.slip:
            action = self.rng.integers(self.n_actions)  # slipped!

        nxt = self._attempt_move(self.state, action)
        self.state = nxt

        if nxt == self.goal:
            return nxt, self.goal_reward, True
        return nxt, self.step_reward, False


# --- a tiny default world so you can see it work immediately -------------

def default_world(seed=None):
    """
    5x5 grid, start top-left, goal bottom-right, a couple of walls.
    0 = free, 1 = wall.
    """
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    return GridWorld(grid, start=(0, 0), goal=(4, 4), slip=0.1, seed=seed)


# --- run this file directly to sanity-check the world -------------------

if __name__ == "__main__":
    env = default_world(seed=0)
    print("State space size:", len(env.free_cells()))
    print("Start:", env.start, " Goal:", env.goal)

    # take a short random walk and watch the transitions
    s = env.reset()
    print("\nRandom walk from start:")
    for t in range(8):
        a = env.rng.integers(env.n_actions)
        s2, r, done = env.step(a)
        print(f"  step {t}: state {s} --action {a}--> {s2}  reward {r:+.2f}  done={done}")
        s = s2
        if done:
            print("  reached goal!")
            break