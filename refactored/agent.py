"""Tabular Temporal-Difference agent for Tic-Tac-Toe self-play."""

from __future__ import annotations

import numpy as np

from env import TicTacToe


class TDAgent:
    """Tabular TD(0) agent with a NumPy state-value table.

    Policy
    ------
    Epsilon-greedy one-step lookahead: for each valid action, simulate placing
    the agent's mark, look up V(next_state), and pick the action with the
    highest successor value (greedy) or a random valid action (with
    probability epsilon).

    Learning
    --------
    TD(0) updates over the recorded trajectory at the end of an episode.
    The terminal state's value is its reward; earlier states bootstrap from
    the next visited state's value:

        V(s) <- V(s) + lr * (V_next - V(s))
    """

    def __init__(
        self,
        player: int,
        epsilon: float = 0.1,
        lr: float = 0.3,
    ) -> None:
        self.player = player
        self.epsilon = epsilon
        self.lr = lr
        self.V = np.zeros(TicTacToe.NUM_STATES)
        self.trajectory: list[tuple[int, bool, float]] = []

    def reset_trajectory(self) -> None:
        """Clear stored transitions at the start of a new episode."""
        self.trajectory.clear()

    def remember(self, state_id: int, done: bool, reward: float) -> None:
        """Append one transition observed by this agent."""
        self.trajectory.append((state_id, done, reward))

    def choose_action(self, env: TicTacToe) -> int:
        """Epsilon-greedy lookahead over valid actions."""
        valid = env.valid_actions()
        if np.random.random() < self.epsilon:
            return valid[np.random.randint(len(valid))]
        # Greedy: simulate each action in-place, restore the board, pick best.
        best_action, best_value = valid[0], -np.inf
        for action in valid:
            row, col = divmod(action, env.SIZE)
            env.board[row, col] = self.player
            value = self.V[env.state_id()]
            env.board[row, col] = 0
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def learn(self) -> None:
        """Apply TD(0) updates over the recorded trajectory (in reverse)."""
        v_next = 0.0
        for state_id, done, reward in reversed(self.trajectory):
            if done:
                # Terminal value is exactly the reward — no future to bootstrap.
                self.V[state_id] = reward
                v_next = reward
                continue
            self.V[state_id] += self.lr * (v_next - self.V[state_id])
            v_next = self.V[state_id]

    def save(self, path: str) -> None:
        """Persist the value table to disk as a NumPy file."""
        np.save(path, self.V)

    def load(self, path: str) -> None:
        """Restore the value table from disk."""
        self.V = np.load(path)
