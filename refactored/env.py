"""Tic-Tac-Toe game environment for tabular reinforcement learning."""

from __future__ import annotations

import numpy as np


class TicTacToe:
    """3x3 board for self-play RL.

    Cells: 0 (empty), 1 (player 1), 2 (player 2).
    Actions: integers 0..8 in row-major order.
    Reward is given to the moving player at terminal states only:
    +1 for a win, 0 for draw or non-terminal step.
    """

    SIZE = 3
    NUM_STATES = 3**9

    def __init__(self) -> None:
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)

    def reset(self) -> None:
        """Clear the board."""
        self.board.fill(0)

    def step(self, action: int, player: int) -> tuple[int, bool, float]:
        """Apply player's move at cell 'action'. Returns (state_id, done, reward)."""
        row, col = divmod(action, self.SIZE)
        self.board[row, col] = player
        winner = self._winner()
        done = winner != 0 or not self.valid_actions()
        reward = 1.0 if winner == player else 0.0
        return self.state_id(), done, reward

    def state_id(self) -> int:
        """Bijective base-3 encoding of the board into a unique integer in [0, 3**9).

        Treats the 9 cells (row-major) as digits of a base-3 number:

            id = c0*3**0 + c1*3**1 + c2*3**2 + ... + c8*3**8

        where each cell value c_i is in {0, 1, 2}. Different boards always
        produce different ids, so the result can be used as a direct index
        into the value table.

        Example:
            board = [[1, 2, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            flatten      -> [1, 2, 0, 0, 1, 0, 0, 0, 0]
            powers       -> [1, 3, 9, 27, 81, 243, 729, 2187, 6561]
            elementwise  -> [1, 6, 0, 0, 81, 0, 0, 0, 0]
            sum          -> 88
        """
        return int(np.sum(3 ** np.arange(9) * self.board.flatten()))

    def valid_actions(self) -> list[int]:
        """Indices of empty cells."""
        return [i for i, v in enumerate(self.board.flatten()) if v == 0]

    def _winner(self) -> int:
        """Return winner's player number (1 or 2), or 0 if no winner yet."""
        b = self.board
        lines = (
            [b[i, :] for i in range(self.SIZE)]  # rows (3)
            + [b[:, i] for i in range(self.SIZE)]  # columns (3)
            + [np.diag(b), np.diag(np.fliplr(b))]  # diagonals (2)
        )
        for line in lines:
            if line[0] != 0 and np.all(line == line[0]):
                return int(line[0])
        return 0

    def render(self) -> None:
        """Print the board to stdout."""
        symbols = {0: " ", 1: "X", 2: "O"}
        for i, row in enumerate(self.board):
            print(" | ".join(symbols[int(v)] for v in row))
            if i < self.SIZE - 1:
                print("-" * 9)
