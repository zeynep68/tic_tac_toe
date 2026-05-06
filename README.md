# Tic-Tac-Toe — Self-Play TD Learning

A small reinforcement-learning project: two agents learn to play tic-tac-toe through self-play, using tabular Temporal-Difference learning.

## Two implementations

- **`old_version/`** — original implementation (NumPy, single-file agent + a separate CartPole DQN notebook) from 2020 / 2021. Kept for reference.
- **`refactored/`** — cleaner rewrite: object-oriented, type-hinted, terminal state values fixed, save/load support, interactive play mode. Recommended.

## Quick start

```bash
cd refactored
python train.py     # train via self-play, ~30 seconds
python play.py      # play interactively against the trained agent
```

See refactored/README.md for details on the design and individual files.
