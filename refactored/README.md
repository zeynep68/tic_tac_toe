# Tic-Tac-Toe — Tabular TD-Learning

Self-play Temporal-Difference agents on a 3x3 board. Each agent maintains its
own state-value table V(s) of shape (3^9,) = (19,683,) and updates it after
every episode with TD(0).

## Files

| File       | Purpose                                                                            |
| ---------- | ---------------------------------------------------------------------------------- |
| `env.py`   | Game environment: board, step/reset, winner check, board encoding                  |
| `agent.py` | `TDAgent`: epsilon-greedy lookahead policy + TD(0) value-table updates + save/load |
| `train.py` | Self-play loop: train two agents and save their value tables                       |
| `play.py`  | Play interactively against a saved agent                                           |

## Run

```bash
python train.py        # train via self-play, writes agent1.npy + agent2.npy
python play.py         # play a single game against the trained agent
```

`train.py` runs 20,000 episodes, prints a final greedy match, and saves both
value tables. `play.py` then loads the saved agent and prompts you for moves
(0..8 = row \* 3 + col). Requires only `numpy`.

## Notes

- **State encoding** is bijective: a 3x3 board with cells in {0, 1, 2} maps to
  a unique integer in [0, 3^9). This integer indexes into the value table.
- **Reward** is given only at terminal states. The moving player's reward is
  mirrored to the opponent in `train.py` so that both agents see consistent
  win/loss signals across the same trajectory.
- **Tabular TD(0)** is sufficient for the 3x3 board (~5,500 reachable states
  out of 19,683 encodable). For larger games the same TD update rule
  generalises to function approximators (e.g. a neural network) — only the
  representation of `V` changes.
