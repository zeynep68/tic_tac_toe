"""Play interactively against a trained TD agent."""

from __future__ import annotations

from agent import TDAgent
from env import TicTacToe


def get_human_action(env: TicTacToe) -> int:
    """Prompt the human for a valid action (0..8 = row*3 + col)."""
    valid = env.valid_actions()
    while True:
        raw = input(f"Your move {valid}: ").strip()
        if raw.isdigit() and int(raw) in valid:
            return int(raw)
        print(f"Invalid. Choose one of {valid}.")


def main(human_player: int = 2) -> None:
    """Play one game; human is player `human_player` (1 = X, 2 = O)."""
    agent_player = 3 - human_player
    agent = TDAgent(player=agent_player, epsilon=0.0)
    agent.load(f"agent{agent_player}.npy")

    env = TicTacToe()
    env.reset()

    print("Cell indices:\n0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n")

    current = 1
    while True:
        env.render()
        print()
        if current == human_player:
            action = get_human_action(env)
        else:
            action = agent.choose_action(env)
            print(f"Agent plays {action}")
        _, done, _ = env.step(action, player=current)
        if done:
            env.render()
            winner = env._winner()
            print(
                "\nDraw!"
                if winner == 0
                else f"\n{'You' if winner == human_player else 'Agent'} win{'' if winner == human_player else 's'}."
            )
            return
        current = 3 - current


if __name__ == "__main__":
    side = input("Play as X (1, goes first) or O (2)? [2] ").strip() or "2"
    main(human_player=int(side))
