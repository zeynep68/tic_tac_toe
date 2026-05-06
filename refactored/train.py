"""Self-play training loop for two tabular TD agents on Tic-Tac-Toe."""

from __future__ import annotations

from agent import TDAgent
from env import TicTacToe


def play_episode(
    env: TicTacToe,
    agents: dict[int, TDAgent],
    render: bool = False,
) -> None:
    """Play one game; both agents record the same trajectory with mirrored rewards."""
    env.reset()
    for agent in agents.values():
        agent.reset_trajectory()

    current = 1
    while True:
        agent = agents[current]
        action = agent.choose_action(env)
        state_id, done, reward = env.step(action, player=current)
        agent.remember(state_id, done, reward)
        agents[3 - current].remember(state_id, done, -reward)
        if render:
            env.render()
            print()
        if done:
            return
        current = 3 - current


def main(num_episodes: int = 40_000) -> None:
    env = TicTacToe()
    agents = {p: TDAgent(player=p) for p in (1, 2)}

    for episode in range(num_episodes):
        play_episode(env, agents)
        for agent in agents.values():
            agent.learn()
        if (episode + 1) % 2_000 == 0:
            print(f"episode {episode + 1}/{num_episodes}")

    print("\nFinal greedy match:")
    for agent in agents.values():
        agent.epsilon = 0.0
    play_episode(env, agents, render=True)

    for player, agent in agents.items():
        agent.save(f"agent{player}.npy")
    print("\nSaved value tables to agent1.npy and agent2.npy.")


if __name__ == "__main__":
    main()
