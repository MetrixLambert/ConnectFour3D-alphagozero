from __future__ import annotations

import numpy as np

from self_play.replay_buffer import SelfPlaySample


def play_self_play_game(env, agent) -> tuple[list[SelfPlaySample], int]:
    """Play one game and return canonical training samples and its winner."""
    observation, info = env.reset()
    agent.reset_episode()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while True:
        player = int(info["current_player"])
        action = agent.select_action(observation, training=True)
        policy = agent.get_policy()
        if policy is None:
            raise RuntimeError("AlphaZeroAgent did not expose its search policy.")
        canonical_state = (observation.astype(np.float32) * player).copy()
        history.append((canonical_state, policy, player))

        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            winner = int(info["winner"])
            break

    samples = [
        SelfPlaySample(
            state=state,
            policy=policy,
            value=0.0 if winner == 0 else float(winner * player),
        )
        for state, policy, player in history
    ]
    return samples, winner
