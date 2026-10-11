from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm


def _play_game(env, player_one, player_two, opening_actions=()) -> int:
    observation, info = env.reset()
    player_one.reset_episode()
    player_two.reset_episode()
    for action in opening_actions:
        observation, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return int(info["winner"])
    while True:
        agent = player_one if info["current_player"] == 1 else player_two
        action = agent.select_action(observation, training=False)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return int(info["winner"])


def evaluate_agents(
    env,
    candidate,
    opponent,
    games: int = 20,
    *,
    opening_random_moves: int = 0,
    seed: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, int]:
    """Evaluate both seat assignments; odd extra games give candidate player 1."""
    results = {"wins": 0, "draws": 0, "losses": 0}
    rng = np.random.default_rng(seed)
    paired_opening: list[int] = []
    game_indices = tqdm(
        range(games),
        desc=progress_desc,
        unit="game",
        leave=False,
        disable=progress_desc is None,
    )
    for game_index in game_indices:
        if opening_random_moves and game_index % 2 == 0:
            env.reset()
            paired_opening = []
            for _ in range(opening_random_moves):
                valid = env.get_valid_actions()
                action = int(rng.choice(valid))
                paired_opening.append(action)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        candidate_is_player_one = game_index % 2 == 0
        if candidate_is_player_one:
            winner = _play_game(env, candidate, opponent, paired_opening)
            candidate_result = winner
        else:
            winner = _play_game(env, opponent, candidate, paired_opening)
            candidate_result = -winner

        if candidate_result > 0:
            results["wins"] += 1
        elif candidate_result < 0:
            results["losses"] += 1
        else:
            results["draws"] += 1
        game_indices.set_postfix(
            wins=results["wins"],
            draws=results["draws"],
            losses=results["losses"],
            refresh=False,
        )
    return results
