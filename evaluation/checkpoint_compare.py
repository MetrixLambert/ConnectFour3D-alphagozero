from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from agents.alpha_beta_agent import AlphaBetaAgent
from agents.alpha_zero_agent import AlphaZeroAgent
from environments.connect_four_3d import ConnectFour3D
from evaluation.evaluate import evaluate_agents
from models.policy_value_network import PolicyValueNetwork


def _load_alpha_zero_agent(
    checkpoint_path: Path,
    *,
    env: ConnectFour3D,
    simulations: int,
    device: torch.device,
    seed: int,
) -> tuple[AlphaZeroAgent, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    network = PolicyValueNetwork(
        board_size=env.board_size,
        num_actions=env.action_space.n,
        channels=int(metadata.get("channels", 32)),
        num_residual_blocks=int(metadata.get("residual_blocks", 2)),
        value_head_channels=int(metadata.get("value_head_channels", 1)),
        value_hidden_size=int(
            metadata.get("value_hidden_size", metadata.get("channels", 32))
        ),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    return AlphaZeroAgent(
        env.observation_space,
        env.action_space,
        env=env,
        network=network,
        num_simulations=simulations,
        device=device,
        seed=seed,
    ), metadata


def evaluate_checkpoints_against_alpha_beta(
    checkpoint_paths: list[Path],
    *,
    games: int,
    simulations: int,
    alpha_beta_depth: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, int | str]]:
    """Evaluate multiple checkpoints under the same Alpha-Beta conditions."""
    results: list[dict[str, int | str]] = []
    progress = tqdm(checkpoint_paths, desc="Checkpoints", unit="checkpoint")

    for checkpoint_path in progress:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        progress.set_postfix(checkpoint=checkpoint_path.name, refresh=False)
        env = ConnectFour3D()
        candidate, metadata = _load_alpha_zero_agent(
            checkpoint_path, env=env, simulations=simulations, device=device, seed=seed
        )
        alpha_beta = AlphaBetaAgent(
            env.observation_space,
            env.action_space,
            env=env,
            max_depth=alpha_beta_depth,
            seed=seed + 2,
        )
        score = evaluate_agents(
            env,
            candidate,
            alpha_beta,
            games,
            progress_desc=f"vs AlphaBeta {checkpoint_path.stem}",
        )
        row: dict[str, int | str] = {
            "checkpoint": str(checkpoint_path),
            "iteration": int(metadata.get("iteration", -1)),
            "games": int(metadata.get("total_games", -1)),
            **score,
        }
        results.append(row)
        tqdm.write(
            f"{checkpoint_path.name}: iteration={row['iteration']} "
            f"W/D/L={row['wins']}/{row['draws']}/{row['losses']}"
        )

    return results


def evaluate_checkpoints_against_checkpoint(
    checkpoint_paths: list[Path],
    opponent_checkpoint: Path,
    *,
    games: int,
    simulations: int,
    device: torch.device,
    seed: int,
    opening_random_moves: int = 2,
) -> list[dict[str, int | str]]:
    """Evaluate candidate checkpoints against one fixed neural checkpoint."""
    results: list[dict[str, int | str]] = []
    progress = tqdm(checkpoint_paths, desc="Checkpoint arena", unit="checkpoint")
    for checkpoint_path in progress:
        checkpoint_path = Path(checkpoint_path)
        progress.set_postfix(checkpoint=checkpoint_path.name, refresh=False)
        env = ConnectFour3D()
        candidate, metadata = _load_alpha_zero_agent(
            checkpoint_path, env=env, simulations=simulations, device=device, seed=seed
        )
        opponent, _ = _load_alpha_zero_agent(
            opponent_checkpoint,
            env=env,
            simulations=simulations,
            device=device,
            seed=seed + 1,
        )
        score = evaluate_agents(
            env,
            candidate,
            opponent,
            games,
            opening_random_moves=opening_random_moves,
            seed=seed + 3,
            progress_desc=f"vs champion {checkpoint_path.stem}",
        )
        row: dict[str, int | str] = {
            "checkpoint": str(checkpoint_path),
            "iteration": int(metadata.get("iteration", -1)),
            "games": int(metadata.get("total_games", -1)),
            **score,
        }
        results.append(row)
        tqdm.write(
            f"{checkpoint_path.name}: iteration={row['iteration']} "
            f"W/D/L={row['wins']}/{row['draws']}/{row['losses']}"
        )
    return results


def format_comparison_table(results: list[dict[str, int | str]]) -> str:
    headers = ("checkpoint", "iteration", "games", "wins", "draws", "losses", "win_rate")
    rows: list[tuple[str, ...]] = []
    for result in results:
        played = int(result["wins"]) + int(result["draws"]) + int(result["losses"])
        win_rate = float(result["wins"]) / played if played else 0.0
        rows.append(
            (
                Path(str(result["checkpoint"])).name,
                str(result["iteration"]),
                str(result["games"]),
                str(result["wins"]),
                str(result["draws"]),
                str(result["losses"]),
                f"{win_rate:.1%}",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join((render(headers), separator, *(render(row) for row in rows)))
