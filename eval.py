from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from agents.alpha_beta_agent import AlphaBetaAgent
from agents.alpha_zero_agent import AlphaZeroAgent
from agents.random_agent import RandomAgent
from environments.connect_four_3d import ConnectFour3D
from evaluation.checkpoint_compare import (
    evaluate_checkpoints_against_alpha_beta,
    evaluate_checkpoints_against_checkpoint,
    format_comparison_table,
)
from evaluation.evaluate import evaluate_agents
from models.policy_value_network import PolicyValueNetwork


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AlphaZero model.")
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Compare multiple checkpoints against the same Alpha-Beta agent.",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--alpha-beta-depth", type=int, default=2)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("alpha_beta", "checkpoint"),
        help="Batch opponents to evaluate: alpha_beta and/or checkpoint.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Also play each batch checkpoint against this fixed checkpoint.",
    )
    parser.add_argument(
        "--arena-opening-moves",
        type=int,
        default=2,
        help="Random opening plies used in paired checkpoint arena games.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.checkpoint is None) == (args.checkpoints is None):
        raise ValueError("Provide exactly one of checkpoint or --checkpoints")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if args.checkpoints is not None:
        opponents = args.opponents
        if opponents is None:
            opponents = ["alpha_beta"]
            if args.opponent_checkpoint is not None:
                opponents.append("checkpoint")
        if "checkpoint" in opponents and args.opponent_checkpoint is None:
            raise ValueError(
                "opponents includes 'checkpoint', but --opponent-checkpoint is missing"
            )
        print(
            f"device={device} checkpoints={len(args.checkpoints)} "
            f"opponents={','.join(opponents)}"
        )
        if "alpha_beta" in opponents:
            results = evaluate_checkpoints_against_alpha_beta(
                args.checkpoints,
                games=args.games,
                simulations=args.simulations,
                alpha_beta_depth=args.alpha_beta_depth,
                device=device,
                seed=args.seed,
            )
            print("\nCheckpoint comparison vs AlphaBetaAgent")
            print(format_comparison_table(results))
        if "checkpoint" in opponents:
            print(
                f"\nEvaluating against checkpoint: {args.opponent_checkpoint.name}"
            )
            arena_results = evaluate_checkpoints_against_checkpoint(
                args.checkpoints,
                args.opponent_checkpoint,
                games=args.games,
                simulations=args.simulations,
                device=device,
                seed=args.seed,
                opening_random_moves=args.arena_opening_moves,
            )
            print(f"\nCheckpoint comparison vs {args.opponent_checkpoint.name}")
            print(format_comparison_table(arena_results))
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})

    env = ConnectFour3D()
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
    candidate = AlphaZeroAgent(
        env.observation_space,
        env.action_space,
        env=env,
        network=network,
        num_simulations=args.simulations,
        device=device,
        seed=args.seed,
    )
    random_agent = RandomAgent(env.observation_space, env.action_space, seed=args.seed + 1)
    alpha_beta = AlphaBetaAgent(
        env.observation_space,
        env.action_space,
        env=env,
        max_depth=args.alpha_beta_depth,
        seed=args.seed + 2,
    )

    print(f"device={device} checkpoint={args.checkpoint}")
    print(
        "vs RandomAgent:",
        evaluate_agents(
            env, candidate, random_agent, args.games, progress_desc="vs RandomAgent"
        ),
    )
    print(
        "vs AlphaBetaAgent:",
        evaluate_agents(
            env,
            candidate,
            alpha_beta,
            args.games,
            progress_desc="vs AlphaBetaAgent",
        ),
    )


if __name__ == "__main__":
    main()
