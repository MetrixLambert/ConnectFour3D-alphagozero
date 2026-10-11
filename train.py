from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from agents.alpha_zero_agent import AlphaZeroAgent
from environments.connect_four_3d import ConnectFour3D
from models.policy_value_network import PolicyValueNetwork
from self_play.replay_buffer import ReplayBuffer
from self_play.self_play import play_self_play_game
from training.trainer import AlphaZeroTrainer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact 3D Connect Four AlphaZero model.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument(
        "--late-game-simulations",
        type=int,
        default=None,
        help="Optional self-play MCTS budget used from --late-game-simulation-move onward.",
    )
    parser.add_argument(
        "--late-game-simulation-move",
        type=int,
        default=20,
        help="Self-play ply at which --late-game-simulations takes effect.",
    )
    parser.add_argument("--training-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument(
        "--recent-sample-window",
        type=int,
        default=10_000,
        help="Number of newest replay samples in the recent stratum.",
    )
    parser.add_argument(
        "--recent-sample-fraction",
        type=float,
        default=0.0,
        help="Batch fraction reserved for recent replay samples; 0 disables.",
    )
    parser.add_argument(
        "--ply-sample-fractions",
        type=float,
        nargs=4,
        metavar=("PLY_0_9", "PLY_10_19", "PLY_20_29", "PLY_30_PLUS"),
        default=None,
        help="Optional fixed batch fractions for ply ranges 0-9, 10-19, 20-29, and 30+.",
    )
    parser.add_argument(
        "--resize-buffer-on-resume",
        action="store_true",
        help="Resize a restored replay buffer to --buffer-size after loading.",
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--value-head-channels", type=int, default=1)
    parser.add_argument("--value-hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperature-drop-move",
        type=int,
        default=10,
        help="Use --post-drop-temperature from this self-play ply onward.",
    )
    parser.add_argument("--post-drop-temperature", type=float, default=0.1)
    parser.add_argument(
        "--symmetry-augmentation",
        action="store_true",
        help="Randomly apply one of 8 horizontal board symmetries per sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("saved_models/latest.pt"))
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="Save a numbered full checkpoint every N global iterations; 0 disables snapshots.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("saved_models/checkpoints"),
        help="Directory for numbered historical checkpoints.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize compatible weights and replay buffer from another architecture.",
    )
    parser.add_argument(
        "--reuse-value-head",
        action="store_true",
        help="Also reuse compatible value-head tensors during architecture initialization.",
    )
    parser.add_argument(
        "--pretraining-steps",
        type=int,
        default=0,
        help="Train on the restored replay buffer before starting self-play.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Write a metrics line every N completed iterations.",
    )
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def get_snapshot_path(
    checkpoint: Path, snapshot_dir: Path, iteration: int
) -> Path:
    suffix = checkpoint.suffix or ".pt"
    return snapshot_dir / f"{checkpoint.stem}_iter_{iteration:06d}{suffix}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if args.snapshot_interval < 0:
        raise ValueError("--snapshot-interval cannot be negative")
    if args.pretraining_steps < 0:
        raise ValueError("--pretraining-steps cannot be negative")
    if args.training_temperature < 0.0 or args.post_drop_temperature < 0.0:
        raise ValueError("training temperatures cannot be negative")
    if args.temperature_drop_move < 0:
        raise ValueError("--temperature-drop-move cannot be negative")
    if args.simulations < 1:
        raise ValueError("--simulations must be positive")
    if args.late_game_simulations is not None and args.late_game_simulations < 1:
        raise ValueError("--late-game-simulations must be positive")
    if args.late_game_simulation_move < 0:
        raise ValueError("--late-game-simulation-move cannot be negative")
    if args.recent_sample_window < 1:
        raise ValueError("--recent-sample-window must be positive")
    if not 0.0 <= args.recent_sample_fraction <= 1.0:
        raise ValueError("--recent-sample-fraction must be between 0 and 1")
    if args.ply_sample_fractions is not None:
        if any(fraction < 0.0 for fraction in args.ply_sample_fractions):
            raise ValueError("--ply-sample-fractions cannot contain negative values")
        if not np.isclose(sum(args.ply_sample_fractions), 1.0):
            raise ValueError("--ply-sample-fractions must sum to 1")
    if args.resume and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    env = ConnectFour3D()
    network = PolicyValueNetwork(
        board_size=env.board_size,
        num_actions=env.action_space.n,
        channels=args.channels,
        num_residual_blocks=args.residual_blocks,
        value_head_channels=args.value_head_channels,
        value_hidden_size=args.value_hidden_size,
    )
    buffer = ReplayBuffer(capacity=args.buffer_size, seed=args.seed)
    trainer = AlphaZeroTrainer(
        network,
        buffer,
        device=device,
        learning_rate=args.learning_rate,
        symmetry_augmentation=args.symmetry_augmentation,
        recent_sample_window=args.recent_sample_window,
        recent_sample_fraction=args.recent_sample_fraction,
        ply_sample_fractions=(
            tuple(args.ply_sample_fractions)
            if args.ply_sample_fractions is not None
            else None
        ),
    )
    resume_metadata = {}
    lineage_metadata = {}
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        resume_metadata = trainer.load_checkpoint(
            args.checkpoint,
            learning_rate=args.learning_rate,
            replay_buffer_capacity=(
                args.buffer_size if args.resize_buffer_on_resume else None
            ),
        )
    elif args.initialize_from is not None:
        if args.checkpoint.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing stage checkpoint: {args.checkpoint}"
            )
        source_metadata, load_report = trainer.initialize_from_checkpoint(
            args.initialize_from,
            excluded_prefixes=() if args.reuse_value_head else (
                "value_features.",
                "value_head.",
            ),
        )
        lineage_metadata = {
            "initialized_from": str(args.initialize_from),
            "source_iteration": int(source_metadata.get("iteration", -1)),
            "source_total_games": int(source_metadata.get("total_games", -1)),
        }
        tqdm.write(
            f"Initialized from {args.initialize_from}; "
            f"loaded_tensors={len(load_report['loaded'])} "
            f"skipped_tensors={len(load_report['skipped'])} "
            f"buffer={len(buffer)}"
        )

    agent = AlphaZeroAgent(
        env.observation_space,
        env.action_space,
        env=env,
        network=network,
        num_simulations=args.simulations,
        late_game_simulations=args.late_game_simulations,
        late_game_simulation_move=args.late_game_simulation_move,
        training_temperature=args.training_temperature,
        temperature_drop_move=args.temperature_drop_move,
        post_drop_temperature=args.post_drop_temperature,
        device=device,
        seed=args.seed,
    )
    tqdm.write(
        f"Training on {device}; "
        f"parameters={sum(p.numel() for p in network.parameters()):,}; "
        f"learning_rate={trainer.optimizer.param_groups[0]['lr']:g}; "
        f"symmetry_augmentation={trainer.symmetry_augmentation}; "
        f"recent_sampling={args.recent_sample_fraction:g}@{args.recent_sample_window}"
        f"; ply_sampling={args.ply_sample_fractions or 'disabled'}"
        f"; self_play_simulations={args.simulations}"
        f"->{args.late_game_simulations or args.simulations}"
        f"@ply{args.late_game_simulation_move}"
    )

    if args.initialize_from is not None and args.pretraining_steps > 0:
        pretraining_losses = trainer.train_steps(
            args.pretraining_steps,
            args.batch_size,
            show_progress=True,
            progress_description="Stage2 pretraining",
        )
        tqdm.write(
            f"pretraining_steps={args.pretraining_steps} "
            f"loss={pretraining_losses['loss']:.4f} "
            f"policy={pretraining_losses['policy_loss']:.4f} "
            f"value={pretraining_losses['value_loss']:.4f}"
        )
        initialization_metadata = {
            **lineage_metadata,
            "iteration": 0,
            "total_games": 0,
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
            "value_head_channels": args.value_head_channels,
            "value_hidden_size": args.value_hidden_size or args.channels,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "symmetry_augmentation": trainer.symmetry_augmentation,
            "training_temperature": args.training_temperature,
            "temperature_drop_move": args.temperature_drop_move,
            "post_drop_temperature": args.post_drop_temperature,
            "recent_sample_window": args.recent_sample_window,
            "recent_sample_fraction": args.recent_sample_fraction,
            "ply_sample_fractions": args.ply_sample_fractions,
            "simulations": args.simulations,
            "late_game_simulations": args.late_game_simulations,
            "late_game_simulation_move": args.late_game_simulation_move,
            "pretraining_steps": args.pretraining_steps,
            "reuse_value_head": args.reuse_value_head,
        }
        trainer.save_checkpoint(
            args.checkpoint,
            metadata=initialization_metadata,
        )
        tqdm.write(f"Saved initialized checkpoint: {args.checkpoint}")

    completed_iterations = int(resume_metadata.get("iteration", 0))
    total_games = int(resume_metadata.get("total_games", 0))
    if args.resume:
        tqdm.write(
            f"Resumed iteration={completed_iterations} games={total_games} "
            f"buffer={len(buffer)}"
        )
    iteration_bar = tqdm(
        range(1, args.iterations + 1),
        desc="Iterations",
        unit="iter",
    )
    for batch_iteration in iteration_bar:
        iteration = completed_iterations + batch_iteration
        outcomes = {1: 0, 0: 0, -1: 0}
        game_bar = tqdm(
            range(args.games_per_iteration),
            desc=f"Self-play {iteration}",
            unit="game",
            leave=False,
        )
        for _ in game_bar:
            samples, winner = play_self_play_game(env, agent)
            buffer.add_game(samples)
            outcomes[winner] += 1
            total_games += 1
            game_bar.set_postfix(buffer=len(buffer), refresh=False)

        losses = trainer.train_steps(
            args.training_steps,
            args.batch_size,
            show_progress=True,
            progress_description=f"Optimize {iteration}",
        )
        metadata = {
            **resume_metadata,
            **lineage_metadata,
            "iteration": iteration,
            "total_games": total_games,
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
            "value_head_channels": args.value_head_channels,
            "value_hidden_size": args.value_hidden_size or args.channels,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "symmetry_augmentation": trainer.symmetry_augmentation,
            "training_temperature": args.training_temperature,
            "temperature_drop_move": args.temperature_drop_move,
            "post_drop_temperature": args.post_drop_temperature,
            "recent_sample_window": args.recent_sample_window,
            "recent_sample_fraction": args.recent_sample_fraction,
            "ply_sample_fractions": args.ply_sample_fractions,
            "simulations": args.simulations,
            "late_game_simulations": args.late_game_simulations,
            "late_game_simulation_move": args.late_game_simulation_move,
            "reuse_value_head": args.reuse_value_head,
        }
        trainer.save_checkpoint(args.checkpoint, metadata=metadata)
        snapshot_path = None
        if (
            args.snapshot_interval > 0
            and iteration % args.snapshot_interval == 0
        ):
            snapshot_path = get_snapshot_path(
                args.checkpoint, args.snapshot_dir, iteration
            )
            trainer.save_checkpoint(snapshot_path, metadata=metadata)
        iteration_bar.set_postfix(
            buffer=len(buffer),
            loss=f"{losses['loss']:.4f}",
            refresh=False,
        )
        if batch_iteration % args.log_interval == 0 or (
            batch_iteration == args.iterations
        ):
            tqdm.write(
                f"iteration={iteration} games={total_games} buffer={len(buffer)} "
                f"W1/D/W2={outcomes[1]}/{outcomes[0]}/{outcomes[-1]} "
                f"loss={losses['loss']:.4f} "
                f"policy={losses['policy_loss']:.4f} "
                f"value={losses['value_loss']:.4f}"
            )
        if snapshot_path is not None:
            tqdm.write(f"Saved snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
