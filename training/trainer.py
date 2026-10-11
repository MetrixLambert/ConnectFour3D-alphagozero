from __future__ import annotations

import os
from pathlib import Path
import time

import torch
from torch import nn
from tqdm.auto import tqdm

from utils.symmetry import augment_batch_with_symmetry


class AlphaZeroTrainer:
    def __init__(
        self,
        network: nn.Module,
        replay_buffer,
        device: torch.device | str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        symmetry_augmentation: bool = False,
        recent_sample_window: int | None = None,
        recent_sample_fraction: float = 0.0,
        ply_sample_fractions: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.network = network
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.symmetry_augmentation = symmetry_augmentation
        self.recent_sample_window = recent_sample_window
        self.recent_sample_fraction = recent_sample_fraction
        self.ply_sample_fractions = ply_sample_fractions
        self.network.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_steps(
        self,
        steps: int,
        batch_size: int = 64,
        show_progress: bool = False,
        progress_description: str = "Training",
    ) -> dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        self.network.train()
        step_iterator = tqdm(
            range(steps),
            desc=progress_description,
            unit="step",
            leave=False,
            disable=not show_progress,
        )
        for _ in step_iterator:
            states, target_policies, target_values = self.replay_buffer.sample(
                batch_size,
                recent_window=self.recent_sample_window,
                recent_fraction=self.recent_sample_fraction,
                ply_fractions=self.ply_sample_fractions,
            )
            if self.symmetry_augmentation:
                states, target_policies = augment_batch_with_symmetry(
                    states,
                    target_policies,
                    self.replay_buffer.rng,
                )
            states_tensor = torch.from_numpy(states).unsqueeze(1).to(self.device)
            policies_tensor = torch.from_numpy(target_policies).to(self.device)
            values_tensor = torch.from_numpy(target_values).to(self.device)

            logits, values = self.network(states_tensor)
            policy_loss = -(
                policies_tensor * torch.log_softmax(logits, dim=1)
            ).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, values_tensor)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            self.optimizer.step()

            totals["loss"] += float(loss.item())
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())

            step_iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                policy=f"{policy_loss.item():.4f}",
                value=f"{value_loss.item():.4f}",
                refresh=False,
            )

        return {name: value / steps for name, value in totals.items()}

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replay_buffer_state_dict": self.replay_buffer.state_dict(),
                "metadata": metadata or {},
            },
            temporary_path,
        )
        for attempt in range(20):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)

    def load_checkpoint(
        self,
        path: str | Path,
        learning_rate: float | None = None,
        replay_buffer_capacity: int | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if learning_rate is not None:
            if learning_rate <= 0.0:
                raise ValueError("learning_rate must be positive")
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        if "replay_buffer_state_dict" in checkpoint:
            self.replay_buffer.load_state_dict(
                checkpoint["replay_buffer_state_dict"]
            )
        if replay_buffer_capacity is not None:
            self.replay_buffer.resize(replay_buffer_capacity)
        return checkpoint.get("metadata", {})

    def initialize_from_checkpoint(
        self,
        path: str | Path,
        excluded_prefixes: tuple[str, ...] = (
            "value_features.",
            "value_head.",
        ),
    ) -> tuple[dict, dict[str, list[str]]]:
        """Load compatible weights and replay data into a changed architecture."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        source_state = checkpoint["model_state_dict"]
        target_state = self.network.state_dict()
        compatible = {
            name: value
            for name, value in source_state.items()
            if name in target_state
            and target_state[name].shape == value.shape
            and not name.startswith(excluded_prefixes)
        }
        result = self.network.load_state_dict(compatible, strict=False)
        if "replay_buffer_state_dict" in checkpoint:
            self.replay_buffer.load_state_dict(
                checkpoint["replay_buffer_state_dict"]
            )
        report = {
            "loaded": sorted(compatible),
            "missing": sorted(result.missing_keys),
            "unexpected": sorted(result.unexpected_keys),
            "skipped": sorted(set(source_state) - set(compatible)),
        }
        return checkpoint.get("metadata", {}), report
