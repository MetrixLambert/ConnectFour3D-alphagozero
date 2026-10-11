from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelfPlaySample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._samples: deque[SelfPlaySample] = deque(maxlen=capacity)
        self._plies: deque[int] = deque(maxlen=capacity)
        self._ply_cache: np.ndarray | None = None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def add_game(self, samples: list[SelfPlaySample]) -> None:
        self._samples.extend(samples)
        self._plies.extend(int(np.count_nonzero(sample.state)) for sample in samples)
        self._ply_cache = None

    def state_dict(self) -> dict:
        """Return all state needed to continue sampling after a restart."""
        return {
            "capacity": self._samples.maxlen,
            "samples": list(self._samples),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore samples and RNG state from :meth:`state_dict`."""
        capacity = int(state["capacity"])
        if capacity < 1:
            raise ValueError("checkpoint replay buffer capacity must be positive")
        self._samples = deque(state.get("samples", ()), maxlen=capacity)
        self._plies = deque(
            (int(np.count_nonzero(sample.state)) for sample in self._samples),
            maxlen=capacity,
        )
        self._ply_cache = None
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]

    def resize(self, capacity: int) -> None:
        """Change capacity while retaining the newest available samples."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._samples = deque(self._samples, maxlen=capacity)
        self._plies = deque(self._plies, maxlen=capacity)
        self._ply_cache = None

    def _get_plies(self) -> np.ndarray:
        if self._ply_cache is None:
            self._ply_cache = np.fromiter(
                self._plies,
                dtype=np.int16,
                count=len(self._plies),
            )
        return self._ply_cache

    @staticmethod
    def _allocate_quotas(batch_size: int, fractions: np.ndarray) -> np.ndarray:
        exact = fractions * batch_size
        quotas = np.floor(exact).astype(np.int64)
        remainder = batch_size - int(quotas.sum())
        if remainder:
            order = np.argsort(-(exact - quotas), kind="stable")
            quotas[order[:remainder]] += 1
        return quotas

    def _sample_by_ply(
        self,
        batch_size: int,
        recent_window: int | None,
        recent_fraction: float,
        ply_fractions: tuple[float, float, float, float],
    ) -> np.ndarray:
        fractions = np.asarray(ply_fractions, dtype=np.float64)
        if fractions.shape != (4,) or np.any(fractions < 0.0):
            raise ValueError("ply_fractions must contain four non-negative values")
        total = float(fractions.sum())
        if not np.isfinite(total) or not np.isclose(total, 1.0):
            raise ValueError("ply_fractions must sum to 1")

        sample_count = len(self._samples)
        recent_start = (
            max(0, sample_count - recent_window)
            if recent_window is not None
            else sample_count
        )
        plies = self._get_plies()
        stage_masks = (
            plies < 10,
            (plies >= 10) & (plies < 20),
            (plies >= 20) & (plies < 30),
            plies >= 30,
        )
        quotas = self._allocate_quotas(batch_size, fractions)
        selected: list[int] = []
        selected_set: set[int] = set()

        for mask, quota in zip(stage_masks, quotas):
            quota = int(quota)
            if quota == 0:
                continue
            candidates = np.flatnonzero(mask)
            recent_candidates = candidates[candidates >= recent_start]
            history_candidates = candidates[candidates < recent_start]
            recent_quota = int(round(quota * recent_fraction))
            history_quota = quota - recent_quota
            stage_selected = [
                *self.rng.choice(
                    recent_candidates,
                    size=min(recent_quota, len(recent_candidates)),
                    replace=False,
                ).tolist(),
                *self.rng.choice(
                    history_candidates,
                    size=min(history_quota, len(history_candidates)),
                    replace=False,
                ).tolist(),
            ]
            if len(stage_selected) < quota:
                remaining_stage = np.asarray(
                    [index for index in candidates if int(index) not in stage_selected],
                    dtype=np.int64,
                )
                fill = min(quota - len(stage_selected), len(remaining_stage))
                stage_selected.extend(
                    self.rng.choice(remaining_stage, size=fill, replace=False).tolist()
                )
            selected.extend(int(index) for index in stage_selected)
            selected_set.update(int(index) for index in stage_selected)

        if len(selected) < batch_size:
            remaining = np.asarray(
                [index for index in range(sample_count) if index not in selected_set],
                dtype=np.int64,
            )
            selected.extend(
                self.rng.choice(
                    remaining,
                    size=batch_size - len(selected),
                    replace=False,
                ).tolist()
            )
        indices = np.asarray(selected, dtype=np.int64)
        self.rng.shuffle(indices)
        return indices

    def sample(
        self,
        batch_size: int,
        recent_window: int | None = None,
        recent_fraction: float = 0.0,
        ply_fractions: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample without replacement, optionally reserving a recent-data quota.

        When both strata contain enough samples, ``recent_fraction`` of the batch
        comes from the newest ``recent_window`` entries and the remainder comes
        from older history. If either stratum is too small, its shortfall is
        filled uniformly from all still-unselected samples.
        """
        if batch_size > len(self._samples):
            raise ValueError("batch_size exceeds the number of stored samples")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be between 0 and 1")
        if recent_window is not None and recent_window < 1:
            raise ValueError("recent_window must be positive")

        sample_count = len(self._samples)
        if ply_fractions is not None:
            indices = self._sample_by_ply(
                batch_size,
                recent_window,
                recent_fraction,
                ply_fractions,
            )
        else:
            recent_count = int(round(batch_size * recent_fraction))
            use_strata = (
                recent_count > 0
                and recent_window is not None
                and recent_window < sample_count
            )
            if not use_strata:
                indices = self.rng.choice(sample_count, size=batch_size, replace=False)
            else:
                recent_start = max(0, sample_count - recent_window)
                recent_candidates = np.arange(recent_start, sample_count)
                history_candidates = np.arange(recent_start)
                selected_recent_count = min(recent_count, len(recent_candidates))
                selected_history_count = min(
                    batch_size - selected_recent_count,
                    len(history_candidates),
                )
                selected = [
                    *self.rng.choice(
                        recent_candidates,
                        size=selected_recent_count,
                        replace=False,
                    ).tolist(),
                    *self.rng.choice(
                        history_candidates,
                        size=selected_history_count,
                        replace=False,
                    ).tolist(),
                ]
                if len(selected) < batch_size:
                    remaining = np.setdiff1d(
                        np.arange(sample_count),
                        np.asarray(selected, dtype=np.int64),
                        assume_unique=False,
                    )
                    selected.extend(
                        self.rng.choice(
                            remaining,
                            size=batch_size - len(selected),
                            replace=False,
                        ).tolist()
                    )
                indices = np.asarray(selected, dtype=np.int64)
                self.rng.shuffle(indices)
        batch = [self._samples[int(index)] for index in indices]
        states = np.stack([item.state for item in batch]).astype(np.float32)
        policies = np.stack([item.policy for item in batch]).astype(np.float32)
        values = np.asarray([item.value for item in batch], dtype=np.float32)
        return states, policies, values
