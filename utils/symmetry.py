from __future__ import annotations

import numpy as np


NUM_HORIZONTAL_SYMMETRIES = 8


def apply_horizontal_symmetry(
    state: np.ndarray,
    policy: np.ndarray,
    symmetry: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one D4 symmetry to x/y while leaving gravity axis z unchanged."""
    if not 0 <= symmetry < NUM_HORIZONTAL_SYMMETRIES:
        raise ValueError(f"symmetry must be in [0, 7], got {symmetry}")
    if state.ndim != 3 or state.shape[0] != state.shape[1]:
        raise ValueError("state must have square x/y axes and shape (x, y, z)")

    board_size = state.shape[0]
    if policy.shape != (board_size * board_size,):
        raise ValueError(
            f"policy must have shape ({board_size * board_size},), "
            f"got {policy.shape}"
        )

    rotations = symmetry % 4
    transformed_state = np.rot90(state, k=rotations, axes=(0, 1))
    transformed_policy = np.rot90(
        policy.reshape(board_size, board_size),
        k=rotations,
        axes=(0, 1),
    )
    if symmetry >= 4:
        transformed_state = np.flip(transformed_state, axis=0)
        transformed_policy = np.flip(transformed_policy, axis=0)

    # rot90/flip can return negative-stride views, which torch.from_numpy rejects.
    return (
        np.ascontiguousarray(transformed_state),
        np.ascontiguousarray(transformed_policy.reshape(-1)),
    )


def augment_batch_with_symmetry(
    states: np.ndarray,
    policies: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly and independently augment every state-policy pair in a batch."""
    if len(states) != len(policies):
        raise ValueError("states and policies batch sizes must match")

    augmented_states = np.empty_like(states)
    augmented_policies = np.empty_like(policies)
    symmetries = rng.integers(
        NUM_HORIZONTAL_SYMMETRIES,
        size=len(states),
    )
    for index, symmetry in enumerate(symmetries):
        state, policy = apply_horizontal_symmetry(
            states[index], policies[index], int(symmetry)
        )
        augmented_states[index] = state
        augmented_policies[index] = policy
    return augmented_states, augmented_policies
