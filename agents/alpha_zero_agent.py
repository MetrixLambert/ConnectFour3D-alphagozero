from __future__ import annotations

import numpy as np
import torch

from agents.base_agent import BaseAgent
from mcts.mcts import MCTS


class AlphaZeroAgent(BaseAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        env,
        network,
        num_simulations: int = 50,
        late_game_simulations: int | None = None,
        late_game_simulation_move: int = 20,
        c_puct: float = 1.5,
        training_temperature: float = 1.0,
        temperature_drop_move: int = 10,
        post_drop_temperature: float = 0.1,
        device: torch.device | str | None = None,
        gamma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(observation_space, action_space, gamma=gamma, seed=seed)
        self.env = env
        self.network = network
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network.to(self.device)
        self.training_temperature = training_temperature
        self.temperature_drop_move = temperature_drop_move
        self.post_drop_temperature = post_drop_temperature
        if late_game_simulations is not None and late_game_simulations < 1:
            raise ValueError("late_game_simulations must be positive")
        if late_game_simulation_move < 0:
            raise ValueError("late_game_simulation_move cannot be negative")
        self.base_simulations = num_simulations
        self.late_game_simulations = late_game_simulations
        self.late_game_simulation_move = late_game_simulation_move
        self.episode_move_count = 0
        self.mcts = MCTS(
            env=env,
            network=network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=self.device,
            seed=seed,
        )
        self.last_policy: np.ndarray | None = None
        self.last_state_value: float | None = None

    @staticmethod
    def get_current_player(board: np.ndarray) -> int:
        player_one = int(np.count_nonzero(board == 1))
        player_two = int(np.count_nonzero(board == -1))
        if player_one == player_two:
            return 1
        if player_one == player_two + 1:
            return -1
        raise ValueError("Invalid board state: players did not alternate correctly.")

    def select_action(
        self, observation: np.ndarray, training: bool = False
    ) -> int:
        player = self.get_current_player(observation)
        self.mcts.num_simulations = (
            self.late_game_simulations
            if training
            and self.late_game_simulations is not None
            and self.episode_move_count >= self.late_game_simulation_move
            else self.base_simulations
        )
        root = self.mcts.search(
            observation,
            player,
            add_exploration_noise=training,
        )
        if not root.children:
            raise ValueError("No valid actions available.")

        # The visit distribution remains the policy training target. Temperature
        # only controls which move is sampled during self-play.
        visit_policy = self.mcts.policy_from_root(root, temperature=1.0)
        self.last_policy = visit_policy
        self.last_state_value = root.value
        if training:
            temperature = (
                self.training_temperature
                if self.episode_move_count < self.temperature_drop_move
                else self.post_drop_temperature
            )
            action_policy = self.mcts.policy_from_root(root, temperature)
            self.episode_move_count += 1
            return int(self.rng.choice(self.num_actions, p=action_policy))
        return int(np.argmax(visit_policy))

    def get_policy(self) -> np.ndarray | None:
        return None if self.last_policy is None else self.last_policy.copy()

    def get_state_value(self) -> np.ndarray | None:
        if self.last_state_value is None:
            return None
        return np.asarray(self.last_state_value, dtype=np.float32)

    def evaluate_state(self, observation: np.ndarray) -> float:
        """Return the raw value-network estimate for the side to move."""
        player = self.get_current_player(observation)
        canonical = np.asarray(observation, dtype=np.float32) * float(player)
        tensor = torch.from_numpy(canonical).unsqueeze(0).unsqueeze(0).to(self.device)
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            _, value = self.network(tensor)
        if was_training:
            self.network.train()
        return float(value.item())

    def reset_episode(self) -> None:
        self.last_policy = None
        self.last_state_value = None
        self.episode_move_count = 0
