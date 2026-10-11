from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from environments.connect_four_3d import ConnectFour3D
    from models.policy_value_network import PolicyValueNetwork


@dataclass
class MCTSNode:
    to_play: int
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    """PUCT search whose values are always from each node's side to move."""

    def __init__(
        self,
        env: "ConnectFour3D",
        network: "PolicyValueNetwork",
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if num_simulations < 1:
            raise ValueError("num_simulations must be positive")
        self.env = env
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    def _network_evaluate(
        self, board: np.ndarray, player: int
    ) -> tuple[np.ndarray, float]:
        canonical = np.asarray(board, dtype=np.float32) * float(player)
        tensor = torch.from_numpy(canonical).unsqueeze(0).unsqueeze(0).to(self.device)
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        if was_training:
            self.network.train()
        return probabilities, float(value.item())

    def _expand(self, node: MCTSNode, board: np.ndarray) -> float:
        valid_actions = self.env.get_valid_actions_from_board(board)
        if len(valid_actions) == 0:
            return 0.0

        policy, value = self._network_evaluate(board, node.to_play)
        valid_policy = policy[valid_actions]
        total = float(valid_policy.sum())
        if not np.isfinite(total) or total <= 0.0:
            valid_policy = np.full(len(valid_actions), 1.0 / len(valid_actions))
        else:
            valid_policy = valid_policy / total

        for action, prior in zip(valid_actions, valid_policy):
            node.children[int(action)] = MCTSNode(
                to_play=-node.to_play,
                prior=float(prior),
            )
        return value

    def _add_root_noise(self, root: MCTSNode) -> None:
        actions = list(root.children)
        if not actions:
            return
        noise = self.rng.dirichlet(
            np.full(len(actions), self.dirichlet_alpha, dtype=np.float64)
        )
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1.0 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(sample)
            )

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        sqrt_parent = np.sqrt(max(1, node.visit_count))
        best_score = -np.inf
        best: list[tuple[int, MCTSNode]] = []
        for action, child in node.children.items():
            score = (
                -child.value
                + self.c_puct
                * child.prior
                * sqrt_parent
                / (1 + child.visit_count)
            )
            if score > best_score + 1e-12:
                best_score = score
                best = [(action, child)]
            elif abs(score - best_score) <= 1e-12:
                best.append((action, child))
        return best[int(self.rng.integers(len(best)))]

    @staticmethod
    def _backpropagate(search_path: list[MCTSNode], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def search(
        self,
        board: np.ndarray,
        player: int,
        add_exploration_noise: bool = False,
    ) -> MCTSNode:
        root = MCTSNode(to_play=int(player))
        self._expand(root, board)
        if add_exploration_noise:
            self._add_root_noise(root)

        for _ in range(self.num_simulations):
            node = root
            simulation_board = board.copy()
            search_path = [node]
            terminal_value: float | None = None

            while node.expanded:
                action, child = self._select_child(node)
                _, won = self.env.make_move(
                    simulation_board, action, node.to_play
                )
                node = child
                search_path.append(node)
                if won:
                    terminal_value = -1.0
                    break

            if terminal_value is None:
                terminal_value = self._expand(node, simulation_board)
            self._backpropagate(search_path, terminal_value)

        return root

    def policy_from_root(
        self, root: MCTSNode, temperature: float
    ) -> np.ndarray:
        policy = np.zeros(self.env.action_space.n, dtype=np.float32)
        actions = np.fromiter(root.children.keys(), dtype=np.int64)
        if len(actions) == 0:
            return policy
        visits = np.array(
            [root.children[int(action)].visit_count for action in actions],
            dtype=np.float64,
        )
        if temperature <= 1e-8:
            maximum = visits.max()
            best = actions[visits == maximum]
            policy[int(self.rng.choice(best))] = 1.0
            return policy

        scaled = np.power(visits, 1.0 / temperature)
        total = scaled.sum()
        if total <= 0.0:
            scaled.fill(1.0 / len(scaled))
        else:
            scaled /= total
        policy[actions] = scaled.astype(np.float32)
        return policy
