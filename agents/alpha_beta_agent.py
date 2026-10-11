import numpy as np

from agents.base_agent import BaseAgent


WIN_SCORE = 1.0


class AlphaBetaAgent(BaseAgent):

    def __init__(
        self,
        observation_space,
        action_space,
        env,
        max_depth=3,
        gamma=0.9,
        seed=None,
    ):
        super().__init__(
            observation_space,
            action_space,
            gamma=gamma,
            seed=seed,
        )

        self.env = env
        self.max_depth = max_depth

    def _evaluate(
        self,
        board,
        player,
    ):
        """
        Heuristic evaluation of a non-terminal state.

        Currently no heuristic is used. Therefore, states
        reaching the depth limit are treated as neutral.
        """
        return 0.0

    def _alpha_beta(
        self,
        board,
        heights,
        player,
        root_player,
        depth,
        alpha,
        beta,
    ):
        # -------------------------
        # Depth limit
        # -------------------------
        if depth == 0:
            return self._evaluate(
                board,
                root_player,
            )

        # -------------------------
        # Valid actions
        # -------------------------
        valid_actions = np.flatnonzero(
            heights.ravel()
            < self.env.board_size
        )

        # Draw
        if len(valid_actions) == 0:
            return 0.0

        # =========================
        # MAX node
        # =========================
        if player == root_player:

            value = -np.inf

            for action in valid_actions:

                position, won = (
                    self.env.make_move_with_heights(
                        board,
                        heights,
                        action,
                        player,
                    )
                )

                if won:
                    child_value = WIN_SCORE

                else:
                    child_value = self._alpha_beta(
                        board,
                        heights,
                        -player,
                        root_player,
                        depth - 1,
                        alpha,
                        beta,
                    )

                # Restore state
                self.env.unmake_move_with_heights(
                    board,
                    heights,
                    position,
                )

                value = max(
                    value,
                    child_value,
                )

                alpha = max(
                    alpha,
                    value,
                )

                # Best possible value
                if value == WIN_SCORE:
                    return WIN_SCORE

                # Beta cutoff
                if alpha >= beta:
                    break

            return value

        # =========================
        # MIN node
        # =========================
        else:

            value = np.inf

            for action in valid_actions:

                position, won = (
                    self.env.make_move_with_heights(
                        board,
                        heights,
                        action,
                        player,
                    )
                )

                if won:
                    child_value = -WIN_SCORE

                else:
                    child_value = self._alpha_beta(
                        board,
                        heights,
                        -player,
                        root_player,
                        depth - 1,
                        alpha,
                        beta,
                    )

                # Restore state
                self.env.unmake_move_with_heights(
                    board,
                    heights,
                    position,
                )

                value = min(
                    value,
                    child_value,
                )

                beta = min(
                    beta,
                    value,
                )

                # Worst possible value
                if value == -WIN_SCORE:
                    return -WIN_SCORE

                # Alpha cutoff
                if alpha >= beta:
                    break

            return value

    def select_action(
        self,
        observation: np.ndarray,
        training: bool = False,
    ) -> int:

        # Copy the board so that the input observation
        # is never modified during search.
        board = observation.copy()

        # Number of occupied cells in each vertical column.
        # This avoids repeatedly searching for drop heights.
        heights = np.count_nonzero(
            board,
            axis=2,
        ).astype(np.int8)

        root_player = self._get_current_player(
            board
        )

        valid_actions = np.flatnonzero(
            heights.ravel()
            < self.env.board_size
        )

        if len(valid_actions) == 0:
            raise ValueError(
                "No valid actions available."
            )

        best_value = -np.inf
        best_actions = []

        # =========================
        # Search root actions
        # =========================
        for action in valid_actions:

            position, won = (
                self.env.make_move_with_heights(
                    board,
                    heights,
                    action,
                    root_player,
                )
            )

            if won:
                value = WIN_SCORE

            else:
                value = self._alpha_beta(
                    board,
                    heights,
                    -root_player,
                    root_player,
                    self.max_depth - 1,
                    -np.inf,
                    np.inf,
                )

            # Restore root state
            self.env.unmake_move_with_heights(
                board,
                heights,
                position,
            )

            # Update best actions
            if value > best_value:

                best_value = value
                best_actions = [action]

            elif value == best_value:

                best_actions.append(action)

        # Random tie-breaking
        return int(
            self.rng.choice(best_actions)
        )

    def _get_current_player(
        self,
        board,
    ):
        num_player_1 = np.count_nonzero(
            board == 1
        )

        num_player_2 = np.count_nonzero(
            board == -1
        )

        if num_player_1 == num_player_2:
            return 1

        elif num_player_1 == num_player_2 + 1:
            return -1

        else:
            raise ValueError(
                "Invalid board state."
            )