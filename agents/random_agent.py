import numpy as np 

from agents.base_agent import BaseAgent  
import numpy as np

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random agent for the 5x5x5 Connect Four environment.

    The agent randomly selects one action from all currently
    available columns.

    Observation:
        board with shape (5, 5, 5)

        board[x, y, z]:
             0 : empty
             1 : player 1
            -1 : player 2

    Action:
        action = 5 * x + y
    """

    def select_action(
        self,
        observation: np.ndarray,
        training: bool = False,
    ) -> int:

        board_size = observation.shape[0]

        valid_actions = []

        for action in range(self.num_actions):

            x = action // board_size
            y = action % board_size

            # A column is available if its top cell is empty.
            if observation[x, y, board_size - 1] == 0:
                valid_actions.append(action)

        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        action = self.rng.choice(valid_actions)

        return int(action)