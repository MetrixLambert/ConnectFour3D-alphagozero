import numpy as np 
import gymnasium as gym
from gymnasium import spaces 

import matplotlib.pyplot as plt  
from matplotlib.patches import Circle 


class ConnectFour3D(gym.Env): 
    """
    5 x 5 x 5 3D Connect Four environment.

    Board encoding:
        0  : empty
        1  : player 1
       -1  : player 2

    Action:
        action in {0, ..., 24}

        action = 5 * x + y

    Gravity:
        A piece is dropped into column (x, y)
        and occupies the lowest available z position.
    """

    def __init__(self): 
        super().__init__() 

        # Board parameters 
        self.board_size = 5 
        self.connect_length = 4 


        # ---- Action Space ---- 
        # 25 vertical columns 
        self.action_space = spaces.Discrete(
            self.board_size * self.board_size 
        )

        # ---- Observation Space ----  
        # board [x, y, z] in {-1, 0, 1} 
        self.observation_space = spaces.Box(
            low = -1, high = 1, shape=(
                self.board_size, self.board_size, self.board_size
            ), 
            dtype = np.int8 
        )


        # ---- Winning directions ---- 
        # Only one direction from each pair d and -d is needed 
        self.directions = np.array(
            [
                # axes
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),

                # xy diagonals
                (1, 1, 0),
                (1, -1, 0),

                # xz diagonals
                (1, 0, 1),
                (1, 0, -1),

                # yz diagonals
                (0, 1, 1),
                (0, 1, -1),

                # 3D diagonals
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
            ],
            dtype=np.int8,
        )

        self.windows = self._generate_windows()
        self.windows_by_position = (
            self._generate_windows_by_position()
        )

        # ---- State ---- 
        self.board = np.zeros(
            self.observation_space.shape, 
            dtype = np.int8 
        )

        self.current_player = 1 

        # None: unfinished 
        # 1: player 1 wins 
        # -1: player 2 wins 
        # 0: draw 
        self.winner = None 
        self.winning_positions = [] 

    def _generate_windows_by_position(self):
        windows_by_position = {}

        for x in range(self.board_size):
            for y in range(self.board_size):
                for z in range(self.board_size):
                    windows_by_position[
                        (x, y, z)
                    ] = []

        for window in self.windows:

            for position in window:

                key = tuple(position)

                windows_by_position[
                    key
                ].append(window)

        return windows_by_position
        
    # get obs 
    def _get_obs(self): 
        return self.board.copy()  

    # get info 
    def _get_info(self): 
        return {
            "current_player": self.current_player,
            "action_mask": self.get_actions_mask(), 
            "winner": self.winner, 
        }
    
    def _action_to_position(self, action): 
        x = action // self.board_size 
        y = action % self.board_size 
        return x, y 

    def _is_valid_action(self, action): 
        if not self.action_space.contains(action): 
            return False 
        x, y = self._action_to_position(action)  
        return self.board[x, y, self.board_size - 1] == 0 

    def get_valid_actions(self): 
        return np.array(
            [
                action for action in range(self.action_space.n) 
                if self._is_valid_action(action)
            ], 
            dtype = np.int8 
        )


    def get_actions_mask(self): 
        mask = np.zeros(
            self.action_space.n, 
            dtype = bool 
        )

        for action in range(self.action_space.n): 
            mask[action] = self._is_valid_action(action)
        return mask 


    def _get_drop_height(self, x, y): 
        for z in range(self.board_size): 
            if self.board[x, y, z] == 0: 
                return z 

        return None 

    def _drop_piece(self, action, player): 
        x, y = self._action_to_position(action)
        z = self._get_drop_height(x, y)
        if z is None: 
            raise(ValueError(f"Column ({x}, {y}) is full"))
        self.board[x, y, z] = player  

        return np.array(
            [x, y, z], 
            dtype = np.int64, 
        )

    def _is_in_bounds(self, position): 
        x, y, z = position 
        return (
            0 <= x < self.board_size and 
            0 <= y < self.board_size and 
            0 <= z < self.board_size 
        )

    def _count_direction(
        self, board, position, direction, player
        ): 
        count = 0 

        current = (np.asarray(position, dtype=np.int64) + direction) 

        while self._is_in_bounds(current): 
            x, y, z = current 
            if board[x, y, z] != player: 
                break 
            count += 1 
            current = current + direction 
        
        return count 

    def _check_win(self, position, player): 
        position = np.asarray(position, dtype=np.int64) 

        for direction in self.directions: 

            line = [tuple(position)] 

            # forward 
            current = position + direction 
            while self._is_in_bounds(current): 
                x, y, z = current 
                if self.board[x, y, z] != player: 
                    break 
                line.append(tuple(current)) 
                current = current + direction 
            
            # backward 
            current = position - direction 
            while self._is_in_bounds(current): 
                x, y, z = current 

                if self.board[x, y, z] != player: 
                    break 

                line.append(tuple(current)) 
                current = current - direction  

            if len(line) >= self.connect_length: 
                self.winning_positions = line 
                return True 

        return False 

    def reset(self, *, seed=None, options=None): 
        super().reset(seed=seed) 

        self.board.fill(0) 

        self.current_player = 1 
        self.winner = None 
        self.winning_positions = [] 

        observation = self._get_obs() 
        info = self._get_info()  

        return observation, info  

    def step(self, action): 

        # 1. check action 
        if not self.action_space.contains(action): 
            raise ValueError(f"Invalid action: {action}")

        if not self._is_valid_action(action): 
            raise ValueError(f"Column for action {action} is full")

        # 2. Acting Player 
        player = self.current_player  

        # 3. Drop piece 
        position = self._drop_piece(action, player) 

        # 4. Check win 
        won = self._check_win(position, player) 

        # 5. Check draw 
        draw = (not won and np.all(self.board != 0)) 

        # 6. Terminate state 
        terminated = won or draw 
        truncated = False  

        # 7. Reward 
        reward = 1.0 if won else 0.0 

        # 8. Winner 
        if won: 
            self.winner = player  
        elif draw: 
            self.winner = 0  

        # 9. Change Player 
        if not terminated: 
            self.current_player = -player   

        # 10. Output 
        observation = self._get_obs()  
        info = self._get_info()   
        
        return (
            observation, reward, terminated, truncated, info  
        )

    def render(self):
        fig, axes = plt.subplots(
            1,
            self.board_size,
            figsize=(15, 3),
        )

        for z in range(self.board_size):

            ax = axes[z]

            # draw grid
            ax.set_xlim(-0.5, self.board_size - 0.5)
            ax.set_ylim(self.board_size - 0.5, -0.5)

            ax.set_xticks(range(self.board_size))
            ax.set_yticks(range(self.board_size))

            ax.set_xticks(
                np.arange(-0.5, self.board_size, 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, self.board_size, 1),
                minor=True,
            )

            ax.grid(which="minor")
            ax.tick_params(which="minor", length=0)

            ax.set_title(f"z = {z}")

            # draw pieces
            for x in range(self.board_size):
                for y in range(self.board_size):

                    player = self.board[x, y, z]

                    if player == 0:
                        continue

                    color = "tab:red" if player == 1 else "tab:blue"

                    is_winner = (
                        x,
                        y,
                        z,
                    ) in self.winning_positions

                    circle = Circle(
                        (y, x),
                        radius=0.32,
                        facecolor=color,
                        edgecolor="gold" if is_winner else "black",
                        linewidth=4 if is_winner else 1,
                    )

                    ax.add_patch(circle)

            ax.set_aspect("equal")

        if self.winner == 1:
            fig.suptitle("Player 1 wins!")

        elif self.winner == -1:
            fig.suptitle("Player 2 wins!")

        elif self.winner == 0:
            fig.suptitle("Draw")

        else:
            fig.suptitle(
                f"Current player: {self.current_player}"
            )

        plt.tight_layout()
        plt.show()

    def _simulate_drop(self, board, action, player): 

        x, y = self._action_to_position(action) 
        for z in range(self.board_size): 
            if board[x, y, z] == 0: 
                board[x, y, z] = player 
                return np.array(
                    [x, y, z], 
                    dtype = np.int64
                )

        return None 

    def _check_win_on_board(
        self,
        board,
        position,
        player,
    ):
        x, y, z = position

        windows = self.windows_by_position[
            (int(x), int(y), int(z))
        ]

        for window in windows:

            values = board[
                window[:, 0],
                window[:, 1],
                window[:, 2],
            ]

            if np.all(values == player):
                return True

        return False
            

    def get_model(self, board, player, action): 
        # copy state 
        next_board = board.copy() 

        # simulate action 
        position = self._simulate_drop(next_board, action, player)  

        # check win 
        won = self._check_win_on_board(next_board, position, player)  

        return next_board, position, won 

    def get_valid_actions_from_board(
        self,
        board,
    ):
        valid_actions = []

        for action in range(self.action_space.n):
            x, y = self._action_to_position(action)

            if board[
                x,
                y,
                self.board_size - 1,
            ] == 0:
                valid_actions.append(action)

        return np.array(
            valid_actions,
            dtype=np.int64,
        )


    def _generate_windows(self):

        windows = []

        for x in range(self.board_size):
            for y in range(self.board_size):
                for z in range(self.board_size):

                    start = np.array(
                        [x, y, z],
                        dtype=np.int64,
                    )

                    for direction in self.directions:

                        end = (
                            start
                            + (self.connect_length - 1)
                            * direction
                        )

                        if not self._is_in_bounds(end):
                            continue

                        window = []

                        for k in range(self.connect_length):

                            position = (
                                start + k * direction
                            )

                            window.append(
                                tuple(position)
                            )

                        windows.append(window)

        return np.array(
            windows,
            dtype=np.int64,
        )


    # def get_all_windows(self, board):
    #     """
    #     For alpha-beta agent (heuristic evaluation function) 
    #     assess all possible windows in the board 
    #     """

    #     windows = []

    #     for x in range(self.board_size):
    #         for y in range(self.board_size):
    #             for z in range(self.board_size):

    #                 start = np.array(
    #                     [x, y, z],
    #                     dtype=np.int64,
    #                 )

    #                 for direction in self.directions:

    #                     end = (
    #                         start
    #                         + (self.connect_length - 1)
    #                         * direction
    #                     )

    #                     if not self._is_in_bounds(end):
    #                         continue

    #                     window = []

    #                     for k in range(
    #                         self.connect_length
    #                     ):
    #                         position = (
    #                             start + k * direction
    #                         )

    #                         px, py, pz = position

    #                         window.append(
    #                             board[px, py, pz]
    #                         )

    #                     windows.append(window)

        return windows

    
    def make_move(
        self,
        board,
        action,
        player,
    ):
        x, y = self._action_to_position(action)

        z = self._get_drop_height_from_board(
            board,
            x,
            y,
        )

        if z is None:
            raise ValueError(
                f"Column ({x}, {y}) is full."
            )

        board[x, y, z] = player

        position = np.array(
            [x, y, z],
            dtype=np.int64,
        )

        won = self._check_win_on_board(
            board,
            position,
            player,
        )

        return position, won

    def unmake_move(
        self,
        board,
        position,
    ):
        x, y, z = position
        board[x, y, z] = 0

    def _get_drop_height_from_board(
        self,
        board,
        x,
        y,
    ):
        for z in range(self.board_size):
            if board[x, y, z] == 0:
                return z

        return None

    def make_move_with_heights(
        self,
        board,
        heights,
        action,
        player,
    ):
        x, y = self._action_to_position(action)

        z = int(heights[x, y])

        if z >= self.board_size:
            raise ValueError(
                f"Column ({x}, {y}) is full."
            )

        # Make move
        board[x, y, z] = player
        heights[x, y] += 1

        position = (x, y, z)

        won = self._check_win_on_board(
            board,
            position,
            player,
        )

        return position, won


    def unmake_move_with_heights(
        self,
        board,
        heights,
        position,
    ):
        x, y, z = position

        heights[x, y] -= 1
        board[x, y, z] = 0



def get_immediate_winning_actions(
    env: ConnectFour3D,
    board: np.ndarray,
    player: int,
):
    """
    utils func for tactical double dqn agent and diagnosis 
    """
    winning_actions = []

    valid_actions = (
        env.get_valid_actions_from_board(
            board
        )
    )

    for action in valid_actions:

        _, _, won = env.get_model(
            board,
            player,
            action,
        )

        if won:
            winning_actions.append(
                int(action)
            )

    return winning_actions



