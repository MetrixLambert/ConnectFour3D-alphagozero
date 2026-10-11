from dataclasses import dataclass  
import numpy as np  
from abc import ABC, abstractmethod  

import gymnasium as gym 

Observation = np.ndarray 
Action = int 

@dataclass
class Transition:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    next_action: Action | None = None
    action_mask: np.ndarray | None = None
    next_action_mask: np.ndarray | None = None
    info: dict | None = None


class BaseAgent(ABC): 
    """Common base class for all agents"""

    def __init__(
        self, 
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float = 0.9, # discount factor
        seed: int | None = None,
    ) -> None: 
    
        if not 0.0 <= gamma <= 1.0: 
            raise ValueError(f"Invalid gamma: {gamma}") 

        self.observation_space = observation_space 
        self.action_space = action_space 

        self.num_actions = action_space.n 
        self.gamma = gamma 
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(
        self, 
        observation: np.ndarray, 
        training: bool = False, 
    ) -> Action: 

        """Select an action based on the current observation"""
        raise NotImplementedError("Subclasses must implement this method")

    def reset_episode(self) -> None: 
        """Reset episode-dependent internal variables"""

    def get_policy(self) -> np.ndarray | None: 
        """Return the current policy as a numpy array of shape (state_shape, num_actions)"""
        return None 

    def get_state_value(self) -> np.ndarray | None: 
        """Return the current state value as a numpy array of shape (state_shape,)"""
        return None 

    def get_action_value(self) -> np.ndarray | None: 
        """Return the current action value as a numpy array of shape (state_shape, num_actions)"""
        return None 


    