from copy import deepcopy

import gymnasium as gym
import numpy as np
import warnings
from scipy.linalg import expm

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment, INVENTORY_INDEX, TIME_INDEX, BID_INDEX, ASK_INDEX
from MultiAgent.MultiAgentTradingEnvironment import MultiAgentTradingEnvironment
from mbt_gym.rewards.RewardFunctions import CjMmCriterion, PnL
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryAndPermanentPriceImpact

class MultiRandomAgent(Agent):
    def __init__(self, env: MultiAgentTradingEnvironment, 
                    AgentID: int, 
                    seed: int = None):
        self.AgentID: str = str(AgentID)
        self.action_space = deepcopy(env.action_space[self.AgentID])
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)

class FixedActionMultiAgent(Agent):
    def __init__(self, 
                fixed_action: np.ndarray, 
                env: MultiAgentTradingEnvironment, 
                AgentID: int):
        self.AgentID: str = str(AgentID)
        self.fixed_action = fixed_action
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.fixed_action.reshape(1, -1), self.env.num_trajectories, axis=0)

class FixedSpreadMultiAgent(Agent):
    def __init__(self, env: MultiAgentTradingEnvironment,
                  half_spread: float = 1.0, 
                  AgentID: float = 1,
                  offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset
        self.env = env
        self.AgentID: str = str(AgentID)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.array([[self.half_spread - self.offset, self.half_spread + self.offset]])
        return np.repeat(action, self.env.num_trajectories, axis=0)