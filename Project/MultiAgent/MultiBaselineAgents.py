from copy import deepcopy

import gymnasium as gym
import numpy as np
import warnings
from scipy.linalg import expm
from typing import Union, Callable, Tuple, Dict

import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import Callable

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment, INVENTORY_INDEX, TIME_INDEX, BID_INDEX, ASK_INDEX
from MultiAgent.MultiAgentTradingEnvironment import MultiAgentTradingEnvironment
from mbt_gym.gym.helpers.generate_trajectory import generate_multiagent_trajectory
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
    
# Possible Policy Gradient Agents
class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return F.softmax(self.output(x),dim=-1) #-1 to take softmax of last dimension
    
class ValueFunctionNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueFunctionNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return self.output(x)
class PolicyGradientMultiAgent(Agent):
    def __init__(
        self,
        AgentID: float = 1,
        action_std: Union[float, Callable] = 0.01,
        optimizer: torch.optim.Optimizer = None,
        env: gym.Env = None,
        lr_scheduler: _LRScheduler = None,
    ):
        self.AgentID: str = str(AgentID)
        self.env = env
        self.input_size = env.observation_space[self.AgentID].shape[0]
        self.action_size = env.action_space[self.AgentID].shape[0]
        self.policy_net = ActorNet(state_size=self.input_size, action_size=self.action_size, hidden_size=self.input_size*self.action_size)

        self.action_std = action_std
        self.optimizer = optimizer or torch.optim.SGD(self.policy_net.parameters(), lr=1e-1)
        self.lr_scheduler = lr_scheduler or StepLR(self.optimizer, step_size=1, gamma=0.995)
        self.noise_dist = torch.distributions.Normal
        self.proportion_completed: float = 0.0

    def get_action(
        self, state: np.ndarray, deterministic: bool = False, include_log_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.tensor]]:
        assert not (deterministic and include_log_probs), "include_log_probs is only an option for deterministic output"
        mean_value = self.policy_net(torch.tensor(state, dtype=torch.float, requires_grad=False))
        std = self.action_std(self.proportion_completed) if isinstance(self.action_std, Callable) else self.action_std
        if deterministic:
            return mean_value.detach().numpy()
        action_dist = torch.distributions.Normal(loc=mean_value, scale=std * torch.ones_like(mean_value))
        action = action_dist.sample()
        if include_log_probs:
            log_probs = action_dist.log_prob(action)
            return action.detach().numpy(), log_probs
        return action.detach().numpy()

    def train(self, agents: Dict[str, Agent], num_epochs: int = 1, reporting_freq: int = 100):
        learning_losses = []
        learning_rewards = []
        self.proportion_completed = 0.0
        for epoch in range(num_epochs):
            observations, actions, rewards, log_probs = generate_multiagent_trajectory(self.env, agents, include_log_probs=True)
            observations, actions, rewards, log_probs = observations[self.AgentID], actions[self.AgentID], rewards[self.AgentID], log_probs[self.AgentID]
            learning_rewards.append(rewards.mean())
            rewards = torch.tensor(rewards)
            future_rewards = self._calculate_future_rewards(rewards)
            loss = -torch.mean(log_probs * future_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if epoch % reporting_freq == 0:
            #     tqdm.write(str(loss.item()))
            learning_losses.append(loss.item())
            self.proportion_completed += 1 / (num_epochs) # num_epochs - 1
            self.lr_scheduler.step()
        return learning_losses, learning_rewards

    @staticmethod
    def _calculate_future_rewards(rewards: torch.tensor):
        flipped_rewards = torch.flip(rewards, dims=(-1,))
        cumulative_flipped = torch.cumsum(flipped_rewards, dim=-1)
        return torch.flip(cumulative_flipped, dims=(-1,))