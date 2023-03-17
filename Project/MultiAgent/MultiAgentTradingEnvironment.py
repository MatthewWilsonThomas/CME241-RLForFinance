from collections import OrderedDict
from typing import Union, Tuple, Callable, List, Dict

import numpy as np
import random as rand

from copy import deepcopy

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import Box
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel
from mbt_gym.gym.info_calculators import InfoCalculator
from mbt_gym.rewards.RewardFunctions import RewardFunction, PnL
from mbt_gym.gym.TradingEnvironment import TradingEnvironment

MARKET_MAKING_ACTION_TYPES = ["touch", "limit", "limit_and_market"]
EXECUTION_ACTION_TYPES = ["speed"]
ACTION_TYPES = MARKET_MAKING_ACTION_TYPES + EXECUTION_ACTION_TYPES

CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3

BID_INDEX = 0
ASK_INDEX = 1
MARKET_BUY_INDEX = 2
MARKET_SELL_INDEX = 3

class MultiAgentTradingEnvironment(TradingEnvironment):
    metadata = {"render.modes": ["human"], "name": "Multi-Agent Trading Environment"}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        reward_function: RewardFunction = None,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        price_impact_model: PriceImpactModel = None,
        action_type: str = "limit",
        max_inventory: int = 10000000,  # representing the mean and variance of it.
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        max_speed: float = None,
        half_spread: float = None,
        random_start: Union[float, int, tuple, list, rv_discrete_frozen, rv_continuous_frozen, Callable] = 0.0,
        info_calculator: InfoCalculator = None,  # episode given as a proportion.
        seed: int = None,
        num_trajectories: int = 1,
        num_agents: int = 0,
    ):
        super(TradingEnvironment, self).__init__()
        
        # MultiAgent Specific info
        self.agents = [str(agent) for agent in range(num_agents)]
        # self.possible_agents = self.agents[:]

        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.n_steps = n_steps
        self.step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function or PnL()
        self.midprice_model = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.step_size, num_trajectories=num_trajectories
        )
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.action_type = action_type
        self._check_stochastic_processes()
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self._check_stochastic_seeds()
        self.max_inventory = max_inventory
        self._check_params()
        self.rng = np.random.default_rng(seed)
        if seed:
            self.seed(seed) 
        self.random_start = random_start
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self._get_max_depth()
        self.max_speed = max_speed or self._get_max_speed()

        self.action_space: Dict[str, spaces.Space] = {
            agent: self._get_action_space() for agent in self.agents
        }
        self.observation_space: Dict[str, spaces.Space] = {
            agent: self._get_observation_space() for agent in self.agents
        }

        self.state: Dict[str, np.ndarray] = {}
        self.initial_cash: Dict[str, float] = {}
        self.initial_inventory: Dict[str, Union[int, Tuple[float, float]]] = {}

        self.half_spread = half_spread
        self.info_calculator = info_calculator
        self.empty_infos = [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}
        ones = np.ones((self.num_trajectories, 1))
        self.multiplier = np.append(-ones, ones, axis=1)

    def register_agents(self, AgentIDs: List[str], initial_cash: Dict[str, float], initial_inventory: Dict[str, int]):
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.possible_agents = AgentIDs
        self.agents = AgentIDs
        for agent in self.agents:
            self.action_space[agent] = self._get_action_space()
            self.observation_space[agent] = self._get_observation_space()
            self.state[agent] = self.initial_state(agent, self.initial_cash[agent], self.initial_inventory[agent])

    def reset(self):
        #TODO: Update this to be multi-agent
        for process in self.stochastic_processes.values():
            process.reset()
        self.agents = self.possible_agents
        self.terminations = {agent: False for agent in self.possible_agents}
        for Agent in self.agents:
            self.action_space[Agent] = self._get_action_space()
            self.observation_space[Agent] = self._get_observation_space()
            self.state[Agent] = self.initial_state(Agent, self.initial_cash[Agent], self.initial_inventory[Agent])
            self.reward_function.reset(self.state[Agent].copy())
        return self.state.copy()

    def step(self, actions: Dict[str, np.ndarray]):
        #TODO: Update this to be multi-agent
        next_state = deepcopy(self.state)
        rewards: Dict = {}
        dones: Dict = {}
        infos: Dict = {}
        items = list(actions.items())
        rand.shuffle(items)
        for AgentID, action in items:
            if action.shape != (self.num_trajectories, self.action_space[AgentID].shape[0]):
                action = action.reshape(self.num_trajectories, self.action_space[AgentID].shape[0])
            current_state = self.state[AgentID].copy()
            next_state[AgentID] = self._update_state(AgentID, action)
            done = self.state[AgentID][0, TIME_INDEX] >= self.terminal_time - self.step_size / 2
            dones[AgentID] = np.full((self.num_trajectories,), done, dtype=bool)
            rewards[AgentID] = self.reward_function.calculate(current_state, action, next_state[AgentID], done)
            infos[AgentID] = (
                self.info_calculator.calculate(current_state, action, rewards[AgentID])
                if self.info_calculator is not None
                else self.empty_infos
            )
        return next_state.copy(), rewards, dones, infos

    def _get_arrivals_and_fills(self, agent: str, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = self.limit_depths(action)
            fills = self.fill_probability_model.get_fills(depths)
        elif self.action_type == "touch":
            fills = self.post_at_touch(action)
        fills = self.remove_max_inventory_fills(agent, fills)
        return arrivals, fills

    def remove_max_inventory_fills(self, agent: str, fills: np.ndarray) -> np.ndarray:
        fill_multiplier = np.concatenate(
            ((1 - self.is_at_max_inventory(agent)).reshape(-1, 1), (1 - self.is_at_min_inventory(agent)).reshape(-1, 1)), axis=1
        )
        return fill_multiplier * fills

    def is_at_max_inventory(self, agent: str):
        return self.state[agent][:, INVENTORY_INDEX] >= self.max_inventory

    def is_at_min_inventory(self, agent: str):
        return self.state[agent][:, INVENTORY_INDEX] <= -self.max_inventory
    
    # The action space depends on the action_type but bids always precede asks for limit and market order actions.
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining state depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, agent: str, action: np.ndarray) -> np.ndarray:
        if self.action_type in MARKET_MAKING_ACTION_TYPES:
            arrivals, fills = self._get_arrivals_and_fills(agent, action)
        else:
            arrivals, fills = None, None
        self._update_agent_state(agent, arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.state[agent]

    def _update_market_state(self, arrivals, fills, action):
        for process_name, process in self.stochastic_processes.items():
            process.update(arrivals, fills, action)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            for agent in self.agents:
                self.state[agent][:, lower_index:upper_index] = process.current_state

    def _update_agent_state(self, agent: str, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        if self.action_type == "limit_and_market":
            mo_buy = np.single(self.market_order_buy(action) > 0.5)
            mo_sell = np.single(self.market_order_sell(action) > 0.5)
            best_bid = self.midprice - self.half_spread
            best_ask = self.midprice + self.half_spread
            self.state[agent][:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
            self.state[agent][:, INVENTORY_INDEX] += mo_buy - mo_sell
        elif self.action_type == "touch":
            self.state[agent][:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.half_spread * self.multiplier), axis=1
            )
            self.state[agent][:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.multiplier, axis=1)
        elif self.action_type in ["limit", "limit_and_market"]:
            self.state[agent][:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.multiplier, axis=1)
            self.state[agent][:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.limit_depths(action) * self.multiplier),
                axis=1,
            )
        if self.action_type in EXECUTION_ACTION_TYPES:
            price_impact = self.price_impact_model.get_impact(action)
            execution_price = self.midprice + price_impact
            volume = action * self.step_size
            self.state[agent][:, CASH_INDEX] -= np.squeeze(volume * execution_price)
            self.state[agent][:, INVENTORY_INDEX] += np.squeeze(volume)
        self._clip_inventory_and_cash()
        self.state[agent][:, TIME_INDEX] += self.step_size

    def _clip_inventory_and_cash(self):
        for agent in self.agents:
            self.state[agent][:, INVENTORY_INDEX] = self._clip(
                self.state[agent][:, INVENTORY_INDEX], -self.max_inventory, self.max_inventory, cash_flag=False
            )
            self.state[agent][:, CASH_INDEX] = self._clip(self.state[agent][:, CASH_INDEX], -self.max_cash, self.max_cash, cash_flag=True)

    def initial_state(self, AgentID: str, initial_cash, intial_inventory) -> np.ndarray:
        scalar_initial_state = np.array([[initial_cash, 0, 0.0]])
        initial_state = np.repeat(scalar_initial_state, self.num_trajectories, axis=0)
        if self.random_start is not None:
            random_start_time = self._get_random_start_time()
            initial_state[:, TIME_INDEX] = random_start_time * np.ones((self.num_trajectories,))
        initial_state[:, INVENTORY_INDEX] = intial_inventory
        for process in self.stochastic_processes.values():
            initial_state = np.append(initial_state, process.initial_vector_state, axis=1)
        return initial_state

    def _get_initial_inventories(self, AgentID: str) -> np.ndarray:
        if isinstance(self.initial_inventory[AgentID], tuple) and len(self.initial_inventory[AgentID]) == 2:
            return self.rng.integers(*self.initial_inventory[AgentID], size=self.num_trajectories)
        elif isinstance(self.initial_inventory[AgentID], int):
            return self.initial_inventory[AgentID] * np.ones((self.num_trajectories,))
        else:
            raise Exception("Initial inventory must be a tuple of length 2 or an int.")