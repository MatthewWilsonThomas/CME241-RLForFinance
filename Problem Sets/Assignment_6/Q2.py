# Optimal Market Making Problem

from typing import Callable, Sequence, Tuple, Iterator
from dataclasses import dataclass
import numpy as np
import itertools
import matplotlib.pyplot as plt
import copy as cp

from rl.distribution import (Constant, Categorical, Uniform, Distribution, FiniteDistribution, Bernoulli, SampledDistribution)
from rl.markov_process import MarkovProcess, State, NonTerminal
from rl.markov_decision_process import MarkovDecisionProcess
from rl.chapter9.order_book import OrderBook, DollarsAndShares
from rl.policy import Policy

@dataclass(frozen=True)
class PriceAndShares:
    price: float
    shares: int

@dataclass(frozen=True)
class MarketAction:
    bid: PriceAndShares
    ask: PriceAndShares

class MarketMaker:
    mid_price: float = 100
    I: int = 0
    PnL: float = 0
    bid_ask_spread: float = 1

S = MarketMaker
A = MarketAction


####---- Optimal Market Making ----####
class OptimalMarketMaking(MarkovDecisionProcess[S, A]):
    T: float = 1
    delta_t: float = 0.005
    gamma: float = 0.1
    sigma: float = 2
    k: float = 1.5
    c: int = 140
    t: float = 0
    
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> SampledDistribution[Tuple[State[S], float]]:
        
        delta_ask: float = ((1 - 2 * state.state.I) * self.gamma * self.sigma**2 * (self.T - self.t) / 2) + 1 / self.gamma * np.log(1 + self.gamma/self.k)
        delta_bid: float = ((2 * state.state.I + 1) * self.gamma * self.sigma**2 * (self.T - self.t) / 2) + 1 / self.gamma * np.log(1 + self.gamma/self.k)

        prob_up: float = self.c * np.exp(-self.k * delta_ask) * self.delta_t
        prob_down: float = self.c * np.exp(-self.k * delta_bid) * self.delta_t

        def sr_sampler_func(
            state=state,
            action=action,
            prob_up=prob_up,
            prob_down=prob_down
        ) -> Tuple[State[S], float]:
            
            next_state = cp.deepcopy(state.state)
            # Update market maker's inventory.
            if Bernoulli(prob_up).sample():
                next_state.I = state.state.I - 1
                next_state.PnL = state.state.PnL + action.ask.price
            if Bernoulli(prob_down).sample():
                next_state.I = state.state.I + 1
                next_state.PnL = state.state.PnL - action.bid.price

            # Update market mid price
            if Bernoulli(0.5).sample():
                next_state.mid_price = state.state.mid_price + self.sigma * np.sqrt(self.delta_t)
            else:
                next_state.mid_price = state.state.mid_price - self.sigma * np.sqrt(self.delta_t)

            # Update the market bid-ask spread with the optimal bid-ask spread values.
            next_state.bid_ask_spread = action.ask.price - action.bid.price

            return (NonTerminal(next_state), next_state.PnL)
        
        self.t = self.t + self.delta_t
        return SampledDistribution(sr_sampler_func)

    def actions(self, state : NonTerminal[S]) -> \
            Iterator[A]:

        indifference_ask_price: float = state.state.mid_price - (2 * state.state.I - 1) + self.gamma * self.sigma**2 * (self.T - self.t) / 2
        indifference_bid_price: float = state.state.mid_price - (2 * state.state.I - 1) - self.gamma * self.sigma**2 * (self.T - self.t) / 2
        
        ask_price: float = indifference_ask_price + 1 / self.gamma * np.log(1 + self.gamma/self.k)
        bid_price: float = indifference_bid_price - 1 / self.gamma * np.log(1 + self.gamma/self.k)

        # Assume bid and ask volumes are 1 in infintessimal time.
        action: A = MarketAction(bid=PriceAndShares(price = bid_price,
                                                      shares = 1), 
                                 ask=PriceAndShares(price = ask_price, 
                                                   shares = 1))

        return iter([action])
    
#####---- Naive Market Making ----####
class NaiveMarketMaking(MarkovDecisionProcess[S, A]):
    T: float = 1
    delta_t: float = 0.005
    gamma: float = 0.1
    sigma: float = 2
    I: float = 0
    k: float = 1.5
    c: int = 140
    t: float = 0
    bid_ask_spread: float
    
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> SampledDistribution[Tuple[State[S], float]]:
        
        delta_ask: float = ((1 - 2 * state.state.I) * self.gamma * self.sigma**2 * (self.T - self.t) / 2) + 1 / self.gamma * np.log(1 + self.gamma/self.k)
        delta_bid: float = ((2 * state.state.I + 1) * self.gamma * self.sigma**2 * (self.T - self.t) / 2) + 1 / self.gamma * np.log(1 + self.gamma/self.k)

        prob_up: float = self.c * np.exp(-self.k * delta_ask) * self.delta_t
        prob_down: float = self.c * np.exp(-self.k * delta_bid) * self.delta_t

        def sr_sampler_func(
            state=state,
            action=action,
            prob_up=prob_up,
            prob_down=prob_down
        ) -> Tuple[State[S], float]:
            
            next_state = cp.deepcopy(state.state)
            # Update market maker's inventory.
            if Bernoulli(prob_up).sample():
                next_state.I = state.state.I - 1
                next_state.PnL = state.state.PnL + action.ask.price
            if Bernoulli(prob_down).sample():
                next_state.I = state.state.I + 1
                next_state.PnL = state.state.PnL - action.bid.price

            # Update market mid price
            if Bernoulli(0.5).sample():
                next_state.mid_price = state.state.mid_price + self.sigma * np.sqrt(self.delta_t)
            else:
                next_state.mid_price = state.state.mid_price - self.sigma * np.sqrt(self.delta_t)

            # Update the market bid-ask spread with the optimal bid-ask spread values.
            next_state.bid_ask_spread = action.ask.price - action.bid.price

            return (NonTerminal(next_state), next_state.PnL)
        
        self.t = self.t + self.delta_t
        return SampledDistribution(sr_sampler_func)

    def actions(self, state : NonTerminal[S]) -> \
            Iterator[A]:
        
        ask_price: float = state.state.mid_price + self.bid_ask_spread/2
        bid_price: float = state.state.mid_price - self.bid_ask_spread/2

        # Assume bid and ask volumes are 1 in infintessimal time.
        action: A = MarketAction(bid=PriceAndShares(price = bid_price,
                                                      shares = 1), 
                                 ask=PriceAndShares(price = ask_price, 
                                                   shares = 1))

        return iter([action])
    
class OptimalPolicy(Policy):
    T: float = 1
    delta_t: float = 0.005
    t: float = 0
    gamma: float = 0.1
    sigma: float = 2
    k: float = 1.5
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        indifference_ask_price: float = state.state.mid_price - (2 * state.state.I - 1) + self.gamma * self.sigma**2 * (self.T - self.t) / 2
        indifference_bid_price: float = state.state.mid_price - (2 * state.state.I - 1) - self.gamma * self.sigma**2 * (self.T - self.t) / 2
        
        ask_price: float = indifference_ask_price + 1 / self.gamma * np.log(1 + self.gamma/self.k)
        bid_price: float = indifference_bid_price - 1 / self.gamma * np.log(1 + self.gamma/self.k)

        # Assume bid and ask volumes are 1 in infintessimal time.
        action: A = MarketAction(bid=PriceAndShares(price = bid_price,
                                                      shares = 1), 
                                 ask=PriceAndShares(price = ask_price, 
                                                   shares = 1))
        
        return Constant(action)
    
class NaivePolicy(Policy):
    T: float = 1
    delta_t: float = 0.005
    t: float = 0
    gamma: float = 0.1
    sigma: float = 2
    k: float = 1.5
    bid_ask_spread: float = 1

    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        ask_price: float = state.state.mid_price + self.bid_ask_spread/2
        bid_price: float = state.state.mid_price - self.bid_ask_spread/2

        # Assume bid and ask volumes are 1 in infintessimal time.
        action: A = MarketAction(bid=PriceAndShares(price = bid_price,
                                                      shares = 1), 
                                 ask=PriceAndShares(price = ask_price, 
                                                   shares = 1))
        
        return Constant(action)

if __name__ == '__main__':
    num_traces: int = 100
    num_steps: int = 100

    # Optimal Market Making
    MarketMDP = OptimalMarketMaking()
    starting_dist = Constant(NonTerminal(MarketMaker()))

    # Generate sample paths.
    optimal_policy = OptimalPolicy()

    action_generator = list(itertools.islice(MarketMDP.action_traces(starting_dist, optimal_policy), num_traces))
    
    sample_paths = [list(itertools.islice(generator, num_steps)) for generator in action_generator]

    prices_opt = [step.state.state.mid_price for step in sample_paths[0]]
    PnL_opt = [[step.state.state.PnL for step in sample_path] for sample_path in sample_paths]
    bid_ask_spreads = [[step.state.state.bid_ask_spread for step in sample_path] for sample_path in sample_paths]
    avg_bid_ask_spread = float(np.mean([np.mean(bid_ask_spread) for bid_ask_spread in bid_ask_spreads]))

    # Naive Market Making
    MarketMDP = NaiveMarketMaking()
    starting_dist = Constant(NonTerminal(MarketMaker()))

    # Generate sample paths.
    naive_policy = NaivePolicy()
    naive_policy.bid_ask_spread = avg_bid_ask_spread

    action_generator = list(itertools.islice(MarketMDP.action_traces(starting_dist, naive_policy), num_traces))
    
    sample_paths = [list(itertools.islice(generator, num_steps)) for generator in action_generator]

    prices_naive = [step.state.state.mid_price for step in sample_paths[0]]
    PnL_naive = [[step.state.state.PnL for step in sample_path] for sample_path in sample_paths]

    plt.plot(prices_opt, alpha = 1, color='blue')
    plt.plot(prices_naive, alpha = 1, color='red')
    plt.title("Sample paths of Mid Price")
    plt.savefig("/Users/mwthomas/Documents/Stanford/Study/CME241-RLForFinance/Problem Sets/Assignment_6/Q2 Mid Price.png")
    plt.figure().clear()

    for PnL in PnL_opt:
        plt.plot(PnL, alpha = 10/len(PnL_opt), color='blue')
    for PnL in PnL_naive:
        plt.plot(PnL, alpha = 10/len(PnL_naive), color='green')
    plt.title("Sample paths of PnL")
    plt.savefig("/Users/mwthomas/Documents/Stanford/Study/CME241-RLForFinance/Problem Sets/Assignment_6/Q2 Profits.png")
    plt.figure().clear()