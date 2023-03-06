from rl.distribution import (Constant, Categorical, Uniform, Distribution, FiniteDistribution,
                             SampledDistribution)
from rl.markov_process import MarkovProcess, State, NonTerminal
from rl.chapter9.order_book import OrderBook, DollarsAndShares, PriceSizePairs

import itertools
import copy
import matplotlib as plt
from numpy.random import poisson, chisquare, normal
from dataclasses import dataclass, replace
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)

S = OrderBook
A = PriceSizePairs
class OrderBookDynamics(MarkovProcess[State[S], A]):

    def transition(self, state: NonTerminal[S]) -> Distribution[NonTerminal[S]]:
        """Implement the transition probability for the Order Book 

        Args:
            state (NonTerminal[S])

        Returns:
            SampledDistribution[State[S]]
        """
        def sr_sampler_func(state=state):
            # Define number of new orders
            count_Buy_LO = poisson(2)
            count_Sell_LO = poisson(2)
            count_Buy_MO = poisson(1)
            count_Sell_MO = poisson(1)

            next_state = copy.deepcopy(state.state)

            for _ in range(count_Buy_LO):
                next_state = next_state.buy_limit_order(normal(state.state.bid_price(), 5), int(2*chisquare(6)))[1]
            for _ in range(count_Sell_LO):
                next_state = next_state.sell_limit_order(normal(state.state.ask_price(), 5), int(2*chisquare(6)))[1]
            for _ in range(count_Buy_MO):
                next_state = next_state.buy_market_order(int(2*chisquare(6)))[1]
            for _ in range(count_Sell_MO):
                next_state = next_state.sell_market_order(int(2*chisquare(6)))[1]

            return NonTerminal(next_state)

        return SampledDistribution(sampler=sr_sampler_func)

if __name__ == '__main__':

    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    # ob0.pretty_print_order_book()
    # ob0.display_order_book()

    OrderBookMP = OrderBookDynamics()
    starting_distribution = Constant(NonTerminal(ob0))

    num_traces = 1

    # Get the python generate
    sample_paths = list(itertools.islice(OrderBookMP.simulate(starting_distribution), 100))
    print(len(sample_paths))
    sample_paths[-1].state.pretty_print_order_book()