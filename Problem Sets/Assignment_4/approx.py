from typing import Iterator, Tuple, TypeVar, Sequence, List, Mapping
from operator import itemgetter
import numpy as np
import itertools

from rl.distribution import Distribution, Choose
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import DeterministicPolicy, FinitePolicy, FiniteDeterministicPolicy

from rl.dynamic_programming import greedy_policy_from_vf, evaluate_mrp_result
from rl.approximate_dynamic_programming import extended_vf, evaluate_finite_mrp

S = TypeVar('S')
A = TypeVar('A')
V = Mapping[NonTerminal[S], float]

ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]

def policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: Tuple[ValueFunctionApprox[S], FinitePolicy[S, A]]
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, by improving the policy repeatedly after 
    using the given FunctionApprox to approximate the  optimal value function 
    at each step.
    '''

    def update(vf_policy: Tuple[ValueFunctionApprox[S], FinitePolicy[S, A]])\
            -> Tuple[ValueFunctionApprox[S], FiniteDeterministicPolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: ValueFunctionApprox[S] = itertools.islice(evaluate_finite_mrp(mrp, gamma, vf), 100)[-1]

        greedy_policy_dict = {}
        for s in mdp.non_terminal_states:
            q_values: Iterator[Tuple[A, float]] = \
                ((a, mdp.mapping[s][a].expectation(
                    lambda s_r: s_r[1] + gamma * extended_vf(vf, s_r[0])
                )) for a in mdp.actions(s))
            greedy_policy_dict[s.state] = \
                max(q_values, key=itemgetter(1))[0]

        improved_pi: FiniteDeterministicPolicy[S, A] = FiniteDeterministicPolicy(greedy_policy_dict)

        return policy_vf, improved_pi

    return iterate(update, approx_0)







def value_iteration_finite(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given finite
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step
    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + gamma * extended_vf(v, s1)

        return v.update(
            [(
                s,
                max(mdp.mapping[s][a].expectation(return_)
                    for a in mdp.actions(s))
            ) for s in mdp.non_terminal_states]
        )

    return iterate(update, approx_0)
