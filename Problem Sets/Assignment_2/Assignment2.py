import argparse
from rl.markov_process import *
from rl.distribution import *
from dataclasses import dataclass
import itertools
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np

parser = argparse.ArgumentParser(
    prog = 'Assignment2.py',
    description = 'Code for problem set 2 of CME241: RL for Finance'
)

parser.add_argument('-q', '--question', type = int, default = 0, help='Which question do you wish to run?')
args = parser.parse_args()
question = args.question

if question == 1 or question == 0:
    ################ Question 1. Snakes and Ladders ########################
    @dataclass(frozen=True)
    class StateSL(State[int]):
        square: int

    SnakesAndLaddersMapping = {
        StateSL(97): StateSL(78), 
        StateSL(95): StateSL(56),
        StateSL(88): StateSL(24),
        StateSL(62): StateSL(18),
        StateSL(48): StateSL(26),
        StateSL(36): StateSL(6),
        StateSL(32): StateSL(10),
        StateSL(1): StateSL(38),
        StateSL(4): StateSL(14),
        StateSL(8): StateSL(30),
        StateSL(21): StateSL(41), 
        StateSL(28): StateSL(76),
        StateSL(50): StateSL(67),
        StateSL(71): StateSL(92),
        StateSL(80): StateSL(99)
    }
    Transition = dict([(StateSL(i) , Categorical({StateSL(i+1): 1/6, StateSL(i+2): 1/6, StateSL(i+3):1/6, StateSL(i+4):1/6, StateSL(i+5):1/6, StateSL(i+6):1/6}) )for i in range(100)])

    for key, value in Transition.copy().items():
        for outcome, prob in value.probabilities.copy().items():
            if outcome in SnakesAndLaddersMapping:
                value.probabilities.pop(outcome)
                outcome = SnakesAndLaddersMapping[outcome]
                value.probabilities[outcome] = 1/6
            if outcome.square > 100:
                del value.probabilities[outcome]
                for outcome_sub, prob in value.probabilities.items():
                    if outcome_sub.square == 100:
                        prob += 1/6

    SnakesAndLaddersGame = FiniteMarkovProcess(Transition)

    ################ Simulation ################
    # Always start on square 0
    starting_distribution = Categorical({NonTerminal(StateSL(0)):1})

    num_traces = 1000

    # Get the python generate
    traces = itertools.islice(SnakesAndLaddersGame.traces(starting_distribution), num_traces)

    # Sample from the generators, enough times so that they always finish the game. 
    sample_paths = [list(itertools.islice(trace, 10000)) for trace in traces]

    # Count the number of steps required to finish the game for each sample path. 
    time_steps_to_finish = [len(path) for path in sample_paths]

    # Plot the sampled traces.
    for trace in sample_paths:
        path = [state.state.square for state in trace]
        plt.plot(path, alpha = 0.01, color='royalblue')
    plt.title("Sample paths of Snakes And Ladders")
    plt.savefig('Assignment_2/Q1. Sample paths of Snakes and Ladders.png')
    plt.figure().clear()

    # # Plot the number of time steps as a histogram to approximate the distribution
    plt.hist(time_steps_to_finish, bins=100, density=True)
    plt.title("Distribution of the number of rolls till the game ends")
    plt.savefig('Assignment_2/Q1. Distribution of time to finish.png')
    plt.figure().clear()

    print(f"Completed {num_traces} simulations of Snakes and Ladders")

    ################ Expected number of rolls ################

    TransitionReward = dict([(StateSL(i) , Categorical({(StateSL(i+1), 1): 1/6, (StateSL(i+2), 1): 1/6, (StateSL(i+3), 1):1/6, (StateSL(i+4), 1):1/6, (StateSL(i+5), 1):1/6, (StateSL(i+6), 1):1/6}) )for i in range(100)])

    for key, value in TransitionReward.copy().items():
        for outcome, prob in value.probabilities.copy().items():
            if outcome[0] in SnakesAndLaddersMapping:
                value.probabilities.pop(outcome)
                outcome = (SnakesAndLaddersMapping[outcome[0]], 1)
                value.probabilities[outcome] = 1/6
            if outcome[0].square > 100:
                del value.probabilities[outcome]
                for outcome_sub, prob in value.probabilities.items():
                    if outcome_sub[0].square == 100:
                        value.probabilities[outcome_sub] += 1/6

    SnakesAndLaddersRewardGame = FiniteMarkovRewardProcess(TransitionReward)

    SnakesAndLaddersRewardGame.display_value_function(1)