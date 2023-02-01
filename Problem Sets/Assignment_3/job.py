import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class job_mdp:

    def __init__(self, pi_list: list, wi_list: list, \
        gamma:float, alpha: float, a_init: float):
        self.pi_list = pi_list
        self.n = len(pi_list)
        self.utilities = np.log(wi_list)
        self.gamma = gamma
        self.alpha = alpha
        self.a_init = a_init
        self.create_transition_matrix(self.a_init)
        self.create_reward_vector(self.a_init)

    def create_transition_matrix(self, a) -> None:
        matrix = np.zeros((self.n+1, self.n+1))

        first_row = [a*self.pi_list[i] for i in range(n)]
        first_row.insert(0, 1-a)
        matrix[0] = first_row

        for i in range(1, self.n+1):
            matrix[i][0] = self.alpha
            matrix[i][i] = 1-self.alpha

        self.transition_matrix = matrix

    def get_transition_matrix(self) -> np.ndarray:
        return self.transition_matrix

    def create_reward_vector(self, a) -> None:
        vector = self.utilities

        vector[0] = a*np.sum([a*b for a,b in zip(self.utilities[1:], self.pi_list)]) \
            + (1-a)*self.utilities[0]

        self.reward_vector = vector

    def get_reward_vector(self) -> np.ndarray:
        return self.reward_vector

    def policy_evaluation(self, a, value_func, num_iter=100) -> np.ndarray:
        for i in range(num_iter):
            value_func = np.add(self.reward_vector,\
                 self.gamma*np.dot(self.transition_matrix, value_func))

        return value_func

    def policy_improvement(self, value_func):
        # discretize a values and find arg max
        a_vals = np.linspace(0, 1, num=1000)
        max, a_max = 0, a_vals[0]

        for a in a_vals:
            new_val = self.reward_vector[0] + np.multiply(np.dot(np.multiply(a, self.pi_list), \
                value_func[1:]), self.gamma) + (1-a)*value_func[0]
            if new_val > max:
                max = new_val
                a_max = a

        return a_max

if __name__=="__main__":

    n = 5
    pi_list = [(1/n)]*n
    wi_list = [25, 50, 35, 60, 200]
    wi_list.insert(0, 100)
    gamma = 0.8
    alpha = 0.5
    policy = 0.5
    value_func = [1]*(n+1)

    job_obj = job_mdp(pi_list, wi_list, gamma, alpha, policy)
    # print(job_obj.get_transition_matrix())

    num_iters = 10

    for i in range(num_iters):

        value_func = job_obj.policy_evaluation(policy, value_func)
        policy = job_obj.policy_improvement(value_func)
        job_obj.create_transition_matrix(policy)
        job_obj.create_reward_vector(policy)

    print(value_func)
