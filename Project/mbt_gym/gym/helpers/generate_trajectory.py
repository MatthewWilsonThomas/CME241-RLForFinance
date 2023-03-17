import gym
import numpy as np
import torch

from typing import Dict

from mbt_gym.agents.Agent import Agent
from MultiAgent.MultiAgentTradingEnvironment import MultiAgentTradingEnvironment


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None, include_log_probs: bool = False):
    if seed is not None:
        env.seed(seed)
    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    observations = np.zeros((env.num_trajectories, obs_space_dim, env.n_steps + 1))
    actions = np.zeros((env.num_trajectories, action_space_dim, env.n_steps))
    rewards = np.zeros((env.num_trajectories, 1, env.n_steps))
    if include_log_probs:
        log_probs = torch.zeros((env.num_trajectories, env.action_space.shape[0], env.n_steps))
    obs = env.reset()
    observations[:, :, 0] = obs
    count = 0
    while True:
        if include_log_probs:
            action, log_prob = agent.get_action(obs, include_log_probs=True)
        else:
            action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        actions[:, :, count] = action
        observations[:, :, count + 1] = obs
        rewards[:, :, count] = reward.reshape(-1, 1)
        if include_log_probs:
            log_probs[:, :, count] = log_prob
        if (env.num_trajectories > 1 and done[0]) or (env.num_trajectories == 1 and done):
            break
        count += 1
    if include_log_probs:
        return observations, actions, rewards, log_probs
    else:
        return observations, actions, rewards


def generate_multiagent_trajectory(env: MultiAgentTradingEnvironment, 
                                agents: Dict[str, Agent] or None, 
                                seed: int = None,
                                include_log_probs: bool = False):

    if seed is not None:
        env.seed(seed)

    agent_names = env.agents

    obs_space_dim = env.observation_space[agent_names[0]].shape[0]
    action_space_dim = env.action_space[agent_names[0]].shape[0]

    observations = {agent_id: np.zeros((env.num_trajectories, obs_space_dim, env.n_steps + 1)) for agent_id in agent_names}
    actions = {agent_id: np.zeros((env.num_trajectories, action_space_dim, env.n_steps)) for agent_id in agent_names}
    rewards = {agent_id: np.zeros((env.num_trajectories, 1, env.n_steps)) for agent_id in agent_names}
    if include_log_probs:
        log_probs = {agent_id: torch.zeros((env.num_trajectories, action_space_dim, env.n_steps)) for agent_id in agent_names}

    obs = env.reset()
    for agent_id in agent_names:
        observations[agent_id][:, :, 0] = obs[agent_id]

    count = 0
    action: Dict[str, np.ndarray] = {}
    log_prob: Dict[str, np.ndarray] = {}

    while True:
        for agent_id in agent_names:
            if include_log_probs:
                action[agent_id], log_prob[agent_id] = agents[agent_id].get_action(obs[agent_id], include_log_probs=True)
            else:
                action[agent_id] = agents[agent_id].get_action(obs[agent_id])

        obs, reward, done, _ = env.step(action)
        for agent_id in agent_names:
            actions[agent_id][:, :, count] = action[agent_id]
            observations[agent_id][:, :, count + 1] = obs[agent_id]
            rewards[agent_id][:, :, count] = reward[agent_id].reshape(-1, 1)

            if include_log_probs:
                log_probs[agent_id][:, :, count] = log_prob[agent_id]

        if (env.num_trajectories > 1 and done[agent_id][0]) or (env.num_trajectories == 1 and done[agent_id]):
            break
        count += 1


    if include_log_probs:
        return observations, actions, rewards, log_probs
    else:
        return observations, actions, rewards