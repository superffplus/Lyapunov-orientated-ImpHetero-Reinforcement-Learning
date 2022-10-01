from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import cfg
from federated import LocalModel
from memory import *
from util import *


class Agent:
    """
    The basic class of agent in RL
    """
    def __init__(self, ob_space, act_space, buffer: Buffer, idx):
        self.ob_space = ob_space
        self.act_space = act_space
        self.buffer = buffer
        self.idx = idx
        self.device = cfg['train']['device']

    @abstractmethod
    def explore(self, observation):
        pass

    @abstractmethod
    def step(self, observation):
        pass

    @abstractmethod
    def learn(self, batch_size):
        pass


class ExpAgent(Agent):
    def __init__(self, ob_space, act_space, buffer: Buffer, idx, local_model: LocalModel):
        super(ExpAgent, self).__init__(ob_space, act_space, buffer, idx)
        self.discount = cfg['agent']['discount']
        self.update_ratio = cfg['agent']['update_ratio']
        self.lr = cfg['train']['lr']
        self.epsilon = cfg['agent']['explore_epsilon']

        self.local_model = local_model
        self.actor = local_model.actor
        self.actor_optimizer = local_model.actor_optimizer
        self.t_actor = local_model.t_actor
        self.t_actor_optimizer = local_model.t_actor_optimizer
        self.critic = local_model.critic
        self.critic_optimizer = local_model.critic_optimizer
        self.t_critic = local_model.t_critic
        self.t_critic_optimizer = local_model.t_critic_optimizer
        self.critic_criterion = nn.MSELoss()

        # ensure t_actor and actor are the same.
        hard_update(self.t_actor, self.actor)
        hard_update(self.t_critic, self.critic)

        # some config about the model
        self.phi = 100  # number of CPU cycles to process 1-bit task
        ch_fact = 10 ** 10
        d_fact = 10 ** 6
        self.bandwidth = 2  # bandwidth MHz
        self.noise_power = self.bandwidth * d_fact * (10 ** (-17.4)) * (10 ** (-3)) * ch_fact
        self.k_factor = (10 ** (-26)) * (d_fact ** 3)  # energy coefficient
        self.max_frequency = 300  # maximum local computing frequency 100MHz
        self.max_power = 0.1  # maximum transmitter power 100mW
        self.vu = 1.1

    def get_random_action(self):
        decisions = np.random.uniform(-4, 4, self.act_space)
        decisions = 1 / (1 + np.exp(-decisions))
        frequencies = np.random.uniform(0, 1, self.act_space)
        intervals = np.random.uniform(0, 1, self.act_space)
        trans_rates = np.random.uniform(0, 1, self.act_space)
        return decisions, frequencies, intervals, trans_rates

    def explore(self, observation):
        r_seed = np.random.uniform(0, 1)
        if r_seed < self.epsilon:
            ob_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            decisions, frequencies, intervals, rates = self.actor(ob_tensor)
            decision_arr = decisions.detach().cpu().numpy()
            frequency_arr = frequencies.detach().cpu().numpy()
            interval_arr = intervals.detach().cpu().numpy()
            rate_arr = rates.detach().cpu().numpy()
            actions = decision_arr, frequency_arr, interval_arr, rate_arr
            rates, energy = self.get_rates_and_energy(observation, decision_arr, frequency_arr, interval_arr, rate_arr)
        else:
            actions = self.get_random_action()
            rates, energy = self.get_soft_rates_and_energy(observation, *actions)
        return rates, energy, actions

    def get_rates_and_energy(self, observation, decisions, frequencies, intervals, trans_rates):
        """
        This is used to calculate the accuracy rates and energy, and is not advised to used in training the actor model
        :param observation: 1*n array or 2*n array with a same copy
        :param decisions: 1*n array or 2*n array with a same copy
        :param frequencies: 1*n array or 2*n array with a same copy
        :param intervals: 1*n array or 2*n array with a same copy
        :param trans_rates: 1*n array or 2*n array with a same copy
        :return:
        """
        observation = observation[0, :]
        decisions = decisions[0, :]
        frequencies = frequencies[0, :]
        intervals = intervals[0, :]
        trans_rates = trans_rates[0, :]

        rates = np.zeros(self.act_space)
        energy = np.zeros(self.act_space)

        # local computing
        local_idx = np.where(decisions < 0.5)
        offload_idx = np.where(decisions >= 0.5)
        schedule_frequency = frequencies[local_idx] * self.max_frequency
        rates[local_idx] = schedule_frequency / 100
        energy[local_idx] = self.k_factor * (schedule_frequency ** 3)

        # offloading
        channel_state = observation[:self.act_space]
        SNR = (channel_state[offload_idx] / self.noise_power)
        trans_rates_max = self.bandwidth / self.vu * np.log2(1 + SNR * self.max_power)
        s_intervals = softmax_interval(offload_idx, intervals)
        rates[offload_idx] = trans_rates[offload_idx] * trans_rates_max * s_intervals
        e_ratio = 1 / SNR * (2 ** (rates[offload_idx] / s_intervals * self.vu / self.bandwidth) - 1)
        # inner_product = (rates[offload_idx] * self.vu) / (self.bandwidth * intervals[offload_idx])
        # e_ratio = self.noise_power * intervals[offload_idx] / channel_state[offload_idx] * (2 ** inner_product - 1)
        energy[offload_idx] = e_ratio * s_intervals
        return rates, energy

    def get_soft_rates_and_energy(self, observation, decisions, frequencies, intervals, trans_rates):
        local_computing_frequencies = frequencies * self.max_frequency
        local_rates = local_computing_frequencies / 100
        local_energy = self.k_factor * (local_computing_frequencies ** 3)

        channel_state = observation[:self.act_space]
        SNR = channel_state / self.noise_power
        trans_rates_max = self.bandwidth / self.vu * np.log2(1 + SNR * self.max_power)
        s_exp = np.exp(intervals)
        s_intervals = s_exp / np.sum(s_exp, axis=-1)
        # s_intervals = softmax_interval(np.ones(self.usr_num), intervals)
        offload_rates = trans_rates * trans_rates_max * s_intervals
        e_ratio = 1 / SNR * (2 ** (offload_rates / s_intervals * self.vu / self.bandwidth) - 1)
        offload_energy = e_ratio * s_intervals

        rates = (1 - decisions) * local_rates + decisions * offload_rates
        energy = (1 - decisions) * local_energy + decisions * offload_energy
        return rates, energy

    def step(self, observation):
        ob_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(-1, self.ob_space)
        decisions, frequencies, intervals, rates = self.t_actor(ob_tensor)
        decision_arr = decisions.detach().cpu().numpy()
        frequency_arr = frequencies.detach().cpu().numpy()
        interval_arr = intervals.detach().cpu().numpy()
        rate_arr = rates.detach().cpu().numpy()
        rates, energy = self.get_rates_and_energy(observation, decision_arr, frequency_arr, interval_arr, rate_arr)
        return rates, energy, [decision_arr, frequency_arr, interval_arr, rate_arr]

    def learn(self, batch_size):
        sampled_training_data = self.buffer.sample_data(batch_size)
        self.ddpg_update(sampled_training_data)

    def ddpg_update(self, training_data):
        states, actions, rewards, next_states = training_data
        state_tensor = torch.as_tensor(states[:, self.idx, :], dtype=torch.float32, device=self.device)
        # sub_actions = actions[:, self.usr_num:]
        action_tensor = torch.as_tensor(actions[:, self.idx, :], dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards[:, self.idx], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states_tensor = torch.as_tensor(next_states[:, self.idx, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_decision, next_frequency, next_interval, next_rate = self.t_actor(next_states_tensor)
            # next_frequency = next_frequency * next_decision
            # next_interval = next_interval * next_decision
            # next_rate = next_rate * next_decision
            next_critic_input_1 = torch.cat([next_states_tensor, next_decision, next_frequency, next_interval, next_rate], dim=-1)
            next_q_value = self.t_critic(next_critic_input_1)
            target_q_value = rewards_tensor + self.discount * next_q_value

        self.critic_optimizer.zero_grad()
        q_value = self.critic(torch.cat([state_tensor, action_tensor], dim=-1))
        loss = self.critic_criterion(q_value, target_q_value)
        loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        n_decision, n_frequency, n_interval, n_rate = self.actor(state_tensor)
        n_action = torch.cat([n_decision, n_frequency, n_interval, n_rate], dim=-1)
        policy_loss = -self.critic(torch.cat([state_tensor, n_action], dim=-1))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.t_actor, self.actor, self.update_ratio)
        soft_update(self.t_critic, self.critic, self.update_ratio)

    def train(self):
        self.actor.train()
        self.t_actor.train()
        self.critic.train()
        self.t_critic.train()

    def eval(self):
        self.actor.eval()
        self.t_actor.eval()
        self.critic.eval()
        self.t_critic.eval()


if __name__ == '__main__':
    # m.to('cuda')
    b = torch.randn(5, 30)
