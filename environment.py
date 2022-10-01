import math
import numpy as np

from config import cfg


class BasicEnv:
    def __init__(self):
        self.server_num = cfg['environment']['server_num']
        self.usr_num = cfg['environment']['device_num']
        self.user_weight = np.array([[1.5 if i % 2 == 0 else 1 for i in range(cfg['environment']['device_num'])] for _ in range(self.server_num)])
        self.energy_th = cfg['environment']['energy_th']
        self.h_metric = [self.generate_channel() for _ in range(self.server_num)]
        self.channel_factor = 10 ** 10
        self.channel_states = self.generate_real_time_channel(0.3) * self.channel_factor
        self.arrival_rate = cfg['environment']['arrival_rate']
        self.energy_queue = np.zeros([self.server_num, self.usr_num])
        self.data_queue = np.zeros([self.server_num, self.usr_num])

    @property
    def action_space(self):
        return self.usr_num

    @property
    def ob_space(self):
        return self.usr_num * 3

    def step(self, action):
        """
        execute the action and return the new state and rewards
        :param action: contains rates (server_num * user_num) and energy (server_num * user_num)
        :return: channel_states, data_queue and energy queue are (server_num * user_num), rewards is (server_num)
        """
        rates, energy = action
        self.channel_states = self.generate_real_time_channel(0.3) * self.channel_factor
        arrival_task = np.random.exponential(self.arrival_rate, [self.server_num, self.usr_num])
        tmp_data_queue = self.data_queue + arrival_task - rates
        self.data_queue = np.maximum(tmp_data_queue, 0)
        tmp_energy_queue = self.energy_queue + (energy - self.energy_th) * 1000
        self.energy_queue = np.maximum(tmp_energy_queue, 0)
        virtual_z_queue = self.data_queue + self.energy_queue * 0.5
        rewards = (virtual_z_queue + 20 * self.user_weight) * rates - virtual_z_queue * energy
        # rewards = 10 * rates - 0.1 * virtual_z_queue * energy
        rewards = np.sum(rewards, axis=1)
        return [self.channel_states, self.data_queue, self.energy_queue], rewards

    def reset(self):
        self.channel_states = self.generate_real_time_channel(0.3) * self.channel_factor
        self.energy_queue = np.zeros((self.server_num, self.usr_num))
        self.data_queue = np.zeros((self.server_num, self.usr_num))
        return self.channel_states, self.data_queue, self.energy_queue

    def generate_channel(self):
        """
        generate h metric
        :return: 
        """
        dist_v = np.linspace(start=120, stop=255, num=self.usr_num)
        fc = 915 * 10 ** 6
        light = 3 * 10 ** 8
        loss_exponent = 3  # path loss exponent
        h = 3 * (light / 4 / math.pi / fc / dist_v) ** loss_exponent
        return h

    def generate_real_time_channel(self, factor):
        """
        generate real-time random wireless channel.
        :param factor: here we set 0.3
        :return: real-time channel state in (server_num, user_num)
        """
        n = self.usr_num
        channels = []
        for h in self.h_metric:
            beta = np.sqrt(h * factor)
            sigma = np.sqrt(h * (1 - factor) / 2)
            x = np.multiply(sigma * np.ones(n), np.random.randn(n)) + beta * np.ones(n)
            y = np.multiply(sigma * np.ones(n), np.random.randn(n))
            g = np.power(x, 2) + np.power(y, 2)
            channels.append(g)
        return np.array(channels)


if __name__ == '__main__':
    env = BasicEnv()
    print()