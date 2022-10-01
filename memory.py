from abc import abstractmethod
import numpy as np
import pandas as pd

from config import cfg


class Buffer:
    """
    The basic class of the buffer to store the trajectory data from training.
    The Buffer class should implement the $store$ and $sample_data$ methods.
    Each server should implement a Buffer class
    """
    def __init__(self):
        self.max_size = cfg['agent']['buffer_size']
        self.size = 0
        self.index_ptr = 0
        self.column_name = ['states', 'actions', 'rewards', 'next_states']

        # Here we assume the buffer is with fixed length
        self.memory = pd.DataFrame(index=range(self.max_size), columns=self.column_name)

    @abstractmethod
    def store(self, state, action, reward, next_state):
        """
        store the trajectory data
        :param state: data in shape (3*user_num), describing the channel information, task queue, virtual energy queue
        :param action: data in shape (4*user_num), containing decisions, frequencies, intervals, powers
        :param reward: reward from the environment
        :param next_state: next state of the environment
        :return:
        """
        pass

    def sample_data(self, batch_size):
        pass


class SimpleBuffer(Buffer):
    def __init__(self):
        super(SimpleBuffer, self).__init__()

    def store(self, state, action, reward, next_state):
        self.memory.iloc[self.index_ptr] = dict(zip(self.column_name, [state, action, reward, next_state]))
        self.index_ptr = (self.index_ptr + 1) % self.max_size
        self.size = s if (s := self.size+1) <= self.max_size else self.max_size

    def sample_data(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class SharedBuffer(Buffer):
    def __init__(self):
        super(SharedBuffer, self).__init__()

    def store(self, state, action, reward, next_state):
        """
        store all information in a shared buffer
        :param state: [channel(server_num*user_num), data_queue(server_num*user_num), energy_queue(server_num*user_num)]
        :param action: a (server_num, 4*user_num) array
        :param reward: a (server_num) array
        :param next_state: an array in the same shape with variable "state"
        :return:
        """
        observation = np.concatenate(state, axis=-1)
        next_observation = np.concatenate(next_state, axis=-1)
        self.memory.iloc[self.index_ptr] = dict(zip(self.column_name, [observation, action, reward, next_observation]))
        self.index_ptr = (self.index_ptr + 1) % self.max_size
        self.size = s if (s := self.size + 1) <= self.max_size else self.max_size

    def sample_data(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# class DistributeBuffer(Buffer):
#     def __init__(self, idx):
#         super(DistributeBuffer, self).__init__()
#         self.idx = idx
