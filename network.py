import torch
import torch.nn as nn

from config import cfg


class DNN(nn.Module):
    def __init__(self, ob_space, hidden_size, act_space, activation):
        super(DNN, self).__init__()
        assert len(hidden_size) > 1
        normal_type = cfg['network']['normal']
        if normal_type == 'bn':
            n1 = nn.BatchNorm1d(hidden_size[0], momentum=None, track_running_stats=False)
        elif normal_type == 'ln':
            n1 = nn.GroupNorm(1, hidden_size[0])
        elif normal_type == 'gn':
            n1 = nn.GroupNorm(4, hidden_size[0])
        else:
            raise NotImplementedError
        layers = [
            nn.Linear(ob_space, hidden_size[0]),
            n1,
            nn.LeakyReLU()
        ]
        for s in range(len(hidden_size) - 1):
            in_size, out_size = hidden_size[s], hidden_size[s + 1]
            if normal_type == 'bn':
                normal_layer = nn.BatchNorm1d(out_size, momentum=None, track_running_stats=False)
            elif normal_type == 'ln':
                normal_layer = nn.GroupNorm(1, out_size)
            elif normal_type == 'gn':
                normal_layer = nn.GroupNorm(4, out_size)
            else:
                raise NotImplementedError
            layers.extend([
                nn.Linear(in_size, out_size),
                normal_layer,
                nn.LeakyReLU()
            ])
        if activation == 'sigmoid':
            activation_layer = nn.Sigmoid()
        elif activation == 'softmax':
            activation_layer = nn.Softmax(dim=-1)
        elif activation == 'relu':
            activation_layer = nn.ReLU()
        else:
            raise NotImplementedError
        layers.extend([
            nn.Linear(hidden_size[-1], act_space),
            activation_layer
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, in_size, out_size):
        super(Block, self).__init__()
        normal_type = cfg['network']['normal']
        self.linear = nn.Linear(in_size, out_size)
        if normal_type == 'bn':
            self.norm = nn.BatchNorm1d(out_size, momentum=None, track_running_stats=False)
        elif normal_type == 'ln':
            self.norm = nn.GroupNorm(1, out_size)
        elif normal_type == 'gn':
            self.norm = nn.GroupNorm(4, out_size)
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ob_space, h_size, act_space, activation):
        super(ResNet, self).__init__()
        assert len(h_size) > 1

        self.block1 = Block(ob_space, h_size[0])
        self.block2 = Block(h_size[0], h_size[1])
        if len(h_size) == 2:
            self.block3 = nn.Linear(h_size[1] + ob_space, act_space)
        elif len(h_size) == 3:
            self.block3 = Block(h_size[1] + ob_space, h_size[2])
            self.block4 = nn.Linear(h_size[2] + h_size[0], act_space)
        elif len(h_size) == 4:
            self.block3 = Block(h_size[1] + ob_space, h_size[2])
            self.block4 = Block(h_size[2] + h_size[0], h_size[3])
            self.block5 = nn.Linear(h_size[3] + ob_space, act_space)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out = self.block3(torch.cat([out2, x], dim=-1))
        out = self.block4(torch.cat([out, out1], dim=-1)) if hasattr(self, 'block4') else out
        out = self.block5(torch.cat([out, x], dim=-1)) if hasattr(self, 'block5') else out
        out = self.activation(out)
        return out


class Actor(nn.Module):
    def __init__(self, decision, frequency, interval, rate):
        super(Actor, self).__init__()
        self.decision_model = decision
        self.local_frequency = frequency
        self.interval_model = interval
        self.transmission_rate = rate
        self.act_space = cfg['environment']['device_num']

    def forward(self, env_tensor):
        decision_vec = self.decision_model(env_tensor)
        frequency_input = env_tensor[:, self.act_space:]
        frequency_vec = self.local_frequency(frequency_input)
        interval_vec = self.interval_model(torch.cat([env_tensor, decision_vec], dim=-1))
        rate_vec = self.transmission_rate(torch.cat([env_tensor, interval_vec], dim=-1))
        return decision_vec, frequency_vec, interval_vec, rate_vec
