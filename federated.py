import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import cfg
from util import *
from network import DNN, ResNet, Actor
from memory import Buffer


class GlobalModel:
    """
    record the global parameter and distribute the latest parameters to each local model.
    """
    def __init__(self, ob_space, act_space):
        self.device = cfg['train']['device']
        hidden_size = cfg['network']['hidden_size']
        if cfg['network']['type'] == 'DNN':
            self.decision_model = DNN(ob_space, hidden_size, act_space, 'sigmoid')
            self.frequency_model = DNN(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.interval_model = DNN(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.rate_model = DNN(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.t_decision_model = DNN(ob_space, hidden_size, act_space, 'sigmoid')
            self.t_frequency_model = DNN(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.t_interval_model = DNN(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.t_rate_model = DNN(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.critic = DNN(ob_space + act_space * 4, hidden_size, act_space, 'relu')
            self.t_critic = DNN(ob_space + act_space * 4, hidden_size, act_space, 'relu')
        elif cfg['network']['type'] == 'resnet':
            self.decision_model = ResNet(ob_space, hidden_size, act_space, 'sigmoid')
            self.frequency_model = ResNet(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.interval_model = ResNet(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.rate_model = ResNet(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.t_decision_model = ResNet(ob_space, hidden_size, act_space, 'sigmoid')
            self.t_frequency_model = ResNet(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.t_interval_model = ResNet(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.t_rate_model = ResNet(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.critic = ResNet(ob_space + act_space * 4, hidden_size, act_space, 'relu')
            self.t_critic = ResNet(ob_space + act_space * 4, hidden_size, act_space, 'relu')
        else:
            raise NotImplementedError

        # should use t_actor and t_critic by default
        self.actor = Actor(self.decision_model, self.frequency_model, self.interval_model, self.rate_model).to(self.device)
        self.t_actor = Actor(self.t_decision_model, self.t_frequency_model, self.t_interval_model, self.t_rate_model).to(self.device)
        self.critic.to(self.device)
        self.t_critic.to(self.device)

        self.modules = get_modules_dict(self)
        self.param_idx_among_scale_rates = {}

    def receive_parameter(self, local_agents_dict):
        """
        receive all parameters from local agents
        :param local_agents_dict: [1:[agent1, agent2], 0.5:[agent3, agent4]]
        :return:
        """
        local_param_dict = {}  # {'decision_model': [model1.decision_model, model2.decision_model], 'frequency_model': [model1.frequency_model, model2.frequency_model]}
        local_param_idx = {}  # {'decision_model': [model1.param_idx, model2.param_idx], 'frequency_model':[model1.param_idx, model2.param_idx]}

        # sort all trainable models according to the submodule name
        for local_agent_under_rate in local_agents_dict.values():
            for local_agent in local_agent_under_rate:
                for module_name in self.modules.keys():
                    if not local_param_dict.get(module_name):
                        local_param_dict[module_name] = [local_agent.local_model.modules[module_name].state_dict()]
                        local_param_idx[module_name] = [local_agent.local_model.param_idx_among_modules[module_name]]
                    else:
                        local_param_dict[module_name].append(local_agent.local_model.modules[module_name].state_dict())
                        local_param_idx[module_name].append(local_agent.local_model.param_idx_among_modules[module_name])

        for sub_module_name, sub_module in self.modules.items():
            new_global_params = OrderedDict()
            for name, param in sub_module.state_dict().items():
                count_tensors = torch.zeros(param.shape, dtype=torch.float32, device=self.device)
                tem_param = torch.zeros(param.shape, dtype=torch.float32, device=self.device)

                if name.split('.')[-1] == 'weight' and param.dim() > 1:
                    for local_model_param, local_model_param_idx in zip(local_param_dict[sub_module_name], local_param_idx[sub_module_name]):
                        tem_param[torch.meshgrid(local_model_param_idx[name])] += copy.deepcopy(local_model_param[name])
                        count_tensors[torch.meshgrid(local_model_param_idx[name])] += 1
                else:
                    for local_model_param, local_model_param_idx in zip(local_param_dict[sub_module_name], local_param_idx[sub_module_name]):
                        tem_param[local_model_param_idx[name]] += copy.deepcopy(local_model_param[name])
                        count_tensors[local_model_param_idx[name]] += 1

                tem_param[count_tensors > 0] = tem_param[count_tensors > 0].div_(count_tensors[count_tensors > 0])
                new_global_params[name] = tem_param
            sub_module.load_state_dict(new_global_params)

    def get_static_param_idx(self, scale_rate):
        params_idx = self.param_idx_among_scale_rates.get(scale_rate)
        if params_idx is None:
            self.param_idx_among_scale_rates[scale_rate] = self.generate_static_param_idx(scale_rate)
        return self.param_idx_among_scale_rates[scale_rate]

    def generate_static_param_idx(self, scale_rate):
        param_idx_among_models = {}   # record the idx of selected parameters among all models
        for sub_module_name, sub_module in self.modules.items():
            output_weight_name = [name for name in sub_module.state_dict().keys() if 'weight' in name.split('.')[-1]][-1]
            output_bias_name = [name for name in sub_module.state_dict().keys() if 'bias' in name.split('.')[-1]][-1]

            param_idx = {}    # record the idx of selected parameters in a single model
            input_idx = None
            if cfg['network']['type'] == 'DNN':
                for name, param in sub_module.state_dict().items():
                    if (param_type := name.split('.')[-1]) == 'weight':
                        if param.dim() > 1:
                            input_idx = np.arange(param.size(1)) if input_idx is None else input_idx
                            output_size = int(np.ceil(param.size(0) * scale_rate))
                            if name == output_weight_name:
                                output_idx = np.arange(param.size(0))
                            else:
                                output_idx = np.random.choice(param.size(0), output_size, replace=False)
                            param_idx[name] = torch.tensor(output_idx, dtype=torch.long, device=self.device), torch.tensor(input_idx, dtype=torch.long, device=self.device)
                            input_idx = output_idx
                        else:
                            param_idx[name] = input_idx
                    elif param_type == 'bias':
                        param_idx[name] = input_idx
            elif cfg['network']['type'] == 'resnet':
                for name, param in sub_module.state_dict().items():
                    if (param_type := name.split('.')[-1]) == 'weight':
                        if param.dim() > 1:
                            output_size = int(np.ceil(param.size(0) * scale_rate))
                            if input_idx is None:
                                input_idx = np.arange(param.size(1))
                            elif output_size != len(input_idx):
                                input_idx = np.concatenate((input_idx, np.arange(param.size(1) - self.ob_space, param.size(1))), axis=-1)
                            if name == output_weight_name:
                                output_idx = np.arange(param.size(0))
                            else:
                                output_idx = np.random.choice(param.size(0), output_size, replace=False)
                            param_idx[name] == torch.tensor(output_idx, dtype=torch.long, device=self.device), torch.tensor(input_idx, dtype=torch.long, device=self.device)
                            input_idx = output_idx
                        else:
                            param[name] = input_idx
                    elif param_type == 'bias':
                        param_idx[name] = input_idx
            else:
                raise NotImplementedError
            param_idx_among_models[sub_module_name] = param_idx
        
        return param_idx_among_models


class LocalModel:
    def __init__(self, ob_space, act_space, scale_rate, global_model: GlobalModel):
        self.global_model = global_model
        self.scale_rate = scale_rate
        self.ob_space = ob_space
        self.act_space = act_space
        self.device = cfg['train']['device']
        self.lr = cfg['train']['lr']

        hidden_size = [int(np.ceil(size * scale_rate)) for size in cfg['network']['hidden_size']]

        if cfg['network']['type'] == 'DNN':
            self.decision_model = DNN(ob_space, hidden_size, act_space, 'sigmoid')
            self.frequency_model = DNN(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.interval_model = DNN(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.rate_model = DNN(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.t_decision_model = DNN(ob_space, hidden_size, act_space, 'sigmoid')
            self.t_frequency_model = DNN(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.t_interval_model = DNN(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.t_rate_model = DNN(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.critic = DNN(ob_space + act_space * 4, hidden_size, act_space, 'relu')
            self.t_critic = DNN(ob_space + act_space * 4, hidden_size, act_space, 'relu')

        elif cfg['network']['type'] == 'resnet':
            self.decision_model = ResNet(ob_space, hidden_size, act_space, 'sigmoid')
            self.frequency_model = ResNet(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.interval_model = ResNet(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.rate_model = ResNet(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.t_decision_model = ResNet(ob_space, hidden_size, act_space, 'sigmoid')
            self.t_frequency_model = ResNet(ob_space-act_space, hidden_size, act_space, 'sigmoid')
            self.t_interval_model = ResNet(ob_space + act_space, hidden_size, act_space, 'softmax')
            self.t_rate_model = ResNet(ob_space + act_space, hidden_size, act_space, 'sigmoid')

            self.critic = ResNet(ob_space + act_space * 4, hidden_size, act_space, 'relu')
            self.t_critic = ResNet(ob_space + act_space * 4, hidden_size, act_space, 'relu')

        else:
            raise NotImplementedError

        self.actor = Actor(self.decision_model, self.frequency_model, self.interval_model, self.rate_model).to(self.device)
        self.t_actor = Actor(self.t_decision_model, self.t_frequency_model, self.t_interval_model, self.t_rate_model).to(self.device)
        self.critic.to(self.device)
        self.t_critic.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=1e-4)
        self.t_actor_optimizer = optim.Adam(self.t_actor.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=1e-4)
        self.t_critic_optimizer = optim.Adam(self.t_critic.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=1e-4)

        self.modules = get_modules_dict(self)

    
    def set_local_param_idx(self, idx):
        federated_param_select = cfg['federated']['param_select']
        if federated_param_select == 'random' or 'sequential':
        # if cfg['federated']['static'] == '':
            self.param_idx_among_modules = self.get_random_param_idx()
        elif federated_param_select == 'static':
            self.param_idx_among_modules = self.global_model.get_static_param_idx(idx)

    def get_random_param_idx(self):
        """
        Randomly obtain the idx of selected parameters
        :return: a dict to describe the idx of selected parameters. its shape is as: {model1:{para_name1:[], para_name_2:[]}}
        """
        random_select = cfg['federated']['param_select']
        param_idx_among_models = {}  # record the idx of selected parameters among all models
        for sub_module_name, sub_module in self.global_model.modules.items():
            output_weight_name = [name for name in sub_module.state_dict().keys() if 'weight' in name.split('.')[-1]][-1]
            output_bias_name = [name for name in sub_module.state_dict().keys() if 'bias' in name.split('.')[-1]][-1]

            param_idx = {}  # record the idx of selected parameters in a single model
            input_idx = None
            if cfg['network']['type'] == 'DNN':
                for name, param in sub_module.state_dict().items():
                    if (param_type := name.split('.')[-1]) == 'weight':
                        if param.dim() > 1:
                            input_idx = np.arange(param.size(1)) if input_idx is None else input_idx
                            output_size = int(np.ceil(param.size(0) * self.scale_rate))
                            if name == output_weight_name:
                                output_idx = np.arange(param.size(0))
                            elif random_select == 'random':
                                output_idx = np.random.choice(param.size(0), output_size, replace=False)
                            else:
                                output_idx = np.arange(param.size(0))[:output_size]
                            param_idx[name] = torch.tensor(output_idx, dtype=torch.long, device=self.device), torch.tensor(input_idx, dtype=torch.long, device=self.device)
                            input_idx = output_idx
                        else:
                            param_idx[name] = input_idx
                    elif param_type == 'bias':
                        param_idx[name] = input_idx
            elif cfg['network']['type'] == 'resnet':
                for name, param in sub_module.state_dict().items():
                    if (param_type := name.split('.')[-1]) == 'weight':
                        if param.dim() > 1:
                            output_size = int(np.ceil(param.size(0) * self.scale_rate))
                            if input_idx is None:
                                input_idx = np.arange(param.size(1))
                            elif output_size != len(input_idx):
                                input_idx = np.concatenate((input_idx, np.arange(param.size(1) - self.ob_space, param.size(1))), axis=-1)
                            if name == output_weight_name:
                                output_idx = np.arange(param.size(0))
                            elif random_select == 'random':
                                output_idx = np.random.choice(param.size(0), output_size, replace=False)
                            else:
                                output_idx = np.arange(param.size(0))[:output_size]
                            param_idx[name] == torch.tensor(output_idx, dtype=torch.long, device=self.device), torch.tensor(input_idx, dtype=torch.long, device=self.device)
                            input_idx = output_idx
                        else:
                            param[name] = input_idx
                    elif param_type == 'bias':
                        param_idx[name] = input_idx
            else:
                raise NotImplementedError

            param_idx_among_models[sub_module_name] = param_idx
        return param_idx_among_models

    def get_fixed_param_idx(self):
        """
        Get fixed idx of selected parameters.
        """
        pass

    def load_parameter(self):
        for sub_module_name, sub_module in self.global_model.modules.items():
            local_parameter = OrderedDict()
            for name, param in sub_module.state_dict().items():
                if name.split('.')[-1] == 'weight' and param.dim() > 1:
                    local_parameter[name] = copy.deepcopy(param[torch.meshgrid(self.param_idx_among_modules[sub_module_name][name])])
                else:
                    local_parameter[name] = copy.deepcopy(param[self.param_idx_among_modules[sub_module_name][name]])

            self.modules[sub_module_name].load_state_dict(local_parameter)
    
    def minimize_kl_between_local_and_global(self, data_buf: Buffer, target, idx):
        """
        Minimize KL distance between local model and global model.\n
        :param data_buf: data buffer
        :param target: minimize kl distance of target net or not
        :param idx: idx of edge device
        """
        batch_size = cfg['train']['batch_size']
        if target:
            local_actor = self.t_actor
            global_actor = self.global_model.t_actor
            actor_optimizer = self.t_actor_optimizer
            
            local_critic = self.t_critic
            global_critic = self.global_model.t_critic
            critic_optimizer = self.t_critic_optimizer
        else:
            local_actor = self.actor
            global_actor = self.global_model.actor
            actor_optimizer = self.actor_optimizer

            local_critic = self.critic
            global_critic = self.global_model.critic
            critic_optimizer = self.critic_optimizer
        for train_epoch in range(50):
            states, actions, rewards, next_states = data_buf.sample_data(batch_size)
            state_tensor = torch.as_tensor(states[:, idx, :], dtype=torch.float32, device=self.device)
            actor_optimizer.zero_grad()
            
            # calculate kl distance on decision model
            decision_tensor = backward_kl_distance(state_tensor, global_actor.decision_model, local_actor.decision_model, True)

            # calculate kl distance on frequency model
            frequency_input_tensor = state_tensor[:, self.act_space:]
            frequency_tensor = backward_kl_distance(frequency_input_tensor, global_actor.local_frequency, local_actor.local_frequency, True)

            # calculate kl distance on interval model
            interval_input_tensor = torch.cat([state_tensor, decision_tensor], dim=-1)
            rate_tensor = backward_kl_distance(interval_input_tensor, global_actor.interval_model, local_actor.interval_model, False)

            # calculate kl distance on rate model
            rate_input_tensor = torch.cat([state_tensor, rate_tensor], dim=-1)
            backward_kl_distance(rate_input_tensor, global_actor.transmission_rate, local_actor.transmission_rate, True)

            actor_optimizer.step()

            with torch.no_grad():
                next_state_tensor = torch.as_tensor(next_states[:, idx, :], dtype=torch.float32, device=self.device)
                next_decision, next_frequency, next_interval, next_rate = global_actor(next_state_tensor)
                next_critic_input_tensor = torch.cat([next_state_tensor, next_decision, next_frequency, next_interval, next_rate], dim=-1)
            
            critic_optimizer.zero_grad()
            # calculate kl distance on critic model
            backward_kl_distance(next_critic_input_tensor, global_critic, local_critic, True)
            critic_optimizer.step()
    
    def minimize_kl_distance(self, data_buf: Buffer, idx):
        if (federated_train_model := cfg['federated']['train_on_model']) == 'target':
            self.minimize_kl_between_local_and_global(data_buf, True, idx)
        elif federated_train_model == 'actor':
            self.minimize_kl_between_local_and_global(data_buf, False, idx)
        elif federated_train_model == 'both':
            self.minimize_kl_between_local_and_global(data_buf, True, idx)
            self.minimize_kl_between_local_and_global(data_buf, False, idx)


if __name__ == '__main__':
    global_model = GlobalModel(30, 10)
    local_model_1 = LocalModel(30, 10, 0.5, global_model)
    local_model_2 = LocalModel(30, 10, 0.5, global_model)
    local_model_3 = LocalModel(30, 10, 0.3, global_model)
    local_model_4 = LocalModel(30, 10, 0.3, global_model)
