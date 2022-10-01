import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from memory import *


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        pass


def get_modules_dict(federated_model):
    """
    return the modules which should be executed federated learning step.\n
    :param federated_model: local model or global model
    :return:
    """

    """
    actor: execute federated learning step on all the models in the actor
    target: execute federated learning step on all the models in the target actor
    both: execute federated learning step on all the models in both actor and target actor
    """
    if (train_on_model := cfg['federated']['train_on_model']) == 'actor':
        modules = {
            'decision_model': federated_model.decision_model,
            'frequency_model': federated_model.frequency_model,
            'interval_model': federated_model.interval_model,
            'rate_model': federated_model.rate_model,
            'critic_model': federated_model.critic
        }
    elif train_on_model == 'target':
        modules = {
            't_decision_model': federated_model.t_decision_model,
            't_frequency_model': federated_model.t_frequency_model,
            't_interval_model': federated_model.t_interval_model,
            't_rate_model': federated_model.t_rate_model,
            't_critic_model': federated_model.t_critic
        }
    elif train_on_model == 'both':
        modules = {
            'decision_model': federated_model.decision_model,
            'frequency_model': federated_model.frequency_model,
            'interval_model': federated_model.interval_model,
            'rate_model': federated_model.rate_model,
            'critic_model': federated_model.critic,
            't_decision_model': federated_model.t_decision_model,
            't_frequency_model': federated_model.t_frequency_model,
            't_interval_model': federated_model.t_interval_model,
            't_rate_model': federated_model.t_rate_model,
            't_critic_model': federated_model.t_critic
        }
    else:
        raise NotImplementedError
    return modules


def kl_distance_update(global_model, local_agent):
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    batch_size = cfg['train']['batch_size']
    for epoch in range(50):
        if isinstance(local_agent.buffer, SimpleBuffer):
            sampled_training_data = local_agent.buffer.sample_data(batch_size)
        elif isinstance(local_agent.buffer, SharedBuffer):
            sampled_training_data = local_agent.buffer.sample_data(batch_size, local_agent.idx)
        else:
            raise NotImplementedError
        states, actions, rewards, next_states = sampled_training_data
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=local_agent.device)
        with torch.no_grad():
            t_decision_tensor, t_frequency_tensor, t_interval_tensor, t_rate_tensor = global_model.t_actor(state_tensor)
            t_action = torch.cat([state_tensor, t_decision_tensor, t_frequency_tensor, t_interval_tensor, t_rate_tensor], dim=-1)
            t_value = global_model.t_critic(t_action)
        train_on_model = cfg['federated']['train_on_model']
        if train_on_model == 'actor' or train_on_model == 'target':
            if train_on_model == 'actor':
                local_agent_decision_model = local_agent.actor.decision_model
                local_agent_frequency_model = local_agent.actor.local_frequency
                local_agent_interval_model = local_agent.actor.interval_model
                local_agent_rate_model = local_agent.actor.transmission_rate
                local_agent_critic_model = local_agent.critic
                a_optimizer = local_agent.actor_optimizer
                c_optimizer = local_agent.critic_optimizer
            else:
                local_agent_decision_model = local_agent.t_actor.decision_model
                local_agent_frequency_model = local_agent.t_actor.local_frequency
                local_agent_interval_model = local_agent.t_actor.interval_model
                local_agent_rate_model = local_agent.t_actor.transmission_rate
                local_agent_critic_model = local_agent.t_critic
                a_optimizer = local_agent.t_actor_optimizer
                c_optimizer = local_agent.t_critic_optimizer

            a_optimizer.zero_grad()
            c_optimizer.zero_grad()

            decision_tensor = local_agent_decision_model(state_tensor)
            frequency_tensor = local_agent_frequency_model(state_tensor)
            interval_tensor = local_agent_interval_model(torch.cat([state_tensor, t_decision_tensor], dim=-1))
            masked_interval_tensor = t_interval_tensor * t_decision_tensor
            rate_tensor = local_agent_rate_model(torch.cat([state_tensor, masked_interval_tensor], dim=-1))
            value_tensor = local_agent_critic_model(t_action)

            kl_loss_on_decision = kl_loss_on_tensor(decision_tensor, t_decision_tensor, kl_loss_fn)
            kl_loss_on_frequency = kl_loss_on_tensor(frequency_tensor, t_frequency_tensor, kl_loss_fn)
            kl_loss_on_interval = kl_loss_on_tensor(interval_tensor, t_interval_tensor, kl_loss_fn)
            kl_loss_on_rate = kl_loss_on_tensor(rate_tensor, t_rate_tensor, kl_loss_fn)
            kl_loss_on_value = kl_loss_on_tensor(value_tensor, t_value, kl_loss_fn)

            kl_loss_on_decision.backward()
            kl_loss_on_frequency.backward()
            kl_loss_on_interval.backward()
            kl_loss_on_rate.backward()
            kl_loss_on_value.backward()

            a_optimizer.step()
            c_optimizer.step()

        elif train_on_model == 'both':
            local_agent.actor_optimizer.zero_grad()
            local_agent.critic_optimizer.zero_grad()

            decision_tensor = local_agent.actor.decision_model(state_tensor)
            frequency_tensor = local_agent.actor.local_frequency(state_tensor)
            interval_tensor = local_agent.actor.interval_model(torch.cat([state_tensor, t_decision_tensor], dim=-1))
            masked_interval_tensor = t_interval_tensor * t_decision_tensor
            rate_tensor = local_agent.actor.transmission_rate(torch.cat([state_tensor, masked_interval_tensor], dim=-1))
            value_tensor = local_agent.critic(t_action)

            local_agent.t_actor_optimizer.zero_grad()
            local_agent.t_critic_optimizer.zero_grad()

            dt_decision_tensor = local_agent.t_actor.decision_model(state_tensor)
            dt_frequency_tensor = local_agent.t_actor.local_frequency(state_tensor)
            dt_interval_tensor = local_agent.t_actor.interval_model(torch.cat([state_tensor, t_decision_tensor], dim=-1))
            dt_masked_interval_tensor = t_interval_tensor * t_decision_tensor
            dt_rate_tensor = local_agent.t_actor.transmission_rate(torch.cat([state_tensor, dt_masked_interval_tensor],
                                                                            dim=-1))
            dt_value_tensor = local_agent.t_critic(t_action)

            kl_loss_on_decision = kl_loss_on_tensor(decision_tensor, t_decision_tensor, kl_loss_fn)
            kl_loss_on_frequency = kl_loss_on_tensor(frequency_tensor, t_frequency_tensor, kl_loss_fn)
            kl_loss_on_interval = kl_loss_on_tensor(interval_tensor, t_interval_tensor, kl_loss_fn)
            kl_loss_on_rate = kl_loss_on_tensor(rate_tensor, t_rate_tensor, kl_loss_fn)
            kl_loss_on_value = kl_loss_on_tensor(value_tensor, t_value, kl_loss_fn)

            kl_loss_on_dt_decision = kl_loss_on_tensor(dt_decision_tensor, t_decision_tensor, kl_loss_fn)
            kl_loss_on_dt_frequency = kl_loss_on_tensor(dt_frequency_tensor, t_frequency_tensor, kl_loss_fn)
            kl_loss_on_dt_interval = kl_loss_on_tensor(dt_interval_tensor, t_interval_tensor, kl_loss_fn)
            kl_loss_on_dt_rate = kl_loss_on_tensor(dt_rate_tensor, t_rate_tensor, kl_loss_fn)
            kl_loss_on_dt_value = kl_loss_on_tensor(dt_value_tensor, t_value, kl_loss_fn)

            kl_loss_on_decision.backward()
            kl_loss_on_frequency.backward()
            kl_loss_on_interval.backward()
            kl_loss_on_rate.backward()
            kl_loss_on_value.backward()

            kl_loss_on_dt_decision.backward()
            kl_loss_on_dt_frequency.backward()
            kl_loss_on_dt_interval.backward()
            kl_loss_on_dt_rate.backward()
            kl_loss_on_dt_value.backward()

            local_agent.actor_optimizer.step()
            local_agent.critic_optimizer.step()
            local_agent.t_actor_optimizer.step()
            local_agent.t_critic_optimizer.step()


def kl_loss_on_tensor(prediction, target, loss_fn):
    log_prediction = F.log_softmax(prediction, dim=-1)
    s_target = F.softmax(target, dim=-1)
    kl_loss = loss_fn(log_prediction, s_target)
    return kl_loss


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def softmax_interval(offload_idx, intervals):
    candidate = intervals[offload_idx]
    s_exp = np.exp(candidate)
    s_result = s_exp / np.sum(s_exp, axis=-1)
    return s_result


def get_min_max_mean_value(*args):
    min_values = []
    max_values = []
    mean_values = []
    for n_array in args:
        min_value = np.min(n_array, axis=-1).tolist()
        max_value = np.max(n_array, axis=-1).tolist()
        mean_value = np.mean(n_array, axis=-1).tolist()

        min_values.append(min_value)
        max_values.append(max_value)
        mean_values.append(mean_value)

    return min_values, max_values, mean_values

def get_split_rate_info():
    """
    Get split rate information from the .yml file
    :return {split_rate1:num1, split_rate2:nume2, ...}
    """
    split_rates = cfg['agent']['split_rate']
    split_rate_info = {}
    for split_rate in split_rates:
        if split_rate_info.get(split_rate) is None:
            split_rate_info[split_rate] = 1
        else:
            split_rate_info[split_rate] += 1
    return split_rate_info

def backward_kl_distance(net_input, target_net, predict_net, softmax):
    """
    This is to calculate the KL distance between target and predict net, and then backward the gradient to the predict net
    :param net_input input data of two nets
    :param target_net target net
    :param predict_net training net
    :param softmax trainig data need to softmax or not
    :return the output of the target net
    """
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    prediction = predict_net(net_input)
    target = target_net(net_input)
    if softmax:
        prediction_log = F.log_softmax(prediction, dim=-1)
        target = F.softmax(target, dim=-1)
    else:
        prediction_log = torch.log(prediction)
    kl_loss = kl_loss_fn(prediction_log, target)
    kl_loss.backward()
    return target.detach()
