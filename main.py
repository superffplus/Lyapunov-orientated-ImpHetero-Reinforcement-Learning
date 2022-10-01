import os
import threading
import json
from collections import OrderedDict

import torch
import numpy as np

from environment import BasicEnv
from config import cfg
from agent import ExpAgent
from federated import *
from memory import *
from util import *


def test(env: BasicEnv, agent_dict: OrderedDict):
    max_test_epoch = cfg['train']['max_train_epoch']
    server_num = cfg['environment']['server_num']
    user_num = cfg['environment']['device_num']

    task_queue_len_scale_ratio = 200
    energy_queue_len_scale_ratio = 1000

    rates_record = np.zeros((server_num, user_num))  # record the computing rate in this time frame
    energy_record = np.zeros((server_num, user_num))  # record the energy consumption in this time frame

    avg_reward = 0
    avg_data_queue_len = np.zeros((server_num, user_num))  # record avg data length in the test
    avg_energy_queue_len = np.zeros((server_num, user_num))  # record avg energy length in the test
    avg_rate = np.zeros((server_num, user_num))  # record avg computing rate in the test
    avg_energy = np.zeros((server_num, user_num))

    channel, task_queue, energy_queue = env.reset()
    for epoch in range(max_test_epoch):
        task_queue = task_queue / task_queue_len_scale_ratio
        energy_queue = energy_queue / energy_queue_len_scale_ratio
        for split_rate, agents_under_split_rate in agent_dict.items():
            for n, single_agent in enumerate(agents_under_split_rate):
                server_count = single_agent.idx
                sub_channel = np.array([channel[server_count], channel[server_count]])
                sub_task_queue = np.array([task_queue[server_count], task_queue[server_count]])
                sub_energy_queue = np.array([energy_queue[server_count], energy_queue[server_count]])
                sub_observation = np.concatenate([sub_channel, sub_task_queue, sub_energy_queue], axis=-1)
                rates, energy, actions = single_agent.step(sub_observation)
                rates_record[server_count] = rates
                energy_record[server_count] = energy

        observation, reward = env.step((rates_record, energy_record))
        channel, task_queue, energy_queue = observation
        avg_data_queue_len += task_queue
        avg_energy_queue_len += energy_queue
        avg_rate += rates_record
        avg_energy += energy_record
        avg_reward += reward

    avg_data_queue_len = avg_data_queue_len / max_test_epoch
    avg_energy_queue_len = avg_energy_queue_len / max_test_epoch
    avg_rate = avg_rate / max_test_epoch
    avg_energy = avg_energy / max_test_epoch
    avg_reward = avg_reward / max_test_epoch

    raw_data_queue_len = {}
    raw_energy_queue_len = {}
    raw_rate = {}
    raw_energy = {}

    for split_rate, agents_under_split_rate in agent_dict.items():
        raw_data_queue_len_under_split_rate = np.zeros((len(agents_under_split_rate), user_num))
        raw_energy_queue_len_under_split_rate = np.zeros((len(agents_under_split_rate), user_num))
        raw_rate_under_split_rate = np.zeros((len(agents_under_split_rate), user_num))
        raw_energy_under_split_rate = np.zeros((len(agents_under_split_rate), user_num))
        for n, agent_under_split_rate in enumerate(agents_under_split_rate):
            raw_data_queue_len_under_split_rate[n] = avg_data_queue_len[agent_under_split_rate.idx]
            raw_energy_queue_len_under_split_rate[n] = avg_energy_queue_len[agent_under_split_rate.idx]
            raw_rate_under_split_rate[n] = avg_rate[agent_under_split_rate.idx]
            raw_energy_under_split_rate[n] = avg_energy[agent_under_split_rate.idx]
        raw_data_queue_len[split_rate] = raw_data_queue_len_under_split_rate.tolist()
        raw_energy_queue_len[split_rate] = raw_energy_queue_len_under_split_rate.tolist()
        raw_rate[split_rate] = raw_rate_under_split_rate.tolist()
        raw_energy[split_rate] = raw_energy_under_split_rate.tolist()
    return [avg_data_queue_len, avg_energy_queue_len, avg_rate, avg_energy, avg_reward], \
           [raw_data_queue_len, raw_energy_queue_len, raw_rate, raw_energy]


def distributed_train(env: BasicEnv, agent_dict, share_buffer: SharedBuffer, g_model: GlobalModel, json_file, pt_file, start_epoch=0):
    max_train_epoch = cfg['train']['max_train_epoch']
    batch_size = cfg['train']['batch_size']
    server_num = cfg['environment']['server_num']
    user_num = cfg['environment']['device_num']
    train_model = cfg['train']['train_model']

    task_queue_len_scale_ratio = 200
    energy_queue_len_scale_ratio = 1000

    rates_record = np.zeros((server_num, user_num))
    energy_record = np.zeros((server_num, user_num))
    action_record = np.zeros((server_num, user_num * 4))

    for epoch in range(start_epoch, max_train_epoch):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_data, raw_data = test(env, agent_dict)
            data_len_at_test, energy_len_at_test, rate_at_test, energy_at_test, reward_at_test = avg_data
            raw_data_len_at_test, raw_energy_len_at_test, raw_rate_at_test, raw_energy_at_test = raw_data
            min_values, max_values, mean_values = get_min_max_mean_value(data_len_at_test, energy_len_at_test, rate_at_test, energy_at_test, reward_at_test)
            min_data_len, min_energy_len, min_rate, min_energy, min_reward = min_values
            max_data_len, max_energy_len, max_rate, max_energy, max_reward = max_values
            mean_data_len, mean_energy_len, mean_rate, mean_energy, mean_reward = mean_values
            record = {
                'data_len': [min_data_len, max_data_len, mean_data_len],
                'energy_len': [min_energy_len, max_energy_len, mean_energy_len],
                'rate': [min_rate, max_rate, mean_rate],
                'energy': [min_energy, max_energy, mean_energy],
                'reward': [min_reward, max_reward, mean_reward],
                'raw_data_len': raw_data_len_at_test,
                'raw_energy_len': raw_energy_len_at_test,
                'raw_rate': raw_rate_at_test,
                'raw_energy': raw_energy_at_test
            }
            print()
            print(f'train {epoch=}')
            print(f'mean data len: {mean_data_len}')
            print(f'mean energy len: {mean_energy_len}')
            print(f'mean rate: {mean_rate}')
            print(f'mean reward: {mean_reward}')
            print()

            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    total_record = json.load(f)
            else:
                total_record = {}
            total_record[epoch] = record
            with open(json_file, 'w') as f:
                json.dump(total_record, f)

            actors_info, t_actors_info, critic_info, t_critic_info = {}, {}, {}, {}
            actor_optimizers_info, t_actor_optimizers_info, critic_optimizers_info, t_critic_optimizers_info = {}, {}, {}, {}

            for split_rate, agent_under_split_rate in agent_dict.items():
                actors_under_split_rate, t_actors_under_split_rate, critic_under_split_rate, t_critic_under_split_rate = [], [], [], []
                actor_optimizers_under_split_rate, t_actor_optimizers_under_split_rate, critic_optimizers_under_split_rate, t_critic_optimizers_under_split_rate = [], [], [], []
                for local_agent in agent_under_split_rate:
                    actors_under_split_rate.append(local_agent.actor.state_dict())
                    t_actors_under_split_rate.append(local_agent.t_actor.state_dict())
                    critic_under_split_rate.append(local_agent.critic.state_dict())
                    t_critic_under_split_rate.append(local_agent.t_critic.state_dict())
                    actor_optimizers_under_split_rate.append(local_agent.actor_optimizer.state_dict())
                    t_actor_optimizers_under_split_rate.append(local_agent.t_actor_optimizer.state_dict())
                    critic_optimizers_under_split_rate.append(local_agent.critic_optimizer.state_dict())
                    t_critic_optimizers_under_split_rate.append(local_agent.t_critic_optimizer.state_dict())

                actors_info[split_rate] = actors_under_split_rate
                t_actors_info[split_rate] = t_actors_under_split_rate
                critic_info[split_rate] = critic_under_split_rate
                t_critic_info[split_rate] = t_critic_under_split_rate

                actor_optimizers_info[split_rate] = actor_optimizers_under_split_rate
                t_actor_optimizers_info[split_rate] = t_actor_optimizers_under_split_rate
                critic_optimizers_info[split_rate] = critic_optimizers_under_split_rate
                t_critic_optimizers_info[split_rate] = t_critic_optimizers_under_split_rate

            saved_information = {
                'epoch': epoch + 1,
                'actor_in_local_model': actors_info,
                't_actor_in_local_model': t_actors_info,
                'critic_in_local_model': critic_info,
                't_critic_in_local_model': t_critic_info,
                'actor_optimizers_in_local_model': actor_optimizers_info,
                't_actor_optimizers_in_local_model': t_actor_optimizers_info,
                'critic_optimizers_in_local_model': critic_optimizers_info,
                't_critic_optimizers_in_local_model': t_critic_optimizers_info,
                'actor_in_global_model': g_model.actor.state_dict(),
                't_actor_in_global_model': g_model.t_actor.state_dict(),
                'critic_in_global_model': g_model.critic.state_dict(),
                't_critic_in_global_model': g_model.t_critic.state_dict()
            }

            torch.save(saved_information, pt_file)

        channel, task_queue, energy_queue = env.reset()
        for sub_epoch in range(200):
            task_queue = task_queue / task_queue_len_scale_ratio
            energy_queue = energy_queue / energy_queue_len_scale_ratio
            for split_rate, agents_under_split_rate in agent_dict.items():
                for agent_under_split_rate in agents_under_split_rate:
                    server_count = agent_under_split_rate.idx
                    sub_channel = np.array([channel[server_count], channel[server_count]])
                    sub_task_queue = np.array([task_queue[server_count], task_queue[server_count]])
                    sub_energy_queue = np.array([energy_queue[server_count], energy_queue[server_count]])
                    sub_observation = np.concatenate([sub_channel, sub_task_queue, sub_energy_queue], axis=-1)
                    rates, energy, actions = agent_under_split_rate.step(sub_observation)
                    rates_record[server_count] = rates
                    energy_record[server_count] = energy
                    action_record[server_count] = np.concatenate(actions, axis=-1)[0, :]

            observation, reward = env.step((rates_record, energy_record))
            n_channel, n_task_queue, n_energy_queue = observation
            share_buffer.store([channel, task_queue, energy_queue], action_record, reward,
                               [n_channel, n_task_queue / task_queue_len_scale_ratio,
                                n_energy_queue / energy_queue_len_scale_ratio])
            channel, task_queue, energy_queue = n_channel, n_task_queue, n_energy_queue

        for split_rate, agent_under_split_rate in agent_dict.items():
            for local_agent in agent_under_split_rate:
                local_agent.train()
                local_agent.learn(batch_size)
                local_agent.eval()
        if train_model == 'federated':
            g_model.receive_parameter(agent_dict)
            for split_rate, agent_under_split_rate in agent_dict.items():
                for local_agent in agent_under_split_rate:
                    local_agent.local_model.load_parameter()
                    if cfg['federated']['kl']:
                        local_agent.local_model.minimize_kl_distance(share_buffer, local_agent.idx)


if __name__ == '__main__':
    file_description_lst = [
        cfg['train']['train_model'],
        'kl' if cfg['federated']['kl'] else 'non-kl',
        str(cfg['train']['lr']),
        cfg['network']['type'],
        str(cfg['environment']['server_num']),
        str(cfg['environment']['device_num']),
        str(cfg['environment']['energy_th']),
        str(cfg['environment']['arrival_rate']),
        cfg['federated']['param_select']
    ]
    json_filename = './data/' + '_'.join(file_description_lst) + '.json'
    pt_filename = './data/' + '_'.join(file_description_lst) + '.pt'

    environment = BasicEnv()
    server_num = cfg['environment']['server_num']

    s_buffer = SharedBuffer()
    model_split_rte = cfg['agent']['split_rate']
    assert len(model_split_rte) == cfg['environment']['server_num']

    global_model = GlobalModel(environment.ob_space, environment.action_space)
    local_model_dict = {}
    for split_rate in model_split_rte:
        query_result_in_local_model_dict = local_model_dict.get(split_rate)
        if query_result_in_local_model_dict is None:
            local_model_dict[split_rate] = [LocalModel(environment.ob_space, environment.action_space, split_rate, global_model)]
        else:
            local_model_dict[split_rate].append(LocalModel(environment.ob_space, environment.action_space, split_rate, global_model))

    """
    If there exists a .pt file, then load the models and optimizers in this file
    """
    if os.path.exists(pt_filename):
        checkpoint = torch.load(pt_filename)
        start_epoch = checkpoint['epoch']

        # load parameters of local models from .pt file
        for split_rate, local_models_in_split_rate in local_model_dict.items():
            for i, local_model_in_split_rate in enumerate(local_models_in_split_rate):
                # load model parameters
                local_model_in_split_rate.actor.load_state_dict(checkpoint['actor_in_local_model'][split_rate][i])
                local_model_in_split_rate.t_actor.load_state_dict(checkpoint['t_actor_in_local_model'][split_rate][i])
                local_model_in_split_rate.critic.load_state_dict(checkpoint['critic_in_local_model'][split_rate][i])
                local_model_in_split_rate.t_critic.load_state_dict(checkpoint['t_critic_in_local_model'][split_rate][i])

                # load optimizer parameters
                local_model_in_split_rate.actor_optimizer.load_state_dict(checkpoint['actor_optimizers_in_local_model'][split_rate][i])
                local_model_in_split_rate.t_actor_optimizer.load_state_dict(checkpoint['t_actor_optimizers_in_local_model'][split_rate][i])
                local_model_in_split_rate.critic_optimizer.load_state_dict(checkpoint['critic_optimizers_in_local_model'][split_rate][i])
                local_model_in_split_rate.t_critic_optimizer.load_state_dict(checkpoint['t_critic_optimizers_in_local_model'][split_rate][i])

        global_model.actor.load_state_dict(checkpoint['actor_in_global_model'])
        global_model.t_actor.load_state_dict(checkpoint['t_actor_in_global_model'])
        global_model.critic.load_state_dict(checkpoint['critic_in_global_model'])
        global_model.t_critic.load_state_dict(checkpoint['t_critic_in_global_model'])

    else:
        start_epoch = 0

    # generate local agents and use OrderedDict to store them
    local_agent_dict = OrderedDict()
    idx = 0
    for split_rate, local_models_in_split_rate in local_model_dict.items():
        local_agents_in_split_rate = []
        for local_model_in_split_rate in local_models_in_split_rate:
            local_agent = ExpAgent(environment.ob_space, environment.action_space, s_buffer, idx, local_model_in_split_rate)
            local_agent.eval()
            local_agents_in_split_rate.append(local_agent)
            local_agent.local_model.set_local_param_idx(idx)
            idx += 1
        local_agent_dict[split_rate] = local_agents_in_split_rate

    distributed_train(environment, local_agent_dict, s_buffer, global_model, json_filename, pt_filename, start_epoch)
