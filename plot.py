import os
import json

import numpy as np
import matplotlib.pyplot as plt

from util import *


def decode_json(filename):
    """
    Read and load data from the json file
    Return record of all information
    The record has a shape as {'min_rate':{split_rate1: narray(epochs*count1), split_rate2: narray(epochs*count2)}, 'max_rate': ...}
    """
    with open(filename, 'r') as f:
        raw_json_data = json.load(f)

    epochs = []
    min_data_len_lst, max_data_len_lst, mean_data_len_lst = [], [], []
    min_energy_len_lst, max_energy_len_lst, mean_energy_len_lst = [], [], []
    min_rate_lst, max_rate_lst, mean_rate_lst = [], [], []
    min_energy_lst, max_energy_lst, mean_energy_lst = [], [], []
    min_reward_lst, max_reward_lst, mean_reward_lst = [], [], []

    for epoch, single_epoch_json_data in raw_json_data.items():
        epochs.append(int(epoch))
        min_data_len, max_data_len, mean_data_len = single_epoch_json_data['data_len']
        min_data_len_lst.append(min_data_len)
        max_data_len_lst.append(max_data_len)
        mean_data_len_lst.append(mean_data_len)
        
        min_energy_len, max_energy_len, mean_energy_len = single_epoch_json_data['energy_len']
        min_energy_len_lst.append(min_energy_len)
        max_energy_len_lst.append(max_energy_len)
        mean_energy_len_lst.append(mean_energy_len)
        
        min_rate, max_rate, mean_rate = single_epoch_json_data['rate']
        min_rate_lst.append(min_rate)
        max_rate_lst.append(max_rate)
        mean_rate_lst.append(mean_rate)
        
        min_energy, max_energy, mean_energy = single_epoch_json_data['energy']
        min_energy_lst.append(min_energy)
        max_energy_lst.append(max_energy)
        mean_energy_lst.append(mean_energy)
        
        min_reward, max_reward, mean_reward = single_epoch_json_data['reward']
        min_reward_lst.append(min_reward)
        max_reward_lst.append(max_reward)
        mean_reward_lst.append(mean_reward)
    
    split_rate_info = get_split_rate_info()
    all_info_dict = {
        'epoch': np.array(epochs),
        'min_data_len': split_data_lst_by_rate(min_data_len_lst, split_rate_info),
        'max_data_len': split_data_lst_by_rate(max_data_len_lst, split_rate_info),
        'mean_data_len': split_data_lst_by_rate(mean_data_len_lst, split_rate_info),
        'min_energy_len': split_data_lst_by_rate(min_energy_len_lst, split_rate_info),
        'max_energy_len': split_data_lst_by_rate(max_energy_len_lst, split_rate_info),
        'mean_energy_len': split_data_lst_by_rate(mean_energy_len_lst, split_rate_info),
        'min_rate': split_data_lst_by_rate(min_rate_lst, split_rate_info),
        'max_rate': split_data_lst_by_rate(max_rate_lst, split_rate_info),
        'mean_rate': split_data_lst_by_rate(mean_rate_lst, split_rate_info),
        'min_energy': split_data_lst_by_rate(min_energy_lst, split_rate_info),
        'max_energy': split_data_lst_by_rate(max_energy_lst, split_rate_info),
        'mean_energy': split_data_lst_by_rate(mean_energy_lst, split_rate_info),
        'min_reward': np.array(min_reward_lst),
        'max_reward': np.array(max_reward_lst),
        'mean_reward': np.array(mean_reward_lst)
    }
    return all_info_dict

def split_data_lst_by_rate(raw_data_lst, cfg_info):
    """
    Split raw data list according to split rates.
    This is to plot curves with different split rates.
    :param raw_data_lst [[epoch 1 data of N device], [epoch 2 data of N device], ...]
    :param cfg_info {split_rate1: count1, split_rate2: count2, ...}
    :return {split_rate1: narray(epochs*count1), split_rate2: narray(epochs*count2)}
    """
    raw_data_array = np.array(raw_data_lst)
    data_array_dict = {}
    count = 0
    for split_rate, split_rate_count in cfg_info.items():
        data_array_dict[split_rate] = raw_data_array[:, count: split_rate_count+count]
        count += split_rate_count
    return data_array_dict

def plot_indicator(all_file_info_dict, tag, colors, legend, save_fig_name):
    """
    Plot a figure according to tag using the data from all json files.
    :param all_file_info_dict {json_dscrp1: {'min_rate':{split_rate1: narray(epochs*count1), split_rate2: narray(epochs*count2)}}}
    :param tag select from ['data_len', 'energy_len', 'rate', 'energy']
    """
    split_rate_info = get_split_rate_info()
    fig, ax = plt.subplots()

    color_count = 0
    
    for label, single_file_info_dict in all_file_info_dict.items():
        epochs = single_file_info_dict['epoch']
        min_data = single_file_info_dict['min_'+tag]
        max_data = single_file_info_dict['max_'+tag]
        mean_data = single_file_info_dict['mean_'+tag]
        min_data_record, max_data_record, mean_data_record = [], [], []
        split_rate_record = []
        for split_rate in split_rate_info.keys():
            min_data_record.append(np.min(min_data[split_rate], axis=-1))
            max_data_record.append(np.max(max_data[split_rate], axis=-1))
            mean_data_record.append(np.mean(mean_data[split_rate], axis=-1))
            split_rate_record.append(split_rate)

        for i in range(len(min_data_record)):
            ax.plot(epochs, mean_data_record[i], color=colors[color_count], label=label+',SR='+str(split_rate_record[i]))
            ax.fill_between(epochs, min_data_record[i], max_data_record[i], color=colors[color_count], alpha=0.2)
            color_count += 1

    ax.set_xlabel('Epoch')
    ax.set_ylabel(legend)
    ax.legend(loc='best')
    plt.savefig(save_fig_name, dpi=600)

def plot_indicator_under_split_rate(all_file_info_dict, tag, split_rate, colors, legend, save_fig_name):
    """
    Plot a figure with a single indicator under one split rate.
    :param all_file_info_dict {json_dscrp1: {'min_rate':{split_rate1: narray(epochs*count1), split_rate2: narray(epochs*count2)}}}
    :param tag select from ['data_len', 'energy_len', 'rate', 'energy']
    :param split_rate the specific split rate
    :param colors candidate colors to plot
    :param legend the legend of figure
    :param save_fig_name the filename of figure
    """
    
    fig, ax = plt.subplots()
    color_count = 0

    # plot curve of each json file
    for json_file, json_file_info_dict in all_file_info_dict.items():
        epochs = json_file_info_dict['epoch']
        min_data = np.min(json_file_info_dict['min_'+tag][split_rate], axis=-1)
        max_data = np.max(json_file_info_dict['max_'+tag][split_rate], axis=-1)
        mean_data = np.mean(json_file_info_dict['mean_'+tag][split_rate], axis=-1)

        label = '_'.join([json_file.split('_')[0], json_file.split('_')[1], json_file.split('_')[-1]])
        ax.plot(epochs, mean_data, color=colors[color_count], label=label)
        # ax.fill_between(epochs, min_data, max_data, color=colors[color_count], alpha=0.2)
        color_count += 1
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(legend)
    ax.legend(loc='best')
    plt.savefig(save_fig_name, dpi=600)
    

def plot_reward_figure(all_file_info_dict, colors):
    split_rate_info = get_split_rate_info()
    fig, ax = plt.subplots()

    color_count = 0
    for json_file, single_file_info_dict in all_file_info_dict.items():
        epochs = single_file_info_dict['epoch']
        label = '_'.join([json_file.split('_')[0], json_file.split('_')[1], json_file.split('_')[-1]])
        min_rewards = single_file_info_dict['min_reward']
        max_rewards = single_file_info_dict['max_reward']
        mean_rewards = single_file_info_dict['mean_reward']
        ax.plot(epochs, mean_rewards, color=colors[color_count], label=label)
        ax.fill_between(epochs, min_rewards, max_rewards, color=colors[color_count], alpha=0.2)
        color_count += 1
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper left')
    plt.savefig('./fig/reward.jpg', dpi=600)



if __name__ == '__main__':
    filenames = ['./data/' + file for file in os.listdir('./data') if file.endswith('json')]
    # colors = ['tab:reb', 'tab:blue', 'tab:purple', 'tab:green', 'tab:yellow', 'tab:orange', 'tab:black', 'tab:gray']
    candidate_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    files_info_with_tags = {}
    for filename in filenames:
        decode_result = decode_json(filename)
        file_tag = os.path.splitext(os.path.split(filename)[1])[0]
        files_info_with_tags[file_tag] = decode_result
    plot_reward_figure(files_info_with_tags, candidate_colors)
    split_rates_info = get_split_rate_info()
    for split_rate in split_rates_info.keys():
        for indicator in ['data_len', 'energy_len', 'rate', 'energy']:
            legend = 'SR=' + str(split_rate)
            saved_filename = './fig/' + indicator + '_' + legend + '.jpg'
            plot_indicator_under_split_rate(files_info_with_tags, indicator, split_rate, candidate_colors, legend, saved_filename)