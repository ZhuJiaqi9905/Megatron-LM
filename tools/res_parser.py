import os
import random
import re
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pprint


models = ['gpt3_350M', 'gpt3_1_3B', 'gpt3_2_7B', 'gpt3_6_7B', 'gpt3_13B']
nstages = {'gpt3_350M': {32: 1, 30: 1, 28: 1, 26: 1, 24: 1, 22: 1, 20: 1, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1},
           'gpt3_1_3B': {32: 2, 30: 2, 28: 2, 26: 2, 24: 2, 22: 2, 20: 2, 18: 2, 16: 2, 14: 2, 12: 2, 10: 2, 8: 2},
           'gpt3_2_7B': {32: 4, 30: 5, 28: 4, 26: 13, 24: 4, 22: 11, 20: 4, 18: 6, 16: 4, 14: 7, 12: 4, 10: 5, 8: 4},
           'gpt3_6_7B': {32: 8, 30: 6, 28: 7, 26: 13, 24: 8, 22: 11, 20: 10, 18: 6, 16: 8, 14: 7, 12: 6, 10: 10, 8: 8},
           'gpt3_13B': {32: 16, 30: 15, 28: 14, 26: 13, 24: 12, 22: 11, 20: 20, 18: 18, 16: 16, 14: 14, 12: 12, 10: 10, 8: 8}}
mbs = {'gpt3_350M': {32: 8, 30: 8, 28: 8, 26: 8, 24: 8, 22: 8, 20: 8, 18: 8, 16: 8, 14: 8, 12: 8, 10: 8, 8: 8},
       'gpt3_1_3B': {32: 4, 30: 4, 28: 4, 26: 4, 24: 4, 22: 4, 20: 4, 18: 4, 16: 4, 14: 4, 12: 4, 10: 4, 8: 4},
       'gpt3_2_7B': {32: 4, 30: 4, 28: 4, 26: 8, 24: 4, 22: 8, 20: 4, 18: 4, 16: 4, 14: 8, 12: 4, 10: 8, 8: 4},
       'gpt3_6_7B': {32: 4, 30: 2, 28: 2, 26: 4, 24: 4, 22: 4, 20: 4, 18: 2, 16: 2, 14: 2, 12: 2, 10: 4, 8: 4},
        'gpt3_13B': {32: 4, 30: 4, 28: 2, 26: 2, 24: 2, 22: 2, 20: 4, 18: 2, 16: 4, 14: 1, 12: 1, 10: 1, 8: 1}}

iteration_time_parser = re.compile(r'iteration        3/       (\d+) \| elapsed time per iteration \(ms\): (?P<iteration_time>\S+)')
save_checkpoint_time_parser = re.compile(r'save checkpoint: (?P<checkpoint_time>\S+)s')
load_checkpoint_time_parser = re.compile(r'load checkpoint: (?P<checkpoint_time>\S+)s')

time_res = {}

def res_parser(file):
    file_parts = file.split('/')[1].split('_')
    model_size = '_'.join(file_parts[3:-2])
    node_num = int(file_parts[2])
    nstage = int(file_parts[-2])
    mbs = int(file_parts[-1])
    load_file = file.split('/')[0] + '/' + file.split('/')[1] + '_load/' + file.split('/')[2]
    if os.path.exists(file) is False:
        print(f'{file} does not exist')
        return
    if os.path.exists(load_file) is False:
        print(f'{load_file} does not exist')
        return
    with open(file, 'r') as fp:
        for line in fp.readlines():
            iteration_time_res = iteration_time_parser.search(line)
            save_checkpoint_time_res = save_checkpoint_time_parser.search(line)
            if iteration_time_res:
                iteration_time = float(iteration_time_res.group('iteration_time'))
                if time_res.get(model_size) is None:
                    time_res[model_size] = {node_num: {nstage: {mbs: {'iteration_time': iteration_time}}}}
                else:
                    if time_res[model_size].get(node_num) is None:
                        time_res[model_size][node_num] = {nstage: {mbs: {'iteration_time': iteration_time}}}
                    else:
                        if time_res[model_size][node_num].get(nstage) is None:
                            time_res[model_size][node_num][nstage] = {mbs: {'iteration_time': iteration_time}}
                        else:
                            time_res[model_size][node_num][nstage][mbs] = {'iteration_time': iteration_time}
            if save_checkpoint_time_res:
                save_checkpoint_time = float(save_checkpoint_time_res.group('checkpoint_time'))
                assert time_res.get(model_size) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size].get(node_num) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num].get(nstage) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num][nstage].get(mbs) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                time_res[model_size][node_num][nstage][mbs]['checkpoint_time'] = save_checkpoint_time
    with open(load_file, 'r') as fp:
        for line in fp.readlines():
            load_checkpoint_time_res = load_checkpoint_time_parser.search(line)
            if load_checkpoint_time_res:
                load_checkpoint_time = float(load_checkpoint_time_res.group('checkpoint_time'))
                assert time_res.get(model_size) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size].get(node_num) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num].get(nstage) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num][nstage].get(mbs) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                time_res[model_size][node_num][nstage][mbs]['load_checkpoint_time'] = load_checkpoint_time
        
import csv
with open('res.csv', 'w', newline='') as csvfile:
    fieldnames = ['models', 'node_num', 'nstage', 'mbs', 'iteration_time', 'checkpoint_time', 'load_checkpoint_time']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for model in models[:]:
        for node_num in range(8, 26, 2):
            if model == 'gpt3_13B' and node_num < 12:
                continue
            nstage = nstages[model][node_num]
            mb = mbs[model][node_num]
            res_parser(f'res/ssh_log_{node_num}_{model}_{nstage}_{mb}/ssh_out_0.log')
            writer.writerow([model, node_num, nstage, mb, time_res[model][node_num][nstage][mb]['iteration_time'], time_res[model][node_num][nstage][mb]['checkpoint_time'], time_res[model][node_num][nstage][mb]['load_checkpoint_time']])
            

pprint.pprint(time_res)