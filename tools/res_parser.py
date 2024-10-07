import os
import random
import re
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pprint


models = ['gpt3_350M', 'gpt3_1_3B', 'gpt3_2_7B', 'gpt3_6_7B']
nstages = {'gpt3_350M': {24: 4, 22: 11, 20: 4, 18: 6, 16: 4, 14: 7, 12: 4, 10: 5, 8: 2},
           'gpt3_1_3B': {24: 4, 22: 11, 20: 4, 18: 6, 16: 4, 14: 7, 12: 4, 10: 5, 8: 4},
           'gpt3_2_7B': {24: 6, 22: 11, 20: 5, 18: 6, 16: 8, 14: 7, 12: 6, 10: 5, 8: 8},
           'gpt3_6_7B': {24: 12, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 8}}
mbs = {'gpt3_350M': {24: 8, 22: 16, 20: 8, 18: 8, 16: 8, 14: 8, 12: 4, 10: 8, 8: 4},
       'gpt3_1_3B': {24: 2, 22: 8, 20: 2, 18: 2, 16: 2, 14: 2, 12: 2, 10: 2, 8: 2},
       'gpt3_2_7B': {24: 1, 22: 2, 20: 1, 18: 2, 16: 4, 14: 2, 12: 2, 10: 1, 8: 1},
       'gpt3_6_7B': {24: 1, 22: 2, 20: 2, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1}}

iteration_time_parser = re.compile(r'iteration        3/       (\d+) \| elapsed time per iteration \(ms\): (?P<iteration_time>\S+)')
checkpoint_time_parser = re.compile(r'save checkpoint: (?P<checkpoint_time>\S+)s')

time_res = {}

def res_parser(file):
    file_parts = file.split('/')[0].split('_')
    model_size = '_'.join(file_parts[3:-2])
    node_num = int(file_parts[2])
    nstage = int(file_parts[-2])
    mbs = int(file_parts[-1])
    with open(file, 'r') as fp:
        for line in fp.readlines():
            iteration_time_res = iteration_time_parser.search(line)
            checkpoint_time_res = checkpoint_time_parser.search(line)
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
            if checkpoint_time_res:
                checkpoint_time = float(checkpoint_time_res.group('checkpoint_time'))
                assert time_res.get(model_size) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size].get(node_num) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num].get(nstage) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                assert time_res[model_size][node_num][nstage].get(mbs) is not None, f'{model_size}, {node_num}, {nstage}, {mbs}'
                time_res[model_size][node_num][nstage][mbs]['checkpoint_time'] = checkpoint_time
        
for model in models[:3]:
    for node_num in range(8, 26, 2):
        res_parser(f'ssh_logs_{node_num}_{model}_{nstages[model][node_num]}_{mbs[model][node_num]}/ssh_out_0.log')

pprint.pprint(time_res)