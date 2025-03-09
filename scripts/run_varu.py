from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys, os, time
import re

# hosts = ['172.31.44.99', '10.20.23.91', '10.20.23.92', '10.20.23.42']
hosts = ['172.31.44.99', '172.31.38.220', '172.31.47.180', '172.31.43.229']
master_addr = hosts[0]
ngpus_per_node = 8
meg_project_dir = '/workspace/Megatron-LM-varuna'
varu_project_dir = '/workspace/varuna'
user = 'ubuntu'
pkey = '/home/ubuntu/.ssh/id_rsa'
timeout = 4800

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
# nstages = {'gpt3_350M': {32: 1, 30: 1, 28: 1, 26: 1, 24: 1, 22: 1, 20: 1, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1},
#            'gpt3_1_3B': {32: 2, 30: 15, 28: 7, 26: 13, 24: 3, 22: 11, 20: 5, 18: 9, 16: 2, 14: 7, 12: 2, 10: 5, 8: 2},
#            'gpt3_2_7B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 6, 22: 11, 20: 5, 18: 6, 16: 8, 14: 7, 12: 6, 10: 5, 8: 4},
#            'gpt3_6_7B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 12, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 8},
#            'gpt3_13B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 12, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 8}}
# mbs = {'gpt3_350M': {32: 8, 30: 8, 28: 8, 26: 8, 24: 8, 22: 8, 20: 8, 18: 8, 16: 8, 14: 8, 12: 8, 10: 8, 8: 8},
#        'gpt3_1_3B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 2, 22: 8, 20: 2, 18: 2, 16: 4, 14: 8, 12: 4, 10: 8, 8: 4},
#        'gpt3_2_7B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 1, 22: 2, 20: 1, 18: 2, 16: 4, 14: 2, 12: 2, 10: 1, 8: 4},
#        'gpt3_6_7B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 1, 22: 2, 20: 2, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 4},
#         'gpt3_13B': {32: 1, 30: 1, 28: 1, 26: 1, 24: 1, 22: 2, 20: 2, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1}}

clients = []
for host in hosts:
    clients.append(SSHClient(host=host, user=user, pkey=pkey))

local = clients[0]

def local_run_cmd(cmd, printCmd=True):
    output = local.run_command(cmd)
    if printCmd:
        print(cmd)
    local.wait_finished(output)
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)

def client_run_cmd(client, cmd, printCmd=True):
    output = client.run_command(cmd)
    if printCmd:
        print(cmd)
    client.wait_finished(output)
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)

def generate_available_machines(number, is_pretrain=True):
    if is_pretrain:
        with open('available_machines.out', 'w') as fp:
            machines = 0
            for host in hosts:
                for i in range(ngpus_per_node):
                    if machines < number:
                        machines += 1
                        fp.write(f'{host}:{i}\n')
                    else:
                        break
                if machines >= number:
                    break

def kill_all():
    cmd = 'cd ' + meg_project_dir + ' && bash ./scripts/kill_all.sh'
    local_run_cmd(cmd)

def cp_log(number, model, nstage, mbs):
    cmd = f'cd {meg_project_dir} && rm -rf res/ssh_logs_{number}_{model}_{nstage}_{mbs} && cp -r res/ssh_logs res/ssh_logs_{number}_{model}_{nstage}_{mbs}'
    local_run_cmd(cmd)

def rm_tmp():
    cmd = f'rm -rf /mnt/varuna/profile*'
    local_run_cmd(cmd)

def install_varuna():
    for client in clients:
        cmd = f'cd {varu_project_dir} && pip install -e .'
        client_run_cmd(client, cmd)

iteration_time_parser = re.compile(r'iteration(\s+)(?P<iterationnum>\d+)/(.+)\| elapsed time per iteration \(ms\): (?P<iterationtime>.+) \| learning rate')
error_parser = re.compile(r'\[Errno (\d+)\] Connection refused')
traceback_parser = re.compile(r'Traceback \(most recent call last\):')

def check_finish(success, fail, directory='res/ssh_logs'):
    if not os.path.exists(directory):
        fail = True
        return success, fail
    with open(f'{directory}/ssh_out_0.log', 'r') as f:
        for line in f:
            if 'Process done with return code 0' in line:
                success = True
                break
    with open(f'{directory}/ssh_err_0.log', 'r') as f:
        if sum(1 for _ in f):
            fail = True
    return success, fail

def run_test(number, model_i, load=False):
    assert number <= len(hosts) * ngpus_per_node, f'number: {number} > len(hosts): {len(hosts)} * ngpus_per_node: {ngpus_per_node}'
    print(f'run {models[model_i]} test {number} nodes load {load}')
    kill_all()
    rm_tmp()
    print('kill all')
    time.sleep(5)
    generate_available_machines(number, False)
    print('finish generate_available_machines')
    # output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/profile_gpt2.sh ' + models[model_i])
    # for line in output.stdout:
    #     print(line)
    # for line in output.stderr:
    #     print(line)
    # local.wait_finished(output)
    # for line in output.stdout:
    #     print(line)
    # for line in output.stderr:
    #     print(line)
    # print('finish profile')
    # kill_all()
    generate_available_machines(number, True)
    if load:
        output = local.run_command(f'cd {meg_project_dir} && bash ./scripts/pretrain_gpt2_varuna_load.sh {models[model_i]} {nstages[models[model_i]][number]} {mbs[models[model_i]][number]} {number}')
    else:
        output = local.run_command(f'cd {meg_project_dir} && bash ./scripts/pretrain_gpt2_varuna.sh {models[model_i]} {nstages[models[model_i]][number]} {mbs[models[model_i]][number]} {number} {master_addr}')
    print('time to sleep')
    success = False
    fail = False
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(20)
        if load:
            success, fail = check_finish(success, fail, f'res/ssh_log_{number}_{models[model_i]}_{nstages[models[model_i]][number]}_{mbs[models[model_i]][number]}_load')
        else:
            success, fail = check_finish(success, fail, f'res/ssh_log_{number}_{models[model_i]}_{nstages[models[model_i]][number]}_{mbs[models[model_i]][number]}')
        if success or fail:
            break
    print(f'time to kill {fail}')
    kill_all()
    time.sleep(10)
    print('finish pretrain')
    if success:
        # cp_log(number, models[model_i], nstages[models[model_i]][number], mbs[models[model_i]][number])
        pass
    if fail:
        print(f'failed with {number} {models[model_i]}, {nstages[models[model_i]][number]}, {mbs[models[model_i]][number]}')
        local.wait_finished(output)
        for line in output.stdout:
            print(line)
        for line in output.stderr:
            print(line)

# run_test(20, 4)

# for i in range(30, 34, 2):
#     for model_size_i in range(len(models)):
#         run_test(i, model_size_i)

for i in range(8, 34, 2):
    for model_i in range(0, len(models)):
        run_test(i, model_i, True)