from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys, os, time
import re

hosts = [91, 92, 90, 42, 46, 47]
for idx, host_suffix in enumerate(hosts):
    hosts[idx] = '172.21.0.' + str(host_suffix)
ports = [2230 + i for i in range(4)]
meg_project_dir = '/workspace/Megatron-LM-varuna'
varu_project_dir = '/workspace/varuna'
user = 'root'
pkey = '/root/.ssh/id_rsa'
timeout = 1200

models = ['gpt3_350M', 'gpt3_1_3B', 'gpt3_2_7B', 'gpt3_6_7B']
nstages = {'gpt3_350M': {24: 4, 22: 11, 20: 4, 18: 6, 16: 4, 14: 7, 12: 4, 10: 5, 8: 2},
           'gpt3_1_3B': {24: 4, 22: 11, 20: 4, 18: 6, 16: 4, 14: 7, 12: 4, 10: 5, 8: 4},
           'gpt3_2_7B': {24: 6, 22: 11, 20: 5, 18: 6, 16: 8, 14: 7, 12: 6, 10: 5, 8: 8},
           'gpt3_6_7B': {24: 12, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 8}}
mbs = {'gpt3_350M': {24: 8, 22: 16, 20: 8, 18: 8, 16: 8, 14: 8, 12: 4, 10: 8, 8: 4},
       'gpt3_1_3B': {24: 2, 22: 8, 20: 2, 18: 2, 16: 2, 14: 2, 12: 2, 10: 2, 8: 2},
       'gpt3_2_7B': {24: 1, 22: 2, 20: 1, 18: 2, 16: 4, 14: 2, 12: 2, 10: 1, 8: 1},
       'gpt3_6_7B': {24: 1, 22: 2, 20: 2, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1}}

clients = []
for host in hosts:
    for port in ports:
        clients.append(SSHClient(host=host, port=port, user=user, pkey=pkey))

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

def generate_available_machines(number, is_pretrain=True):
    if is_pretrain:
        with open('available_machines.out', 'w') as fp:
            machines = 0
            for host in hosts:
                for port in ports:
                    if machines < number:
                        machines += 1
                        fp.write(str(host) + ':' + str(port) + '\n')
                    else:
                        break
                if machines >= number:
                    break

def kill_all():
    cmd = 'cd ' + meg_project_dir + ' && bash ./scripts/kill_all.sh'
    local_run_cmd(cmd)

def cp_log(number, model, nstage, mbs):
    cmd = f'cd {meg_project_dir} && rm -rf ssh_logs_{number}_{model}_{nstage}_{mbs} && cp -r ssh_logs ssh_logs_{number}_{model}_{nstage}_{mbs}'
    local_run_cmd(cmd)

def rm_tmp():
    cmd = f'rm -rf /mnt/gpu-91/varuna/profile*'
    local_run_cmd(cmd)

iteration_time_parser = re.compile(r'iteration(\s+)(?P<iterationnum>\d+)/(.+)\| elapsed time per iteration \(ms\): (?P<iterationtime>.+) \| learning rate')
error_parser = re.compile(r'\[Errno (\d+)\] Connection refused')
traceback_parser = re.compile(r'Traceback \(most recent call last\):')

def check_finish(success, fail, directory='ssh_logs'):
    with open(f'{directory}/ssh_out_0.log', 'r') as f:
        for line in f:
            if 'Process done with return code 0' in line:
                success = True
                break
    with open(f'{directory}/ssh_err_0.log', 'r') as f:
        for line in f:
            if 'Error' in line:
                fail = True
    return success, fail

def run_test(number, model_i, load=False):
    print(f'run {models[model_i]} test {number} nodes')
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
    output = local.run_command(f'cd {meg_project_dir} && bash ./scripts/pretrain_gpt2_varuna.sh {models[model_i]} {nstages[models[model_i]][number]} {mbs[models[model_i]][number]} {number}')
    print('time to sleep')
    success = False
    fail = False
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(20)
        success, fail = check_finish(success, fail, f'ssh_logs_{number}_{models[model_i]}_{nstages[models[model_i]][number]}_{mbs[models[model_i]][number]}')
        if success or fail:
            break
    print('time to kill')
    kill_all()
    print('finish pretrain')
    if success:
        # cp_log(number, models[model_i], nstages[models[model_i]][number], mbs[models[model_i]][number])
        pass
    if fail:
        print(f'failed with {number} {models[model_i]}, {nstages[models[model_i]][number]}, {mbs[models[model_i]][number]}')
    if load:
        output = local.run_command(f'cd {meg_project_dir} && bash ./scripts/pretrain_gpt2_varuna_load.sh {models[model_i]} {nstages[models[model_i]][number]} {mbs[models[model_i]][number]} {number}')
        local.wait_finished(output)
        for line in output.stdout:
            print(line)
        for line in output.stderr:
            print(line)

run_test(8, 0)
# run_test(20, 2)

# for model_i in range(0, 2):
#     for i in (8, 12):
#         run_test(i, model_i)

# for model_i in range(0, len(models)):
#     run_test(16, model_i)