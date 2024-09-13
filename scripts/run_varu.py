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
timeout = 600

models = ['gpt3_350M', 'gpt3_1_3B', 'gpt3_2_7B', 'gpt3_6_7B']
nstages = {'gpt3_350M': {24: 8, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 1},
           'gpt3_1_3B': {24: 8, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 4},
           'gpt3_2_7B': {24: 8, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 4},
           'gpt3_6_7B': {24: 8, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 4}}
mbs = {'gpt3_350M': {24: 16, 22: 16, 20: 8, 18: 8, 16: 8, 14: 4, 12: 4, 10: 16, 8: 16},
       'gpt3_1_3B': {24: 8, 22: 8, 20: 4, 18: 4, 16: 4, 14: 2, 12: 2, 10: 2, 8: 2},
       'gpt3_2_7B': {24: 8, 22: 8, 20: 4, 18: 4, 16: 4, 14: 2, 12: 2, 10: 2, 8: 2},
       'gpt3_6_7B': {24: 2, 22: 2, 20: 2, 18: 1, 16: 1, 14: 1, 12: 1, 10: 1, 8: 1}}

clients = []
for host in hosts:
    for port in ports:
        clients.append(SSHClient(host=host, port=port, user=user, pkey=pkey))

local = clients[0]

def generate_available_machines(number, is_pretrain=True):
    if number == 16:
        if is_pretrain:
            output = local.run_command('cd ' + meg_project_dir + ' && cp available_machines_pretrain.out available_machines.out')
            local.wait_finished(output)
            for line in output.stdout:
                print(line)
            for line in output.stderr:
                print(line)
        else:
            output = local.run_command('cd ' + meg_project_dir + ' && cp available_machines_profile.out available_machines.out')
            local.wait_finished(output)
            for line in output.stdout:
                print(line)
            for line in output.stderr:
                print(line)
        return
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
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/kill_all.sh')
    local.wait_finished(output)
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)

def cp_log(number, model):
    output = local.run_command('cd ' + meg_project_dir + ' && rm -rf ssh_logs_' + str(number) + '_' + model + ' && cp -r ssh_logs ssh_logs_' + str(number) + '_' + model)
    local.wait_finished(output)

def rm_tmp():
    output = local.run_command(f'rm -rf /mnt/gpu-91/varuna/profile*')
    local.wait_finished(output)
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)

iteration_time_parser = re.compile(r'iteration(\s+)(?P<iterationnum>\d+)/(.+)\| elapsed time per iteration \(ms\): (?P<iterationtime>.+) \| learning rate')
error_parser = re.compile(r'\[Errno (\d+)\] Connection refused')
traceback_parser = re.compile(r'Traceback \(most recent call last\):')

def check_finish(number):
    timeout = time.time() + 60 * (5 + (2 * number) / 4)
    while True:
        with open('ssh_logs/ssh_out_0.log', 'r') as f:
            for line in f:
                iteration_time_match = iteration_time_parser.search(line)
                error_match = error_parser.search(line)
                if iteration_time_match:
                    iterationnum = int(iteration_time_match.group('iterationnum'))
                    if iterationnum == 5:
                        return 0
                if error_match:
                    return -1
        for i in range(number):
            with open(f'ssh_logs/ssh_err_{i}.log') as f:
                for line in f:
                    traceback_match = traceback_parser.search(line)
                    if traceback_match:
                        return -1
        if time.time() > timeout:
            return -2
        time.sleep(30)

def run_test(number, model_i):
    print(f'run {models[model_i]} test {number} nodes')
    kill_all()
    rm_tmp()
    print('kill all')
    time.sleep(5)
    # generate_available_machines(number, False)
    # print('finish generate_available_machines')
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
    output = local.run_command(f'cd {meg_project_dir} && bash ./scripts/pretrain_gpt2_varuna.sh {models[model_i]} {nstages[models[model_i]][number]} {mbs[models[model_i]][number]}')
    print('time to sleep')
    found = False
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(20)
        with open('ssh_logs/ssh_out_0.log', 'r') as f:
            for line in f:
                if 'Process done with return code 0' in line:
                    found = True
                    break
            if found:
                break
    # out = check_finish(number)
    # if out == -1:
    #     print('error')
    # elif out == -2:
    #     print('timeout')
    # else:
    #     print('success')
    print('time to kill')
    kill_all()
    print('finish pretrain')
    cp_log(number, models[model_i])

run_test(8, 0)
# run_test(10, 0)

# for model_i in range(0, 2):
#     for i in (8, 12):
#         run_test(i, model_i)

# for model_i in range(0, len(models)):
#     run_test(16, model_i)