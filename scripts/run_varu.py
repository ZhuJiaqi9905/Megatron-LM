from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys, os, time
import re

hosts = [91, 90, 92, 42]
for idx, host_suffix in enumerate(hosts):
    hosts[idx] = '172.21.0.' + str(host_suffix)
ports = [2230 + i for i in range(4)]
meg_project_dir = '/workspace/Megatron-LM-varuna'
varu_project_dir = '/workspace/varuna'
user = 'root'
pkey = '/root/.ssh/id_rsa'

models = ['gpt3_350M', 'gpt3_1_3B', 'gpt3_2_7B']
nstages = {24: 8, 22: 11, 20: 5, 18: 6, 16: 4, 14: 7, 12: 6, 10: 5, 8: 4} # nodes: nstages

clients = []
for host in hosts:
    for port in ports:
        clients.append(SSHClient(host=host, port=port, user=user, pkey=pkey))

local = clients[0]

def generate_available_machines(number):
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
    generate_available_machines(number)
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
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/pretrain_gpt2_varuna.sh ' + models[model_i])
    print('time to sleep')
    time.sleep(60)
    out = check_finish(number)
    if out == -1:
        print('error')
    elif out == -2:
        print('timeout')
    else:
        print('success')
    print('time to kill')
    kill_all()
    print('finish pretrain')
    cp_log(number, models[model_i])

run_test(12, 1)

# for model_i in range(0, 2):
#     for i in (8, 12):
#         run_test(i, model_i)

# for model_i in range(0, len(models)):
#     run_test(16, model_i)