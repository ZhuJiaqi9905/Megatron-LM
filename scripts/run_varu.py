from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys, os, time

hosts = [91, 90, 92, 46, 42, 47]
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
    outputs = []
    for client in clients:
        outputs.append(client.run_command('rm -f /tmp/*'))
    for idx, client in enumerate(clients):
        client.wait_finished(outputs[idx])

def run_test(number, model_i):
    print(f'run {models[model_i]} test {number} nodes')
    kill_all()
    print('kill all')
    time.sleep(5)
    rm_tmp()
    print('rm tmp')
    generate_available_machines(number)
    print('finish generate_available_machines')
    # output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/profile_gpt2.sh')
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
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/pretrain_gpt2_varuna.sh ' + models[model_i] + ' ' + str(nstages[number]))
    print('time to sleep')
    time.sleep(60 * 5)
    print('time to kill')
    kill_all()
    print('finish pretrain')
    cp_log(number, models[model_i])

# run_test(24, 0)

for model_i in range(len(models)):
    for i in range(8, 24, 2):
        run_test(i, model_i)