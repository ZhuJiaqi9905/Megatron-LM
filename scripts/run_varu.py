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
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./script/kill_all.sh')
    local.wait_finished(output)

def cp_log(number):
    output = local.run_command('cd ' + meg_project_dir + ' && cp -r ssh_logs ssh_logs_' + str(number))
    local.wait_finished(output)

def run_test(number):
    print(f'run test {number} nodes')
    kill_all()
    generate_available_machines(number)
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/profile_gpt2.sh')
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)
    local.wait_finished(output)
    for line in output.stdout:
        print(line)
    for line in output.stderr:
        print(line)
    print('finish profile')
    kill_all()
    output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/pretrain_gpt2_varuna.sh')
    time.sleep(60 * 5)
    print('time to kill')
    kill_output = local.run_command('cd ' + meg_project_dir + ' && bash ./scripts/kill_all.sh')
    local.wait_finished(kill_output)
    for line in kill_output.stdout:
        print(line)
    for line in kill_output.stderr:
        print(line)
    local.wait_finished(output)
    print('finish pretrain')
    cp_log(number)

run_test(16)

# for i in range(8, 24, 2):
#     run_test(i)