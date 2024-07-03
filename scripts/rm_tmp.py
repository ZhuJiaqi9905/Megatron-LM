from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys

hosts = []
ports = []
with open('available_machines.out', 'r') as f:
    for line in f.readlines():
        hosts.append(line.split(':')[0])
        ports.append(int(line.split(':')[1]))

meg_project_dir = '/workspace/Megatron-LM-varuna'
varu_project_dir = '/workspace/varuna'
user = 'root'
pkey = '/root/.ssh/id_rsa'

clients = []
for idx in range(len(hosts)):
    print(hosts[idx], ports[idx])
    clients.append(SSHClient(host=hosts[idx], port=ports[idx], user=user, pkey=pkey))

for idx in range(len(hosts)):
    print("==============================")
    print(hosts[idx], ports[idx], f'rm -rf /mnt/gpu-91/varuna/profile_rank_{idx}')
    print("==============================")
    output = clients[idx].run_command(f'rm -rf /mnt/gpu-91/varuna/profile_rank_{idx}')
    clients[idx].wait_finished(output)