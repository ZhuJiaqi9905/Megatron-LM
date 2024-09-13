from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys

hosts = [91, 92, 42, 46]
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
        print(host, port)
        clients.append(SSHClient(host=host, port=port, user=user, pkey=pkey))


for port in range(len(ports)):
    outputs = []
    local_clients = []
    print("==============================")
    print(port)
    print("==============================")
    for host in range(len(hosts)):
        outputs.append(clients[host * len(ports) + port].run_command('cd ' + varu_project_dir + ' && /opt/conda/bin/pip uninstall --yes varuna && /opt/conda/bin/pip install .'))
        local_clients.append(clients[host * len(ports) + port])
    for idx, output in enumerate(outputs):
        print(idx)
        local_clients[idx].wait_finished(output)
        for line in output.stdout:
            print(line)
        for line in output.stderr:
            print(line)