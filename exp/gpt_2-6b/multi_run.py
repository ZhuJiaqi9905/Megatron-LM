import asyncio
import os
from typing import List
import aiofiles
import asyncssh
import subprocess
import itertools


async def run_command_on_node(node, command, label, node_rank: int):
    output_file = f"/workspace/python/Megatron-LM/logs/aws/{label}/"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = f"{output_file}/{node_rank}.log"

    # command = "date && sleep 10s && date"
    # Execute the command remotely
    try:
        async with asyncssh.connect(node, port=2222, known_hosts=None) as conn:
            async with aiofiles.open(
                output_file, "w"
            ) as log_file, conn.create_process(
                command,
                term_type="xterm",
            ) as process:
                print(f"Agent {node} output will be written at {output_file}")
                async for data in process.stdout:
                    await log_file.write(data)
                    await log_file.flush()
            print(f"run {command} on node {node}") 
    except (OSError, asyncssh.Error) as exc:
        print(f"Error connecting to {node}: {exc}")


async def run_model_tasks(nodes, layer_file, label):
    # Command template
    print(f"{layer_file} broadcast test begin")
    master_addr = "172.21.0.42"
    master_port = 10078
    command_template = '/bin/bash -ic "conda run --no-capture-output -n oobleck python /workspace/Oobleck/simulate/broadcast_test.py --master-ip {}  --master-port {} --node-rank {} --layer-file {} --gpus-per-node 4 --num-nodes 4"'
    # Create tasks for running commands on nodes
    tasks = []
    for node_rank, node in enumerate(nodes):
        command = command_template.format(
            master_addr, master_port, node_rank, layer_file
        )
        print(f"run command {command} on node {node}")
        task = asyncio.create_task(run_command_on_node(node, command, label))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"All tasks for {layer_file} completed.")

async def run_task(tp: int, mbs: int, seq_len: int, nodes: List[str], gpus_per_node: int):
    total_gpus = len(nodes) * gpus_per_node 

    pp = total_gpus // tp
    
    world_size = tp * pp 
    assert(world_size == total_gpus)
    master_addr = nodes[0]
    tasks = []
    for node_rank, node in enumerate(nodes):
        command = f'/bin/bash /workspace/python/Megatron-LM/run_multi.sh {tp} {mbs} {seq_len} {pp} {master_addr} {len(nodes)} {node_rank}'
        task = asyncio.create_task(run_command_on_node(node, command, f"GPT_2-6B_TP{tp}_MBS{mbs}_SEQ{seq_len}", node_rank))
        tasks.append(task)
    await asyncio.gather(*tasks)
    print(f"All tasks for TP{tp} MBS{mbs} SEQ_LEN{seq_len} completed.")
    
async def main():
    nodes = ["172.21.0.42", "172.21.0.46"]
    
    tp_values = [1, 2, 4, 8]
    micro_batch_sizes = [1, 2, 4]
    sequence_lengths = [32 * 1024, 64 * 1024]
    gpus_per_node = 8


    for tp, mbs, seq_len in itertools.product(tp_values, micro_batch_sizes, sequence_lengths):
        await run_task(tp, mbs, seq_len, nodes, gpus_per_node)
        await asyncio.sleep(5)
    

if __name__ == "__main__":
    asyncio.run(main())