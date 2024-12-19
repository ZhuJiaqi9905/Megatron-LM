import itertools
import os 
import subprocess 

tp_values = [2**i for i in range(4)]
micro_batch_sizes = [2**i for i in range(5)]
sequence_lengths = [1024 * (2**i) for i in range(2, 4)]



for tp, mbs, seq_len in itertools.product(tp_values, micro_batch_sizes, sequence_lengths):
    cmd = ["bash", f"{os.path.dirname(os.path.abspath(__file__))}/run.sh", str(tp), str(mbs), str(seq_len)]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command {' '.join(cmd)} failed with error: {e}")
        
print("All combinations processed.")