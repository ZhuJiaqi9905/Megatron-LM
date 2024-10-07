#!/bin/bash
ports=(2230 2231 2232 2233)
addrs=(172.21.0.92 172.21.0.90 172.21.0.91 172.21.0.42 172.21.0.46 172.21.0.47)
for port in "${ports[@]}"; do
    for addr in "${addrs[@]}"; do
        ssh -p ${port} ${addr} "cd /workspace/Megatron-LM-varuna && ./scripts/kill.sh" 
    done
    
done

# ssh -p 2230 172.21.0.91  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.92  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.90  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.46  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"      
