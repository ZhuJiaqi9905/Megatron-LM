#!/bin/bash
addrs=(172.31.44.99)
for addr in "${addrs[@]}"; do
    ssh ubuntu@${addr} "cd /workspace/Megatron-LM-varuna && ./scripts/kill.sh" 
done

# ssh -p 2230 172.21.0.91  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.92  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.90  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"
# ssh -p 2230 172.21.0.46  "cd /workspace/Megatron-LM-varuna && \
#       ./scripts/kill.sh"      
