#!/bin/bash
ssh -p 2230 172.21.0.91  "cd /workspace/Megatron-LM-varuna && \
      ./scripts/kill.sh"
ssh -p 2230 172.21.0.92  "cd /workspace/python/Megatron-LM-varuna && \
      ./scripts/kill.sh"
ssh -p 2230 172.21.0.90  "cd /workspace/python/Megatron-LM-varuna && \
      ./scripts/kill.sh"
ssh -p 2230 172.21.0.46  "cd /workspace/python/Megatron-LM-varuna && \
      ./scripts/kill.sh"      
