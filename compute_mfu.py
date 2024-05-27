import os, sys, math, random, json, shutil

from time import perf_counter, sleep 

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--gpus-per-node', type=int, default=8)
    parser.add_argument('--gbs', type=int, default=4096)
    parser.add_argument('--enc-seq-len', type=int, default=4096)
    parser.add_argument('--hs', type=int, default=8192)
    parser.add_argument('--ffn_hs', type=int, default=13824)
    parser.add_argument('--layers', type=int, default=80)
    parser.add_argument('--vocab', type=int, default=32000)
    parser.add_argument('--time-per-step', type=float)
    parser.add_argument('-t', '--total-time', type=float)
    parser.add_argument('--mfu', type=float)
    parser.add_argument('--form-factor', default='bf16')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f'args={args}')

    nodes = args.nodes
    gpus_per_node = args.gpus_per_node
    gbs = args.gbs 
    enc_seq_len = args.enc_seq_len 
    hs = args.hs 
    ffn_hs = args.ffn_hs 
    layers = args.layers
    vocab = args.vocab 

    GPU_TFLOPS = {
        'bf16': 989,
        'fp16': 989,
        'fp8': 1979,
        'tf32': 495,
    }
    if args.time_per_step:
        time_per_step = args.time_per_step 
    elif args.steps and args.total_time:
        time_per_step = args.total_time / args.steps 
    elif args.mfu:
        time_per_step = None 
    else:
        raise Exception('[args.time_per_step] or [args.steps and args.total_time] or [args.mfu] must be set')
    model_flo = ((8 * gbs * enc_seq_len *hs * hs + 6 * gbs * enc_seq_len * hs * ffn_hs + 4 * gbs * enc_seq_len * enc_seq_len * hs) * (3 * layers) + (6 *gbs * enc_seq_len * hs * vocab))
    model_flo_per_gpu = model_flo / (nodes *gpus_per_node)
    model_tflo_per_gpu = model_flo_per_gpu / 1e12

    if time_per_step:
        model_flops = model_flo / time_per_step
        model_flops_per_gpu = model_flops / (nodes *gpus_per_node)
        model_tflops_per_gpu = model_flops_per_gpu / 1e12 
        mfu = model_tflops_per_gpu / GPU_TFLOPS[args.form_factor]
    else:
        mfu = args.mfu 
        time_per_step = model_tflo_per_gpu / GPU_TFLOPS[args.form_factor] / mfu
        model_flops = model_flo / time_per_step
        model_flops_per_gpu = model_flops / (nodes *gpus_per_node)
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
    

    print(f"                 MFU: {mfu *100:.1f}%")
    print(f"model_tflops_per_gpu: {model_tflops_per_gpu:.1f}")
    print(f"       time_per_step: {time_per_step:.2f}")
    print(f"           samples/s: {gbs/time_per_step:.1f}")
    print(f"            tokens/s: {gbs*enc_seq_len/time_per_step:.1f}")

main()