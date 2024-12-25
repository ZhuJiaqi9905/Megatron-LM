#!/usr/bin/python
import os
import re
from typing import Dict, List, Optional, Tuple
import itertools
import pandas as pd
import copy
import json
# Regular expression to extract required information from the filename
filename_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(GPT_\S+)_TP(\d+)_MBS(\d+)_SEQ(\d+).log")
MODEL_CONFIG = {
    'GPT_2-6B': {
    "model_name": "GPT_2-6B",
    "num_layers": 34,
    "parameters": {
      "total_parameters_bytes": sum([524288000] + [314705920] * 32 + [524288000]),
      "parameters_per_layer_bytes": [524288000] + [314705920] * 32 + [524288000]
    }
  },
    'GPT_7B': {
    "model_name": "GPT_7B",
    "num_layers": 34,
    "parameters": {
      "total_parameters_bytes": sum([838860800] + [805519360] * 32 + [838860800]),
      "parameters_per_layer_bytes": [838860800] + [805519360] * 32 + [838860800]
    }
  },
    'GPT_13B': {
    "model_name": "GPT_13B",
    "num_layers": 42,
    "parameters": {
      "total_parameters_bytes": sum([1048576000] + [1258557440] * 40 + [1048576000]),
      "parameters_per_layer_bytes": [1048576000] + [1258557440] * 40 + [1048576000]
    }
  },
}

def extract_floats(line: str) -> Optional[Tuple[float, float]]:
    # 匹配括号中的数字对，例如 (6523.78, 13430.91)
    match = re.search(r"\(([\d\.]+),\s*([\d\.]+)\)", line)
    if match:
        return tuple(map(float, match.groups()))
    return None

# 提取单个数字，例如 "elapsed time per iteration (ms): 37273.3"
def extract_elapsed_time(line: str) -> Optional[float]:
    match = re.search(r"elapsed time per iteration \(ms\):\s*([\d\.]+)", line)
    if match:
        return float(match.group(1))
    return None

# Function to parse the log file and extract the desired information
def parse_log_file(file_path: str):
    data = {
        'total_time_ms': [],
        'forward_backward_time_ms': [],
        'batch_generator_time_ms': [],
        'layernorm_grads_all_reduce_time_ms': [],
        'embedding_grads_all_reduce_time_ms': [],
        'optimizer_time_ms': [],
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Extract the relevant values from the log content
    for line in lines:
        if 'elapsed time per iteration (ms): ' in line:
            data['total_time_ms'].append(extract_elapsed_time(line))
        elif 'forward-backward' in line:
            data['forward_backward_time_ms'].append(extract_floats(line)[1])
        elif 'batch-generator' in line:
            data['batch_generator_time_ms'].append(extract_floats(line)[1])
        elif 'layernorm-grads-all-reduce' in line:
            data['layernorm_grads_all_reduce_time_ms'].append(extract_floats(line)[1])
        elif 'embedding-grads-all-reduce' in line:
            data['embedding_grads_all_reduce_time_ms'].append(extract_floats(line)[1])
        elif '    optimizer ....' in line:
            data['optimizer_time_ms'].append(extract_floats(line)[1])
    # print(f"{file_path}: {data}")
    return data

# Function to calculate the average of the last three values
def calculate_average(data ):
    averages = {}
    for key, value in data.items():
        if len(value) == 0:
            averages[key] = 0
            continue
        avg = sum(value[1:]) / len(value[1:])
        averages[key] = avg
    return averages 

# Process all log files in the directory
def process_logs(log_folder: str):
    results = []
    
    # List all files in the directory
    for file_name in os.listdir(log_folder):
        if file_name.endswith('.log'):
            match = filename_pattern.match(file_name)
            if match:
                # Extract information from filename
                timestamp, GPT_model, TP, MBS, SEQ = match.groups()
                file_path = os.path.join(log_folder, file_name)
                
                # Parse the log file and calculate averages
                log_data = parse_log_file(file_path)
                averages = calculate_average(log_data)
                results.append({
                    'timestamp': timestamp,
                    'model': GPT_model,
                    'tp': int(TP),
                    'mbs': int(MBS),
                    'seq_len': int(SEQ),
                    'data': averages
                })
    
    return results
def get_df_layer_compute(df, model_size, mbs, seq_len):
    '''
    in ms.
    '''
    embedding_forward = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['fwd_compute'].values[0]

    embedding_backward = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['bwd_compute'].values[0]
    embedding_compute = (embedding_forward + embedding_backward) / 1000
    transformer_data = df[df['op'].isin(['TELayerNormSelfAttentionDropout', 'TELayerNormMlpDropout'])]
    transformer_grouped = transformer_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length_q', 'sequence_length_kv'], as_index=False).agg({
        'fwd_compute': 'sum',
        'bwd_compute': 'sum'
    })
    transformer_fwd_compute = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &
        (transformer_grouped['sequence_length_kv'] == seq_len) 
        ]['fwd_compute'].iloc[0]

    transformer_bwd_compute = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &  
        (transformer_grouped['sequence_length_kv'] == seq_len)  
        ]['bwd_compute'].iloc[0]   
    
    transformer_compute = (transformer_fwd_compute + transformer_bwd_compute) / 1000
    post_process_forward = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['fwd_compute'].values[0]

    post_process_backward = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['bwd_compute'].values[0]
    post_process_compute = (post_process_forward + post_process_backward) / 1000
    return embedding_compute, transformer_compute, post_process_compute

def get_df_layer_weight_size(df, model_size, mbs, seq_len):
    '''
    in MB.
    '''
    embedding_weight = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['weight'].values[0] 
    transformer_data = df[df['op'].isin(['TELayerNormSelfAttentionDropout', 'TELayerNormMlpDropout'])]
    transformer_grouped = transformer_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length_q', 'sequence_length_kv'], as_index=False).agg({
        'weight': 'sum',
    })
    transformer_weight = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &
        (transformer_grouped['sequence_length_kv'] == seq_len) 
        ]['weight'].iloc[0]
    post_process_weight = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['weight'].values[0] 
    return embedding_weight, transformer_weight, post_process_weight

def get_df_layer_activation_size(df, model_size, mbs, seq_len):
    '''
    in MB.
    '''
    embedding_activation = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['fwd_allocated'].values[0] 
    transformer_data = df[df['op'].isin(['TELayerNormSelfAttentionDropout', 'TELayerNormMlpDropout'])]
    transformer_grouped = transformer_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length_q', 'sequence_length_kv'], as_index=False).agg({
        'fwd_allocated': 'sum',
    })
    transformer_activation = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &
        (transformer_grouped['sequence_length_kv'] == seq_len) 
        ]['fwd_allocated'].iloc[0]
    post_process_activation = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['fwd_allocated'].values[0] 
    return embedding_activation, transformer_activation, post_process_activation

def get_df_layer_reserved_size(df, model_size, mbs, seq_len):
    '''
    in MB.
    '''
    embedding_fwd_reserved= df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['fwd_reserved'].values[0] 
    embedding_bwd_reserved= df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'LanguageModelEmbedding')
                           ]['bwd_reserved'].values[0] 
    embedding_reserved = embedding_fwd_reserved + embedding_bwd_reserved
    transformer_data = df[df['op'].isin(['TELayerNormSelfAttentionDropout', 'TELayerNormMlpDropout'])]
    transformer_grouped = transformer_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length_q', 'sequence_length_kv'], as_index=False).agg({
        'fwd_reserved': 'max',
        'bwd_reserved': 'max'
    })
    transformer_fwd_reserved = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &
        (transformer_grouped['sequence_length_kv'] == seq_len) 
        ]['fwd_reserved'].iloc[0]
    transformer_bwd_reserved = transformer_grouped[
        (transformer_grouped['model_name'] == 'gpt') &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length_q'] == seq_len) &
        (transformer_grouped['sequence_length_kv'] == seq_len) 
        ]['bwd_reserved'].iloc[0]
    transformer_reserved = transformer_fwd_reserved + transformer_bwd_reserved
    post_fwd_reserved = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['fwd_reserved'].values[0] 
    post_bwd_reserved = df[(df['model_name'] == 'gpt') &
                           (df['model_size'] == model_size) &
                           (df['micro_batch_size'] == mbs) &
                           (df['sequence_length_q'] == seq_len) & 
                           (df['sequence_length_kv'] == seq_len) &
                           (df['op'] == 'TELayerNormPostProcess')
                           ]['bwd_reserved'].values[0] 
    post_process_reserved = post_fwd_reserved + post_bwd_reserved
    return embedding_reserved, transformer_reserved, post_process_reserved  

    
def get_df_layer_memory(df, model_size, mbs, seq_len):
    embedding_weight, transformer_weight, post_process_weight = get_df_layer_weight_size(df, model_size, mbs, seq_len)
    embedding_activation, transformer_activation, post_process_activation = get_df_layer_activation_size(df, model_size, mbs, seq_len) 
    embedding_reserved, transformer_reserved, post_process_reserved  = get_df_layer_reserved_size(df, model_size, mbs, seq_len)
    
    embedding_reserved = max(embedding_weight, embedding_reserved)
    transformer_reserved = max(transformer_weight, transformer_reserved)
    post_process_reserved = max(post_process_weight, post_process_reserved)
   
    # weight + main_param + optimizer + gradient + activation + reserved
    embedding_mem = embedding_weight * (1 + 2 + 4 + 1) + embedding_activation + embedding_reserved
    transformer_mem =  transformer_weight * (1 + 2 + 4 + 1) + transformer_activation + transformer_reserved 
    post_process_mem = post_process_weight * (1 + 2 + 4 + 1) + post_process_activation + post_process_reserved 
    return embedding_mem, transformer_mem, post_process_mem

def gen_metis_profile_files(mega_res:List[Dict], csv_folder: str, device: str):
    global MODEL_CONFIG
    
    models = ['GPT_2-6B']
    tp_values = [1, 2, 4, 8]
    seq_lens = [4096, 8192]
    micro_batch_sizes = [1, 2, 4]
    for model, tp, seq_len, mbs in itertools.product(models, tp_values, seq_lens, micro_batch_sizes):
        print(f'{model} TP_{tp} MBS_{mbs} SEQLEN_{seq_len}')
        result = {
            'model': MODEL_CONFIG[model],
            'execution_time': None,
            'execution_memory': {} 
        }
        result['execution_time'] = copy.deepcopy(next((res for res in mega_res if (res['model'] == model and res['tp'] == tp and res['mbs'] == mbs and res['seq_len'] == seq_len)), None)['data'])
        df = pd.read_csv(f'{csv_folder}/{model}_tp_{tp}.csv') 
        embedding_compute, transformer_compute, post_process_compute = get_df_layer_compute(df, model.split('_')[1], mbs, seq_len)
        result['execution_time']['layer_compute_total_ms'] = [embedding_compute] + [transformer_compute] * (result['model']['num_layers'] - 2) + [post_process_compute]

        embedding_mem, transformer_mem, post_process_mem = get_df_layer_memory(df, model.split('_')[1], mbs, seq_len)
        result['execution_memory']['layer_memory_total_mb'] = [embedding_mem] + [transformer_mem]* (result['model']['num_layers'] - 2) + [post_process_mem] 
        result['execution_memory']['total_memory'] = sum(result['execution_memory']['layer_memory_total_mb'])

        metis_file_folder = f"../metis/{model}/{seq_len}/"
        os.makedirs(metis_file_folder, exist_ok=True)
        with open(f"{metis_file_folder}/DeviceType.{device.upper()}_tp{tp}_bs{mbs}.json", 'w') as f:
            json.dump(result, f)
            
        print(result)
        # exit()

# Print the results
if __name__ == '__main__':
    
    for device in ['a10g', 'l40s']:
        log_folder = f'./important/{device}/'
        log_results = process_logs(log_folder)
        csv_folder = f'../profiler/aws/{device}/'
        gen_metis_profile_files(log_results, csv_folder, device)
        
    # # Display the results
    # for result in log_results:
    #     print(f"File: {result['timestamp']}_{result['GPT_model']}_TP{result['TP']}_MBS{result['MBS']}_SEQ{result['SEQ']}.log")
    #     print(f"GPT model: {result['GPT_model']}, TP: {result['TP']}, MBS: {result['MBS']}, SEQ: {result['SEQ']}")
    #     for key, avg_data in result['data'].items():
    #         print(f"{key} average: {avg_data:.2f}")
    #     print()
