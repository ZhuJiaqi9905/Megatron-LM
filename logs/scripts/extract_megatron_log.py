#!/usr/bin/python
import os
import re
from typing import Dict

# Folder where the log files are located
log_folder = './important/a10g/'

# Regular expression to extract required information from the filename
filename_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(GPT_\S+)_TP(\d+)_MBS(\d+)_SEQ(\d+).log")

def extract_floats(line):
    # 匹配括号中的数字对，例如 (6523.78, 13430.91)
    match = re.search(r"\(([\d\.]+),\s*([\d\.]+)\)", line)
    if match:
        return tuple(map(float, match.groups()))
    return None

# 提取单个数字，例如 "elapsed time per iteration (ms): 37273.3"
def extract_elapsed_time(line):
    match = re.search(r"elapsed time per iteration \(ms\):\s*([\d\.]+)", line)
    if match:
        return float(match.group(1))
    return None

# Function to parse the log file and extract the desired information
def parse_log_file(file_path):
    data = {
        'GPT_model': None,
        'TP': None,
        'MBS': None,
        'SEQ': None,
        'total_time_ms': [],
        'forward_backward_time_ms': [],
        'batch-generator': [],
        'forward-recv': None,
        'forward-send': None,
        'backward-recv': None,
        'backward-send': None,
        'layernorm-grads-all-reduce': None,
        'embedding-grads-all-reduce': None,
        'all-grads-sync': None,
        'optimizer': None,
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()
    print(f"{file_path}") 
    # Extract the relevant values from the log content
    for line in lines:
        if 'elapsed time per iteration (ms): ' in line:
            data['total_time_ms'].append(extract_elapsed_time(line))
        elif 'forward-backward' in line:
            data['forward_backward_time_ms'].append(extract_floats(line)[1])
        elif 'batch-generator' in line:
            data['batch_generator_time_ms'].append(extract_floats(line)[1])
        elif 'forward-recv' in line:
            data['forward-recv'] = extract_floats(line)
        elif 'forward-send' in line:
            data['forward-send'] = extract_floats(line)
        elif 'backward-recv' in line:
            data['backward-recv'] = extract_floats(line)
        elif 'backward-send' in line:
            data['backward-send'] = extract_floats(line)
        elif 'layernorm-grads-all-reduce' in line:
            data['layernorm-grads-all-reduce'] = extract_floats(line)
        elif 'embedding-grads-all-reduce' in line:
            data['embedding-grads-all-reduce'] = extract_floats(line)
        elif 'all-grads-sync' in line:
            data['all-grads-sync'] = extract_floats(line)
        elif '    optimizer ....' in line:
            data['optimizer'] = extract_floats(line)
    print(f"{file_path}: {data}")
    exit()
    return data

# Function to calculate the average of the last three values
def calculate_average(data ):
    averages = {}
    for key, value in data.items():
        avg = sum(value[1:]) / len(value[1:])
        averages[key] = avg
    return averages 

# Process all log files in the directory
def process_logs():
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
                    'GPT_model': GPT_model,
                    'TP': TP,
                    'MBS': MBS,
                    'SEQ': SEQ,
                    'averages': averages
                })
    
    return results

# Print the results
if __name__ == '__main__':
    log_results = process_logs()
    
    # Display the results
    for result in log_results:
        print(f"File: {result['timestamp']}_{result['GPT_model']}_TP{result['TP']}_MBS{result['MBS']}_SEQ{result['SEQ']}.log")
        print(f"GPT model: {result['GPT_model']}, TP: {result['TP']}, MBS: {result['MBS']}, SEQ: {result['SEQ']}")
        for key, avg in result['averages'].items():
            print(f"{key} average: {avg:.2f}")
        print()
