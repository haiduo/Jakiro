import json
import sys
import numpy as np
from transformers import AutoTokenizer

# Get the benchmark_name parameter
bench_name = sys.argv[1]
model_path = sys.argv[2]
jsonl_file_eagle = sys.argv[3]
jsonl_file_base = sys.argv[4]

# ## debug vicuna
# bench_name = "mt_bench"
# model_path = "/home/haiduo/model/vicuna/vicuna-7b-v1.3"
# jsonl_file_eagle = "/home/haiduo/code/Jakiro/eagle/outputs/A40/our_Jakiro/vicuna_7b/mt_bench/eagle2-vicuna-7b-fp16-temperature-0.0.jsonl"
# jsonl_file_base = "/home/haiduo/code/Jakiro/eagle/outputs/A40/paper_baseline/vicuna_7b/mt_bench/eagle2-vicuna-7b-fp16-baseline-temperature-0.0.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Read the comparison json file
data = []
with open(jsonl_file_eagle, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds = []
accept_lengths_list = []
for datapoint in data:
    tokens = sum(datapoint["choices"][0]["new_tokens"])
    times = sum(datapoint["choices"][0]["wall_time"])

    ### compute core
    if sum(datapoint["choices"][0]["idxs"]) == 0:
        accept_lengths_list.extend([0])
    else:
        accept_lengths_list.extend([sum(datapoint["choices"][0]["new_tokens"])/sum(datapoint["choices"][0]["idxs"])]) 
    
    speeds.append(tokens / times)

# Read benchmark data
data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

total_time = 0
total_token = 0
speeds0 = []
for datapoint in data:
    answer = datapoint["choices"][0]["turns"]
    tokens = sum((len(tokenizer(i).input_ids) - 1) for i in answer)
    times = sum(datapoint["choices"][0]["wall_time"])
    speeds0.append(tokens / times)
    total_time += times
    total_token += tokens

# print("#speed:", np.array(speeds).mean())
# print("#speed0:", np.array(speeds0).mean())
print("#ratio:", np.array(speeds).mean() / np.array(speeds0).mean())
print("#Mean accepted tokens:", np.mean(accept_lengths_list))
