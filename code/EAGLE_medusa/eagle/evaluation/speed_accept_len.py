# speed_spec.py
import json
import sys
import numpy as np
from transformers import AutoTokenizer

# 获取bench_name参数
bench_name = sys.argv[1]
model_path = sys.argv[2]
jsonl_file_eagle = sys.argv[3]
jsonl_file_base = sys.argv[4]

# ## debug 
# ##　vicuna
# bench_name = "mt_bench"
# model_path = "/home/haiduo/model/vicuna/vicuna-7b-v1.3"
# jsonl_file_eagle = "/home/haiduo/code/EAGLE_medusa/eagle/outputs/our/vicuna_7b/moe_eagle/llama-mlp/medusa_right_loss=0.05_0.5_epochs=35-cosin-lr=9e-5_wd=1e-3/depth-5_60/check4/gsm8k/eagle2-vicuna-7b-fp16-temperature-0.0.jsonl"
# jsonl_file_base = "/home/haiduo/code/EAGLE/eagle/outputs/paper/depth-5/vicuna_7b/gsm8k/eagle2-vicuna-7b-fp16-baseline-temperature-0.0.jsonl"

# # ####　llama2
# # bench_name = "mt_bench"
# model_path = "/home/haiduo/model/llama/Llama-2-13b-chat-hf"
# jsonl_file_eagle = "/home/haiduo/code/EAGLE_medusa/eagle/outputs/our/llama2_13b/medusa_llama-mlp_cosin_epochs=20_lr=9e-5_wd=1e-3/depth_5_50/check4/mt_bench/ess-llama-2-chat-13b-fp16-temperature-1.0.jsonl"
# jsonl_file_base = "/home/haiduo/code/EAGLE/eagle/outputs/paper/depth-5/llama2_13b/mt_bench/eagle2-llama2-13b-fp16-baseline-temperature-1.0.jsonl"

# # ####　llama3
# # bench_name = "mt_bench"
# model_path = "/home/haiduo/model/llama/Meta-Llama-3-8B-Instruct"
# jsonl_file_eagle = "/home/haiduo/code/EAGLE_medusa/eagle/outputs/our/Llama-3-8B-Instruct/moe_eagle/medusa_llama_mlp_cosin_epochs=20_lr=9e-5_wd=1e-3/tree_5_60/check4/mt_bench/llama38b2_40-temperature-1.0.jsonl"
# jsonl_file_base = "/home/haiduo/code/EAGLE/eagle/outputs/paper/depth-5/llama3_8b/mt_bench/llama38b2_40-fp16-baseline-temperature-0.0.jsonl"


tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# 读取对比的json文件
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
    ### 方案1 我们的实现
    # accept_lengths_list.extend(datapoint["choices"][0]["accept_length"])

    ### eagle的实现
    if sum(datapoint["choices"][0]["idxs"]) == 0:
        accept_lengths_list.extend([0])
    else:
        accept_lengths_list.extend([sum(datapoint["choices"][0]["new_tokens"])/sum(datapoint["choices"][0]["idxs"])]) 

    # ### jinze的实现
    # if sum(datapoint["choices"][0]["idxs"]) == 0:
    #     accept_lengths_list.extend([0])
    # else:
    #     accept = []
    #     for new_tokens, idex in zip(datapoint["choices"][0]["new_tokens"],datapoint["choices"][0]["idxs"]):
    #         accept.extend([new_tokens/idex])
    #     accept_lengths_list.extend([sum(accept)/len(accept)]) 
    
    speeds.append(tokens / times)

# 读取基准数据
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
