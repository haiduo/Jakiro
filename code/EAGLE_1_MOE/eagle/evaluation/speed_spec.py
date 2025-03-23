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
    accept_lengths_list.extend(datapoint["choices"][0]["accept_lengths"])
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
