import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("/home/haiduo/model/vicuna/vicuna-7b-v1.3")
jsonl_file = "/home/haiduo/code/EAGLE/eagle/outputs/results/vicuna_7b/orig_eagle/60/mt_bench/ess-vicuna-7b-fp16-temperature-0.0.jsonl"
jsonl_file_base = "/home/haiduo/code/EAGLE/eagle/outputs/results/vicuna_7b_paper/1_gpu/mt_bench/eagle2-vicuna-7b-fp16-baseline-temperature-0.0.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)

data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
# print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())


