import json
import numpy as np

json_files=["/home/haiduo/code/EAGLE-1/eagle/outputs/depth-6/simple_moe_e2k2_gpt_align/pw-0.4_lr-3e-5_2/mt_bench/eagle1-vicuna-7b-fp16-tmp-temperature-1.0-alpha.jsonl",]


for jsonl_file in json_files:
    data=[]
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    alphas=[0 for _ in range(6)] # 改5为6
    alphas_num=[0 for _ in range(6)]
    accept_lengths = 0
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        ids = sum(datapoint["choices"][0]['idxs'])
        alpha=datapoint["choices"][0]['alpha']
        alpha_num = datapoint["choices"][0]['alpha_num']
        accept_length = datapoint["choices"][0]['accept_length']
        accept_lengths += sum(accept_length) / len(accept_length)
        for i in range(len(alpha)):
            alphas[i]+=alpha[i]
            alphas_num[i] += alpha_num[i]

    accept_lengths = accept_lengths/len(data)
    ar=np.array(alphas)/np.array(alphas_num)
    print("accept rate:", np.round(ar, 2))
    print("accept length:", np.round(accept_lengths, 2))