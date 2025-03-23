#!/bin/bash

ea_model_path=/home/haiduo/code/EAGLE_MOE/eagle/outputs/checkpoints/llama3-8b/moe_eagle/medusa_llama_mlp_cosin_epochs=20_lr=9e-5_wd=1e-3
base_model_path=/home/haiduo/model/llama/Meta-Llama-3-8B-Instruct
model_id=llama38b2_40
model_id_baseline=llama38b2_40-fp16-baseline

answer_file=/home/haiduo/code/EAGLE_medusa/eagle/outputs/our/depth-5/medusa_llama_mlp_cosin_epochs=20_lr=9e-5_wd=1e-3/check2
answer_file_base=/home/haiduo/code/EAGLE/eagle/outputs/paper/depth-5/llama3_8b
temperature=1.0
total_token=60 #[60,50,48]
GPU_DEVICES=1

# 定义 bench_name 列表
bench_names=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")

### 测试 EAGLE 的 speed ###
for bench_name in "${bench_names[@]}"; do
    echo "Processing $bench_name for EAGLE..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea_model_path "$ea_model_path" \
    --base_model_path "$base_model_path" \
    --model_id "$model_id" \
    --bench_name "$bench_name" \
    --temperature "$temperature" \
    --answer_file "$answer_file" \
    --total_token "$total_token"
done

# ### 测试 baseline 的 speed ###
# for bench_name in "${bench_names[@]}"; do
#     echo "Processing $bench_name for baseline..."
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eagle.evaluation.gen_baseline_answer_llama3chat \
#     --ea_model_path "$ea_model_path" \
#     --base_model_path "$base_model_path" \
#     --model_id "$model_id_baseline" \
#     --bench_name "$bench_name" \
#     --temperature "$temperature" \
#     --answer_file "$answer_file_base"
# done

### 计算每个 bench_name 的 ratio 和 Mean accepted tokens ###
total_ratio=0
total_accepted_tokens=0
count=0

for bench_name in "${bench_names[@]}"; do
    eagle_name="${model_id}-temperature-${temperature}"
    jsonl_file_eagle="${answer_file}/${bench_name}/${eagle_name}.jsonl"

    baseline_name="${model_id_baseline}-temperature-${temperature}"
    jsonl_file_base="${answer_file_base}/${bench_name}/${baseline_name}.jsonl"

    echo "Processing $bench_name ..."

    # 执行 Python 脚本并捕获输出，同时显示原始输出
    output=$(python -m eagle.evaluation.speed_accept_len "$bench_name" "$base_model_path" "$jsonl_file_eagle" "$jsonl_file_base")

    # 显示 Python 脚本的原始输出
    echo "$output"

    # 提取 ratio 和 Mean accepted tokens 的值
    ratio=$(echo "$output" | grep -oP '(?<=#ratio: )\d+\.\d+')
    mean_accepted_tokens=$(echo "$output" | grep -oP '(?<=#Mean accepted tokens: )\d+\.\d+')

    # 如果提取失败，设置默认值为0
    ratio=${ratio:-0}
    mean_accepted_tokens=${mean_accepted_tokens:-0}

    # 累加
    total_ratio=$(echo "$total_ratio + $ratio" | bc)
    total_accepted_tokens=$(echo "$total_accepted_tokens + $mean_accepted_tokens" | bc)
    count=$((count + 1))
done

# 计算平均值
average_ratio=$(echo "scale=4; $total_ratio / $count" | bc)
average_accepted_tokens=$(echo "scale=4; $total_accepted_tokens / $count" | bc)

# 打印结果
echo "########## Averages Across All Benchmarks ##########"
echo "Average #ratio: $average_ratio"
echo "Average #Mean accepted tokens: $average_accepted_tokens"
