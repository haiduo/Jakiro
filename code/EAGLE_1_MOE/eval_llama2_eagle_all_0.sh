#!/bin/bash

ea_model_path=/home/haiduo/code/EAGLE_MOE/eagle/outputs/checkpoints/llama2_13b/simple_moe_e2k2_gpt-mlp/orig
base_model_path=/home/haiduo/model/llama/Llama-2-13b-chat-hf
model_id=eagle1-llama2-13b-fp16
model_id_baseline=eagle1-llama2-13b-fp16-baseline

answer_file=/home/haiduo/code/EAGLE-1/eagle/outputs/paper/depth-6/llama2_13b/6_1/orig/check1
answer_file_base=/home/haiduo/code/EAGLE-1/eagle/outputs/paper/depth-5/llama2_13b
temperature=1.0
GPU_DEVICES=0

# 定义 bench_name 列表
bench_names=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")

### 测试 EAGLE 的 speed ###
for bench_name in "${bench_names[@]}"; do
    echo "Processing $bench_name for EAGLE..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eagle.evaluation.gen_ea_answer_llama2chat \
    --ea_model_path "$ea_model_path" \
    --base_model_path "$base_model_path" \
    --model_id "$model_id" \
    --bench_name "$bench_name" \
    --temperature "$temperature" \
    --answer_file "$answer_file"
done

# ### 测试 baseline 的 speed ###
# for bench_name in "${bench_names[@]}"; do
#     echo "Processing $bench_name for baseline..."
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eagle.evaluation.gen_baseline_answer_llama2chat \
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

    echo "Processing $bench_name for speedup ratio and Mean accepted tokens calculation..."

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
