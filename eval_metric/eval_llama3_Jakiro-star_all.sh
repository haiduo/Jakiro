#!/bin/bash

ea_model_path=/home/haiduo/code/Jakiro/eagle/outputs/checkpoints/llama3-8b/moe_eagle/Jakiro_llama_mlp_cosin_epochs=20_lr=9e-5_wd=1e-3
base_model_path=/home/haiduo/model/llama/Meta-Llama-3-8B-Instruct
model_id=llama38b2_40
model_id_baseline=llama38b2_40-fp16-baseline

answer_file=/home/haiduo/code/Jakiro/eagle/outputs/our/depth-5/Jakiro_llama_mlp_cosin_epochs=20_lr=9e-5_wd=1e-3
answer_file_base=/home/haiduo/code/Jakiro/eagle/outputs/paper/depth-5/llama3_8b
temperature=0.0
total_token=60 #[60,50,48]
GPU_DEVICES=0

# Define the BenchMark_name list
bench_names=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")

### Test Jakiro's speed ###
for bench_name in "${bench_names[@]}"; do
    echo "Processing $bench_name for Jakiro..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m jakiro_star.evaluation.gen_ea_answer_llama3chat \
    --ea_model_path "$ea_model_path" \
    --base_model_path "$base_model_path" \
    --model_id "$model_id" \
    --bench_name "$bench_name" \
    --temperature "$temperature" \
    --answer_file "$answer_file" \
    --total_token "$total_token"
done

### Test Baseline's speed ###
for bench_name in "${bench_names[@]}"; do
    echo "Processing $bench_name for baseline..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m jakiro_star.evaluation.gen_baseline_answer_llama3chat \
    --ea_model_path "$ea_model_path" \
    --base_model_path "$base_model_path" \
    --model_id "$model_id_baseline" \
    --bench_name "$bench_name" \
    --temperature "$temperature" \
    --answer_file "$answer_file_base"
done

### Calculate the Speedup ratio and Mean accepted tokens for each BenchMark_name ###
total_ratio=0
total_accepted_tokens=0
count=0

for bench_name in "${bench_names[@]}"; do
    eagle_name="${model_id}-temperature-${temperature}"
    jsonl_file_eagle="${answer_file}/${bench_name}/${eagle_name}.jsonl"

    baseline_name="${model_id_baseline}-temperature-${temperature}"
    jsonl_file_base="${answer_file_base}/${bench_name}/${baseline_name}.jsonl"

    echo "Processing $bench_name ..."

    # Execute the Python script and capture the output, while displaying the original output
    output=$(python -m jakiro_star.evaluation.speed_accept_len "$bench_name" "$base_model_path" "$jsonl_file_eagle" "$jsonl_file_base")

    # Display the raw output of the Python script
    echo "$output"

    # Extract the values of ratio and Mean accepted tokens
    ratio=$(echo "$output" | grep -oP '(?<=#ratio: )\d+\.\d+')
    mean_accepted_tokens=$(echo "$output" | grep -oP '(?<=#Mean accepted tokens: )\d+\.\d+')

    # If extraction fails, set the default value to 0
    ratio=${ratio:-0}
    mean_accepted_tokens=${mean_accepted_tokens:-0}

    # Accumulate results
    total_ratio=$(echo "$total_ratio + $ratio" | bc)
    total_accepted_tokens=$(echo "$total_accepted_tokens + $mean_accepted_tokens" | bc)
    count=$((count + 1))
done

# Calculate the average
average_ratio=$(echo "scale=4; $total_ratio / $count" | bc)
average_accepted_tokens=$(echo "scale=4; $total_accepted_tokens / $count" | bc)

# Print the result
echo "########## Averages Across All Benchmarks ##########"
echo "Average #ratio: $average_ratio"
echo "Average #Mean accepted tokens: $average_accepted_tokens"
