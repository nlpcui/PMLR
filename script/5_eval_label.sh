#!/bin/bash

mkdir -p logs

datasets=("bill" "wikitext")
models=("lda" "ctm" "bertopic")
num_clusters=(50) #50 100)
sampling_strategies=("top_k" "weighted")
positive_samples=(10) #(3 7 10)

# First, count total combinations to run (including repeats)
total_combinations=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${num_clusters[@]}"; do
      for sampling in "${sampling_strategies[@]}"; do
        repeats=1
        [ "$sampling" == "weighted" ] && repeats=3
        for ((rep=1; rep<=repeats; rep++)); do
          for pos in "${positive_samples[@]}"; do
            total_combinations=$((total_combinations+1))
          done
        done
      done
    done
  done
done

current_combination=0
total_jobs=0
skipped_jobs=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${num_clusters[@]}"; do
      for sampling in "${sampling_strategies[@]}"; do
        repeats=1
        [ "$sampling" == "weighted" ] && repeats=3
        for ((rep=1; rep<=repeats; rep++)); do
          for pos in "${positive_samples[@]}"; do
            current_combination=$((current_combination+1))
            weights_file="output/topic_models/${model}_${dataset}_${k}.json"
            ann_file="output/annotations/${dataset}/${dataset}_${model}_${k}_${sampling}_pos${pos}_neg0_${rep}.json"
            result_dir="output/eval_result/${dataset}"
            result_file="${result_dir}/$(basename "$ann_file")"
            echo "[$current_combination/$total_combinations] Checking: $dataset $model $k $sampling pos$pos rep$rep"
            if [ -f "$ann_file" ] && [ -f "$weights_file" ]; then
              total_jobs=$((total_jobs+1))
              if [ -f "$result_file" ]; then
                echo "  Result already exists: $result_file, skipping. (Skipped so far: $skipped_jobs)"
                skipped_jobs=$((skipped_jobs+1))
                continue
              fi
              echo "  Evaluating: $dataset $model $k $sampling pos$pos rep$rep"
              mkdir -p "$result_dir"
              start_time=$(date +%s)
              python -m src.eval_label \
                --dataset "$dataset" \
                --saved_weights "$weights_file" \
                --topic_annotations "$ann_file" \
                2>&1 | tee -a logs/eval_label_subset.log
              end_time=$(date +%s)
              elapsed=$((end_time - start_time))
              echo "  Finished: $dataset $model $k $sampling pos$pos rep$rep in ${elapsed} seconds." | tee -a logs/eval_label_subset.log
            else
              echo "  Missing file: $ann_file or $weights_file, skipping. (Skipped so far: $skipped_jobs)"
              skipped_jobs=$((skipped_jobs+1))
            fi
          done
        done
      done
    done
  done
done

echo "Total combinations: $total_combinations" | tee -a logs/eval_label_subset.log
echo "Total jobs attempted: $total_jobs" | tee -a logs/eval_label_subset.log
echo "Jobs skipped (already done or missing files): $skipped_jobs" | tee -a logs/eval_label_subset.log
echo "Jobs actually run: $((total_jobs-skipped_jobs))" | tee -a logs/eval_label_subset.log