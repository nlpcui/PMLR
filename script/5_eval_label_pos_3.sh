#!/bin/bash

mkdir -p logs

datasets=("bill" "wikitext")
models=("lda" "ctm" "bertopic")
num_clusters=(25) #50 100)
sampling_strategies=("top_k" "weighted")
positive_samples=(3) #(3 7 10)

# python -m src.eval_label \
#   --dataset wikitext \
#   --saved_weights output/topic_models/lda_wikitext_25.json \
#   --topic_annotations output/annotations/wikitext/wikitext_lda_25_top_k_pos3_neg0_1.json \
#   2>&1 | tee -a logs/eval_label_subset.log

# Uncomment below to run all combinations:
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${num_clusters[@]}"; do
      for sampling in "${sampling_strategies[@]}"; do
        repeats=1
        [ "$sampling" == "weighted" ] && repeats=3
        for ((rep=1; rep<=repeats; rep++)); do
          for pos in "${positive_samples[@]}"; do
            weights_file="output/topic_models/${model}_${dataset}_${k}.json"
            ann_file="output/annotations/${dataset}/${dataset}_${model}_${k}_${sampling}_pos${pos}_neg0_${rep}.json"
            if [ -f "$ann_file" ] && [ -f "$weights_file" ]; then
              echo "Evaluating: $dataset $model $k $sampling pos$pos rep$rep"
              start_time=$(date +%s)
              python -m src.eval_label \
                --dataset "$dataset" \
                --saved_weights "$weights_file" \
                --topic_annotations "$ann_file" \
                2>&1 | tee -a logs/eval_label_subset_pos_3.log
              end_time=$(date +%s)
              elapsed=$((end_time - start_time))
              echo "Finished: $dataset $model $k $sampling pos$pos rep$rep in ${elapsed} seconds." | tee -a logs/eval_label_subset.log
            else
              echo "Missing file: $ann_file or $weights_file, skipping."
            fi
          done
        done
      done
    done
  done
done