#!/bin/bash

datasets=("bill" "wikitext")
models=("lda" "ctm" "bertopic")
num_clusters=(25 50 100)
sampling_strategies=("top_k" "weighted")
positive_samples=(3 7 10)

prompt_model_id='gpt-4o-mini'
temperature=0.5

# Count total jobs
total_jobs=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${num_clusters[@]}"; do
      for sampling in "${sampling_strategies[@]}"; do
        repeats=1
        [ "$sampling" == "weighted" ] && repeats=3
        for ((rep=1; rep<=repeats; rep++)); do
          for pos in "${positive_samples[@]}"; do
            total_jobs=$((total_jobs+1))
          done
        done
      done
    done
  done
done

job=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${num_clusters[@]}"; do
      for sampling in "${sampling_strategies[@]}"; do
        repeats=1
        [ "$sampling" == "weighted" ] && repeats=3
        for ((rep=1; rep<=repeats; rep++)); do
          for pos in "${positive_samples[@]}"; do
            # Compose sampled doc file path
            sample_file="output/sampled_docs/${dataset}_${model}_${k}_${sampling}_pos${pos}_neg0_${rep}.json"
            outdir="output/annotations/${dataset}"
            outfile="${outdir}/$(basename "$sample_file")"
            job=$((job+1))
            percent=$(( 100 * job / total_jobs ))
            if [ -f "$outfile" ]; then
              echo "[$job/$total_jobs - ${percent}%] Skipping $outfile (already exists)"
              continue
            fi
            echo "[$job/$total_jobs - ${percent}%] Running: $dataset $model $k $sampling pos$pos rep$rep"
            start_time=$(date +%s)
            python -m src.label_topic \
              --dataset "$dataset" \
              --prompt_model_id "$prompt_model_id" \
              --temperature "$temperature" \
              --samples "$sample_file"
            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            echo "Finished $outfile in ${elapsed} seconds."
          done
        done
      done
    done
  done
done

# run this to append the log file: bash script/4_label_topic_subset.sh 2>&1 | tee -a logs/label_topic_subset.log