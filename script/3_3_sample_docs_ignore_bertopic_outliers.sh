#!/bin/bash

mkdir -p output/sampled_docs
mkdir -p logs

datasets=("wikitext" "bill")
model="bertopic"
topic_nums=(25 50 100 200)
pos_k_combos=(3 7 10)
neg_k_combos=(0 3)
strategy="weighted"

LOGFILE="logs/sample_docs_errors_bertopic_ignore_outlier.log"
> $LOGFILE

job=0
for dataset in "${datasets[@]}"; do
  for k in "${topic_nums[@]}"; do
    for pos_k in "${pos_k_combos[@]}"; do
      for neg_k in "${neg_k_combos[@]}"; do
        runs=3
        for run in $(seq 1 $runs); do
          job=$((job+1))
          outpath="output/sampled_docs/${dataset}_${model}_${k}_${strategy}_pos${pos_k}_neg${neg_k}_${run}.json"
          if [ ! -f "$outpath" ]; then
            echo "[$job] $model | $dataset | K=$k | strategy=$strategy | pos_k=$pos_k | neg_k=$neg_k | run=$run"
            python -m src.sample_docs \
              --dataset=$dataset \
              --strategy=$strategy \
              --pos_k=$pos_k \
              --neg_k=$neg_k \
              --topic_model=$model \
              --num_topics=$k \
              --output_path=$outpath 2>> $LOGFILE
          fi
        done
      done
    done
  done
done

echo "Sampling complete. Check $LOGFILE for errors."