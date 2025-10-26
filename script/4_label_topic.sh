#!/bin/bash

mkdir -p logs

prompt_model_id='gpt-4o-mini'
temperature=0.5
samples_dir=output/sampled_docs

total_files=$(ls "$samples_dir"/*.json | wc -l)
current=0

for sample_file in "$samples_dir"/*.json; do
  current=$((current+1))
  fname=$(basename "$sample_file")
  if [[ $fname == wikitext* ]]; then
    dataset="wikitext"
  elif [[ $fname == bill* ]]; then
    dataset="bill"
  else
    echo "Unknown dataset for file $fname, skipping."
    continue
  fi
  echo "[$current/$total_files] Running label_topic for dataset: $dataset, file: $sample_file"
  start_time=$(date +%s)
  # Use 'time -p' for simple timing output
  { time -p python -m src.label_topic \
    --dataset=${dataset} \
    --prompt_model_id=${prompt_model_id} \
    --temperature=${temperature} \
    --samples="$sample_file" ; } 2>&1
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "Finished $fname in ${elapsed} seconds."
done | tee logs/label_topic.log

echo "Labeling complete. Check logs/label_topic.log for details."